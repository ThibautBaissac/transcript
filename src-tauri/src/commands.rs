use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use transcript_core::audio::{CaptureHandle, start_capture};
use transcript_core::{
    Engine, ModelId, ModelInfo, TranscribeOptions, TranscriptResult, format_srt, format_txt,
    model_info, resolve_model,
};

use crate::transcripts::{self, TranscriptRecord, TranscriptSource, TranscriptSummary};

const LEVEL_MIN_INTERVAL_MS: u64 = 33; // ~30 Hz — one canvas frame.

#[derive(Default)]
pub struct AppState {
    capture: Arc<Mutex<Option<CaptureHandle>>>,
    engine: Arc<Mutex<Option<Engine>>>,
    last_recording: Arc<Mutex<Option<StoredRecording>>>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: ModelId,
    pub display: String,
    pub ggml_present: bool,
    pub coreml_present: bool,
}

impl ModelEntry {
    fn for_id(id: ModelId) -> Self {
        let info = model_info(id).ok();
        Self::build(id, info.as_ref())
    }

    fn build(id: ModelId, info: Option<&ModelInfo>) -> Self {
        Self {
            id,
            display: id.display_name().to_string(),
            ggml_present: info.is_some_and(|i| i.ggml_present),
            coreml_present: info.is_some_and(|i| i.coreml_present),
        }
    }
}

#[tauri::command]
pub fn list_models() -> Vec<ModelEntry> {
    ModelId::ALL.iter().copied().map(ModelEntry::for_id).collect()
}

#[tauri::command]
pub fn model_status(id: ModelId) -> Result<ModelEntry, String> {
    let info = model_info(id).map_err(|e| e.to_string())?;
    Ok(ModelEntry::build(id, Some(&info)))
}

#[derive(Debug, Serialize, Clone)]
pub struct DownloadProgress {
    pub model: ModelId,
    pub stage: &'static str,
    pub downloaded: u64,
    pub total: Option<u64>,
}

#[tauri::command]
pub async fn download_model(id: ModelId, app: AppHandle) -> Result<ModelEntry, String> {
    let app_for_cb = app.clone();
    let info = resolve_model(id, move |stage, downloaded, total| {
        let _ = app_for_cb.emit(
            "download-progress",
            DownloadProgress {
                model: id,
                stage: stage.as_str(),
                downloaded,
                total,
            },
        );
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(ModelEntry::build(info.id, Some(&info)))
}

#[derive(Debug, Serialize, Clone)]
pub struct RecordingLevel {
    pub rms: f32,
}

#[tauri::command]
pub fn start_recording(state: State<'_, AppState>, app: AppHandle) -> Result<(), String> {
    let mut guard = state.capture.lock();
    if guard.is_some() {
        return Err("already recording".into());
    }
    // Throttle: the cpal audio callback fires at buffer rate (often 100+ Hz). Emitting
    // a Tauri event per buffer floods the IPC channel. We cap at ~30 Hz using a single
    // shared AtomicU64 of elapsed-ms-at-last-emit — no lock on the audio thread.
    let start = Instant::now();
    let last_emit_ms = Arc::new(AtomicU64::new(0));
    let app_for_cb = app.clone();
    let last_emit_for_cb = last_emit_ms.clone();
    let handle = start_capture(move |rms| {
        let now_ms = start.elapsed().as_millis() as u64;
        let last = last_emit_for_cb.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < LEVEL_MIN_INTERVAL_MS {
            return;
        }
        last_emit_for_cb.store(now_ms, Ordering::Relaxed);
        let _ = app_for_cb.emit("recording-level", RecordingLevel { rms });
    })
    .map_err(|e| e.to_string())?;
    *guard = Some(handle);
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct RecordingInfo {
    pub duration_seconds: f32,
    pub sample_rate: u32,
    pub channels: u16,
}

#[tauri::command]
pub fn stop_recording(state: State<'_, AppState>) -> Result<RecordingInfo, String> {
    let handle = state
        .capture
        .lock()
        .take()
        .ok_or_else(|| "not recording".to_string())?;
    let sample_rate = handle.sample_rate;
    let channels = handle.channels;
    let samples = handle.stop_and_take();
    let frames = samples.len() / channels as usize;
    let duration_seconds = frames as f32 / sample_rate as f32;
    *state.last_recording.lock() = Some(StoredRecording {
        samples,
        sample_rate,
        channels,
    });
    Ok(RecordingInfo {
        duration_seconds,
        sample_rate,
        channels,
    })
}

struct StoredRecording {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

#[tauri::command]
pub async fn transcribe_current_recording(
    state: State<'_, AppState>,
    model: ModelId,
    lang: Option<String>,
) -> Result<TranscriptResult, String> {
    // Leave the buffer in place so a subsequent save_transcript can persist it as
    // a sibling WAV.
    let (samples, sample_rate, channels) = {
        let guard = state.last_recording.lock();
        let rec = guard
            .as_ref()
            .ok_or_else(|| "no recording available".to_string())?;
        (rec.samples.clone(), rec.sample_rate, rec.channels)
    };
    let engine_slot = state.engine.clone();
    run_blocking(move || {
        let engine = ensure_engine(&engine_slot, model)?;
        engine.transcribe_recorded(&samples, sample_rate, channels, &opts(lang))
    })
    .await
}

#[tauri::command]
pub async fn transcribe_file(
    state: State<'_, AppState>,
    path: String,
    model: ModelId,
    lang: Option<String>,
) -> Result<TranscriptResult, String> {
    let engine_slot = state.engine.clone();
    let path = PathBuf::from(path);
    run_blocking(move || {
        let engine = ensure_engine(&engine_slot, model)?;
        engine.transcribe_file(&path, &opts(lang))
    })
    .await
}

#[tauri::command]
pub fn format_transcript(format: String, result: TranscriptResult) -> Result<String, String> {
    match format.as_str() {
        "txt" => Ok(format_txt(&result)),
        "srt" => Ok(format_srt(&result)),
        _ => Err(format!("unknown format: {format}")),
    }
}

fn opts(lang: Option<String>) -> TranscribeOptions {
    TranscribeOptions {
        language: lang,
        ..Default::default()
    }
}

/// Returns a lifetime-bound handle to the cached engine, loading the requested model
/// if the slot is empty or currently holds a different one.
fn ensure_engine(
    slot: &Mutex<Option<Engine>>,
    model: ModelId,
) -> anyhow::Result<parking_lot::MappedMutexGuard<'_, Engine>> {
    let mut guard = slot.lock();
    if guard.as_ref().map(|e| e.model() != model).unwrap_or(true) {
        *guard = Some(Engine::load(model)?);
    }
    Ok(parking_lot::MutexGuard::map(guard, |opt| {
        opt.as_mut().expect("engine loaded above")
    }))
}

async fn run_blocking<T, F>(f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_transcript(
    state: State<'_, AppState>,
    model: String,
    source: TranscriptSource,
    duration_secs: Option<f32>,
    result: TranscriptResult,
) -> Result<TranscriptRecord, String> {
    let audio = if matches!(source, TranscriptSource::Recording) {
        state.last_recording.lock().take().map(|r| transcripts::SavedAudio {
            samples: r.samples,
            sample_rate: r.sample_rate,
            channels: r.channels,
        })
    } else {
        None
    };
    run_blocking(move || transcripts::save(model, source, duration_secs, result, audio)).await
}

#[tauri::command]
pub async fn list_transcripts() -> Result<Vec<TranscriptSummary>, String> {
    run_blocking(transcripts::list).await
}

#[tauri::command]
pub async fn load_transcript(id: String) -> Result<TranscriptRecord, String> {
    run_blocking(move || transcripts::load(&id)).await
}

#[tauri::command]
pub async fn delete_transcript(id: String) -> Result<(), String> {
    run_blocking(move || transcripts::delete(&id)).await
}

#[tauri::command]
pub async fn get_transcript_audio_path(
    id: String,
    source: TranscriptSource,
) -> Result<Option<String>, String> {
    run_blocking(move || {
        Ok(transcripts::audio_path_if_exists(&id, &source)?
            .map(|p| p.to_string_lossy().into_owned()))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn model_entry_build_copies_flags_from_info() {
        let info = ModelInfo {
            id: ModelId::BaseEn,
            ggml_path: PathBuf::from("/tmp/ggml-base.en.bin"),
            coreml_dir: Some(PathBuf::from("/tmp/ggml-base.en-encoder.mlmodelc")),
            ggml_present: true,
            coreml_present: false,
        };
        let e = ModelEntry::build(ModelId::BaseEn, Some(&info));
        assert_eq!(e.id, ModelId::BaseEn);
        assert_eq!(e.display, ModelId::BaseEn.display_name());
        assert!(e.ggml_present);
        assert!(!e.coreml_present);
    }

    #[test]
    fn model_entry_build_with_none_info_reports_absent() {
        let e = ModelEntry::build(ModelId::SmallEn, None);
        assert!(!e.ggml_present);
        assert!(!e.coreml_present);
        assert!(!e.display.is_empty());
    }

    #[test]
    fn model_entry_for_id_matches_model_info() {
        // Must agree with the filesystem view from `model_info` — if model_info says
        // the ggml is there, ModelEntry should too.
        let info = model_info(ModelId::BaseEn).unwrap();
        let e = ModelEntry::for_id(ModelId::BaseEn);
        assert_eq!(e.id, info.id);
        assert_eq!(e.ggml_present, info.ggml_present);
        assert_eq!(e.coreml_present, info.coreml_present);
    }

    #[test]
    fn list_models_returns_entry_per_variant() {
        let models = list_models();
        assert_eq!(models.len(), ModelId::ALL.len());
        let got: Vec<ModelId> = models.iter().map(|m| m.id).collect();
        let want: Vec<ModelId> = ModelId::ALL.to_vec();
        assert_eq!(got, want);
    }

    #[test]
    fn model_status_returns_entry_for_known_id() {
        let e = model_status(ModelId::BaseEn).unwrap();
        assert_eq!(e.id, ModelId::BaseEn);
    }

    #[test]
    fn opts_wraps_lang_preserving_other_defaults() {
        let o = opts(Some("fr".into()));
        assert_eq!(o.language.as_deref(), Some("fr"));
        assert_eq!(o.temperature, 0.0);
        assert!(o.threads.is_none());
        assert!(o.initial_prompt.is_none());

        let o = opts(None);
        assert!(o.language.is_none());
    }

    #[test]
    fn download_progress_serializes_with_stage_string() {
        // Frontend consumes `stage` as a string tag — lock the JSON shape.
        let p = DownloadProgress {
            model: ModelId::BaseEn,
            stage: "ggml",
            downloaded: 42,
            total: Some(100),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"stage\":\"ggml\""));
        assert!(json.contains("\"downloaded\":42"));
        assert!(json.contains("\"total\":100"));
    }

    #[test]
    fn recording_level_serializes_as_rms_field() {
        let r = RecordingLevel { rms: 0.25 };
        assert_eq!(serde_json::to_string(&r).unwrap(), r#"{"rms":0.25}"#);
    }

    #[test]
    fn recording_info_serializes_all_fields() {
        let i = RecordingInfo {
            duration_seconds: 1.5,
            sample_rate: 48_000,
            channels: 2,
        };
        let json = serde_json::to_string(&i).unwrap();
        assert!(json.contains("duration_seconds"));
        assert!(json.contains("sample_rate"));
        assert!(json.contains("channels"));
    }

    #[test]
    fn format_transcript_dispatches_by_name() {
        use transcript_core::Segment;
        let result = TranscriptResult {
            text: "hello world".into(),
            segments: vec![Segment { start: 0.0, end: 1.5, text: "hello".into() }],
            language: "en".into(),
        };

        assert_eq!(
            format_transcript("txt".into(), result.clone()).unwrap(),
            "hello world"
        );

        let srt = format_transcript("srt".into(), result.clone()).unwrap();
        assert!(srt.contains(" --> "), "srt: {srt}");
        assert!(srt.contains("hello"), "srt: {srt}");

        let err = format_transcript("xml".into(), result).unwrap_err();
        assert!(err.contains("xml"), "err: {err}");
    }

    #[tokio::test]
    async fn run_blocking_returns_ok_on_success() {
        let v = run_blocking(|| Ok::<_, anyhow::Error>(42)).await.unwrap();
        assert_eq!(v, 42);
    }

    #[tokio::test]
    async fn run_blocking_surfaces_error_as_string() {
        let err = run_blocking(|| {
            Err::<(), _>(anyhow::anyhow!("boom"))
        })
        .await
        .unwrap_err();
        assert!(err.contains("boom"));
    }
}
