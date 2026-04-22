use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use transcript_core::audio::{CaptureHandle, start_capture};
use transcript_core::{
    Engine, ModelId, ModelInfo, TranscribeOptions, TranscriptResult, model_info, resolve_model,
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
    let rec = state
        .last_recording
        .lock()
        .take()
        .ok_or_else(|| "no recording available".to_string())?;
    let engine_slot = state.engine.clone();
    run_blocking(move || {
        let engine = ensure_engine(&engine_slot, model)?;
        engine.transcribe_recorded(&rec.samples, rec.sample_rate, rec.channels, &opts(lang))
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
    model: String,
    source: TranscriptSource,
    duration_secs: Option<f32>,
    result: TranscriptResult,
) -> Result<TranscriptRecord, String> {
    run_blocking(move || transcripts::save(model, source, duration_secs, result)).await
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
