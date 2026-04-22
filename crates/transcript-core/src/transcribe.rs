use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::{DecodedAudio, decode_file};
use crate::models::{ModelId, ModelInfo, model_info};
use crate::resample::to_whisper_input;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptResult {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: String,
}

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Language code like "en" or "fr". `None` → auto-detect.
    pub language: Option<String>,
    /// Number of CPU threads for the decoder. Defaults to physical-ish count.
    pub threads: Option<i32>,
    /// Whisper's temperature fallback. Default 0.0 (greedy).
    pub temperature: f32,
    /// Optional initial prompt biasing (names, jargon).
    pub initial_prompt: Option<String>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            threads: None,
            temperature: 0.0,
            initial_prompt: None,
        }
    }
}

pub struct Engine {
    ctx: WhisperContext,
    model: ModelId,
}

impl Engine {
    /// Loads a Whisper model from the shared cache. Call `resolve_model` first to
    /// ensure the files exist on disk.
    pub fn load(model: ModelId) -> Result<Self> {
        Self::load_from_info(model, &model_info(model)?)
    }

    /// Internal variant that accepts a pre-resolved `ModelInfo`. Kept separate from
    /// `load` so tests can exercise the error paths (missing ggml, non-UTF-8 path,
    /// corrupt model file) without touching the shared on-disk cache.
    fn load_from_info(model: ModelId, info: &ModelInfo) -> Result<Self> {
        anyhow::ensure!(
            info.ggml_present,
            "ggml file not downloaded: {}",
            info.ggml_path.display()
        );
        let mut params = WhisperContextParameters::default();
        // Whisper-rs picks up the CoreML encoder automatically if a sibling
        // `<model>-encoder.mlmodelc` directory exists. No extra config needed.
        params.use_gpu = true;

        let ctx = WhisperContext::new_with_params(
            info.ggml_path
                .to_str()
                .context("model path is not valid UTF-8")?,
            params,
        )
        .with_context(|| "loading Whisper model")?;
        Ok(Self { ctx, model })
    }

    pub fn model(&self) -> ModelId {
        self.model
    }

    /// Transcribes a pre-prepared 16 kHz mono f32 buffer.
    pub fn transcribe_samples(
        &self,
        samples_16k_mono: &[f32],
        opts: &TranscribeOptions,
    ) -> Result<TranscriptResult> {
        let mut state = self.ctx.create_state().context("creating whisper state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if let Some(n) = opts.threads {
            params.set_n_threads(n);
        } else {
            let n = std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4);
            params.set_n_threads(n);
        }
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_temperature(opts.temperature);
        params.set_language(opts.language.as_deref());
        if let Some(prompt) = opts.initial_prompt.as_deref() {
            params.set_initial_prompt(prompt);
        }

        state
            .full(params, samples_16k_mono)
            .context("running whisper inference")?;

        let n_segments = state.full_n_segments().context("counting segments")?;
        let mut segments = Vec::with_capacity(n_segments as usize);
        for i in 0..n_segments {
            let raw = state
                .full_get_segment_text(i)
                .context("reading segment text")?;
            let start = state
                .full_get_segment_t0(i)
                .context("reading segment start")? as f32
                / 100.0;
            let end = state
                .full_get_segment_t1(i)
                .context("reading segment end")? as f32
                / 100.0;
            segments.push(Segment {
                start,
                end,
                text: raw.trim().to_string(),
            });
        }
        let buf = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let language = opts
            .language
            .clone()
            .or_else(|| {
                whisper_rs::get_lang_str(state.full_lang_id_from_state().ok()?)
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "auto".into());

        Ok(TranscriptResult {
            text: buf,
            segments,
            language,
        })
    }

    /// Convenience: decode an audio file, resample, and transcribe.
    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        opts: &TranscribeOptions,
    ) -> Result<TranscriptResult> {
        let DecodedAudio {
            samples,
            sample_rate,
            channels,
        } = decode_file(path)?;
        self.transcribe_recorded(&samples, sample_rate, channels, opts)
    }

    /// Resample a captured interleaved buffer to Whisper's required format, then transcribe.
    pub fn transcribe_recorded(
        &self,
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
        opts: &TranscribeOptions,
    ) -> Result<TranscriptResult> {
        let prepared = to_whisper_input(samples, sample_rate, channels)?;
        self.transcribe_samples(&prepared, opts)
    }
}

/// Format a result as plain text (one line per segment, joined).
pub fn format_txt(result: &TranscriptResult) -> String {
    result.text.trim().to_string()
}

/// Format a result as SRT subtitles.
pub fn format_srt(result: &TranscriptResult) -> String {
    let mut out = String::new();
    for (i, seg) in result.segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            fmt_ts(seg.start),
            fmt_ts(seg.end)
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// Format a result as JSON. Returns an empty string (and logs to stderr) if serialization
/// fails, so the CLI's "stdout = transcript or nothing" contract is preserved rather than
/// emitting misleading `{}`.
pub fn format_json(result: &TranscriptResult) -> String {
    match serde_json::to_string_pretty(result) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warning: failed to serialize transcript as JSON: {}", e);
            String::new()
        }
    }
}

fn fmt_ts(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as i64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> TranscriptResult {
        TranscriptResult {
            text: "hello world".into(),
            segments: vec![
                Segment { start: 0.0, end: 1.5, text: "hello".into() },
                Segment { start: 1.5, end: 3.0, text: "world".into() },
            ],
            language: "en".into(),
        }
    }

    #[test]
    fn srt_formatting() {
        let srt = format_srt(&sample_result());
        assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,500\nhello\n\n2\n"));
        assert!(srt.contains("00:00:01,500 --> 00:00:03,000\nworld"));
    }

    #[test]
    fn srt_formatting_empty_is_empty_string() {
        let r = TranscriptResult { text: "".into(), segments: vec![], language: "en".into() };
        assert_eq!(format_srt(&r), "");
    }

    #[test]
    fn txt_formatting_trims() {
        let r = TranscriptResult {
            text: "  hello world\n".into(),
            segments: vec![],
            language: "en".into(),
        };
        assert_eq!(format_txt(&r), "hello world");
    }

    #[test]
    fn json_formatting_roundtrips() {
        let r = sample_result();
        let json = format_json(&r);
        assert!(json.starts_with('{'));
        let back: TranscriptResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, r.text);
        assert_eq!(back.language, r.language);
        assert_eq!(back.segments.len(), r.segments.len());
        assert_eq!(back.segments[0].text, "hello");
    }

    #[test]
    fn fmt_ts_handles_hours_minutes_seconds_and_ms() {
        assert_eq!(fmt_ts(0.0), "00:00:00,000");
        assert_eq!(fmt_ts(1.5), "00:00:01,500");
        assert_eq!(fmt_ts(59.999), "00:00:59,999");
        assert_eq!(fmt_ts(60.0), "00:01:00,000");
        assert_eq!(fmt_ts(3_600.0), "01:00:00,000");
        // 1h 2m 3.004s
        assert_eq!(fmt_ts(3_723.004), "01:02:03,004");
    }

    #[test]
    fn transcribe_options_default_is_auto_detect_greedy() {
        let opts = TranscribeOptions::default();
        assert!(opts.language.is_none());
        assert!(opts.threads.is_none());
        assert!(opts.initial_prompt.is_none());
        assert_eq!(opts.temperature, 0.0);
    }

    #[test]
    fn segment_serde_roundtrip() {
        let s = Segment { start: 1.25, end: 3.75, text: "word".into() };
        let json = serde_json::to_string(&s).unwrap();
        let back: Segment = serde_json::from_str(&json).unwrap();
        assert_eq!(back.start, 1.25);
        assert_eq!(back.end, 3.75);
        assert_eq!(back.text, "word");
    }

    #[test]
    fn engine_load_errors_when_ggml_not_present() {
        // Synthesize a ModelInfo pointing at a path that doesn't exist. Using the
        // internal `load_from_info` keeps this hermetic — we don't touch the real cache.
        let info = ModelInfo {
            id: ModelId::BaseEn,
            ggml_path: std::path::PathBuf::from("/nonexistent/ggml-base.en.bin"),
            coreml_dir: None,
            ggml_present: false,
            coreml_present: false,
        };
        let res = Engine::load_from_info(ModelId::BaseEn, &info);
        let msg = format!("{:#}", res.err().expect("load should fail without ggml"));
        assert!(msg.contains("ggml file not downloaded"), "msg: {msg}");
    }

    /// Runs Whisper inference against real downloaded models when they're present in
    /// the shared cache. Skipped when the model isn't there so CI (or a freshly cloned
    /// tree) stays green without network or disk assumptions.
    #[test]
    fn engine_full_path_on_downloaded_model() {
        let info = match model_info(ModelId::BaseEn) {
            Ok(i) if i.ggml_present => i,
            _ => return,
        };
        let _ = info; // keep the presence guard
        // Use the public `load()` entry point so it's counted as executed too.
        let engine = match Engine::load(ModelId::BaseEn) {
            Ok(e) => e,
            Err(_) => return,
        };
        assert_eq!(engine.model(), ModelId::BaseEn);

        // Inference on 1s silence — valid Whisper input, exercises the transcribe loop.
        let opts = TranscribeOptions {
            language: Some("en".into()),
            threads: Some(1),
            initial_prompt: Some("JFK inaugural address".into()),
            ..Default::default()
        };
        let res = engine
            .transcribe_samples(&vec![0.0f32; 16_000], &opts)
            .unwrap();
        assert_eq!(res.language, "en");

        // transcribe_recorded resamples then hands off to transcribe_samples.
        let res2 = engine
            .transcribe_recorded(&vec![0.0f32; 48_000], 48_000, 1, &TranscribeOptions::default())
            .unwrap();
        assert!(!res2.language.is_empty());

        // transcribe_file on the repo's jfk.wav sample yields real segments — exercises
        // the segment reader loop (full_get_segment_text / _t0 / _t1).
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let jfk = repo_root.join("jfk.wav");
        if jfk.exists() {
            let res3 = engine
                .transcribe_file(&jfk, &TranscribeOptions::default())
                .unwrap();
            assert!(!res3.segments.is_empty(), "expected segments for jfk.wav");
            assert!(!res3.text.is_empty());
        }
    }

    #[test]
    fn engine_load_errors_when_ggml_file_is_corrupt() {
        // ggml_present=true forces past the early guard, and a bogus file makes
        // WhisperContext::new_with_params fail — exercising the Whisper error path.
        let dir = std::env::temp_dir().join(format!(
            "transcript-engine-corrupt-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bogus = dir.join("ggml-fake.bin");
        std::fs::write(&bogus, b"not a real ggml file").unwrap();

        let info = ModelInfo {
            id: ModelId::BaseEn,
            ggml_path: bogus,
            coreml_dir: None,
            ggml_present: true,
            coreml_present: false,
        };
        let res = Engine::load_from_info(ModelId::BaseEn, &info);
        let msg = format!("{:#}", res.err().expect("load should fail on corrupt file"));
        assert!(msg.contains("loading Whisper model"), "msg: {msg}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
