use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::{DecodedAudio, decode_file};
use crate::models::{ModelId, model_info};
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
        let info = model_info(model)?;
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

    #[test]
    fn srt_formatting() {
        let r = TranscriptResult {
            text: "hello world".into(),
            segments: vec![
                Segment { start: 0.0, end: 1.5, text: "hello".into() },
                Segment { start: 1.5, end: 3.0, text: "world".into() },
            ],
            language: "en".into(),
        };
        let srt = format_srt(&r);
        assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,500\nhello\n\n2\n"));
    }
}
