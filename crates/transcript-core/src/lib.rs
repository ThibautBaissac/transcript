pub mod audio;
pub mod models;
pub mod resample;
pub mod transcribe;

pub use models::{
    DownloadStage, ModelId, ModelInfo, cache_dir, model_info, resolve_model, transcripts_dir,
};
pub use transcribe::{
    Engine, Segment, TranscribeOptions, TranscriptResult, format_json, format_srt, format_txt,
};

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
