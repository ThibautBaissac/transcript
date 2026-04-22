use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use transcript_core::audio::{RecordEvent, StopReason, record_until_silence};
use transcript_core::{
    DownloadStage, Engine, ModelId, TranscribeOptions, TranscriptResult, format_json, format_srt,
    format_txt, resolve_model,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    Txt,
    Srt,
    Json,
}

#[derive(Parser, Debug)]
#[command(
    name = "transcript",
    about = "Local Whisper transcription (whisper.cpp + CoreML). Pass an audio file, or use --record to capture from the mic.",
    version
)]
struct Args {
    /// Path to an audio file (wav, mp3, m4a, flac, opus, webm, ogg, ...). Omit with --record.
    audio: Option<PathBuf>,

    /// Capture from the default microphone instead of reading a file.
    #[arg(long, conflicts_with = "audio")]
    record: bool,

    /// Maximum recording duration in seconds (only with --record).
    #[arg(long, default_value = "60", requires = "record")]
    max_secs: f64,

    /// Silence duration in seconds that triggers auto-stop (only with --record).
    #[arg(long, default_value = "2.5", requires = "record")]
    silence_secs: f64,

    /// RMS amplitude below which audio is considered silent (0.0 - 1.0).
    #[arg(long, default_value = "0.015", requires = "record")]
    silence_threshold: f32,

    /// Suppress the start/stop audible cues (only with --record).
    #[arg(long, requires = "record")]
    no_cue: bool,

    /// Model to use. One of: base.en, small.en, medium.en, large-v3, large-v3-turbo.
    #[arg(short, long, default_value = "large-v3-turbo")]
    model: String,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = Format::Txt)]
    format: Format,

    /// Language code (e.g. en, fr). Omit to auto-detect.
    #[arg(short, long)]
    lang: Option<String>,

    /// Initial prompt to bias decoding (names, jargon).
    #[arg(long)]
    prompt: Option<String>,

    /// Threads for the decoder. Defaults to physical cores.
    #[arg(long)]
    threads: Option<i32>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    if !args.record && args.audio.is_none() {
        bail!("provide an <AUDIO> path or use --record");
    }

    let model = ModelId::parse(&args.model)?;

    // Ensure model is available locally. Progress on stderr so stdout stays pipe-clean.
    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::with_template("{msg:>14} {bar:40.cyan/blue} {bytes}/{total_bytes} {eta}")
            .unwrap()
            .progress_chars("=> "),
    );
    let mut current_stage: Option<DownloadStage> = None;
    resolve_model(model, |stage, downloaded, total| {
        if current_stage != Some(stage) {
            current_stage = Some(stage);
            bar.set_message(stage.as_str());
            bar.set_length(total.unwrap_or(0));
            bar.set_position(0);
        }
        bar.set_position(downloaded);
    })
    .await
    .with_context(|| "resolving model")?;
    bar.finish_and_clear();

    let engine = tokio::task::spawn_blocking(move || Engine::load(model))
        .await
        .context("engine thread panicked")??;

    let opts = TranscribeOptions {
        language: args.lang,
        threads: args.threads,
        temperature: 0.0,
        initial_prompt: args.prompt,
    };

    let result = if args.record {
        eprintln!(
            "🎙  Recording (speak now — auto-stops after {:.1}s silence, max {:.0}s)…",
            args.silence_secs, args.max_secs
        );
        if !args.no_cue {
            notify(
                "Transcript",
                "Recording — speak now",
                "Auto-stops on silence",
            );
            play_cue("/System/Library/Sounds/Tink.aiff");
        }
        let max_duration = Duration::from_secs_f64(args.max_secs);
        let silence_duration = Duration::from_secs_f64(args.silence_secs);
        let threshold = args.silence_threshold;
        let show_cues = !args.no_cue;
        let recorded = tokio::task::spawn_blocking(move || {
            record_until_silence(
                max_duration,
                silence_duration,
                threshold,
                |ev| if let RecordEvent::Stopped { reason, duration_seconds } = ev {
                    let how = stop_reason_label(reason);
                    if show_cues {
                        play_cue("/System/Library/Sounds/Pop.aiff");
                        notify(
                            "Transcript",
                            &format!("Stopped — {:.1}s ({})", duration_seconds, how),
                            "Transcribing…",
                        );
                    }
                    eprintln!("✓ captured {:.1}s ({})", duration_seconds, how);
                },
            )
        })
        .await
        .context("record thread panicked")??;
        eprintln!("transcribing…");
        tokio::task::spawn_blocking(move || {
            engine.transcribe_recorded(
                &recorded.samples,
                recorded.sample_rate,
                recorded.channels,
                &opts,
            )
        })
        .await
        .context("transcription thread panicked")??
    } else {
        let audio_path = args.audio.unwrap();
        eprintln!("transcribing {}…", audio_path.display());
        tokio::task::spawn_blocking(move || engine.transcribe_file(&audio_path, &opts))
            .await
            .context("transcription thread panicked")??
    };

    let output = format_output(args.format, &result);

    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    writeln!(lock, "{}", output)?;
    Ok(())
}

/// Maps a CLI `Format` choice to a string rendering of the transcript. Kept as a pure
/// function so the branch selection can be unit tested without running the whole pipeline.
fn format_output(fmt: Format, result: &TranscriptResult) -> String {
    match fmt {
        Format::Txt => format_txt(result),
        Format::Srt => format_srt(result),
        Format::Json => format_json(result),
    }
}

/// Maps a `StopReason` to a short human label used in the CLI's status message and
/// macOS notification subtitle.
fn stop_reason_label(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Silence => "silence",
        StopReason::MaxDuration => "max duration",
    }
}

/// Plays a short system sound synchronously via `afplay` — used to signal that the
/// microphone has started capturing, so the user knows when to speak. Blocks for ~200 ms.
/// Silently ignores failures (missing file, afplay absent).
fn play_cue(path: &str) {
    let _ = std::process::Command::new("afplay")
        .arg(path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// Shows a macOS Notification Center banner via `osascript`. Non-blocking (`spawn`) so
/// the notification appears concurrently with the start of capture, not after. Silently
/// ignores failures (user hasn't granted notification permission, osascript absent).
fn notify(title: &str, message: &str, subtitle: &str) {
    // AppleScript quoting: escape double quotes in user-supplied strings.
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        "display notification \"{}\" with title \"{}\" subtitle \"{}\"",
        esc(message),
        esc(title),
        esc(subtitle),
    );
    let _ = std::process::Command::new("osascript")
        .arg("-e")
        .arg(script)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_command_validates() {
        // `debug_assert` inside clap fires if the attribute tree is malformed. This
        // guards against a future arg refactor silently breaking the CLI.
        Args::command().debug_assert();
    }

    #[test]
    fn default_args_pick_turbo_model_and_txt_format() {
        let args = Args::try_parse_from(["transcript", "file.wav"]).unwrap();
        assert_eq!(args.model, "large-v3-turbo");
        assert!(matches!(args.format, Format::Txt));
        assert_eq!(args.audio.as_deref(), Some(std::path::Path::new("file.wav")));
        assert!(!args.record);
    }

    #[test]
    fn record_and_audio_are_mutually_exclusive() {
        let err = Args::try_parse_from(["transcript", "--record", "file.wav"]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot be used with") || msg.contains("conflicts"), "msg: {msg}");
    }

    #[test]
    fn format_json_is_parsed() {
        let args = Args::try_parse_from(["transcript", "-f", "json", "file.wav"]).unwrap();
        assert!(matches!(args.format, Format::Json));
    }

    #[test]
    fn format_srt_is_parsed() {
        let args = Args::try_parse_from(["transcript", "--format", "srt", "file.wav"]).unwrap();
        assert!(matches!(args.format, Format::Srt));
    }

    #[test]
    fn record_mode_accepts_no_audio_path() {
        let args = Args::try_parse_from(["transcript", "--record"]).unwrap();
        assert!(args.record);
        assert!(args.audio.is_none());
    }

    #[test]
    fn play_cue_silent_on_missing_file_does_not_panic() {
        // Non-existent path → `afplay` exits non-zero → we ignore it silently.
        play_cue("/definitely/not/a/sound.aiff");
    }

    #[test]
    fn notify_does_not_panic_on_quotes_or_backslashes() {
        // Exercises the AppleScript escape path. We can't assert the notification was
        // shown (it's fire-and-forget spawn), but we can assert we didn't panic.
        notify("ti\"tle", "mes\\sage", "sub\"ti\\tle");
    }

    fn sample_result() -> TranscriptResult {
        TranscriptResult {
            text: "hello".into(),
            segments: vec![transcript_core::Segment {
                start: 0.0,
                end: 1.0,
                text: "hello".into(),
            }],
            language: "en".into(),
        }
    }

    #[test]
    fn format_output_dispatches_by_variant() {
        let r = sample_result();
        assert_eq!(format_output(Format::Txt, &r), "hello");
        let srt = format_output(Format::Srt, &r);
        assert!(srt.contains("00:00:00,000 --> 00:00:01,000"));
        let json = format_output(Format::Json, &r);
        assert!(json.starts_with('{'));
        assert!(json.contains("hello"));
    }

    #[test]
    fn stop_reason_labels_are_stable() {
        assert_eq!(stop_reason_label(StopReason::Silence), "silence");
        assert_eq!(stop_reason_label(StopReason::MaxDuration), "max duration");
    }
}
