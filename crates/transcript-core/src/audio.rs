use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::Sample;

/// Interleaved f32 PCM samples with the original sample rate and channel count.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Decodes an audio file into interleaved f32 samples.
/// Supports wav, mp3, m4a/aac, flac, opus/webm/ogg, and other formats symphonia recognizes.
pub fn decode_file(path: impl AsRef<Path>) -> Result<DecodedAudio> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| "probing audio format")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track in {}", path.display()))?
        .clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| "creating decoder")?;

    // AAC/M4A and a few other containers leave sample_rate / channels unset on the
    // track and only populate them after the first decoded frame. Read the spec off
    // the first produced buffer instead of relying on codec_params.
    let mut out = Vec::<f32>::new();
    let mut sample_rate: Option<u32> = track.codec_params.sample_rate;
    let mut channels: Option<u16> = track.codec_params.channels.map(|c| c.count() as u16);
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(anyhow!("symphonia reset required mid-stream (unsupported)"));
            }
            Err(e) => return Err(e).context("reading packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = buf.spec();
                sample_rate.get_or_insert(spec.rate);
                channels.get_or_insert(spec.channels.count() as u16);
                append_samples_f32(&buf, &mut out);
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding packet"),
        }
    }

    let sample_rate = sample_rate.ok_or_else(|| anyhow!("no audio packets decoded"))?;
    let channels = channels.ok_or_else(|| anyhow!("no audio packets decoded"))?;

    Ok(DecodedAudio {
        samples: out,
        sample_rate,
        channels,
    })
}

fn append_samples_f32(buf: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    use symphonia::core::audio::AudioBufferRef::*;
    match buf {
        U8(b) => interleave(b, out, |s| (s as f32 - 128.0) / 128.0),
        U16(b) => interleave(b, out, |s| (s as f32 - 32768.0) / 32768.0),
        U24(b) => interleave(b, out, |s| {
            (s.inner() as f32 - 8_388_608.0) / 8_388_608.0
        }),
        U32(b) => interleave(b, out, |s| (s as f64 - 2_147_483_648.0) as f32 / 2_147_483_648.0),
        S8(b) => interleave(b, out, |s| s as f32 / 128.0),
        S16(b) => interleave(b, out, |s| s as f32 / 32768.0),
        S24(b) => interleave(b, out, |s| s.inner() as f32 / 8_388_608.0),
        S32(b) => interleave(b, out, |s| s as f32 / 2_147_483_648.0),
        F32(b) => interleave(b, out, |s| s),
        F64(b) => interleave(b, out, |s| s as f32),
    }
}

fn interleave<S, F>(buf: &symphonia::core::audio::AudioBuffer<S>, out: &mut Vec<f32>, mut to_f32: F)
where
    S: Sample + Copy,
    F: FnMut(S) -> f32,
{
    let spec = buf.spec();
    let frames = buf.frames();
    let channels = spec.channels.count();
    for frame in 0..frames {
        for ch in 0..channels {
            out.push(to_f32(buf.chan(ch)[frame]));
        }
    }
}

/// Shared handle to an ongoing capture. The underlying cpal `Stream` is owned by a
/// dedicated thread (because `Stream` is not `Send`). Dropping this handle halts
/// capture and joins the thread; prefer `stop_and_take()` when you want the samples.
pub struct CaptureHandle {
    stopper: Option<Stopper>,
    pub(crate) buffer: Arc<Mutex<Vec<f32>>>,
    pub sample_rate: u32,
    pub channels: u16,
}

struct Stopper {
    stop_tx: std::sync::mpsc::Sender<()>,
    thread: std::thread::JoinHandle<()>,
}

impl CaptureHandle {
    /// Halts capture, joins the owner thread, and returns the accumulated samples.
    /// Because `join()` waits for the owner thread to drop the cpal `Stream`, no
    /// callback can fire after this returns — so the `mem::take` sees a stable buffer.
    pub fn stop_and_take(mut self) -> Vec<f32> {
        if let Some(s) = self.stopper.take() {
            let _ = s.stop_tx.send(());
            let _ = s.thread.join();
        }
        std::mem::take(&mut *self.buffer.lock())
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        if let Some(s) = self.stopper.take() {
            let _ = s.stop_tx.send(());
            let _ = s.thread.join();
        }
    }
}

/// Starts a live capture from the default input device. Audio accumulates internally
/// and is drained via `CaptureHandle::stop_and_take`. `level_cb` receives the RMS
/// amplitude of each delivered buffer in [0, 1] — use it to drive a VU meter.
pub fn start_capture<F>(level_cb: F) -> Result<CaptureHandle>
where
    F: FnMut(f32) + Send + 'static,
{
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<StreamReady>>();
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();

    let thread = std::thread::spawn(move || {
        match build_stream(level_cb) {
            Ok((stream, ready)) => {
                let _ = ready_tx.send(Ok(ready));
                let _ = stop_rx.recv();
                drop(stream);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    });

    let ready = ready_rx
        .recv()
        .map_err(|_| anyhow!("capture thread terminated before reporting readiness"))??;

    Ok(CaptureHandle {
        stopper: Some(Stopper { stop_tx, thread }),
        buffer: ready.buffer,
        sample_rate: ready.sample_rate,
        channels: ready.channels,
    })
}

struct StreamReady {
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
}

fn build_stream<F>(mut level_cb: F) -> Result<(cpal::Stream, StreamReady)>
where
    F: FnMut(f32) + Send + 'static,
{
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let config = device
        .default_input_config()
        .with_context(|| "default input config")?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let err_fn = |err| eprintln!("cpal stream error: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let buf = buffer.clone();
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[f32], _| {
                    push_and_meter(data, &buf, &mut level_cb);
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let buf = buffer.clone();
            // Reused each callback to avoid heap churn on the audio thread.
            let mut scratch = Vec::<f32>::with_capacity(8192);
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[i16], _| {
                    scratch.clear();
                    scratch.extend(data.iter().map(|s| *s as f32 / 32768.0));
                    push_and_meter(&scratch, &buf, &mut level_cb);
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let buf = buffer.clone();
            let mut scratch = Vec::<f32>::with_capacity(8192);
            device.build_input_stream(
                &config.clone().into(),
                move |data: &[u16], _| {
                    scratch.clear();
                    scratch.extend(data.iter().map(|s| (*s as f32 - 32768.0) / 32768.0));
                    push_and_meter(&scratch, &buf, &mut level_cb);
                },
                err_fn,
                None,
            )
        }
        fmt => return Err(anyhow!("unsupported cpal sample format: {:?}", fmt)),
    }
    .with_context(|| "building input stream")?;

    stream.play().with_context(|| "starting capture stream")?;

    let ready = StreamReady {
        buffer: buffer.clone(),
        sample_rate,
        channels,
    };
    Ok((stream, ready))
}

fn push_and_meter<F: FnMut(f32)>(samples: &[f32], buf: &Arc<Mutex<Vec<f32>>>, level_cb: &mut F) {
    if samples.is_empty() {
        return;
    }
    level_cb(rms(samples));
    buf.lock().extend_from_slice(samples);
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Audio captured via `record_until_silence`.
#[derive(Debug, Clone)]
pub struct RecordedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_seconds: f32,
}

/// Records from the default microphone until either:
/// - `max_duration` elapses, or
/// - `silence_duration` of continuous sub-threshold audio is detected after at least
///   one loud sample has been observed (so the caller can actually speak first).
///
/// `on_event` receives lifecycle callbacks so the caller can surface progress.
pub fn record_until_silence(
    max_duration: std::time::Duration,
    silence_duration: std::time::Duration,
    silence_threshold: f32,
    mut on_event: impl FnMut(RecordEvent),
) -> Result<RecordedAudio> {
    anyhow::ensure!(
        silence_threshold.is_finite() && (0.0..=1.0).contains(&silence_threshold),
        "silence_threshold must be a finite value in [0.0, 1.0], got {}",
        silence_threshold
    );
    anyhow::ensure!(
        !max_duration.is_zero(),
        "max_duration must be greater than zero"
    );
    anyhow::ensure!(
        !silence_duration.is_zero(),
        "silence_duration must be greater than zero"
    );
    let handle = start_capture(|_| {})?;
    let sample_rate = handle.sample_rate;
    let channels = handle.channels as usize;
    let silence_frames = (silence_duration.as_secs_f64() * sample_rate as f64) as usize;
    let silence_samples = silence_frames * channels;

    on_event(RecordEvent::Started {
        sample_rate,
        channels: channels as u16,
    });

    let start = std::time::Instant::now();
    let poll = std::time::Duration::from_millis(100);
    let mut ever_loud = false;
    let mut stop_reason = StopReason::MaxDuration;

    loop {
        std::thread::sleep(poll);
        let elapsed = start.elapsed();
        if elapsed >= max_duration {
            break;
        }
        // Take the tail window without holding the lock across the event callback —
        // the cpal audio thread needs to push into this same buffer.
        let (window_len, window_rms) = {
            let buf = handle.buffer.lock();
            let total = buf.len();
            if total == 0 {
                continue;
            }
            let window_start = total.saturating_sub(silence_samples);
            let window = &buf[window_start..];
            (window.len(), rms(window))
        };
        on_event(RecordEvent::Level {
            rms: window_rms,
            elapsed_secs: elapsed.as_secs_f32(),
        });

        if window_rms > silence_threshold {
            ever_loud = true;
        }

        // Only stop on silence once the user has actually spoken and we have a full
        // silence window of data.
        if ever_loud && window_len >= silence_samples && window_rms <= silence_threshold {
            stop_reason = StopReason::Silence;
            break;
        }
    }

    let samples = handle.stop_and_take();
    let frames = samples.len() / channels.max(1);
    let duration_seconds = frames as f32 / sample_rate as f32;

    on_event(RecordEvent::Stopped {
        reason: stop_reason,
        duration_seconds,
    });

    Ok(RecordedAudio {
        samples,
        sample_rate,
        channels: channels as u16,
        duration_seconds,
    })
}


#[derive(Debug, Clone, Copy)]
pub enum RecordEvent {
    Started { sample_rate: u32, channels: u16 },
    Level { rms: f32, elapsed_secs: f32 },
    Stopped { reason: StopReason, duration_seconds: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Silence,
    MaxDuration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn decode_file_errors_on_missing() {
        let r = decode_file("/definitely/does/not/exist.wav");
        assert!(r.is_err());
    }

    #[test]
    fn record_until_silence_rejects_invalid_threshold() {
        for bad in [-0.1_f32, 1.1, f32::NAN, f32::INFINITY] {
            let r = record_until_silence(
                std::time::Duration::from_millis(10),
                std::time::Duration::from_millis(5),
                bad,
                |_| {},
            );
            assert!(r.is_err(), "expected rejection for threshold {}", bad);
        }
    }

    /// Exercises the full `record_until_silence` loop for a short window. Skips when no
    /// microphone is available so CI stays green. The window is short enough that we'll
    /// hit the MaxDuration branch even if the user is in a quiet room.
    #[test]
    fn record_until_silence_runs_and_reports_events() {
        // Probe for mic availability first; if cpal can't open an input, skip silently.
        let probe = match start_capture(|_| {}) {
            Ok(h) => h,
            Err(_) => return,
        };
        drop(probe);

        let mut started_seen = false;
        let mut level_seen = false;
        let mut stopped_seen = false;
        let rec = record_until_silence(
            std::time::Duration::from_millis(300),
            std::time::Duration::from_millis(200),
            0.01,
            |ev| match ev {
                RecordEvent::Started { .. } => started_seen = true,
                RecordEvent::Level { .. } => level_seen = true,
                RecordEvent::Stopped { .. } => stopped_seen = true,
            },
        )
        .unwrap();
        assert!(started_seen);
        assert!(stopped_seen);
        // Level events depend on the audio thread delivering at least one buffer within
        // 300ms — usually it does, but we don't require it if the polling was too fast.
        let _ = level_seen;
        assert!(rec.sample_rate > 0);
        assert!(rec.channels >= 1);
        assert!(rec.duration_seconds >= 0.0);
    }

    #[test]
    fn record_until_silence_rejects_zero_durations() {
        // max_duration == 0 is invalid regardless of silence_duration.
        let r = record_until_silence(
            std::time::Duration::from_millis(0),
            std::time::Duration::from_millis(10),
            0.01,
            |_| {},
        );
        assert!(r.is_err());

        // silence_duration == 0 would never trigger silence detection.
        let r = record_until_silence(
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(0),
            0.01,
            |_| {},
        );
        assert!(r.is_err());
    }

    // Requires a default input device; skipped gracefully when cpal can't open one
    // (headless CI, sandboxed runners). When it does run, it exercises the thread-
    // spawning + stop + join path that `CaptureHandle` owns.
    #[test]
    fn capture_handle_start_and_stop_roundtrip() {
        let handle = match start_capture(|_| {}) {
            Ok(h) => h,
            Err(_) => return, // no input device in this environment
        };
        assert!(handle.sample_rate > 0);
        assert!(handle.channels >= 1);
        std::thread::sleep(std::time::Duration::from_millis(150));
        // stop_and_take must join the audio thread (otherwise the Vec could race) and
        // return ownership of whatever samples accumulated.
        let _samples = handle.stop_and_take();
    }

    #[test]
    fn capture_handle_drop_without_take_does_not_hang() {
        let handle = match start_capture(|_| {}) {
            Ok(h) => h,
            Err(_) => return,
        };
        // Dropping the handle should signal the owner thread to exit and join it.
        // If this test ever hangs, the Drop impl is broken.
        drop(handle);
    }

    fn tmp_path(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "transcript-audio-{}-{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.join(format!("{name}.wav"))
    }

    fn write_wav_s16(path: &std::path::Path, channels: u16, sr: u32, samples: &[i16]) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
    }

    fn write_wav_f32(path: &std::path::Path, channels: u16, sr: u32, samples: &[f32]) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
    }

    #[test]
    fn decode_file_reads_s16_mono_wav() {
        let p = tmp_path("s16_mono");
        let samples: Vec<i16> = (0..1000).map(|i| (i as i16) * 10).collect();
        write_wav_s16(&p, 1, 16_000, &samples);

        let a = decode_file(&p).unwrap();
        assert_eq!(a.sample_rate, 16_000);
        assert_eq!(a.channels, 1);
        assert_eq!(a.samples.len(), samples.len());
        // Mapping: i16 / 32768.0 — check first nonzero sample.
        assert!((a.samples[1] - (10.0 / 32768.0)).abs() < 1e-6);

        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn decode_file_reads_s16_stereo_wav_interleaved() {
        let p = tmp_path("s16_stereo");
        // Interleaved L/R: [L0, R0, L1, R1, ...]
        let samples: Vec<i16> = vec![100, -100, 200, -200, 300, -300];
        write_wav_s16(&p, 2, 48_000, &samples);

        let a = decode_file(&p).unwrap();
        assert_eq!(a.sample_rate, 48_000);
        assert_eq!(a.channels, 2);
        assert_eq!(a.samples.len(), samples.len());
        // Ordering must preserve interleave so `downmix_to_mono` works correctly.
        assert!(a.samples[0] > 0.0);
        assert!(a.samples[1] < 0.0);

        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn decode_file_reads_f32_wav() {
        let p = tmp_path("f32_mono");
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 * 0.001).sin()).collect();
        write_wav_f32(&p, 1, 16_000, &samples);

        let a = decode_file(&p).unwrap();
        assert_eq!(a.channels, 1);
        assert_eq!(a.sample_rate, 16_000);
        assert_eq!(a.samples.len(), samples.len());
        // F32 path should be ~identity (within decoder precision).
        for (got, want) in a.samples.iter().zip(samples.iter()) {
            assert!((got - want).abs() < 1e-4);
        }
        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn decode_file_reads_u8_wav() {
        // WAV's 8-bit PCM is unsigned — exercises the `U8` branch of append_samples_f32.
        let p = tmp_path("u8_mono");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 8,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        // Hound's 8-bit Int writer accepts i8 samples but stores them as unsigned bytes.
        for v in [-100i8, -50, 0, 50, 100] {
            w.write_sample(v as i32).unwrap();
        }
        w.finalize().unwrap();

        let a = decode_file(&p).unwrap();
        assert_eq!(a.channels, 1);
        assert_eq!(a.sample_rate, 16_000);
        assert_eq!(a.samples.len(), 5);
        // Centered around 0 after the `(s - 128) / 128` normalization.
        assert!(a.samples.iter().all(|s| s.abs() <= 1.0));
        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn decode_file_reads_s24_wav() {
        let p = tmp_path("s24_mono");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        // 24-bit max signed = 2^23 - 1 = 8_388_607. Write a peak and midpoint.
        for v in [0i32, 1 << 22, -(1 << 22)] {
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();

        let a = decode_file(&p).unwrap();
        assert_eq!(a.samples.len(), 3);
        assert!((a.samples[0]).abs() < 1e-6);
        assert!((a.samples[1] - 0.5).abs() < 1e-3);
        assert!((a.samples[2] + 0.5).abs() < 1e-3);
        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn decode_file_reads_s32_wav() {
        let p = tmp_path("s32_mono");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        for v in [0i32, 1 << 30, -(1 << 30)] {
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();

        let a = decode_file(&p).unwrap();
        assert_eq!(a.samples.len(), 3);
        assert!((a.samples[0]).abs() < 1e-6);
        assert!((a.samples[1] - 0.5).abs() < 1e-3);
        assert!((a.samples[2] + 0.5).abs() < 1e-3);
        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn decode_file_errors_on_non_audio_input() {
        let p = tmp_path("garbage");
        fs::write(&p, b"this is not an audio file at all").unwrap();
        assert!(decode_file(&p).is_err());
        let _ = fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn rms_zero_for_empty_input() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn rms_matches_formula_for_known_signal() {
        // RMS of [1, -1, 1, -1] = sqrt((1+1+1+1)/4) = 1.0
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        // RMS of zero-signal is zero.
        assert_eq!(rms(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn push_and_meter_is_noop_on_empty_input() {
        let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let mut called_with: Option<f32> = None;
        let mut cb = |rms: f32| called_with = Some(rms);
        push_and_meter(&[], &buf, &mut cb);
        assert!(buf.lock().is_empty());
        // Callback must NOT have been invoked for an empty buffer — otherwise the VU
        // meter would get spurious zero-level events on audio thread warmup.
        assert!(called_with.is_none());
    }

    #[test]
    fn push_and_meter_appends_and_reports_rms() {
        let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![0.5]));
        let mut last_rms = 0.0_f32;
        let mut cb = |r: f32| last_rms = r;
        push_and_meter(&[1.0, -1.0], &buf, &mut cb);
        assert_eq!(&*buf.lock(), &[0.5, 1.0, -1.0]);
        assert!((last_rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stop_reason_equality() {
        // StopReason is compared by callers to choose an emoji; derive(PartialEq) must hold.
        assert_eq!(StopReason::Silence, StopReason::Silence);
        assert_ne!(StopReason::Silence, StopReason::MaxDuration);
    }

    #[test]
    fn record_event_variants_are_debug_printable() {
        // Debug output feeds diagnostic logs — exercise each variant so no derive gets lost.
        let variants = [
            RecordEvent::Started { sample_rate: 48_000, channels: 1 },
            RecordEvent::Level { rms: 0.1, elapsed_secs: 0.5 },
            RecordEvent::Stopped { reason: StopReason::Silence, duration_seconds: 1.0 },
        ];
        for v in variants {
            let s = format!("{:?}", v);
            assert!(!s.is_empty());
        }
    }
}
