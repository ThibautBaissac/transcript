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

    #[test]
    fn decode_file_errors_on_missing() {
        let r = decode_file("/definitely/does/not/exist.wav");
        assert!(r.is_err());
    }
}
