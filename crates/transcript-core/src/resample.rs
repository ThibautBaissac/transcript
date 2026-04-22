use anyhow::{Context, Result};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use crate::WHISPER_SAMPLE_RATE;

/// Takes interleaved f32 samples at `input_sr` with `channels` channels, and returns a
/// mono, 16 kHz f32 buffer suitable for whisper-rs.
pub fn to_whisper_input(
    interleaved: &[f32],
    input_sr: u32,
    channels: u16,
) -> Result<Vec<f32>> {
    let mono = downmix_to_mono(interleaved, channels);
    if input_sr == WHISPER_SAMPLE_RATE {
        return Ok(mono);
    }
    resample_mono(&mono, input_sr, WHISPER_SAMPLE_RATE)
}

fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / ch;
    let mut out = Vec::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * ch;
        let sum: f32 = interleaved[base..base + ch].iter().sum();
        out.push(sum / ch as f32);
    }
    out
}

fn resample_mono(samples: &[f32], input_sr: u32, output_sr: u32) -> Result<Vec<f32>> {
    // Speech-tuned sinc resampler with modest sinc length — quality is plenty for ASR
    // and keeps CPU cost low on full-length recordings.
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = output_sr as f64 / input_sr as f64;

    // Process in fixed-size chunks at the input rate; 1024 frames keeps memory bounded.
    let chunk_size = 1024;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .with_context(|| "constructing rubato resampler")?;

    let mut output = Vec::with_capacity((samples.len() as f64 * ratio) as usize + chunk_size);
    let mut idx = 0usize;
    while idx + chunk_size <= samples.len() {
        let chunk = &samples[idx..idx + chunk_size];
        let out = resampler
            .process(&[chunk], None)
            .with_context(|| "resampling chunk")?;
        output.extend_from_slice(&out[0]);
        idx += chunk_size;
    }
    if idx < samples.len() {
        // Pad the trailing partial chunk with silence, then trim the output back
        // proportionally so we don't emit the silence we just added.
        let mut tail = samples[idx..].to_vec();
        tail.resize(chunk_size, 0.0);
        let out = resampler
            .process(&[&tail], None)
            .with_context(|| "resampling trailing chunk")?;
        let real = ((samples.len() - idx) as f64 * ratio).round() as usize;
        output.extend_from_slice(&out[0][..real.min(out[0].len())]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_already_16k_mono() {
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let out = to_whisper_input(&input, 16_000, 1).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn downmix_stereo() {
        // stereo frames (L, R) at 16k → mono 16k
        let input = vec![1.0, -1.0, 0.5, 0.5, 0.0, 0.0];
        let out = to_whisper_input(&input, 16_000, 2).unwrap();
        assert_eq!(out, vec![0.0, 0.5, 0.0]);
    }

    #[test]
    fn resample_48k_to_16k_rough_length() {
        let input: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = to_whisper_input(&input, 48_000, 1).unwrap();
        // 1 second at 48k → ~1 second at 16k → ~16000 samples, allow slack for tail
        assert!(
            (out.len() as i64 - 16_000).abs() < 1_500,
            "unexpected output length: {}",
            out.len()
        );
    }

    #[test]
    fn empty_input_returns_empty() {
        // No frames → downmix is empty → resampler is never constructed (early return).
        let out = to_whisper_input(&[], 16_000, 1).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn channels_zero_is_treated_as_mono() {
        // `channels.max(1)` guards against spec glitches; passthrough at 16k should survive.
        let input = vec![0.1, 0.2];
        let out = to_whisper_input(&input, 16_000, 0).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn stereo_resample_48k_to_16k_downmixes_and_shortens() {
        // 0.5s of stereo at 48k → 0.5s mono at 16k, so ~8000 samples (with tail slack).
        let frames = 24_000; // 0.5s at 48k
        let mut interleaved = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let v = (i as f32 * 0.02).sin();
            interleaved.push(v); // L
            interleaved.push(-v); // R (sum→0, so mono is zero-mean)
        }
        let out = to_whisper_input(&interleaved, 48_000, 2).unwrap();
        assert!(
            (out.len() as i64 - 8_000).abs() < 1_500,
            "unexpected output length: {}",
            out.len()
        );
    }

    #[test]
    fn resample_leaves_residual_chunk_runs_trailing_branch() {
        // Input size not a multiple of the 1024-frame chunk size → the "trailing partial
        // chunk" branch fires. 1500 frames forces one full chunk + a 476-frame tail.
        let input: Vec<f32> = (0..1500).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = to_whisper_input(&input, 48_000, 1).unwrap();
        // 1500 * 16/48 = 500, allow slack.
        assert!(
            (out.len() as i64 - 500).abs() < 200,
            "unexpected trailing output length: {}",
            out.len()
        );
    }
}
