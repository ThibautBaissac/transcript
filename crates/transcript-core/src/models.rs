use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const APP_DIR_NAME: &str = "com.thibautbaissac.transcript";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelId {
    BaseEn,
    SmallEn,
    MediumEn,
    LargeV3,
    LargeV3Turbo,
}

impl ModelId {
    pub const ALL: &'static [ModelId] = &[
        ModelId::BaseEn,
        ModelId::SmallEn,
        ModelId::MediumEn,
        ModelId::LargeV3,
        ModelId::LargeV3Turbo,
    ];

    pub fn slug(self) -> &'static str {
        match self {
            ModelId::BaseEn => "base.en",
            ModelId::SmallEn => "small.en",
            ModelId::MediumEn => "medium.en",
            ModelId::LargeV3 => "large-v3",
            ModelId::LargeV3Turbo => "large-v3-turbo",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ModelId::BaseEn => "Base (English, ~150 MB)",
            ModelId::SmallEn => "Small (English, ~500 MB)",
            ModelId::MediumEn => "Medium (English, ~1.5 GB)",
            ModelId::LargeV3 => "Large v3 (multilingual, ~3 GB)",
            ModelId::LargeV3Turbo => "Large v3 Turbo (multilingual, ~1.5 GB, recommended)",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|m| m.slug().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                anyhow!(
                    "unknown model '{}' — valid: {}",
                    s,
                    Self::ALL
                        .iter()
                        .map(|m| m.slug())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }

    /// CoreML encoder is available for every currently-supported model. Kept as a
    /// method so future model additions without a CoreML companion can return `false`.
    pub fn has_coreml_encoder(self) -> bool {
        true
    }

    fn ggml_filename(self) -> String {
        format!("ggml-{}.bin", self.slug())
    }

    fn coreml_zip_filename(self) -> String {
        format!("ggml-{}-encoder.mlmodelc.zip", self.slug())
    }

    fn coreml_dir_filename(self) -> String {
        format!("ggml-{}-encoder.mlmodelc", self.slug())
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: ModelId,
    pub ggml_path: PathBuf,
    pub coreml_dir: Option<PathBuf>,
    pub ggml_present: bool,
    pub coreml_present: bool,
}

/// Returns the shared directory where both the CLI and GUI cache Whisper models.
/// macOS: ~/Library/Application Support/com.thibautbaissac.transcript/models
pub fn cache_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().ok_or_else(|| anyhow!("could not resolve OS data directory"))?;
    Ok(base.join(APP_DIR_NAME).join("models"))
}

/// Returns the directory where the GUI persists saved transcripts, sibling to `cache_dir()`.
/// macOS: ~/Library/Application Support/com.thibautbaissac.transcript/transcripts
pub fn transcripts_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().ok_or_else(|| anyhow!("could not resolve OS data directory"))?;
    Ok(base.join(APP_DIR_NAME).join("transcripts"))
}

pub fn model_info(id: ModelId) -> Result<ModelInfo> {
    let dir = cache_dir()?;
    let ggml_path = dir.join(id.ggml_filename());
    let coreml_dir = id
        .has_coreml_encoder()
        .then(|| dir.join(id.coreml_dir_filename()));
    let coreml_present = coreml_dir.as_ref().is_some_and(|p| p.exists());
    Ok(ModelInfo {
        id,
        ggml_present: ggml_path.exists(),
        coreml_present,
        ggml_path,
        coreml_dir,
    })
}

/// Ensures the model (and its CoreML encoder, if applicable) is present on disk,
/// downloading if necessary. Returns the local ggml file path.
///
/// `progress_cb` receives (stage, downloaded_bytes, total_bytes_or_none).
pub async fn resolve_model<F>(id: ModelId, mut progress_cb: F) -> Result<ModelInfo>
where
    F: FnMut(DownloadStage, u64, Option<u64>),
{
    let dir = cache_dir()?;
    tokio::fs::create_dir_all(&dir).await?;

    let ggml_path = dir.join(id.ggml_filename());
    if !ggml_path.exists() {
        let url = format!("{}/{}", HF_BASE, id.ggml_filename());
        download_to_file(&url, &ggml_path, |d, t| {
            progress_cb(DownloadStage::Ggml, d, t)
        })
        .await
        .with_context(|| format!("downloading {}", id.ggml_filename()))?;
    }

    let coreml_dir = if id.has_coreml_encoder() {
        let target = dir.join(id.coreml_dir_filename());
        if !target.exists() {
            let zip_path = dir.join(id.coreml_zip_filename());
            let url = format!("{}/{}", HF_BASE, id.coreml_zip_filename());
            download_to_file(&url, &zip_path, |d, t| {
                progress_cb(DownloadStage::CoreMl, d, t)
            })
            .await
            .with_context(|| format!("downloading {}", id.coreml_zip_filename()))?;
            unzip(&zip_path, &dir).with_context(|| "unzipping CoreML encoder")?;
            if let Err(e) = tokio::fs::remove_file(&zip_path).await {
                eprintln!(
                    "warning: could not remove {}: {}",
                    zip_path.display(),
                    e
                );
            }
        }
        Some(target)
    } else {
        None
    };

    let coreml_present = coreml_dir.as_ref().map(|p| p.exists()).unwrap_or(false);
    Ok(ModelInfo {
        id,
        ggml_present: true,
        coreml_present,
        ggml_path,
        coreml_dir,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DownloadStage {
    Ggml,
    CoreMl,
}

impl DownloadStage {
    pub fn as_str(self) -> &'static str {
        match self {
            DownloadStage::Ggml => "ggml",
            DownloadStage::CoreMl => "coreml",
        }
    }
}

async fn download_to_file<F>(url: &str, dest: &Path, mut progress_cb: F) -> Result<()>
where
    F: FnMut(u64, Option<u64>),
{
    let tmp = dest.with_extension("partial");
    // Wrap the body so any early-exit path (HTTP error, IO error, killed mid-stream)
    // removes the orphan .partial instead of letting it accumulate across retries.
    let result = download_to_file_inner(url, &tmp, &mut progress_cb).await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
        return result;
    }
    if let Err(e) = tokio::fs::rename(&tmp, dest).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e).with_context(|| format!("renaming {} -> {}", tmp.display(), dest.display()));
    }
    Ok(())
}

async fn download_to_file_inner<F>(url: &str, tmp: &Path, progress_cb: &mut F) -> Result<()>
where
    F: FnMut(u64, Option<u64>),
{
    let resp = reqwest::get(url)
        .await
        .with_context(|| format!("GET {}", url))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {}", url))?;
    let total = resp.content_length();
    let mut file = tokio::fs::File::create(tmp).await?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_cb(downloaded, total);
    }
    file.flush().await?;
    file.sync_all().await?;
    Ok(())
}

fn unzip(zip_path: &Path, dest_dir: &Path) -> Result<()> {
    // Use the system `unzip` command — avoids pulling in a zip crate. Every mac has it.
    let status = std::process::Command::new("unzip")
        .arg("-o")
        .arg(zip_path)
        .arg("-d")
        .arg(dest_dir)
        .status()
        .with_context(|| "running unzip")?;
    if !status.success() {
        return Err(anyhow!(
            "unzip exited with status {:?} for {:?}",
            status.code(),
            zip_path
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parse_accepts_all_slugs_roundtrip() {
        for id in ModelId::ALL {
            assert_eq!(ModelId::parse(id.slug()).unwrap(), *id);
        }
    }

    #[test]
    fn parse_is_case_insensitive() {
        assert_eq!(ModelId::parse("BASE.EN").unwrap(), ModelId::BaseEn);
        assert_eq!(ModelId::parse("Large-V3-Turbo").unwrap(), ModelId::LargeV3Turbo);
    }

    #[test]
    fn parse_errors_on_unknown() {
        let err = ModelId::parse("tiny.en").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown model 'tiny.en'"), "msg: {msg}");
        // Every valid slug should appear in the error hint so users can recover.
        for id in ModelId::ALL {
            assert!(msg.contains(id.slug()), "missing slug {} in: {msg}", id.slug());
        }
    }

    #[test]
    fn display_matches_slug() {
        assert_eq!(format!("{}", ModelId::BaseEn), "base.en");
        assert_eq!(format!("{}", ModelId::LargeV3Turbo), "large-v3-turbo");
    }

    #[test]
    fn all_contains_five_variants_with_distinct_slugs() {
        assert_eq!(ModelId::ALL.len(), 5);
        let slugs: Vec<&str> = ModelId::ALL.iter().map(|m| m.slug()).collect();
        let mut deduped = slugs.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(slugs.len(), deduped.len(), "duplicate slugs: {slugs:?}");
    }

    #[test]
    fn display_name_is_nonempty_for_all_variants() {
        for id in ModelId::ALL {
            assert!(!id.display_name().is_empty());
        }
    }

    #[test]
    fn has_coreml_encoder_true_for_all_currently_supported() {
        for id in ModelId::ALL {
            assert!(id.has_coreml_encoder(), "{:?} should have coreml", id);
        }
    }

    #[test]
    fn ggml_filename_follows_convention() {
        assert_eq!(ModelId::BaseEn.ggml_filename(), "ggml-base.en.bin");
        assert_eq!(
            ModelId::LargeV3Turbo.coreml_zip_filename(),
            "ggml-large-v3-turbo-encoder.mlmodelc.zip"
        );
        assert_eq!(
            ModelId::LargeV3Turbo.coreml_dir_filename(),
            "ggml-large-v3-turbo-encoder.mlmodelc"
        );
    }

    #[test]
    fn download_stage_as_str_variants() {
        assert_eq!(DownloadStage::Ggml.as_str(), "ggml");
        assert_eq!(DownloadStage::CoreMl.as_str(), "coreml");
    }

    #[test]
    fn modelid_serde_uses_kebab_case() {
        assert_eq!(
            serde_json::to_string(&ModelId::LargeV3Turbo).unwrap(),
            "\"large-v3-turbo\""
        );
        let parsed: ModelId = serde_json::from_str("\"base-en\"").unwrap();
        assert_eq!(parsed, ModelId::BaseEn);
    }

    #[test]
    fn download_stage_serde_uses_kebab_case() {
        assert_eq!(serde_json::to_string(&DownloadStage::CoreMl).unwrap(), "\"core-ml\"");
    }

    #[test]
    fn cache_dir_and_transcripts_dir_share_app_parent() {
        let c = cache_dir().unwrap();
        let t = transcripts_dir().unwrap();
        assert_eq!(c.file_name().and_then(|s| s.to_str()), Some("models"));
        assert_eq!(t.file_name().and_then(|s| s.to_str()), Some("transcripts"));
        assert_eq!(c.parent(), t.parent());
        assert_eq!(
            c.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()),
            Some(APP_DIR_NAME),
        );
    }

    #[test]
    fn model_info_reports_absent_files_without_error() {
        // model_info must work even when the cache dir hasn't been created, so the GUI can
        // display "not downloaded" rows before the user downloads anything.
        let info = model_info(ModelId::BaseEn).unwrap();
        assert_eq!(info.id, ModelId::BaseEn);
        assert!(info.ggml_path.ends_with("ggml-base.en.bin"));
        assert_eq!(info.ggml_present, info.ggml_path.exists());
        let expected_coreml = info.ggml_path.parent().unwrap().join("ggml-base.en-encoder.mlmodelc");
        assert_eq!(info.coreml_dir.as_deref(), Some(expected_coreml.as_path()));
        assert_eq!(info.coreml_present, expected_coreml.exists());
    }

    #[test]
    fn unzip_errors_on_nonexistent_file() {
        let tmp = std::env::temp_dir();
        let err = unzip(Path::new("/nonexistent/path/foo.zip"), &tmp);
        assert!(err.is_err());
    }

    // Runs `zip` if present; ships with macOS so this is reliable on dev machines.
    // Skipped when `zip` is missing to keep CI-like environments green.
    #[test]
    fn unzip_extracts_a_real_zip() {
        let workdir = std::env::temp_dir()
            .join(format!("transcript-unzip-{}", std::process::id()));
        let _ = fs::remove_dir_all(&workdir);
        fs::create_dir_all(&workdir).unwrap();
        let inner = workdir.join("payload.txt");
        fs::write(&inner, b"hello").unwrap();
        let zip_path = workdir.join("bundle.zip");
        let zipped = std::process::Command::new("zip")
            .arg("-j")
            .arg(&zip_path)
            .arg(&inner)
            .status();
        let Ok(status) = zipped else {
            let _ = fs::remove_dir_all(&workdir);
            return; // `zip` CLI unavailable.
        };
        if !status.success() {
            let _ = fs::remove_dir_all(&workdir);
            return;
        }
        let dest = workdir.join("out");
        fs::create_dir_all(&dest).unwrap();
        unzip(&zip_path, &dest).unwrap();
        assert_eq!(fs::read(dest.join("payload.txt")).unwrap(), b"hello");
        let _ = fs::remove_dir_all(&workdir);
    }

    #[tokio::test]
    async fn download_to_file_cleans_up_partial_on_http_error() {
        // 127.0.0.1:1 is an unassignable port; the connection attempt fails synchronously
        // in reqwest, ensuring we hit the error-cleanup branch without relying on DNS or
        // flaky network conditions.
        let tmp = std::env::temp_dir()
            .join(format!("transcript-dl-{}.bin", std::process::id()));
        let _ = fs::remove_file(&tmp);
        let _ = fs::remove_file(tmp.with_extension("partial"));

        let res = download_to_file("http://127.0.0.1:1/nope", &tmp, |_, _| {}).await;
        assert!(res.is_err());
        assert!(!tmp.with_extension("partial").exists(), "orphan .partial not cleaned up");
        assert!(!tmp.exists(), "destination should not exist after failed download");
    }

    // Serves a single canned HTTP response on a free loopback port. Returns the bound
    // port so the caller can build a URL against it. Dies after the first connection —
    // we intentionally don't `accept` a second time, so there's nothing to clean up.
    fn serve_canned_once(body: Vec<u8>, status: &'static str) -> u16 {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut scratch = [0u8; 4096];
                let _ = stream.read(&mut scratch);
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            }
        });
        port
    }

    #[tokio::test]
    async fn download_to_file_succeeds_and_reports_progress() {
        let body = b"0123456789abcdef".to_vec();
        let port = serve_canned_once(body.clone(), "200 OK");

        let dest = std::env::temp_dir().join(format!(
            "transcript-dl-ok-{}-{}.bin",
            std::process::id(),
            port
        ));
        let _ = fs::remove_file(&dest);

        let mut last_downloaded = 0u64;
        let mut last_total: Option<u64> = None;
        download_to_file(
            &format!("http://127.0.0.1:{port}/"),
            &dest,
            |d, t| {
                last_downloaded = d;
                last_total = t;
            },
        )
        .await
        .unwrap();

        // File should be written atomically and the .partial should be gone.
        assert!(dest.exists());
        assert!(!dest.with_extension("partial").exists());
        assert_eq!(fs::read(&dest).unwrap(), body);
        // Progress callback fired with the final bytes.
        assert_eq!(last_downloaded, body.len() as u64);
        assert_eq!(last_total, Some(body.len() as u64));
        let _ = fs::remove_file(&dest);
    }

    #[tokio::test]
    async fn download_to_file_errors_on_http_404() {
        // 4xx response → error_for_status fails → .partial is cleaned up.
        let port = serve_canned_once(b"not found".to_vec(), "404 Not Found");
        let dest = std::env::temp_dir().join(format!(
            "transcript-dl-404-{}-{}.bin",
            std::process::id(),
            port
        ));
        let _ = fs::remove_file(&dest);

        let res = download_to_file(&format!("http://127.0.0.1:{port}/"), &dest, |_, _| {}).await;
        assert!(res.is_err());
        assert!(!dest.exists());
        assert!(!dest.with_extension("partial").exists());
    }

    #[tokio::test]
    async fn resolve_model_is_noop_when_files_already_present() {
        // If both ggml and coreml are already on disk, `resolve_model` should skip the
        // download branches entirely. We only run this check when that's actually the case
        // on the current machine — otherwise the function would reach out to HF and we'd
        // be doing an integration test against the network.
        let info_before = model_info(ModelId::BaseEn).unwrap();
        if !(info_before.ggml_present && info_before.coreml_present) {
            return;
        }
        let mut progress_called = false;
        let info_after = resolve_model(ModelId::BaseEn, |_, _, _| progress_called = true)
            .await
            .unwrap();
        assert!(info_after.ggml_present);
        assert!(info_after.coreml_present);
        // No download means no progress callback invocation.
        assert!(!progress_called);
    }

    #[test]
    fn unzip_errors_on_non_zip_file() {
        // Create a file that is definitely not a zip archive and verify the system-level
        // `unzip` reports a non-zero exit status — we translate that into anyhow::Err.
        let workdir = std::env::temp_dir()
            .join(format!("transcript-unzip-bad-{}", std::process::id()));
        let _ = fs::remove_dir_all(&workdir);
        fs::create_dir_all(&workdir).unwrap();
        let bad_zip = workdir.join("not-a-zip.zip");
        fs::write(&bad_zip, b"this is plain text, not a zip archive at all").unwrap();
        let dest = workdir.join("out");
        fs::create_dir_all(&dest).unwrap();
        let err = unzip(&bad_zip, &dest).unwrap_err();
        assert!(err.to_string().contains("unzip"), "err: {err}");
        let _ = fs::remove_dir_all(&workdir);
    }
}
