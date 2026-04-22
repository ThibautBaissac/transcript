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
    let resp = reqwest::get(url)
        .await
        .with_context(|| format!("GET {}", url))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {}", url))?;
    let total = resp.content_length();
    let tmp = dest.with_extension("partial");
    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_cb(downloaded, total);
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&tmp, dest).await?;
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
