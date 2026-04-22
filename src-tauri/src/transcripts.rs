use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use transcript_core::{TranscriptResult, transcripts_dir};

const PREVIEW_CHARS: usize = 120;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum TranscriptSource {
    Recording,
    File(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptRecord {
    pub id: String,
    pub created_at: String,
    pub model: String,
    pub source: TranscriptSource,
    pub duration_secs: Option<f32>,
    pub result: TranscriptResult,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptSummary {
    pub id: String,
    pub created_at: String,
    pub model: String,
    pub source: TranscriptSource,
    pub duration_secs: Option<f32>,
    pub language: String,
    pub preview: String,
}

pub fn save(
    model: String,
    source: TranscriptSource,
    duration_secs: Option<f32>,
    result: TranscriptResult,
) -> Result<TranscriptRecord> {
    let dir = transcripts_dir()?;
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let (id, created_at) = new_id_and_timestamp();
    let record = TranscriptRecord {
        id,
        created_at,
        model,
        source,
        duration_secs,
        result,
    };
    // Atomic write: serialize to a sibling `.partial`, fsync, then rename. Same pattern
    // as `models::download_to_file`. `sync_all` before rename ensures the content is
    // durable on disk before the dir entry promotes it — otherwise a power loss between
    // write and flush can leave a zero-length file that `list()` would silently skip.
    let final_path = record_path(&dir, &record.id);
    let tmp_path = final_path.with_extension("json.partial");
    {
        let mut file = fs::File::create(&tmp_path)
            .with_context(|| format!("writing {}", tmp_path.display()))?;
        serde_json::to_writer_pretty(&mut file, &record).with_context(|| "serializing record")?;
        file.sync_all().with_context(|| "fsync record")?;
    }
    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("renaming into {}", final_path.display()))?;
    Ok(record)
}

/// Partial deserialization shape — used by `list()` to skip parsing the segments
/// array, which dominates file size for long recordings.
#[derive(Deserialize)]
struct ListShape {
    id: String,
    created_at: String,
    model: String,
    source: TranscriptSource,
    duration_secs: Option<f32>,
    result: ResultHead,
}

#[derive(Deserialize)]
struct ResultHead {
    language: String,
    text: String,
}

pub fn list() -> Result<Vec<TranscriptSummary>> {
    let dir = transcripts_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                eprintln!("transcripts: skipping unreadable entry: {err}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|s| serde_json::from_str::<ListShape>(&s).map_err(Into::into))
        {
            Ok(shape) => out.push(summarize(shape)),
            Err(err) => eprintln!("transcripts: skipping {}: {err:#}", path.display()),
        }
    }
    // ISO 8601 sorts lexicographically, so `cmp` on the string gives chronological order.
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

pub fn load(id: &str) -> Result<TranscriptRecord> {
    let dir = transcripts_dir()?;
    let path = record_path(&dir, id);
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn delete(id: &str) -> Result<()> {
    let dir = transcripts_dir()?;
    let path = record_path(&dir, id);
    fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))
}

fn summarize(shape: ListShape) -> TranscriptSummary {
    let mut chars = shape.result.text.chars();
    let mut preview: String = chars.by_ref().take(PREVIEW_CHARS).collect();
    if chars.next().is_some() {
        preview.push('…');
    }
    TranscriptSummary {
        id: shape.id,
        created_at: shape.created_at,
        model: shape.model,
        source: shape.source,
        duration_secs: shape.duration_secs,
        language: shape.result.language,
        preview,
    }
}

fn record_path(dir: &std::path::Path, id: &str) -> PathBuf {
    // Guard against path traversal: ids must be plain filename stems.
    let safe: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | 'T'))
        .collect();
    dir.join(format!("{safe}.json"))
}

fn new_id_and_timestamp() -> (String, String) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let (y, mo, d, h, mi, se) = unix_to_ymdhms(now.as_secs());
    let iso = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{se:02}Z");
    // Nano-derived 4-hex-digit suffix so multiple saves in the same second don't clobber.
    let rand = (now.subsec_nanos() ^ (now.as_secs() as u32)) & 0xFFFF;
    let id = format!("{y:04}-{mo:02}-{d:02}T{h:02}-{mi:02}-{se:02}-{rand:04x}");
    (id, iso)
}

/// Convert seconds since Unix epoch to civil UTC (Y, M, D, H, M, S).
/// Howard Hinnant's proleptic Gregorian algorithm.
fn unix_to_ymdhms(ts: u64) -> (u16, u8, u8, u8, u8, u8) {
    let se = (ts % 60) as u8;
    let total_min = ts / 60;
    let mi = (total_min % 60) as u8;
    let total_hr = total_min / 60;
    let h = (total_hr % 24) as u8;
    let days = (total_hr / 24) as i64;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    let y = if m <= 2 { y + 1 } else { y };
    if !(1..=9999).contains(&y) {
        return (1970, 1, 1, 0, 0, 0);
    }
    (y as u16, m, d, h, mi, se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ymdhms_known_epochs() {
        assert_eq!(unix_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
        // 946_684_800 == 2000-01-01T00:00:00Z (well-known)
        assert_eq!(unix_to_ymdhms(946_684_800), (2000, 1, 1, 0, 0, 0));
        // Leap-year day: 2024-02-29T12:34:56Z == 1_709_210_096
        assert_eq!(unix_to_ymdhms(1_709_210_096), (2024, 2, 29, 12, 34, 56));
    }

    #[test]
    fn record_path_rejects_traversal() {
        let dir = std::path::Path::new("/tmp/x");
        let p = record_path(dir, "../../etc/passwd");
        assert_eq!(p, dir.join("etcpasswd.json"));
    }
}
