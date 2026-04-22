use std::fs;
use std::path::{Path, PathBuf};
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
    save_in(&transcripts_dir()?, model, source, duration_secs, result)
}

fn save_in(
    dir: &Path,
    model: String,
    source: TranscriptSource,
    duration_secs: Option<f32>,
    result: TranscriptResult,
) -> Result<TranscriptRecord> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

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
    let final_path = record_path(dir, &record.id);
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
    list_in(&transcripts_dir()?)
}

fn list_in(dir: &Path) -> Result<Vec<TranscriptSummary>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
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
    load_from(&transcripts_dir()?, id)
}

fn load_from(dir: &Path, id: &str) -> Result<TranscriptRecord> {
    let path = record_path(dir, id);
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn delete(id: &str) -> Result<()> {
    delete_from(&transcripts_dir()?, id)
}

fn delete_from(dir: &Path, id: &str) -> Result<()> {
    let path = record_path(dir, id);
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

fn record_path(dir: &Path, id: &str) -> PathBuf {
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
    use transcript_core::Segment;

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
        let dir = Path::new("/tmp/x");
        let p = record_path(dir, "../../etc/passwd");
        assert_eq!(p, dir.join("etcpasswd.json"));
    }

    #[test]
    fn record_path_preserves_clean_id_chars() {
        let dir = Path::new("/tmp/x");
        let id = "2024-01-15T10-20-30-ab12";
        assert_eq!(record_path(dir, id), dir.join("2024-01-15T10-20-30-ab12.json"));
    }

    #[test]
    fn new_id_and_timestamp_format_matches_expected() {
        let (id, iso) = new_id_and_timestamp();
        // ISO 8601 shape: "YYYY-MM-DDTHH:MM:SSZ"
        assert_eq!(iso.len(), 20, "unexpected iso length: {iso}");
        assert!(iso.ends_with('Z'));
        assert_eq!(&iso[4..5], "-");
        assert_eq!(&iso[10..11], "T");
        assert_eq!(&iso[13..14], ":");
        // id shape: "YYYY-MM-DDTHH-MM-SS-xxxx" — 24 chars
        assert_eq!(id.len(), 24, "unexpected id length: {id}");
        assert!(id.as_bytes().iter().all(|b| b.is_ascii_alphanumeric() || matches!(*b, b'-' | b'T')));
    }

    fn sample_result(text: &str, language: &str) -> TranscriptResult {
        TranscriptResult {
            text: text.into(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: text.into(),
            }],
            language: language.into(),
        }
    }

    fn summary_of(text: &str) -> TranscriptSummary {
        summarize(ListShape {
            id: "x".into(),
            created_at: "2024-01-01T00:00:00Z".into(),
            model: "base.en".into(),
            source: TranscriptSource::Recording,
            duration_secs: Some(1.5),
            result: ResultHead {
                language: "en".into(),
                text: text.into(),
            },
        })
    }

    #[test]
    fn summarize_short_text_has_no_ellipsis() {
        let s = summary_of("hello world");
        assert_eq!(s.preview, "hello world");
        assert_eq!(s.language, "en");
        assert_eq!(s.model, "base.en");
    }

    #[test]
    fn summarize_text_at_exactly_preview_cap_has_no_ellipsis() {
        let text: String = "a".repeat(PREVIEW_CHARS);
        let s = summary_of(&text);
        assert_eq!(s.preview.chars().count(), PREVIEW_CHARS);
        assert!(!s.preview.ends_with('…'));
    }

    #[test]
    fn summarize_long_text_truncates_with_ellipsis() {
        let text: String = "a".repeat(PREVIEW_CHARS + 50);
        let s = summary_of(&text);
        assert_eq!(s.preview.chars().count(), PREVIEW_CHARS + 1);
        assert!(s.preview.ends_with('…'));
    }

    #[test]
    fn summarize_preserves_multibyte_chars() {
        // Multi-byte chars shouldn't be split mid-codepoint (char-iterator truncation).
        let text: String = "é".repeat(PREVIEW_CHARS + 10);
        let s = summary_of(&text);
        assert_eq!(s.preview.chars().count(), PREVIEW_CHARS + 1);
        assert!(s.preview.ends_with('…'));
    }

    #[test]
    fn unix_to_ymdhms_clamps_year_out_of_range() {
        // Year 10000+ is out of range → clamp to unix epoch.
        let huge = 253_402_300_800_u64; // 10000-01-01
        let (y, mo, d, h, mi, se) = unix_to_ymdhms(huge);
        assert_eq!((y, mo, d, h, mi, se), (1970, 1, 1, 0, 0, 0));
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("transcript-test-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn save_and_load_roundtrip_in_tmp_dir() {
        let dir = tmp_dir("save_load");
        let rec = save_in(
            &dir,
            "base.en".into(),
            TranscriptSource::Recording,
            Some(2.5),
            sample_result("hello", "en"),
        )
        .unwrap();

        let loaded = load_from(&dir, &rec.id).unwrap();
        assert_eq!(loaded.id, rec.id);
        assert_eq!(loaded.model, "base.en");
        assert_eq!(loaded.duration_secs, Some(2.5));
        assert_eq!(loaded.result.text, "hello");
        assert!(matches!(loaded.source, TranscriptSource::Recording));

        // No leftover .partial file after a successful write.
        let any_partial = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("partial"));
        assert!(!any_partial);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_persists_file_source_variant() {
        let dir = tmp_dir("file_source");
        let rec = save_in(
            &dir,
            "small.en".into(),
            TranscriptSource::File("/path/to/audio.mp3".into()),
            None,
            sample_result("x", "en"),
        )
        .unwrap();
        let loaded = load_from(&dir, &rec.id).unwrap();
        match loaded.source {
            TranscriptSource::File(p) => assert_eq!(p, "/path/to/audio.mp3"),
            _ => panic!("expected File variant"),
        }
        assert_eq!(loaded.duration_secs, None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_in_returns_empty_when_dir_missing() {
        let dir = tmp_dir("missing");
        assert!(!dir.exists());
        let v = list_in(&dir).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn list_in_sorts_by_created_at_descending() {
        let dir = tmp_dir("list_sort");
        fs::create_dir_all(&dir).unwrap();

        // Write two records with different created_at; listing must be newest-first.
        let rec_a = TranscriptRecord {
            id: "2024-01-01T00-00-00-0001".into(),
            created_at: "2024-01-01T00:00:00Z".into(),
            model: "base.en".into(),
            source: TranscriptSource::Recording,
            duration_secs: None,
            result: sample_result("older", "en"),
        };
        let rec_b = TranscriptRecord {
            id: "2024-06-01T00-00-00-0002".into(),
            created_at: "2024-06-01T00:00:00Z".into(),
            model: "base.en".into(),
            source: TranscriptSource::Recording,
            duration_secs: None,
            result: sample_result("newer", "en"),
        };
        for r in [&rec_a, &rec_b] {
            let p = record_path(&dir, &r.id);
            fs::write(&p, serde_json::to_vec_pretty(r).unwrap()).unwrap();
        }

        let summaries = list_in(&dir).unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].id, rec_b.id); // newer first
        assert_eq!(summaries[1].id, rec_a.id);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_in_skips_non_json_and_invalid_json() {
        let dir = tmp_dir("list_skip");
        fs::create_dir_all(&dir).unwrap();

        // Non-json file — must be skipped by extension filter.
        fs::write(dir.join("readme.txt"), b"not a transcript").unwrap();
        // Invalid json — must be logged-and-skipped, not fail the whole list.
        fs::write(dir.join("corrupt.json"), b"{not valid").unwrap();
        // One valid record should survive.
        let good = TranscriptRecord {
            id: "good".into(),
            created_at: "2024-01-01T00:00:00Z".into(),
            model: "base.en".into(),
            source: TranscriptSource::Recording,
            duration_secs: None,
            result: sample_result("ok", "en"),
        };
        fs::write(
            dir.join("good.json"),
            serde_json::to_vec_pretty(&good).unwrap(),
        )
        .unwrap();

        let summaries = list_in(&dir).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, "good");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_removes_file_and_then_errors() {
        let dir = tmp_dir("delete");
        let rec = save_in(
            &dir,
            "base.en".into(),
            TranscriptSource::Recording,
            None,
            sample_result("bye", "en"),
        )
        .unwrap();
        let path = record_path(&dir, &rec.id);
        assert!(path.exists());
        delete_from(&dir, &rec.id).unwrap();
        assert!(!path.exists());
        // Second delete must fail — removing a missing file is an error.
        assert!(delete_from(&dir, &rec.id).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_from_errors_on_missing_id() {
        let dir = tmp_dir("load_missing");
        fs::create_dir_all(&dir).unwrap();
        assert!(load_from(&dir, "does-not-exist").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn transcript_source_serde_roundtrip() {
        // Tagged-enum shape is part of the JSON contract with the frontend — lock it down.
        let rec = TranscriptSource::Recording;
        assert_eq!(serde_json::to_string(&rec).unwrap(), r#"{"kind":"recording"}"#);
        let f = TranscriptSource::File("a.mp3".into());
        assert_eq!(
            serde_json::to_string(&f).unwrap(),
            r#"{"kind":"file","value":"a.mp3"}"#
        );
    }
}
