//! Read-only versioned prompt/style preset manifest.
//!
//! Presets live beside the model roots under `<data_dir>/models/presets/`.
//! This module only parses and validates that operator-owned JSON; it never
//! writes, fetches, or resolves a model. The server exposes the parsed rows
//! through read-only routes and composes them into existing generation
//! requests without introducing a second client format.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Wire version of the preset manifest. Unknown versions are refused, never
/// guessed at.
pub const PRESET_SCHEMA_VERSION: u32 = 1;

/// Manifest location relative to the resolved data directory.
pub const PRESET_MANIFEST_REL: &str = "models/presets/manifest.json";

/// 256 KiB is far above any realistic preset list; a larger file is a
/// malformed manifest, not something to parse into the request path.
const MAX_MANIFEST_BYTES: u64 = 256 * 1024;

/// One versioned negative-prompt/style fragment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Preset {
    /// Stable operator id used by `GET /v1/presets/{id}` and the
    /// `X-Honkbox-Preset` header.
    pub id: String,
    /// Monotonic revision of this preset; bump when the text changes.
    pub version: u32,
    /// Optional human-readable label (the id is the API handle).
    #[serde(default)]
    pub title: Option<String>,
    /// Text appended to the request's negative prompt.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Text appended to the request's prompt.
    #[serde(default)]
    pub style_prompt: Option<String>,
    /// Free-form operator tags; not interpreted by the server.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Versioned preset document. Parsed from one local JSON file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetManifest {
    /// Schema version this file was written against.
    pub schema_version: u32,
    /// Presets in stable id order after [`load_presets`] validates them.
    #[serde(default)]
    pub presets: Vec<Preset>,
}

/// Named reasons a preset manifest cannot be used.
#[derive(Debug, PartialEq, Eq)]
pub enum PresetLoadError {
    /// A newer or older schema version was found.
    UnsupportedSchema {
        /// Schema version found in the operator file.
        found: u32,
    },
    /// The file is not valid UTF-8 JSON.
    InvalidJson {
        /// Parser reason, safe to log.
        reason: String,
    },
    /// A row is missing required fields or carries no usable text.
    InvalidPreset {
        /// Validation reason, safe to log.
        reason: String,
    },
    /// Two rows share the same id.
    DuplicateId {
        /// Offending operator id.
        id: String,
    },
    /// The manifest exceeds the read bound.
    TooLarge {
        /// Observed file size in bytes.
        bytes: u64,
    },
    /// The file could not be read. Host paths are logged, never returned.
    Io {
        /// Sanitized I/O reason.
        reason: String,
    },
}

impl fmt::Display for PresetLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema { found } => {
                write!(formatter, "unsupported preset schema version {found}")
            }
            Self::InvalidJson { reason } => write!(formatter, "invalid preset JSON: {reason}"),
            Self::InvalidPreset { reason } => write!(formatter, "invalid preset row: {reason}"),
            Self::DuplicateId { id } => write!(formatter, "duplicate preset id {id:?}"),
            Self::TooLarge { bytes } => {
                write!(formatter, "preset manifest too large: {bytes} bytes")
            }
            Self::Io { reason } => write!(formatter, "preset manifest read failed: {reason}"),
        }
    }
}

impl std::error::Error for PresetLoadError {}

/// Absolute path of the operator-owned manifest.
#[must_use]
pub fn preset_manifest_path(data_dir: &Path) -> PathBuf {
    data_dir.join(PRESET_MANIFEST_REL)
}

/// Empty manifest for the `schema_version` this binary understands.
#[must_use]
pub fn empty_preset_manifest() -> PresetManifest {
    PresetManifest {
        schema_version: PRESET_SCHEMA_VERSION,
        presets: Vec::new(),
    }
}

/// Parse and validate the manifest. A missing file is an empty manifest,
/// matching the lenient model-root scan: no operator file means no presets,
/// not a failed request.
///
/// # Errors
///
/// Returns a named [`PresetLoadError`] when the file exists but has an
/// unsupported schema, malformed JSON, invalid or duplicate rows, exceeds
/// the read bound, or cannot be read.
pub fn load_presets(data_dir: &Path) -> Result<Option<PresetManifest>, PresetLoadError> {
    let path = preset_manifest_path(data_dir);
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(PresetLoadError::Io {
                reason: err.to_string(),
            });
        }
    };
    if !metadata.is_file() {
        return Err(PresetLoadError::Io {
            reason: "manifest path is not a file".to_owned(),
        });
    }
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(PresetLoadError::TooLarge {
            bytes: metadata.len(),
        });
    }
    let raw = std::fs::read_to_string(&path).map_err(|err| PresetLoadError::Io {
        reason: err.to_string(),
    })?;
    let mut manifest: PresetManifest =
        serde_json::from_str(&raw).map_err(|err| PresetLoadError::InvalidJson {
            reason: err.to_string(),
        })?;
    if manifest.schema_version != PRESET_SCHEMA_VERSION {
        return Err(PresetLoadError::UnsupportedSchema {
            found: manifest.schema_version,
        });
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for preset in &manifest.presets {
        if preset.id.trim().is_empty() {
            return Err(PresetLoadError::InvalidPreset {
                reason: "empty id".to_owned(),
            });
        }
        if preset.version == 0 {
            return Err(PresetLoadError::InvalidPreset {
                reason: format!("preset {:?} has version 0", preset.id),
            });
        }
        let has_text = [
            preset.negative_prompt.as_deref(),
            preset.style_prompt.as_deref(),
        ]
        .iter()
        .any(|text| text.is_some_and(|text| !text.trim().is_empty()));
        if !has_text {
            return Err(PresetLoadError::InvalidPreset {
                reason: format!(
                    "preset {:?} has neither negative_prompt nor style_prompt",
                    preset.id
                ),
            });
        }
        if !seen.insert(preset.id.clone()) {
            return Err(PresetLoadError::DuplicateId {
                id: preset.id.clone(),
            });
        }
    }
    manifest
        .presets
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(Some(manifest))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_data_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "honkbox-presets-{tag}-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(dir.join("models/presets")).expect("preset parent dir");
        dir
    }

    fn write_manifest(dir: &Path, body: &str) {
        std::fs::write(preset_manifest_path(dir), body).expect("manifest write");
    }

    #[test]
    fn valid_manifest_loads_sorted_and_missing_is_none() {
        let dir = temp_data_dir("valid");
        assert_eq!(load_presets(&dir).expect("missing manifest"), None);
        write_manifest(
            &dir,
            r#"{"schema_version":1,"presets":[
                {"id":"z","version":2,"style_prompt":"cinematic"},
                {"id":"a","version":1,"negative_prompt":"blur"}
            ]}"#,
        );
        let manifest = load_presets(&dir)
            .expect("valid manifest")
            .expect("manifest present");
        assert_eq!(manifest.schema_version, PRESET_SCHEMA_VERSION);
        assert_eq!(
            manifest
                .presets
                .iter()
                .map(|preset| preset.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn malformed_manifests_are_named() {
        let cases = [
            (
                r#"{"schema_version":9,"presets":[]}"#,
                PresetLoadError::UnsupportedSchema { found: 9 },
            ),
            (
                r#"{"schema_version":1,"presets":[{"id":"x","version":1}]}"#,
                PresetLoadError::InvalidPreset {
                    reason: "preset \"x\" has neither negative_prompt nor style_prompt".to_owned(),
                },
            ),
            (
                r#"{"schema_version":1,"presets":[
                    {"id":"x","version":1,"style_prompt":"a"},
                    {"id":"x","version":2,"style_prompt":"b"}
                ]}"#,
                PresetLoadError::DuplicateId { id: "x".to_owned() },
            ),
        ];
        for (index, (body, expected)) in cases.iter().enumerate() {
            let dir = temp_data_dir(&format!("bad-{index}"));
            write_manifest(&dir, body);
            assert_eq!(load_presets(&dir).expect_err("named error"), *expected);
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
