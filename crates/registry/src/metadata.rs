//! Weight-file header probing for catalog metadata (feature-survey NEW-6).
//!
//! GGUF: magic + version + tensor count, plus the `general.name` /
//! `general.architecture` KV strings and the training context length
//! (`<arch>.context_length`) when cheap. Safetensors: the 8-byte
//! length prefix + JSON header tensor count / dtype summary.
//!
//! Every parser is deliberately lenient: absent, invalid, or oversized
//! headers yield `None` (or a partially-filled summary once the prologue
//! parsed) — probing never fails a scan.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF header summary.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct GgufMetadata {
    /// GGUF spec version (1–3 today).
    pub version: u32,
    /// Tensor count declared in the header.
    pub tensor_count: u64,
    /// `general.name` KV string, when present and readable.
    pub name: Option<String>,
    /// `general.architecture` KV string, when present and readable.
    pub architecture: Option<String>,
    /// Training context length (`<arch>.context_length` KV); `None` when the
    /// header omits it or its architecture prefix is ambiguous.
    pub context_length: Option<u64>,
}

/// Safetensors header summary.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SafetensorsMetadata {
    /// Tensor entries in the JSON header (`__metadata__` excluded).
    pub tensor_count: u64,
    /// Unique dtype names, sorted (`F32`, `F16`, `BF16`, …).
    pub dtypes: Vec<String>,
    /// Base-model tag from the safetensors `__metadata__` header, normalized
    /// to `sd15`/`sdxl`/`sd3` when recognized; raw text when unknown.
    pub base_model: Option<String>,
    /// Raw `general.name` identity value, preserved verbatim.
    pub identity_name: Option<String>,
    /// Raw `model_name` identity value, preserved verbatim.
    pub identity_model_name: Option<String>,
    /// Raw `model_id` identity value, preserved verbatim.
    pub identity_model_id: Option<String>,
    /// Every present raw identity key/value pair, sorted by key.
    pub raw_identity: Vec<(String, String)>,
}

/// Header-derived metadata for one weight file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "format", rename_all = "snake_case")]
pub enum ModelFileMetadata {
    /// `.gguf`: llama.cpp container header.
    Gguf(GgufMetadata),
    /// `.safetensors`: `HuggingFace` container header.
    Safetensors(SafetensorsMetadata),
}

/// Probe one weight file's header, dispatched by (lowercase) extension.
/// Unknown extensions and any read/parse failure yield `None`.
pub(crate) fn probe(path: &Path, ext: &str) -> Option<ModelFileMetadata> {
    match ext {
        "gguf" => probe_gguf(path).map(ModelFileMetadata::Gguf),
        "safetensors" => probe_safetensors(path).map(ModelFileMetadata::Safetensors),
        _ => None,
    }
}

/// GGUF magic (`GGUF`).
const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Highest GGUF spec version recognized (3 today).
const GGUF_MAX_VERSION: u32 = 3;
/// KV-entry scan cap: bail out of the walk rather than scan a huge file.
const GGUF_MAX_KVS: u64 = 4096;
/// Strings longer than this are never buffered (keys and the `general.*`
/// values are short); oversized string values are skipped instead.
const GGUF_MAX_STRING: u64 = 64 * 1024;
/// Array element-count cap (tokenizer arrays are big; we skip, not read).
const GGUF_MAX_ARRAY_ELEMS: u64 = 1 << 24;
/// Safetensors header-size cap: larger prefixes are treated as invalid.
const SAFETENSORS_MAX_HEADER: u64 = 16 * 1024 * 1024;

fn read_u32(file: &mut std::fs::File) -> Option<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut std::fs::File) -> Option<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

/// Seek forward over `bytes` without reading them.
fn skip_bytes(file: &mut std::fs::File, bytes: u64) -> Option<()> {
    let offset = i64::try_from(bytes).ok()?;
    file.seek(SeekFrom::Current(offset)).ok().map(|_| ())
}

/// Fixed byte width of a scalar GGUF value type (the `gguf_type` numbering).
fn gguf_scalar_size(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1), // u8 / i8 / bool
        2 | 3 => Some(2),     // u16 / i16
        4..=6 => Some(4),     // u32 / i32 / f32
        10..=12 => Some(8),   // u64 / i64 / f64
        _ => None,            // string (8) / array (9) / unknown
    }
}

/// Read one GGUF string (u64 length + bytes); over-long strings fail.
fn read_gguf_string(file: &mut std::fs::File) -> Option<String> {
    let len = read_u64(file)?;
    if len > GGUF_MAX_STRING {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    file.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Skip one GGUF value of `value_type` (recursing into arrays, depth-capped).
fn skip_gguf_value(file: &mut std::fs::File, value_type: u32, depth: u8) -> Option<()> {
    if let Some(size) = gguf_scalar_size(value_type) {
        return skip_bytes(file, size);
    }
    match value_type {
        8 => {
            let len = read_u64(file)?;
            skip_bytes(file, len)
        }
        9 if depth < 4 => {
            let elem_type = read_u32(file)?;
            let count = read_u64(file)?;
            if count > GGUF_MAX_ARRAY_ELEMS {
                return None;
            }
            if let Some(size) = gguf_scalar_size(elem_type) {
                skip_bytes(file, size.checked_mul(count)?)
            } else {
                for _ in 0..count {
                    skip_gguf_value(file, elem_type, depth + 1)?;
                }
                Some(())
            }
        }
        _ => None,
    }
}

/// Parse the GGUF prologue and walk metadata KVs until the wanted values are
/// captured, the KV cap is hit, or the walk breaks on malformed data.
/// Prologue failures yield `None`; walk failures keep what was read.
fn probe_gguf(path: &Path) -> Option<GgufMetadata> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).ok()?;
    if magic != GGUF_MAGIC {
        return None;
    }
    let version = read_u32(&mut file)?;
    if version == 0 || version > GGUF_MAX_VERSION {
        return None;
    }
    let tensor_count = read_u64(&mut file)?;
    let kv_count = read_u64(&mut file)?.min(GGUF_MAX_KVS);
    let mut name: Option<String> = None;
    let mut architecture: Option<String> = None;
    // `<arch>.context_length` candidates: the architecture key usually
    // precedes its arch-specific keys, but the pairing is only decided after
    // the walk, so collect (prefix, value) and resolve below.
    let mut context_lengths: Vec<(String, u64)> = Vec::new();
    for _ in 0..kv_count {
        let Some(key) = read_gguf_string(&mut file) else {
            break;
        };
        let Some(value_type) = read_u32(&mut file) else {
            break;
        };
        if let Some(prefix) = key
            .strip_suffix(".context_length")
            .filter(|_| context_lengths.len() < 8)
        {
            // GGUF integers are unsigned 32- or 64-bit in practice.
            let value = match value_type {
                4 => read_u32(&mut file).map(u64::from),
                10 => read_u64(&mut file),
                _ => None,
            };
            match value {
                Some(value) => context_lengths.push((prefix.to_owned(), value)),
                None => break,
            }
            continue;
        }
        let wanted = match key.as_str() {
            "general.name" if name.is_none() => Some(&mut name),
            "general.architecture" if architecture.is_none() => Some(&mut architecture),
            _ => None,
        };
        if value_type == 8 {
            if let Some(slot) = wanted {
                match read_gguf_string(&mut file) {
                    Some(value) => *slot = Some(value),
                    None => break,
                }
            } else if skip_gguf_value(&mut file, 8, 0).is_none() {
                break;
            }
        } else if skip_gguf_value(&mut file, value_type, 0).is_none() {
            break;
        }
    }
    // Pair the captured candidates with the architecture: prefer the exact
    // `<arch>` prefix; without an architecture, a single candidate is still
    // unambiguous while several are not (they stay unused rather than
    // guessing).
    let context_length = resolve_context_length(architecture.as_deref(), &context_lengths);
    Some(GgufMetadata {
        version,
        tensor_count,
        name,
        architecture,
        context_length,
    })
}

/// Pick the training context length from the captured `<prefix>.context_length`
/// candidates: the entry whose prefix matches `architecture` exactly, or the
/// sole candidate when the header named no architecture. Several candidates
/// without an architecture stay unused rather than guessed.
fn resolve_context_length(architecture: Option<&str>, candidates: &[(String, u64)]) -> Option<u64> {
    match architecture {
        Some(arch) => candidates
            .iter()
            .find(|(prefix, _)| prefix == arch)
            .map(|(_, value)| *value),
        None => {
            if candidates.len() == 1 {
                Some(candidates[0].1)
            } else {
                None
            }
        }
    }
}

/// Read the 8-byte length prefix + JSON header; summarize tensor count and
/// dtypes. Malformed prefixes, oversized headers, or non-JSON headers are
/// `None`; individual entries without a `dtype` are skipped.
fn probe_safetensors(path: &Path) -> Option<SafetensorsMetadata> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > SAFETENSORS_MAX_HEADER {
        return None;
    }
    let mut header_json = vec![0u8; usize::try_from(header_len).ok()?];
    file.read_exact(&mut header_json).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header_json).ok()?;
    let entries = header.as_object()?;
    let mut tensor_count = 0u64;
    let mut dtypes: Vec<String> = Vec::new();
    for (key, entry) in entries {
        if key == "__metadata__" {
            continue;
        }
        let Some(dtype) = entry.get("dtype").and_then(serde_json::Value::as_str) else {
            continue;
        };
        tensor_count += 1;
        if !dtypes.iter().any(|d| d == dtype) {
            dtypes.push(dtype.to_owned());
        }
    }
    dtypes.sort_unstable();
    let meta = entries
        .get("__metadata__")
        .and_then(serde_json::Value::as_object);
    let base_model = meta.and_then(|meta| {
        [
            "ss_base_model",
            "base_model",
            "ss_base_model_version",
            "modelspec.base_model",
        ]
        .into_iter()
        .find_map(|key| {
            meta.get(key)
                .and_then(serde_json::Value::as_str)
                .and_then(normalize_base_model)
        })
    });
    let identity = |key: &str| -> Option<String> {
        meta.and_then(|meta| meta.get(key))
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned)
    };
    let identity_name = identity("general.name");
    let identity_model_name = identity("model_name");
    let identity_model_id = identity("model_id");
    // Raw identity pairs, emitted in key order (`general.name` < `model_id`
    // < `model_name`).
    let raw_identity: Vec<(String, String)> = [
        ("general.name", identity_name.as_deref()),
        ("model_id", identity_model_id.as_deref()),
        ("model_name", identity_model_name.as_deref()),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key.to_owned(), value.to_owned())))
    .collect();
    Some(SafetensorsMetadata {
        tensor_count,
        dtypes,
        base_model,
        identity_name,
        identity_model_name,
        identity_model_id,
        raw_identity,
    })
}

/// Canonicalize the common ComfyUI/diffusers base-model spellings. Unknown
/// text is preserved so the wire surface can warn instead of silently pass.
fn normalize_base_model(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    let lower = value.to_ascii_lowercase();
    let known = if lower.contains("sdxl") || lower.contains("sd_xl") || lower.contains("sd-xl") {
        Some("sdxl")
    } else if lower.contains("sd3") || lower.contains("sd 3") {
        Some("sd3")
    } else if lower.contains("1.5") || lower.contains("v1-5") || lower.contains("v1_5") {
        Some("sd15")
    } else {
        None
    };
    Some(known.unwrap_or(value).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-test scratch file (tests run in parallel within one process).
    fn write_scratch(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("honkbox-meta-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).expect("write scratch file");
        path
    }

    /// Test-only GGUF metadata value.
    enum TestKv {
        Str(&'static str),
        U32(u32),
        U64(u64),
        StrArray(&'static [&'static str]),
    }

    /// Append one GGUF string: u64 length + bytes.
    fn push_str(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    /// GGUF fixture: magic, version 3, one tensor, then the given KVs.
    fn gguf_fixture(kvs: &[(&str, TestKv)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, kv) in kvs {
            push_str(&mut out, key);
            match kv {
                TestKv::Str(value) => {
                    out.extend_from_slice(&8u32.to_le_bytes()); // string
                    push_str(&mut out, value);
                }
                TestKv::U32(value) => {
                    out.extend_from_slice(&4u32.to_le_bytes()); // u32
                    out.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::U64(value) => {
                    out.extend_from_slice(&10u32.to_le_bytes()); // u64
                    out.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::StrArray(values) => {
                    out.extend_from_slice(&9u32.to_le_bytes()); // array
                    out.extend_from_slice(&8u32.to_le_bytes()); // of strings
                    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in *values {
                        push_str(&mut out, value);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn gguf_parses_name_arch_and_skips_arrays() {
        let path = write_scratch(
            "gguf-ok.gguf",
            &gguf_fixture(&[
                ("general.architecture", TestKv::Str("llama")),
                ("tokenizer.ggml.tokens", TestKv::StrArray(&["<s>", "</s>"])),
                ("general.name", TestKv::Str("TestModel")),
                ("general.quantization_version", TestKv::U64(2)),
            ]),
        );
        let metadata = probe(&path, "gguf");
        std::fs::remove_file(&path).ok();
        assert_eq!(
            metadata,
            Some(ModelFileMetadata::Gguf(GgufMetadata {
                version: 3,
                tensor_count: 1,
                name: Some("TestModel".to_owned()),
                architecture: Some("llama".to_owned()),
                context_length: None,
            }))
        );
    }

    #[test]
    fn gguf_captures_context_length_for_architecture() {
        // Real-world shapes: `<arch>.context_length` is u32 for llama-family,
        // u64 elsewhere; the architecture key pairs the value.
        let path = write_scratch(
            "gguf-ctx-u32.gguf",
            &gguf_fixture(&[
                ("general.architecture", TestKv::Str("qwen3")),
                ("qwen3.context_length", TestKv::U32(40_960)),
                ("qwen3.embedding_length", TestKv::U32(4096)),
                ("general.name", TestKv::Str("Qwen3")),
            ]),
        );
        let meta = gguf_meta(probe(&path, "gguf"));
        std::fs::remove_file(&path).ok();
        assert_eq!(meta.name.as_deref(), Some("Qwen3"));
        assert_eq!(meta.architecture.as_deref(), Some("qwen3"));
        assert_eq!(meta.context_length, Some(40_960));

        let path = write_scratch(
            "gguf-ctx-u64.gguf",
            &gguf_fixture(&[
                ("general.architecture", TestKv::Str("llama")),
                ("llama.context_length", TestKv::U64(131_072)),
            ]),
        );
        let meta = gguf_meta(probe(&path, "gguf"));
        std::fs::remove_file(&path).ok();
        assert_eq!(meta.context_length, Some(131_072));
    }

    /// Unwrap a probed GGUF summary (panics with the value on mismatch).
    fn gguf_meta(metadata: Option<ModelFileMetadata>) -> GgufMetadata {
        match metadata {
            Some(ModelFileMetadata::Gguf(meta)) => meta,
            other => panic!("expected gguf metadata, got {other:?}"),
        }
    }

    #[test]
    fn gguf_context_length_pairing_rules() {
        // Architecture named after the candidate: mismatched prefixes unused.
        let path = write_scratch(
            "gguf-ctx-mismatch.gguf",
            &gguf_fixture(&[
                ("general.architecture", TestKv::Str("llama")),
                ("bert.context_length", TestKv::U32(512)),
            ]),
        );
        let meta = gguf_meta(probe(&path, "gguf"));
        std::fs::remove_file(&path).ok();
        assert_eq!(meta.context_length, None);

        // No architecture: a single candidate is unambiguous, several are not.
        let path = write_scratch(
            "gguf-ctx-single.gguf",
            &gguf_fixture(&[("bert.context_length", TestKv::U32(512))]),
        );
        let meta = gguf_meta(probe(&path, "gguf"));
        std::fs::remove_file(&path).ok();
        assert_eq!(meta.context_length, Some(512));

        let path = write_scratch(
            "gguf-ctx-ambiguous.gguf",
            &gguf_fixture(&[
                ("bert.context_length", TestKv::U32(512)),
                ("gpt.context_length", TestKv::U32(1024)),
            ]),
        );
        let meta = gguf_meta(probe(&path, "gguf"));
        std::fs::remove_file(&path).ok();
        assert_eq!(meta.context_length, None);
    }

    #[test]
    fn gguf_bad_magic_and_truncation_are_none() {
        let mut bytes = gguf_fixture(&[("general.name", TestKv::Str("x"))]);
        bytes[0] = b'X';
        let path = write_scratch("gguf-magic.gguf", &bytes);
        let metadata = probe(&path, "gguf");
        std::fs::remove_file(&path).ok();
        assert_eq!(metadata, None);

        // Cut inside the prologue (tensor_count).
        let bytes = gguf_fixture(&[]);
        let path = write_scratch("gguf-trunc.gguf", &bytes[..9]);
        let metadata = probe(&path, "gguf");
        std::fs::remove_file(&path).ok();
        assert_eq!(metadata, None);
    }

    #[test]
    fn safetensors_counts_tensors_and_dtypes() {
        let header = br#"{"__metadata__":{"format":"pt","ss_base_model":"SDXL 1.0"},"emb":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"blk.0.w":{"dtype":"F16","shape":[2],"data_offsets":[16,20]},"blk.1.w":{"dtype":"F32","shape":[2],"data_offsets":[20,24]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        let path = write_scratch("st-ok.safetensors", &bytes);
        let metadata = probe(&path, "safetensors");
        std::fs::remove_file(&path).ok();
        assert_eq!(
            metadata,
            Some(ModelFileMetadata::Safetensors(SafetensorsMetadata {
                tensor_count: 3,
                dtypes: vec!["F16".to_owned(), "F32".to_owned()],
                base_model: Some("sdxl".to_owned()),
                identity_name: None,
                identity_model_name: None,
                identity_model_id: None,
                raw_identity: Vec::new(),
            }))
        );
    }

    #[test]
    fn safetensors_unknown_base_model_tag_is_preserved() {
        let header = br#"{"__metadata__":{"base_model":"flux-dev"},"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        let path = write_scratch("st-unknown.safetensors", &bytes);
        let metadata = probe(&path, "safetensors");
        std::fs::remove_file(&path).ok();
        assert_eq!(
            metadata,
            Some(ModelFileMetadata::Safetensors(SafetensorsMetadata {
                tensor_count: 1,
                dtypes: vec!["F16".to_owned()],
                base_model: Some("flux-dev".to_owned()),
                identity_name: None,
                identity_model_name: None,
                identity_model_id: None,
                raw_identity: Vec::new(),
            }))
        );
    }

    #[test]
    fn safetensors_raw_identity_is_preserved_separately_from_base_model() {
        let header = br#"{"__metadata__":{"general.name":"Weird Name","model_name":"ModelX","model_id":"org/repo","base_model":"SDXL 1.0"},"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        let path = write_scratch("st-identity.safetensors", &bytes);
        let metadata = probe(&path, "safetensors");
        std::fs::remove_file(&path).ok();
        let Some(ModelFileMetadata::Safetensors(meta)) = metadata else {
            panic!("expected safetensors metadata, got {metadata:?}");
        };
        assert_eq!(meta.identity_name.as_deref(), Some("Weird Name"));
        assert_eq!(meta.identity_model_name.as_deref(), Some("ModelX"));
        assert_eq!(meta.identity_model_id.as_deref(), Some("org/repo"));
        assert_eq!(meta.base_model.as_deref(), Some("sdxl"));
        assert_eq!(
            meta.raw_identity,
            vec![
                ("general.name".to_owned(), "Weird Name".to_owned()),
                ("model_id".to_owned(), "org/repo".to_owned()),
                ("model_name".to_owned(), "ModelX".to_owned()),
            ]
        );
    }

    #[test]
    fn safetensors_bad_prefix_is_none() {
        // Zero length prefix (invalid) and one over the size cap.
        for bad in [0u64, SAFETENSORS_MAX_HEADER + 1] {
            let path = write_scratch("st-bad.safetensors", &bad.to_le_bytes());
            let metadata = probe(&path, "safetensors");
            std::fs::remove_file(&path).ok();
            assert_eq!(metadata, None);
        }
    }

    #[test]
    fn other_extensions_and_missing_files_are_none() {
        assert_eq!(probe(Path::new("/nonexistent"), "ckpt"), None);
        assert_eq!(probe(Path::new("/nonexistent"), "gguf"), None);
    }
}
