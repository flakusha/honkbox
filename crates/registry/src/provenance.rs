//! Offline identity provenance for catalog entries.
//!
//! One verdict per registry slug, derived only from local header bytes, an
//! HF-cache path shape, and locally curated identity rows. Nothing here reads
//! the clock, touches the filesystem or fetches anything; `probed_at` is
//! supplied by the caller and excluded from the canonical projection.

use std::collections::BTreeSet;
use std::path::{Component, Path};

use crate::metadata::ModelFileMetadata;
use crate::{CatalogEntry, ModelKind};

/// Version of the persisted provenance record shape.
pub const PROVENANCE_SCHEMA_VERSION: u32 = 1;
/// Version of the header/path probe logic that produced the record.
pub const PROBE_VERSION: u32 = 1;

/// How much of a model's identity the local evidence establishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityStatus {
    /// Local evidence pins the entry to one identity.
    Identified,
    /// Local evidence points at more than one candidate identity.
    Ambiguous,
    /// Local evidence establishes nothing about the entry's identity.
    Unknown,
}

/// Strength of the identity verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Exact identity match, corroborated by a verified digest.
    High,
    /// Exact identity match, or a header-only identification.
    Medium,
    /// Partial evidence, such as a base model or cache path alone.
    Low,
    /// No usable evidence, or conflicting evidence.
    Unknown,
}

/// A locally curated identity claim (never fetched at probe time).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CuratedIdentity {
    /// Catalog model kind the claim applies to.
    pub kind: String,
    /// Upstream repository identifier, in `owner/name` form.
    pub repo_id: String,
    /// Full upstream revision the claim is pinned to.
    pub revision: String,
    /// Path of the file within the upstream revision.
    pub source_path: String,
    /// Expected digest of the file, when known.
    pub digest: Option<String>,
    /// Declared model name, when known.
    pub name: Option<String>,
    /// Declared model architecture, when known.
    pub architecture: Option<String>,
}

/// The complete offline provenance verdict for one catalog entry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProvenanceRecord {
    /// Catalog slug the record describes.
    pub slug: String,
    /// Version of the persisted record shape.
    pub schema_version: u32,
    /// Version of the probe logic that produced the record.
    pub probe_version: u32,
    /// Caller-supplied probe timestamp; excluded from the canonical projection.
    #[serde(skip)]
    pub probed_at: u64,
    /// Size of the model file in bytes.
    pub file_size: u64,
    /// Verified digest of the model file, when known.
    pub file_sha256: Option<String>,
    /// Source provider of the model file, such as `hf` or `user_supplied`.
    pub provider: String,
    /// Upstream repository identifier, when the path shape reveals one.
    pub repo_id: Option<String>,
    /// Upstream revision, when the path shape reveals one.
    pub revision: Option<String>,
    /// File path within the upstream revision, when known.
    pub source_path: Option<String>,
    /// Model name declared by the file header, when present.
    pub declared_name: Option<String>,
    /// Model architecture declared by the file header, when present.
    pub architecture: Option<String>,
    /// Base model declared by the file header, when present.
    pub base_model: Option<String>,
    /// How much of the entry's identity the local evidence establishes.
    pub identity_status: IdentityStatus,
    /// Strength of the identity verdict.
    pub confidence: Confidence,
    /// Stable machine-readable reason for the verdict.
    pub reason_code: String,
    /// Sorted evidence tokens supporting the verdict.
    pub evidence: Vec<String>,
}

impl ProvenanceRecord {
    /// Stable wire/determinism projection: `probed_at` is deliberately absent.
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("provenance record is serializable")
    }
}

/// Repo/revision/file decoded from `models--<owner>--<name>/snapshots/<sha>/<file>`.
struct HfPath {
    repo_id: String,
    revision: String,
    source_path: String,
}

/// Identity values carried by a parsed header.
struct HeaderIdentity {
    evidence: &'static str,
    declared_name: Option<String>,
    architecture: Option<String>,
    base_model: Option<String>,
}

/// Canonical string for a [`crate::ModelKind`] (mirrors its `Serialize` impl).
fn kind_str(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "llm",
        ModelKind::Sd => "sd",
        ModelKind::ControlNet => "controlnet",
        ModelKind::Lora => "lora",
        ModelKind::Embedding => "embedding",
        ModelKind::Vae => "vae",
        ModelKind::Stt => "stt",
        ModelKind::Tts => "tts",
        ModelKind::Ocr => "ocr",
    }
}

/// Decode a valid HF-cache snapshot path from normal path components only.
/// A path string is corroboration, never proof that remote bytes existed.
fn parse_hf_path(path: &Path) -> Option<HfPath> {
    let components: Vec<&str> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    let start = components
        .iter()
        .rposition(|c| c.starts_with("models--") && c.splitn(3, "--").count() == 3)?;
    let mut repo = components[start].splitn(3, "--");
    let _ = repo.next(); // "models"
    let owner = repo.next()?;
    let name = repo.next()?;
    if owner.is_empty() || name.is_empty() {
        return None;
    }
    if components.get(start + 1).copied()? != "snapshots" {
        return None;
    }
    let revision = components.get(start + 2).copied()?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return None;
    }
    let file_parts = &components[start + 3..];
    if file_parts.is_empty() {
        return None;
    }
    Some(HfPath {
        repo_id: format!("{owner}/{name}"),
        revision: revision.to_owned(),
        source_path: file_parts.join("/"),
    })
}

/// Raw safetensors identity keys, preserved verbatim by the probe.
fn safetensors_identity(m: &crate::metadata::SafetensorsMetadata) -> Option<String> {
    m.identity_name
        .clone()
        .or_else(|| m.identity_model_name.clone())
        .or_else(|| m.identity_model_id.clone())
        .or_else(|| {
            m.raw_identity
                .iter()
                .find(|(key, _)| key == "general.name")
                .map(|(_, value)| value.clone())
        })
        .or_else(|| {
            m.raw_identity
                .iter()
                .find(|(key, _)| key == "model_name")
                .map(|(_, value)| value.clone())
        })
        .or_else(|| {
            m.raw_identity
                .iter()
                .find(|(key, _)| key == "model_id")
                .map(|(_, value)| value.clone())
        })
}

fn header_identity(metadata: Option<&ModelFileMetadata>) -> Option<HeaderIdentity> {
    match metadata? {
        ModelFileMetadata::Gguf(g) => Some(HeaderIdentity {
            evidence: "gguf_header",
            declared_name: g.name.clone(),
            architecture: g.architecture.clone(),
            base_model: None,
        }),
        ModelFileMetadata::Safetensors(s) => Some(HeaderIdentity {
            evidence: "safetensors_header",
            declared_name: safetensors_identity(s),
            architecture: None,
            base_model: s.base_model.clone(),
        }),
    }
}

/// A curated row matches only on kind + repo + full revision + source path,
/// with digest agreement required whenever the row supplies a digest.
fn curated_matches(
    c: &CuratedIdentity,
    kind: &str,
    hf: Option<&HfPath>,
    digest: Option<&str>,
) -> bool {
    c.kind == kind
        && hf.is_some_and(|p| {
            c.repo_id == p.repo_id && c.revision == p.revision && c.source_path == p.source_path
        })
        && c.digest.as_deref().is_none_or(|d| digest == Some(d))
}

fn curated_conflicts_with_header(c: &CuratedIdentity, h: &HeaderIdentity) -> bool {
    c.name
        .as_deref()
        .is_some_and(|n| h.declared_name.as_deref().is_some_and(|d| n != d))
        || c.architecture
            .as_deref()
            .is_some_and(|a| h.architecture.as_deref().is_some_and(|d| a != d))
}

fn normalized(s: &str) -> String {
    s.chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

/// Compare a header name with the repo-name and file-stem components of a
/// valid HF path; a name matching neither is a conflict, not corroboration.
fn header_conflicts_with_path(name: Option<&str>, hf: &HfPath) -> bool {
    let Some(name) = name else {
        return false;
    };
    let name = normalized(name);
    if name.is_empty() {
        return false;
    }
    let repo_name = hf.repo_id.rsplit('/').next().unwrap_or(&hf.repo_id);
    let file = hf.source_path.rsplit('/').next().unwrap_or(&hf.source_path);
    let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
    name != normalized(repo_name) && name != normalized(stem)
}

/// Derive the verdict tuple from the collected local evidence.
fn compute_verdict(
    match_count: usize,
    header: Option<&HeaderIdentity>,
    single: Option<&CuratedIdentity>,
    verified_digest: Option<&str>,
    header_path_conflict: bool,
    has_hf_path: bool,
    supported_ext: bool,
) -> (IdentityStatus, Confidence, &'static str) {
    if match_count > 1 {
        (
            IdentityStatus::Ambiguous,
            Confidence::Unknown,
            "duplicate_curated_match",
        )
    } else if let Some(h) = header {
        header_verdict(h, single, verified_digest, header_path_conflict)
    } else if let Some(c) = single {
        curated_verdict(c, verified_digest)
    } else if has_hf_path {
        (IdentityStatus::Ambiguous, Confidence::Low, "hf_path_only")
    } else if verified_digest.is_some() {
        (IdentityStatus::Unknown, Confidence::Unknown, "no_evidence")
    } else if supported_ext {
        (
            IdentityStatus::Unknown,
            Confidence::Unknown,
            "malformed_header",
        )
    } else {
        (
            IdentityStatus::Unknown,
            Confidence::Unknown,
            "unsupported_format",
        )
    }
}

/// Verdict for an entry whose header declares a name or an architecture.
fn header_verdict(
    h: &HeaderIdentity,
    single: Option<&CuratedIdentity>,
    verified_digest: Option<&str>,
    header_path_conflict: bool,
) -> (IdentityStatus, Confidence, &'static str) {
    if h.declared_name.is_none() && h.architecture.is_none() {
        if h.base_model.is_some() {
            (
                IdentityStatus::Ambiguous,
                Confidence::Low,
                "base_model_only",
            )
        } else {
            (IdentityStatus::Unknown, Confidence::Unknown, "no_evidence")
        }
    } else if single.is_some_and(|c| curated_conflicts_with_header(c, h)) {
        (
            IdentityStatus::Ambiguous,
            Confidence::Unknown,
            "header_curated_conflict",
        )
    } else if header_path_conflict {
        (
            IdentityStatus::Ambiguous,
            Confidence::Unknown,
            "header_path_conflict",
        )
    } else if single
        .is_some_and(|c| verified_digest.is_some() && c.digest.as_deref() == verified_digest)
    {
        (
            IdentityStatus::Identified,
            Confidence::High,
            "exact_identity_digest_match",
        )
    } else if single.is_some() {
        (
            IdentityStatus::Identified,
            Confidence::Medium,
            "exact_identity_match",
        )
    } else {
        (
            IdentityStatus::Identified,
            Confidence::Medium,
            "header_only",
        )
    }
}

/// Verdict for an entry backed by exactly one curated row and no header.
fn curated_verdict(
    c: &CuratedIdentity,
    verified_digest: Option<&str>,
) -> (IdentityStatus, Confidence, &'static str) {
    if verified_digest.is_some() && c.digest.as_deref() == verified_digest {
        (
            IdentityStatus::Identified,
            Confidence::High,
            "exact_identity_digest_match",
        )
    } else {
        (
            IdentityStatus::Identified,
            Confidence::Medium,
            "exact_identity_match",
        )
    }
}

/// Derive one offline provenance verdict. No IO, no clock, no network.
pub fn evaluate(
    entry: &CatalogEntry,
    curated: &[CuratedIdentity],
    verified_digest: Option<&str>,
    probed_at: u64,
) -> ProvenanceRecord {
    let hf = parse_hf_path(&entry.path);
    let header = header_identity(entry.metadata.as_ref());
    let ext = entry
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let supported_ext = ext.eq_ignore_ascii_case("gguf") || ext.eq_ignore_ascii_case("safetensors");

    let mut evidence = BTreeSet::new();
    if hf.is_some() {
        evidence.insert("hf_cache_path");
    }
    if let Some(h) = &header {
        evidence.insert(h.evidence);
    }
    if verified_digest.is_some() {
        evidence.insert("verified_digest");
    }

    let matches: Vec<&CuratedIdentity> = curated
        .iter()
        .filter(|c| curated_matches(c, kind_str(entry.kind), hf.as_ref(), verified_digest))
        .collect();
    if !matches.is_empty() {
        evidence.insert("curated");
    }
    let single = matches.first().copied().filter(|_| matches.len() == 1);

    // A single curated row binds repo/revision/path explicitly, so the
    // heuristic header-name-vs-path-name comparison only applies when no
    // curated row matched.
    let header_path_conflict = match (single, header.as_ref(), hf.as_ref()) {
        (None, Some(h), Some(p)) => header_conflicts_with_path(h.declared_name.as_deref(), p),
        _ => false,
    };

    let mut declared_name = header.as_ref().and_then(|h| h.declared_name.clone());
    let mut architecture = header.as_ref().and_then(|h| h.architecture.clone());
    if let Some(c) = single
        && header.is_none()
    {
        if declared_name.is_none() {
            declared_name.clone_from(&c.name);
        }
        if architecture.is_none() {
            architecture.clone_from(&c.architecture);
        }
    }

    let verdict = compute_verdict(
        matches.len(),
        header.as_ref(),
        single,
        verified_digest,
        header_path_conflict,
        hf.is_some(),
        supported_ext,
    );

    ProvenanceRecord {
        slug: entry.slug.clone(),
        schema_version: PROVENANCE_SCHEMA_VERSION,
        probe_version: PROBE_VERSION,
        probed_at,
        file_size: entry.bytes,
        file_sha256: verified_digest.map(str::to_owned),
        provider: if hf.is_some() {
            "hf".to_owned()
        } else {
            "user_supplied".to_owned()
        },
        repo_id: hf.as_ref().map(|p| p.repo_id.clone()),
        revision: hf.as_ref().map(|p| p.revision.clone()),
        source_path: hf.as_ref().map(|p| p.source_path.clone()),
        declared_name,
        architecture,
        base_model: header.as_ref().and_then(|h| h.base_model.clone()),
        identity_status: verdict.0,
        confidence: verdict.1,
        reason_code: verdict.2.to_owned(),
        evidence: evidence.into_iter().map(str::to_owned).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{GgufMetadata, SafetensorsMetadata};

    const REV: &str = "0123456789abcdef0123456789abcdef01234567";
    const HF_PATH: &str = "/cache/huggingface/hub/models--acme--Widget-7B/snapshots/0123456789abcdef0123456789abcdef01234567/Widget-7B-Q4.gguf";
    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn entry(path: &str, kind: ModelKind, metadata: Option<ModelFileMetadata>) -> CatalogEntry {
        CatalogEntry {
            slug: "test-slug".to_owned(),
            kind,
            category: "test".to_owned(),
            path: std::path::PathBuf::from(path),
            bytes: 42,
            metadata,
        }
    }

    fn gguf(name: &str, arch: &str) -> ModelFileMetadata {
        ModelFileMetadata::Gguf(GgufMetadata {
            version: 3,
            tensor_count: 1,
            name: Some(name.to_owned()),
            architecture: Some(arch.to_owned()),
            context_length: None,
        })
    }

    fn safetensors(name: Option<&str>, base: Option<&str>) -> ModelFileMetadata {
        ModelFileMetadata::Safetensors(SafetensorsMetadata {
            tensor_count: 1,
            dtypes: vec!["F16".to_owned()],
            base_model: base.map(str::to_owned),
            identity_name: name.map(str::to_owned),
            identity_model_name: None,
            identity_model_id: None,
            raw_identity: name
                .map(|n| vec![("general.name".to_owned(), n.to_owned())])
                .unwrap_or_default(),
        })
    }

    fn curated(digest: Option<&str>) -> CuratedIdentity {
        CuratedIdentity {
            kind: "llm".to_owned(),
            repo_id: "acme/Widget-7B".to_owned(),
            revision: REV.to_owned(),
            source_path: "Widget-7B-Q4.gguf".to_owned(),
            digest: digest.map(str::to_owned),
            name: Some("Widget-7B".to_owned()),
            architecture: Some("llama".to_owned()),
        }
    }

    fn hf_entry(metadata: Option<ModelFileMetadata>) -> CatalogEntry {
        entry(HF_PATH, ModelKind::Llm, metadata)
    }

    #[test]
    fn curated_with_digest_is_high() {
        let record = evaluate(
            &hf_entry(Some(gguf("Widget-7B", "llama"))),
            &[curated(Some(DIGEST))],
            Some(DIGEST),
            7,
        );
        assert_eq!(record.slug, "test-slug");
        assert_eq!(record.identity_status, IdentityStatus::Identified);
        assert_eq!(record.confidence, Confidence::High);
        assert_eq!(record.reason_code, "exact_identity_digest_match");
        assert_eq!(record.file_sha256.as_deref(), Some(DIGEST));
        assert_eq!(record.provider, "hf");
        assert_eq!(record.repo_id.as_deref(), Some("acme/Widget-7B"));
        assert_eq!(record.revision.as_deref(), Some(REV));
        assert_eq!(
            record.evidence,
            vec!["curated", "gguf_header", "hf_cache_path", "verified_digest",]
        );
    }

    #[test]
    fn curated_without_digest_is_medium() {
        let record = evaluate(
            &hf_entry(Some(gguf("Widget-7B", "llama"))),
            &[curated(None)],
            None,
            7,
        );
        assert_eq!(record.identity_status, IdentityStatus::Identified);
        assert_eq!(record.confidence, Confidence::Medium);
        assert_eq!(record.reason_code, "exact_identity_match");
        assert_eq!(record.file_sha256, None);
    }

    #[test]
    fn header_only_is_medium() {
        let record = evaluate(
            &entry(
                "/models/Widget-7B-Q4.gguf",
                ModelKind::Llm,
                Some(gguf("Widget-7B", "llama")),
            ),
            &[],
            None,
            7,
        );
        assert_eq!(record.identity_status, IdentityStatus::Identified);
        assert_eq!(record.confidence, Confidence::Medium);
        assert_eq!(record.reason_code, "header_only");
        assert_eq!(record.provider, "user_supplied");
    }

    #[test]
    fn base_model_only_is_ambiguous() {
        let record = evaluate(
            &entry(
                "/models/plain.safetensors",
                ModelKind::Sd,
                Some(safetensors(None, Some("sd15"))),
            ),
            &[],
            None,
            7,
        );
        assert_eq!(record.identity_status, IdentityStatus::Ambiguous);
        assert_eq!(record.confidence, Confidence::Low);
        assert_eq!(record.reason_code, "base_model_only");
        assert_eq!(record.declared_name, None);
        assert_eq!(record.base_model.as_deref(), Some("sd15"));
    }

    #[test]
    fn hf_path_only_is_low() {
        let record = evaluate(&hf_entry(None), &[], None, 7);
        assert_eq!(record.identity_status, IdentityStatus::Ambiguous);
        assert_eq!(record.confidence, Confidence::Low);
        assert_eq!(record.reason_code, "hf_path_only");
        assert_eq!(record.provider, "hf");
        assert_eq!(record.source_path.as_deref(), Some("Widget-7B-Q4.gguf"));
        assert_eq!(record.evidence, vec!["hf_cache_path"]);
    }

    #[test]
    fn header_path_conflict_is_ambiguous() {
        let record = evaluate(
            &hf_entry(Some(gguf("Totally-Other", "llama"))),
            &[],
            None,
            7,
        );
        assert_eq!(record.identity_status, IdentityStatus::Ambiguous);
        assert_eq!(record.confidence, Confidence::Unknown);
        assert_eq!(record.reason_code, "header_path_conflict");
    }

    #[test]
    fn duplicate_curated_match_is_ambiguous() {
        let record = evaluate(
            &hf_entry(Some(gguf("Widget-7B", "llama"))),
            &[curated(None), curated(None)],
            None,
            7,
        );
        assert_eq!(record.identity_status, IdentityStatus::Ambiguous);
        assert_eq!(record.confidence, Confidence::Unknown);
        assert_eq!(record.reason_code, "duplicate_curated_match");
    }

    #[test]
    fn header_curated_conflict_is_ambiguous() {
        let mut bad = curated(Some(DIGEST));
        bad.name = Some("Wrong-Name".to_owned());
        let record = evaluate(
            &hf_entry(Some(gguf("Widget-7B", "llama"))),
            &[bad],
            Some(DIGEST),
            7,
        );
        assert_eq!(record.reason_code, "header_curated_conflict");
        assert_eq!(record.identity_status, IdentityStatus::Ambiguous);
        assert_eq!(record.confidence, Confidence::Unknown);
    }

    #[test]
    fn malformed_header_for_supported_formats() {
        for path in ["/models/x.gguf", "/models/x.safetensors"] {
            let record = evaluate(&entry(path, ModelKind::Llm, None), &[], None, 7);
            assert_eq!(record.reason_code, "malformed_header", "{path}");
            assert_eq!(record.identity_status, IdentityStatus::Unknown);
            assert_eq!(record.confidence, Confidence::Unknown);
        }
    }

    #[test]
    fn unsupported_format_is_unknown() {
        let record = evaluate(&entry("/models/x.onnx", ModelKind::Llm, None), &[], None, 7);
        assert_eq!(record.reason_code, "unsupported_format");
        assert_eq!(record.identity_status, IdentityStatus::Unknown);
        assert_eq!(record.confidence, Confidence::Unknown);
    }

    #[test]
    fn no_evidence_is_unknown() {
        let record = evaluate(
            &entry(
                "/models/x.safetensors",
                ModelKind::Sd,
                Some(safetensors(None, None)),
            ),
            &[],
            None,
            7,
        );
        assert_eq!(record.reason_code, "no_evidence");
        assert_eq!(record.identity_status, IdentityStatus::Unknown);
    }

    #[test]
    fn canonical_json_ignores_probed_at() {
        let e = hf_entry(Some(gguf("Widget-7B", "llama")));
        let first = evaluate(&e, &[curated(Some(DIGEST))], Some(DIGEST), 1);
        let second = evaluate(&e, &[curated(Some(DIGEST))], Some(DIGEST), 2);
        assert_eq!(first.probed_at, 1);
        assert_eq!(second.probed_at, 2);
        assert_eq!(first.canonical_json(), second.canonical_json());
        assert!(!first.canonical_json().contains("probed_at"));
    }
}
