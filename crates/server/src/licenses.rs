//! Curated, machine-readable model license records (TASK-model-license-registry).
//!
//! One authoritative source: the `license*` fields on `catalog.json`'s
//! `ManifestEntry`. Every lookup joins on `(kind, file name)` — the same
//! stable key `catalog::build` and `resolve_loras` use — with a slug fallback
//! for consumers that have no live scan catalog. No weights are ever read.

use std::sync::LazyLock;

use honkbox_api_types::{ModelCatalog, ModelKind};

use crate::catalog::ManifestEntry;

/// One curated license record, shared by the serving gates, capabilities and
/// bundle-manifest renderers. All facts are curated; absent facts stay `None`
/// rather than being guessed.
#[allow(dead_code)] // shared contract: capabilities/manifest consumers land separately
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LicenseRecord {
    pub license: String,
    pub license_url: Option<String>,
    pub territory: Option<String>,
    pub commercial_terms: String,
    pub gated: bool,
    pub source_url: Option<String>,
    pub sha256: Option<String>,
    pub attribution_text: Option<String>,
}

/// Build records from an already-parsed manifest — the one conversion used by
/// both the shipped snapshot and inline test fixtures.
pub(crate) fn records_from(manifest: &[ManifestEntry]) -> Vec<LicenseRecord> {
    manifest.iter().map(record_of).collect()
}

fn record_of(entry: &ManifestEntry) -> LicenseRecord {
    LicenseRecord {
        license: entry.license.clone(),
        license_url: entry.license_url.clone(),
        territory: entry.territory.clone(),
        commercial_terms: entry.commercial_terms.clone(),
        gated: entry.gated,
        source_url: entry.source_url.clone(),
        sha256: entry.sha256.clone(),
        attribution_text: entry.attribution_text.clone(),
    }
}

static RECORDS: LazyLock<Vec<LicenseRecord>> =
    LazyLock::new(|| records_from(crate::catalog::manifest()));

/// Every curated record, borrowed from the immutable shipped snapshot.
#[allow(dead_code)] // shared contract: capabilities/manifest consumers land separately
pub(crate) fn records() -> impl Iterator<Item = &'static LicenseRecord> {
    RECORDS.iter()
}

fn file_name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn kind_name(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "llm",
        ModelKind::Sd => "sd",
        ModelKind::ControlNet => "controlnet",
        ModelKind::Lora => "lora",
        ModelKind::Embedding => "embedding",
        ModelKind::Vae => "vae",
        ModelKind::Stt => "stt",
        ModelKind::Tts => "tts",
    }
}

fn kind_matches(kind: &str, expected: ModelKind) -> bool {
    kind.trim().eq_ignore_ascii_case(kind_name(expected))
}

/// Look up by `(kind, weight file name)`, ignoring directory prefixes.
#[allow(dead_code)] // shared contract: manifest consumers land separately
pub(crate) fn record_for_entry(kind: &str, file_name: &str) -> Option<LicenseRecord> {
    crate::catalog::manifest()
        .iter()
        .find(|m| kind_matches(kind, m.kind) && file_name_of(&m.file) == file_name)
        .map(record_of)
}

/// Resolve a request slug to its curated record. Prefers the live catalog
/// entry's `(kind, path)`; falls back to a curated manifest id or a slug that
/// embeds the weight file name (callers without a scan catalog).
pub(crate) fn record_for_slug(slug: &str, catalog: &ModelCatalog) -> Option<LicenseRecord> {
    let slug = slug.trim();
    if let Some(entry) = catalog.data.iter().find(|e| e.id == slug || e.slug == slug)
        && let Some(record) = record_for_entry(kind_name(entry.kind), file_name_of(&entry.path))
    {
        return Some(record);
    }
    let manifest = crate::catalog::manifest();
    manifest
        .iter()
        .find(|m| m.id == slug)
        .or_else(|| {
            manifest
                .iter()
                .find(|m| file_name_of(&m.file) == file_name_of(slug))
        })
        .map(record_of)
}

/// Whether a record permits bundling/redistribution. Local-only use is a
/// separate question: an undeclared local file is served with a warning.
#[allow(dead_code)] // shared contract: manifest consumers land separately
pub(crate) fn is_redistributable(record: &LicenseRecord) -> bool {
    let license = record.license.trim();
    !license.is_empty()
        && !license.eq_ignore_ascii_case("all-rights-reserved")
        && !record
            .commercial_terms
            .eq_ignore_ascii_case("non-commercial")
        && !record
            .commercial_terms
            .eq_ignore_ascii_case("all-rights-reserved")
        && record.territory.is_none()
        && !record.gated
}
