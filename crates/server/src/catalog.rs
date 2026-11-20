//! Curated model catalog + fit pills (NEW-5): a checked-in JSON manifest
//! (`catalog.json`, `include_str!`), merged with the local scan snapshot
//! into `GET /v1/catalog`.
//!
//! Review-by-PR is the signature — no crypto, no auto-refresh: entries are
//! added/verified by PR and carry a `verified_date` (entries unverified for
//! 12 months are dropped at release). Matching against the scan snapshot is
//! by (modality, file name): downloads land flat in the managed roots, so
//! the local file name is the stable join key; a HF repo id is not
//! recoverable from arbitrary scan roots.
//!
//! Fit pills are advisory heuristics over weight bytes with per-kind
//! overhead factors (ReviewedNewItems-Approach §2): `llm` ×1.35 (KV cache +
//! activations at modest `n_ctx`), `sd` +2.5 GiB (activations at 1024²
//! dominate), `stt`/`tts` ×1.25. Integer math only (no float casts). The
//! budget comes from `[models] vram_budget_bytes` / `HONKBOX_VRAM_BUDGET`;
//! without one, `est_vram_bytes`/`pill` are omitted and entries show bytes
//! + quant only. Pills never gate anything — advisory DTO fields only.

use std::sync::LazyLock;

use honkbox_api_types::{CatalogEntry, CatalogList, CatalogLocal, ModelCatalog, ModelKind, Pill};

// Declared here (not at the crate root) so every `#[path]` test harness that
// includes routes.rs transitively compiles the license contract too.
#[path = "asset_packs.rs"]
pub(crate) mod asset_packs;
#[path = "license_gate.rs"]
pub(crate) mod license_gate;
#[path = "licenses.rs"]
pub(crate) mod licenses;

/// Checked-in curated asset-pack manifest (kept separate from the model
/// array so existing `catalog.json` consumers keep their array shape).
const PACKS_JSON: &str = include_str!("packs.json");

/// Curated packs, parsed once; a broken checked-in list is a programmer
/// error, like a broken model manifest.
#[must_use]
pub(crate) fn packs() -> &'static [asset_packs::PackEntry] {
    static PACKS: LazyLock<Vec<asset_packs::PackEntry>> =
        LazyLock::new(|| serde_json::from_str(PACKS_JSON).expect("bad packs.json"));
    &PACKS
}

/// Checked-in curated manifest (review-by-PR is the signature).
const MANIFEST_JSON: &str = include_str!("catalog.json");

/// `sd` activation overhead at 1024² (2.5 GiB, integer bytes).
const SD_OVERHEAD_BYTES: u64 = 2_684_354_560;

/// One parsed manifest row (shape mirrors `api_types::CatalogEntry` minus
/// the locally computed fields).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManifestEntry {
    /// Stable manifest id.
    pub id: String,
    /// `HuggingFace` repo id (`owner/name`).
    pub repo: String,
    /// File path inside `repo`.
    pub file: String,
    /// Modality.
    pub kind: ModelKind,
    /// Quantization/format tag.
    pub quant: String,
    /// Model license tag.
    pub license: String,
    /// Human-readable terms URL, when the source publishes one.
    #[serde(default)]
    pub license_url: Option<String>,
    /// Territory restriction (`excludes-eu-uk-kr`) or `None`.
    #[serde(default)]
    pub territory: Option<String>,
    /// `commercial` | `non-commercial` | `all-rights-reserved`.
    #[serde(default)]
    pub commercial_terms: String,
    /// HF repo-level gating (`auto` | `manual`); needs an accepted token.
    #[serde(default)]
    pub gated: bool,
    /// Canonical source (HF repo) URL.
    #[serde(default)]
    pub source_url: Option<String>,
    /// Published sha256, when the source publishes one (never fabricated).
    #[serde(default)]
    pub sha256: Option<String>,
    /// Mandatory attribution text for CC-BY-family sources.
    #[serde(default)]
    pub attribution_text: Option<String>,
    /// Manual verification month (`YYYY-MM`).
    pub verified_date: String,
    /// Short human note.
    pub notes: String,
}

/// The parsed manifest, loaded once (a broken checked-in manifest is a
/// programmer error, caught by the manifest unit test at review time).
pub fn manifest() -> &'static [ManifestEntry] {
    static MANIFEST: LazyLock<Vec<ManifestEntry>> = LazyLock::new(|| {
        serde_json::from_str(MANIFEST_JSON).expect("checked-in catalog manifest parses")
    });
    &MANIFEST
}

/// Advisory VRAM estimate for a weight file's bytes, per-kind factors over
/// integer math (saturating at `u64::MAX`).
#[must_use]
pub fn est_vram_bytes(kind: ModelKind, bytes: u64) -> u64 {
    match kind {
        ModelKind::Llm => bytes
            .checked_mul(135)
            .map_or(u64::MAX, |scaled| scaled / 100),
        ModelKind::Sd => bytes.saturating_add(SD_OVERHEAD_BYTES),
        ModelKind::Stt | ModelKind::Tts => bytes
            .checked_mul(125)
            .map_or(u64::MAX, |scaled| scaled / 100),
        // Non-download modalities never appear in the manifest; a 1:1
        // estimate is the honest neutral for anything that slips in.
        ModelKind::ControlNet | ModelKind::Lora | ModelKind::Embedding | ModelKind::Vae => bytes,
    }
}

/// Fit pill for weight `bytes` vs a `budget`, given the advisory `est`:
/// `est <= budget` → ok, `bytes <= budget < est` → slow, else fail.
#[must_use]
pub fn pill_for(bytes: u64, est: u64, budget: u64) -> Pill {
    if est <= budget {
        Pill::Ok
    } else if bytes <= budget {
        Pill::Slow
    } else {
        Pill::Fail
    }
}

/// Last path segment (works for both scan paths and manifest repo files).
fn file_name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Manifest entries merged with the scan snapshot, sorted by id. Local
/// presence matches by (kind, file name); pills compute only from the
/// matched file's bytes and the configured budget (`None` → advisory
/// fields omitted).
#[must_use]
pub fn build(snapshot: &ModelCatalog, budget: Option<u64>) -> CatalogList {
    let mut data: Vec<CatalogEntry> = manifest()
        .iter()
        .map(|m| {
            let local = snapshot
                .data
                .iter()
                .find(|entry| {
                    entry.kind == m.kind && file_name_of(&entry.path) == file_name_of(&m.file)
                })
                .map(|entry| {
                    let est = budget.map(|_| est_vram_bytes(m.kind, entry.bytes));
                    let pill =
                        est.map(|est| pill_for(entry.bytes, est, budget.unwrap_or_default()));
                    CatalogLocal {
                        // The matched snapshot entry's registry slug (its
                        // `id` may be an operator alias; REQ-2).
                        slug: entry.slug.clone(),
                        bytes: entry.bytes,
                        est_vram_bytes: est,
                        pill,
                    }
                });
            CatalogEntry {
                id: m.id.clone(),
                repo: m.repo.clone(),
                file: m.file.clone(),
                kind: m.kind,
                quant: m.quant.clone(),
                license: m.license.clone(),
                verified_date: m.verified_date.clone(),
                notes: m.notes.clone(),
                local,
            }
        })
        .collect();
    data.sort_by(|a, b| a.id.cmp(&b.id));
    CatalogList {
        object: "list".to_owned(),
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use honkbox_api_types::{ModelCatalog, ModelCatalogEntry};

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Snapshot helper (no header metadata — pills need bytes only).
    fn snapshot_entry(kind: ModelKind, path: &str, bytes: u64) -> ModelCatalogEntry {
        ModelCatalogEntry {
            id: path.trim_start_matches('/').to_owned(),
            slug: path.trim_start_matches('/').to_owned(),
            kind,
            owned_by: "local".to_owned(),
            context_length: None,
            max_output: None,
            thinking: false,
            tool_calling: false,
            modalities: vec!["text".to_owned()],
            capabilities: honkbox_api_types::ModelCapabilities::for_kind(kind),
            category: "root".to_owned(),
            path: path.to_owned(),
            bytes,
            metadata: None,
        }
    }

    fn snapshot(entries: Vec<ModelCatalogEntry>) -> ModelCatalog {
        ModelCatalog {
            object: "list".to_owned(),
            data: entries,
        }
    }

    /// The checked-in manifest parses, stays ≤30 entries, and every row is
    /// complete (ids unique, dates current, review fields filled).
    #[test]
    fn manifest_is_valid_and_bounded() {
        let manifest = manifest();
        assert!(!manifest.is_empty(), "manifest must not be empty");
        assert!(manifest.len() <= 30, "≤30 entries per the NEW-5 review");
        let mut ids: Vec<&str> = Vec::new();
        for entry in manifest {
            assert!(!entry.id.is_empty() && !entry.repo.is_empty());
            assert!(entry.repo.split_once('/').is_some(), "repo is owner/name");
            assert!(!entry.file.is_empty() && !entry.quant.is_empty());
            assert!(!entry.license.is_empty() && !entry.notes.is_empty());
            assert_eq!(entry.verified_date, "2026-11", "verified this release");
            assert!(matches!(
                entry.kind,
                ModelKind::Llm | ModelKind::Sd | ModelKind::Stt | ModelKind::Tts
            ));
            ids.push(&entry.id);
        }
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), manifest.len(), "ids unique");
    }

    /// Per-kind estimate factors: llm ×1.35, sd +2.5 GiB, stt/tts ×1.25.
    #[test]
    fn est_vram_per_kind_factors() {
        assert_eq!(est_vram_bytes(ModelKind::Llm, 100), 135);
        assert_eq!(est_vram_bytes(ModelKind::Stt, 100), 125);
        assert_eq!(est_vram_bytes(ModelKind::Tts, 100), 125);
        assert_eq!(est_vram_bytes(ModelKind::Sd, 1024), 1024 + 2_684_354_560);
        assert_eq!(
            est_vram_bytes(ModelKind::Sd, 4 * GIB),
            4 * GIB + 2_684_354_560
        );
        // Saturating, never wrapping/panicking.
        assert_eq!(est_vram_bytes(ModelKind::Llm, u64::MAX), u64::MAX);
        assert_eq!(est_vram_bytes(ModelKind::Sd, u64::MAX), u64::MAX);
    }

    /// Pill boundaries: `est <= budget` ok, `bytes <= budget < est` slow,
    /// `bytes > budget` fail — inclusive at both edges.
    #[test]
    fn pill_boundaries() {
        let (bytes, est) = (100, 135);
        assert_eq!(pill_for(bytes, est, 135), Pill::Ok);
        assert_eq!(pill_for(bytes, est, 134), Pill::Slow);
        assert_eq!(pill_for(bytes, est, 100), Pill::Slow);
        assert_eq!(pill_for(bytes, est, 99), Pill::Fail);
    }

    /// Merge: present entries carry the matched slug + bytes, absent ones
    /// have no `local`; matching is by (kind, file name).
    #[test]
    fn merge_matches_present_and_absent() {
        let snap = snapshot(vec![
            snapshot_entry(
                ModelKind::Llm,
                "/models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                808,
            ),
            snapshot_entry(ModelKind::Stt, "/models/stt/ggml-base.bin", 147),
        ]);
        let list = build(&snap, None);
        let by_id = |id: &str| {
            list.data
                .iter()
                .find(|entry| entry.id == id)
                .unwrap_or_else(|| panic!("{id} in list"))
        };
        let llama = by_id("llama-3.2-1b-q4");
        let local = llama.local.as_ref().expect("llama present");
        assert_eq!(local.bytes, 808);
        assert!(local.slug.ends_with("Llama-3.2-1B-Instruct-Q4_K_M.gguf"));
        let whisper = by_id("whisper-base");
        let local = whisper.local.as_ref().expect("whisper present");
        assert_eq!(local.bytes, 147);
        assert!(local.slug.ends_with("ggml-base.bin"));
        // Same file name under the WRONG kind must not match.
        let wrong = snapshot(vec![snapshot_entry(
            ModelKind::Tts,
            "/models/tts/ggml-base.bin",
            147,
        )]);
        assert!(
            build(&wrong, None)
                .data
                .iter()
                .all(|entry| entry.id != "whisper-base" || entry.local.is_none())
        );
        // No matching file anywhere → no `local`.
        let flux = by_id("flux1-schnell-bf16");
        assert!(flux.local.is_none(), "flux absent from the snapshot");
    }

    /// Budget on: advisory fields present and correct; budget off: omitted
    /// even when the file is present.
    #[test]
    fn budget_gates_advisory_fields() {
        let snap = snapshot(vec![snapshot_entry(
            ModelKind::Llm,
            "/models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            100,
        )]);
        // No budget: bytes only.
        let list = build(&snap, None);
        let local = &list
            .data
            .iter()
            .find(|e| e.id == "llama-3.2-1b-q4")
            .expect("entry")
            .local;
        let local = local.as_ref().expect("present");
        assert_eq!(local.est_vram_bytes, None);
        assert_eq!(local.pill, None);
        // Budget 135: green at the boundary.
        let list = build(&snap, Some(135));
        let local = list
            .data
            .iter()
            .find(|e| e.id == "llama-3.2-1b-q4")
            .expect("entry")
            .local
            .as_ref()
            .expect("present");
        assert_eq!(local.est_vram_bytes, Some(135));
        assert_eq!(local.pill, Some(Pill::Ok));
        // Budget 134: yellow.
        let list = build(&snap, Some(134));
        let local = list
            .data
            .iter()
            .find(|e| e.id == "llama-3.2-1b-q4")
            .expect("entry")
            .local
            .as_ref()
            .expect("present");
        assert_eq!(local.pill, Some(Pill::Slow));
        // Budget 99: red.
        let list = build(&snap, Some(99));
        let local = list
            .data
            .iter()
            .find(|e| e.id == "llama-3.2-1b-q4")
            .expect("entry")
            .local
            .as_ref()
            .expect("present");
        assert_eq!(local.pill, Some(Pill::Fail));
    }

    /// Output sorted by id; envelope shape matches the other lists.
    #[test]
    fn output_sorted_with_envelope() {
        let list = build(&snapshot(Vec::new()), None);
        assert_eq!(list.object, "list");
        let mut ids: Vec<&str> = list.data.iter().map(|entry| entry.id.as_str()).collect();
        assert!(ids.is_sorted(), "entries sorted by id");
        let sorted = ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, sorted);
    }
}
