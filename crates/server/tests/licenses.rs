//! License registry + serving-gate contract tests (TASK-model-license-registry).
//!
//! Pure-function tests: an inline manifest fixture in a per-test temp dir, a
//! synthesized catalog, and no network / no shared mutable state. The one
//! shipped-data assertion is that every curated row carries a license.

#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use axum::http::StatusCode;
use catalog::license_gate::{LicenseRefusal, check, gate};
use catalog::licenses::{
    LicenseRecord, is_redistributable, record_for_entry, record_for_slug, records, records_from,
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

/// Tag-scoped temp dir, removed on drop (including the failure path).
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("honkbox-license-{tag}-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("per-test temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Five curated-style rows: commercial, territory, gated+NC, NC, no license.
const FIXTURE_MANIFEST: &str = r#"[
  {"id":"fixture-commercial","repo":"example/commercial","file":"commercial.gguf","kind":"llm","quant":"q4","license":"apache-2.0","license_url":"https://spdx.org/licenses/Apache-2.0.html","territory":null,"commercial_terms":"commercial","gated":false,"source_url":"https://example.test/commercial","sha256":null,"attribution_text":null,"verified_date":"2026-11","notes":"fixture"},
  {"id":"fixture-territory","repo":"example/territory","file":"territory.safetensors","kind":"sd","quant":"bf16","license":"tencent-hunyuan-community","license_url":null,"territory":"excludes-eu-uk-kr","commercial_terms":"commercial","gated":false,"source_url":"https://example.test/territory","sha256":null,"attribution_text":null,"verified_date":"2026-11","notes":"fixture"},
  {"id":"fixture-gated","repo":"example/gated","file":"gated.safetensors","kind":"sd","quant":"bf16","license":"flux-non-commercial-license","license_url":"https://example.test/license","territory":null,"commercial_terms":"non-commercial","gated":true,"source_url":"https://example.test/gated","sha256":null,"attribution_text":null,"verified_date":"2026-11","notes":"fixture"},
  {"id":"fixture-nc","repo":"example/nc","file":"nc.safetensors","kind":"sd","quant":"fp16","license":"cc-by-nc-4.0","license_url":"https://example.test/license","territory":null,"commercial_terms":"non-commercial","gated":false,"source_url":"https://example.test/nc","sha256":null,"attribution_text":"fixture attribution","verified_date":"2026-11","notes":"fixture"},
  {"id":"fixture-no-license","repo":"example/unknown","file":"unknown.safetensors","kind":"sd","quant":"fp16","license":"","license_url":null,"territory":null,"commercial_terms":"all-rights-reserved","gated":false,"source_url":null,"sha256":null,"attribution_text":null,"verified_date":"2026-11","notes":"fixture"}
]"#;

fn record(license: &str, terms: &str) -> LicenseRecord {
    LicenseRecord {
        license: license.to_owned(),
        license_url: Some("https://example.test/license".to_owned()),
        territory: None,
        commercial_terms: terms.to_owned(),
        gated: false,
        source_url: Some("https://example.test/source".to_owned()),
        sha256: None,
        attribution_text: None,
    }
}

fn inline_catalog() -> ModelCatalog {
    let data = catalog::manifest()
        .iter()
        .map(|row| ModelCatalogEntry {
            id: row.id.clone(),
            slug: row.id.clone(),
            kind: row.kind,
            owned_by: "fixture".to_owned(),
            context_length: None,
            max_output: None,
            thinking: false,
            tool_calling: false,
            modalities: Vec::new(),
            capabilities: ModelCapabilities::for_kind(row.kind),
            category: "fixture".to_owned(),
            path: format!("/models/{}", row.file),
            bytes: 1,
            metadata: None,
        })
        .collect();
    ModelCatalog {
        object: "list".to_owned(),
        data,
    }
}

#[test]
fn shipped_manifest_has_a_license_on_every_row() {
    let manifest = catalog::manifest();
    assert!(!manifest.is_empty());
    for row in manifest {
        assert!(!row.license.trim().is_empty(), "{} has no license", row.id);
    }
    assert_eq!(records().count(), manifest.len());
}

#[test]
fn inline_manifest_fixture_round_trips_into_records() {
    let dir = TempDir::new("fixture");
    let path = dir.path().join("manifest.json");
    fs::write(&path, FIXTURE_MANIFEST).expect("write fixture");
    let text = fs::read_to_string(&path).expect("read fixture");
    let rows: Vec<catalog::ManifestEntry> = serde_json::from_str(&text).expect("parse fixture");
    assert_eq!(rows.len(), 5);

    let recs = records_from(&rows);
    assert_eq!(recs[0].commercial_terms, "commercial");
    assert_eq!(recs[1].territory.as_deref(), Some("excludes-eu-uk-kr"));
    assert!(recs[2].gated);
    assert_eq!(
        recs[3].attribution_text.as_deref(),
        Some("fixture attribution")
    );
    assert_eq!(recs[4].license, "");
    assert!(is_redistributable(&recs[0]));
    assert!(!is_redistributable(&recs[1]));
    assert!(!is_redistributable(&recs[2]));
    assert!(!is_redistributable(&recs[3]));
    assert!(!is_redistributable(&recs[4]));
}

#[test]
fn lookups_join_on_kind_and_file_name() {
    let catalog = inline_catalog();
    let sdxl = record_for_entry("sd", "sd_xl_base_1.0.safetensors").expect("sdxl record");
    assert_eq!(sdxl.license, "openrail++");
    assert!(record_for_entry("llm", "sd_xl_base_1.0.safetensors").is_none());
    let by_slug = record_for_slug("sdxl-base-fp16", &catalog).expect("slug record");
    assert_eq!(by_slug, sdxl);
    let embedded = record_for_slug("comfy:checkpoints/sd_xl_base_1.0.safetensors", &catalog)
        .expect("slug with embedded file name");
    assert_eq!(embedded, sdxl);
    assert!(record_for_slug("not-a-model", &catalog).is_none());
}

#[test]
fn missing_record_refusal_is_named_412() {
    let refusal = check(
        "fixture-no-license",
        Some(record("", "commercial")),
        &config::Config::default(),
    )
    .unwrap_err();
    assert_eq!(refusal, LicenseRefusal::MissingRecord);
    assert_eq!(refusal.code(), "model_license_missing");
    assert_eq!(LicenseRefusal::status(), StatusCode::PRECONDITION_FAILED);
    assert!(refusal.to_string().contains("model_license_missing"));
}

#[test]
fn territory_refusal_needs_a_declared_outside_jurisdiction() {
    let mut restricted = record("tencent-hunyuan-community", "commercial");
    restricted.territory = Some("excludes-eu-uk-kr".to_owned());
    let eu = config::Config {
        territory: "eu".to_owned(),
        commercial: true,
    };
    let refusal = check("hunyuan", Some(restricted.clone()), &eu).unwrap_err();
    assert_eq!(
        refusal,
        LicenseRefusal::Territory {
            slug: "hunyuan".to_owned(),
            territory: "excludes-eu-uk-kr".to_owned(),
        }
    );
    assert_eq!(refusal.code(), "model_license_territory");
    assert!(refusal.message().contains("hunyuan"));
    assert!(refusal.message().contains("excludes-eu-uk-kr"));
    // The default `unknown` jurisdiction fails closed too.
    assert!(
        check(
            "hunyuan",
            Some(restricted.clone()),
            &config::Config::default()
        )
        .is_err()
    );
    let us = config::Config {
        territory: "us".to_owned(),
        commercial: true,
    };
    assert!(check("hunyuan", Some(restricted), &us).is_ok());
}

#[test]
fn gated_refusal_names_terms_and_missing_token() {
    let mut gated = record("flux-non-commercial-license", "non-commercial");
    gated.gated = true;
    let research = config::Config {
        territory: "us".to_owned(),
        commercial: false,
    };
    let refusal = check("flux2", Some(gated), &research).unwrap_err();
    match &refusal {
        LicenseRefusal::Gated {
            slug,
            terms_url,
            token_missing,
        } => {
            assert_eq!(slug, "flux2");
            assert_eq!(terms_url.as_deref(), Some("https://example.test/license"));
            assert!(*token_missing);
        }
        other => panic!("expected gated refusal, got {other:?}"),
    }
    assert_eq!(refusal.code(), "model_license_gated");
    assert_eq!(LicenseRefusal::status(), StatusCode::PRECONDITION_FAILED);
    assert!(refusal.message().contains("flux2"));
}

#[test]
fn non_commercial_refuses_commercial_but_research_can_opt_out() {
    let nc = record("cc-by-nc-4.0", "non-commercial");
    let refusal = check("zero123", Some(nc.clone()), &config::Config::default()).unwrap_err();
    assert_eq!(refusal.code(), "model_license_non_commercial");
    assert_eq!(LicenseRefusal::status(), StatusCode::PRECONDITION_FAILED);
    assert!(refusal.message().contains("zero123"));
    assert!(refusal.message().contains("https://example.test/license"));
    let research = config::Config {
        territory: "unknown".to_owned(),
        commercial: false,
    };
    assert!(check("zero123", Some(nc), &research).is_ok());
}

#[test]
fn undeclared_local_file_is_local_only_not_refused() {
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: Vec::new(),
    };
    let local = gate("local-only-lora", &catalog, &config::Config::default())
        .expect("undeclared local files stay local-only");
    assert_eq!(local.license, "all-rights-reserved");
    assert!(!is_redistributable(&local));
}

#[test]
fn config_knobs_default_and_env_wins() {
    let mut warnings = Vec::new();
    let defaults = config::resolve_licenses(None, None, None, None, &mut warnings);
    assert_eq!(defaults, config::Config::default());
    let overridden = config::resolve_licenses(
        Some(" eu "),
        Some(true),
        Some("us"),
        Some("false"),
        &mut warnings,
    );
    assert_eq!(overridden.territory, "us");
    assert!(!overridden.commercial);
    let invalid = config::resolve_licenses(None, None, Some(""), Some("nonsense"), &mut warnings);
    assert_eq!(invalid.territory, "unknown");
    assert!(invalid.commercial);
    assert_eq!(warnings.len(), 1);
}
