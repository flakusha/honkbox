//! Integration coverage for registry-only provenance, via public APIs only.

use honkbox_registry::{
    CatalogEntry, Confidence, CuratedIdentity, Db, FoundModel, IdentityStatus, ModelFileMetadata,
    PROBE_VERSION, PROVENANCE_SCHEMA_VERSION, ProvenanceRecord, RootKind, evaluate,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique temp dir per test; removed on drop (including on panic).
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "honkbox-prov-it-{tag}-{}-{nanos}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self { dir }
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.join(rel)
    }

    fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(rel);
        std::fs::create_dir_all(path.parent().expect("scratch parent")).expect("create parent");
        std::fs::write(&path, bytes).expect("write scratch file");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// Minimal valid GGUF header: version 3, one tensor, string KVs only.
fn gguf(kvs: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, value) in kvs {
        push_str(&mut out, key);
        out.extend_from_slice(&8u32.to_le_bytes()); // GGUF string type
        push_str(&mut out, value);
    }
    out
}

/// Minimal safetensors file: header length + JSON header + tensor bytes.
fn safetensors(metadata_json: &str) -> Vec<u8> {
    let json = format!(
        "{{\"weight\":{{\"dtype\":\"F16\",\"shape\":[1]}},\"__metadata__\":{metadata_json}}}"
    );
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(&[0u8; 2]);
    out
}

fn found(root: RootKind, category: &str, slug: &str, path: PathBuf, bytes: u64) -> FoundModel {
    FoundModel {
        root,
        category: category.into(),
        path,
        slug: slug.into(),
        bytes,
    }
}

#[test]
fn gguf_header_and_curated_digest_yield_high_exact_identity() {
    let scratch = Scratch::new("gguf");
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let rel = format!("models--acme--widget/snapshots/{revision}/model.gguf");
    let bytes = gguf(&[
        ("general.name", "widget-7b"),
        ("general.architecture", "llama"),
    ]);
    let path = scratch.write(&rel, &bytes);
    let entry = CatalogEntry::from(&found(
        RootKind::Llm,
        "acme/widget",
        &format!("llm:{rel}"),
        path,
        bytes.len() as u64,
    ));
    let curated = CuratedIdentity {
        kind: "llm".into(),
        repo_id: "acme/widget".into(),
        revision: revision.into(),
        source_path: "model.gguf".into(),
        digest: Some("sha256:abcd".into()),
        name: Some("widget-7b".into()),
        architecture: Some("llama".into()),
    };

    let record = evaluate(&entry, &[curated], Some("sha256:abcd"), 42);
    assert_eq!(record.identity_status, IdentityStatus::Identified);
    assert_eq!(record.confidence, Confidence::High);
    assert_eq!(record.reason_code, "exact_identity_digest_match");
    assert_eq!(record.provider, "hf");
    assert_eq!(record.repo_id.as_deref(), Some("acme/widget"));
    assert_eq!(record.revision.as_deref(), Some(revision));
    assert_eq!(record.source_path.as_deref(), Some("model.gguf"));
    assert_eq!(record.declared_name.as_deref(), Some("widget-7b"));
    assert_eq!(record.architecture.as_deref(), Some("llama"));
    assert!(record.evidence.iter().any(|e| e == "gguf_header"));
    assert!(record.evidence.iter().any(|e| e == "curated"));
    assert!(record.evidence.iter().any(|e| e == "verified_digest"));
}

#[test]
fn safetensors_raw_identity_is_verbatim_and_base_model_normalized() {
    let scratch = Scratch::new("safetensors");
    let bytes = safetensors(
        r#"{"general.name":"My / SD 1.5 checkpoint","model_name":"my-model","model_id":"acme/my-model","base_model":"SD 1.5"}"#,
    );
    let path = scratch.write("plain/model.safetensors", &bytes);
    let entry = CatalogEntry::from(&found(
        RootKind::Comfy,
        "checkpoints",
        "comfy:checkpoints/model.safetensors",
        path,
        bytes.len() as u64,
    ));

    let Some(ModelFileMetadata::Safetensors(meta)) = &entry.metadata else {
        panic!("expected safetensors metadata");
    };
    assert_eq!(meta.base_model.as_deref(), Some("sd15"));
    assert_eq!(
        meta.identity_name.as_deref(),
        Some("My / SD 1.5 checkpoint")
    );
    assert_eq!(meta.identity_model_name.as_deref(), Some("my-model"));
    assert_eq!(meta.identity_model_id.as_deref(), Some("acme/my-model"));
    assert_eq!(
        meta.raw_identity,
        vec![
            (
                "general.name".to_owned(),
                "My / SD 1.5 checkpoint".to_owned()
            ),
            ("model_id".to_owned(), "acme/my-model".to_owned()),
            ("model_name".to_owned(), "my-model".to_owned()),
        ]
    );

    let record = evaluate(&entry, &[], None, 11);
    assert_eq!(
        record.declared_name.as_deref(),
        Some("My / SD 1.5 checkpoint")
    );
    assert_eq!(record.base_model.as_deref(), Some("sd15"));
}

#[test]
fn hf_cache_path_without_header_or_curated_is_ambiguous_low() {
    let scratch = Scratch::new("hf-path");
    let revision = "fedcba9876543210fedcba9876543210fedcba98";
    let rel = format!("models--acme--widget/snapshots/{revision}/tokenizer.bin");
    let path = scratch.write(&rel, b"weights");
    let entry = CatalogEntry::from(&found(
        RootKind::Llm,
        "acme/widget",
        &format!("llm:{rel}"),
        path,
        7,
    ));
    assert!(entry.metadata.is_none());

    let record = evaluate(&entry, &[], None, 7);
    assert_eq!(record.identity_status, IdentityStatus::Ambiguous);
    assert_eq!(record.confidence, Confidence::Low);
    assert_eq!(record.reason_code, "hf_path_only");
    assert_eq!(record.provider, "hf");
    assert_eq!(record.repo_id.as_deref(), Some("acme/widget"));
    assert_eq!(record.revision.as_deref(), Some(revision));
    assert_eq!(record.source_path.as_deref(), Some("tokenizer.bin"));
    assert_eq!(record.evidence, vec!["hf_cache_path".to_owned()]);
}

#[test]
fn provenance_survives_reopen_and_is_invalidated_on_byte_change() {
    let scratch = Scratch::new("db");
    let db_path = scratch.path("registry.sqlite3");
    let bytes = gguf(&[("general.name", "local"), ("general.architecture", "llama")]);
    let path = scratch.write("plain/model.gguf", &bytes);
    let slug = "comfy:checkpoints/model.gguf";
    let entry = CatalogEntry::from(&found(
        RootKind::Comfy,
        "checkpoints",
        slug,
        path.clone(),
        bytes.len() as u64,
    ));
    let record = evaluate(&entry, &[], None, 11);
    assert_eq!(record.slug, slug);

    {
        let db = Db::open(&db_path).expect("open db");
        db.upsert_entries(std::slice::from_ref(&entry), 11)
            .expect("upsert entry");
        db.upsert_provenance(std::slice::from_ref(&record))
            .expect("upsert provenance");
    }
    {
        let db = Db::open(&db_path).expect("reopen db");
        assert_eq!(db.load_entries().expect("load entries").len(), 1);
        let loaded = db.load_provenance().expect("load provenance");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].slug, slug);
        assert_eq!(loaded[0].canonical_json(), record.canonical_json());
    }

    let changed = gguf(&[
        ("general.name", "local-v2"),
        ("general.architecture", "llama"),
    ]);
    std::fs::write(&path, &changed).expect("rewrite fixture with new bytes");
    let changed_entry = CatalogEntry::from(&found(
        RootKind::Comfy,
        "checkpoints",
        slug,
        path,
        changed.len() as u64,
    ));
    {
        let db = Db::open(&db_path).expect("reopen db for invalidation");
        db.upsert_entries(std::slice::from_ref(&changed_entry), 12)
            .expect("upsert changed entry");
    }
    {
        let db = Db::open(&db_path).expect("reopen db after invalidation");
        assert_eq!(
            db.load_provenance().expect("load provenance"),
            Vec::<ProvenanceRecord>::new()
        );
    }
}

#[test]
fn canonical_json_ignores_probed_at() {
    let scratch = Scratch::new("canonical");
    let bytes = gguf(&[("general.name", "x"), ("general.architecture", "llama")]);
    let path = scratch.write("plain/model.gguf", &bytes);
    let entry = CatalogEntry::from(&found(
        RootKind::Comfy,
        "checkpoints",
        "comfy:checkpoints/model.gguf",
        path,
        bytes.len() as u64,
    ));
    let early: ProvenanceRecord = evaluate(&entry, &[], None, 1);
    let late: ProvenanceRecord = evaluate(&entry, &[], None, 9_999_999);

    assert_eq!(PROVENANCE_SCHEMA_VERSION, 1);
    assert_eq!(PROBE_VERSION, 1);
    assert_eq!(early.schema_version, PROVENANCE_SCHEMA_VERSION);
    assert_eq!(early.probe_version, PROBE_VERSION);
    assert_ne!(early.probed_at, late.probed_at);
    assert_eq!(early.canonical_json(), late.canonical_json());
    assert!(!early.canonical_json().contains("probed_at"));
}
