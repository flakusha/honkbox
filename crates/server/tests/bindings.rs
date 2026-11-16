//! REQ-2 operator bindings: the registry-cache roundtrip (aliases +
//! capability overrides survive a restart with empty model roots).
//!
//! Its own binary on purpose: the global `HONKBOX_REGISTRY_DB` override is
//! process-local, and the other server test binaries toggle it in parallel.

// No lib target on this binary crate, so include the modules directly.
// This binary exercises only the binding/cache path of `routes.rs`, so
// every included module carries `allow(dead_code)` for the unused rest.
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/downloads.rs"]
mod downloads;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/metrics.rs"]
mod metrics;
#[allow(dead_code)]
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use crate::config::ModelCapabilitiesOverride;
use honkbox_api_types::ModelKind;
use routes::refresh_catalog;

/// One scan hit for a synthetic file path.
fn scan_hit(slug: &str, path: &str) -> honkbox_registry::FoundModel {
    honkbox_registry::FoundModel {
        root: honkbox_registry::RootKind::Llm,
        category: "root".to_owned(),
        path: std::path::PathBuf::from(path),
        slug: slug.to_owned(),
        bytes: 1,
    }
}

/// One `[[models.entry]]` declaration.
fn model_entry(
    slug: Option<&str>,
    file_match: Option<&str>,
    aliases: &[&str],
    capabilities: ModelCapabilitiesOverride,
) -> config::ModelEntryConfig {
    config::ModelEntryConfig {
        slug: slug.map(str::to_owned),
        file_match: file_match.map(str::to_owned),
        aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        capabilities,
    }
}

/// The registry cache round-trips bindings: a restart with empty roots
/// serves the cached catalog with cached aliases + overrides (metadata is
/// re-probed only on the next real scan, so it stays absent here).
#[test]
fn cache_roundtrips_aliases_and_overrides() {
    // Hermetic cache path; nothing else runs in this binary.
    let base = std::env::temp_dir().join(format!("honkbox-req2-roundtrip-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("test dir");
    honkbox_registry::db::set_db_override(Some(base.join("registry.sqlite3")));
    let file = base.join("m.gguf");
    std::fs::write(&file, b"x").expect("write");
    let found = vec![scan_hit("llm:m.gguf", &file.display().to_string())];
    let entries = vec![model_entry(
        Some("llm:m.gguf"),
        None,
        &["qwen3-embed"],
        ModelCapabilitiesOverride {
            chat: Some(false),
            embeddings: Some(true),
            ..Default::default()
        },
    )];
    // First boot: live scan, cache synced.
    let catalog = refresh_catalog(&found, Some(&base), &entries).expect("first boot");
    assert_eq!(catalog.data[0].id, "qwen3-embed");
    assert_eq!(catalog.data[0].slug, "llm:m.gguf");
    assert_eq!(catalog.data[0].kind, ModelKind::Embedding);

    // Restart with empty roots: the cached fallback carries the alias and
    // the capability reclassification.
    let catalog = refresh_catalog(&[], Some(&base), &entries).expect("empty roots");
    assert_eq!(catalog.data.len(), 1);
    assert_eq!(catalog.data[0].id, "qwen3-embed");
    assert_eq!(catalog.data[0].slug, "llm:m.gguf");
    assert_eq!(catalog.data[0].kind, ModelKind::Embedding);
    assert!(catalog.data[0].capabilities.embeddings);
    assert!(!catalog.data[0].capabilities.chat);
    honkbox_registry::db::set_db_override(None);
    std::fs::remove_dir_all(&base).ok();
}

/// Raw-slug dispatch through the production catalog loader while an alias
/// owns the entry's `id`: both the alias and the registry slug hit the
/// lookup (the subsequent load fails on the fake path — `Load`, not
/// `UnknownSlug`), and a truly unknown value still misses.
#[test]
fn catalog_loader_resolves_alias_and_slug() {
    use honkbox_api_types::{ModelCapabilities, ModelCatalogEntry};

    let catalog = std::sync::Arc::new(std::sync::RwLock::new(std::sync::Arc::new(
        honkbox_api_types::ModelCatalog {
            object: "list".to_owned(),
            data: vec![ModelCatalogEntry {
                id: "embed-0.6b".to_owned(),
                slug: "llm:m.gguf".to_owned(),
                kind: ModelKind::Llm,
                owned_by: "local".to_owned(),
                context_length: None,
                max_output: None,
                thinking: false,
                tool_calling: false,
                modalities: vec!["text".to_owned()],
                capabilities: ModelCapabilities::for_kind(ModelKind::Llm),
                category: "root".to_owned(),
                path: "/nonexistent/m.gguf".to_owned(),
                bytes: 1,
                metadata: None,
            }],
        },
    )));
    let loader = swap::catalog_llm_loader(catalog);

    // Raw registry slug: lookup must hit (load failure names the path).
    // (`expect_err` is unavailable — the Ok side isn't `Debug`.)
    let Err(err) = loader("llm:m.gguf") else {
        panic!("fake weights must not load");
    };
    assert!(
        matches!(err, swap::SwapError::Load(ref message) if message.contains("m.gguf")),
        "raw slug missed the lookup: {err}"
    );

    // The alias hits the same entry.
    let Err(err) = loader("embed-0.6b") else {
        panic!("fake weights must not load");
    };
    assert!(
        matches!(err, swap::SwapError::Load(_)),
        "alias missed the lookup: {err}"
    );

    // Unknown values still miss the lookup entirely.
    let Err(err) = loader("nope.gguf") else {
        panic!("unknown slug must not resolve");
    };
    assert!(matches!(err, swap::SwapError::UnknownSlug(_)), "{err}");
}
