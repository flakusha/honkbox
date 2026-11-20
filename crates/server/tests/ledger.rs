//! Ledger admission contracts at the HTTP surface (TASK-resource-ledger-
//! bounded-queue): the hard-capacity 503 for a NEVER-fitting demand, the
//! proof that a configured budget which FITS admits normally, and the
//! default-shape fast path. This binary owns its own process globals, so it
//! may reconfigure `residency::global()` without racing sibling suites
//! (each cargo test binary is a separate process); its own tests serialize
//! on a lock because each one installs a different ledger config.

// No lib target on this binary crate, so include the modules directly.
#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "../src/artifacts.rs"]
mod artifacts;
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/container.rs"]
mod container;
#[allow(dead_code)]
#[path = "../src/downloads.rs"]
mod downloads;
#[allow(dead_code)]
#[path = "../src/formats.rs"]
mod formats;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/jobs_registry.rs"]
mod jobs_registry;
#[allow(dead_code)]
#[path = "../src/metrics.rs"]
mod metrics;
#[allow(dead_code)]
#[path = "../src/ora.rs"]
mod ora;
#[allow(dead_code)]
#[path = "../src/raster.rs"]
mod raster;
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;

use std::sync::{Arc, LazyLock, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes this binary's tests: each installs a different ledger config
/// on the process-global, so they must never overlap.
static LEDGER_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One fake LLM entry (served id == registry slug).
fn entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        modality: honkbox_api_types::ModelModalities::empty(),
        id: id.to_owned(),
        slug: id.to_owned(),
        kind: ModelKind::Llm,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities: ModelCapabilities::for_kind(ModelKind::Llm),
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

fn catalog_with(ids: &[&str]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: ids.iter().map(|id| entry(id)).collect(),
    }
}

fn test_app(catalog: ModelCatalog) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            None.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

fn completion_body(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })
}

/// `POST /v1/chat/completions`; returns (status, headers, body text).
async fn completions(app: Router, body: Value) -> (StatusCode, HeaderMap, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
    (
        status,
        headers,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

/// The serve-startup default shape (one GPU slot, unmanaged budgets), made
/// explicit so this test is order-independent under the lock.
#[tokio::test]
async fn default_ledger_admits_the_fast_path() {
    let _lock = LEDGER_LOCK.lock().await;
    honkbox_residency::global().configure(honkbox_residency::LedgerConfig::new());
    let (status, headers, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The 429-only middleware hint must NOT appear on a 200.
    assert!(headers.get("retry-after").is_none());
}

/// A demand that can NEVER fit the configured VRAM budget is a 503 naming
/// the binding dimension — even though the GPU slot itself is idle — and
/// carries no `Retry-After` (middleware patches 429s only; a config error
/// has no meaningful retry hint).
#[tokio::test]
async fn never_fitting_demand_is_503_naming_the_budget() {
    let _lock = LEDGER_LOCK.lock().await;
    let mut config = honkbox_residency::LedgerConfig::new();
    config.capacities.vram_bytes = 100;
    config
        .kind_mut(honkbox_residency::ResourceKind::Llm)
        .demand
        .vram_bytes = 200;
    honkbox_residency::global().configure(config);

    let (status, headers, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert!(body.contains("llm"), "{body}");
    assert!(body.contains("vram budget"), "{body}");
    assert!(headers.get("retry-after").is_none(), "{headers:?}");

    // The ledger counted the hard rejection (bounded-label exposition).
    let exposition = honkbox_residency::global().exposition();
    assert!(
        exposition.contains("honkbox_ledger_rejections_total{kind=\"llm\",reason=\"capacity\"} 1")
    );
}

/// A configured budget that FITS admits normally — budgets gate, they do
/// not reject fitting work.
#[tokio::test]
async fn fitting_budget_admits_normally() {
    let _lock = LEDGER_LOCK.lock().await;
    let mut config = honkbox_residency::LedgerConfig::new();
    config.capacities.vram_bytes = 100;
    config
        .kind_mut(honkbox_residency::ResourceKind::Llm)
        .demand
        .vram_bytes = 30;
    honkbox_residency::global().configure(config);

    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
