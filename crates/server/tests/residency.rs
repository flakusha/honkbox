//! REQ-5 residency control (explicit unload, `/v1/memory`, llama-swap
//! compat `/running` + `/api/models/unload`) and REQ-6 Option B (streaming
//! busy honesty: a held slot is a real HTTP 429 on both stream routes,
//! never a 200 with an error frame).

// No lib target on this binary crate, so include the modules directly.
// `allow(dead_code)`: each test binary exercises a slice of `routes.rs` /
// `swap.rs`, so the unused remainder must not trip `-D warnings`.
#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
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
#[allow(dead_code)]
#[path = "../src/artifacts.rs"]
mod artifacts;
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::{
    ChatRequest as WorkerChatRequest, ChatResponse, LlmError, LlmPipeline, SamplerParams, StubLlm,
};
use serde_json::Value;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};
use swap::{EvictOutcome, LlmLoader, SwapManager, SwapSettings};

/// Serializes the residency-slot tests: they share the process-global GPU
/// slot, and cargo runs test fns in parallel threads. Async mutex so the
/// guard may be held across `.await`.
static SLOT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Construction/drop counters shared by the pipelines a test builds:
/// `loads` counts constructions, `live` the pipelines not yet dropped
/// (cache entries + in-flight holders).
#[derive(Debug, Default)]
struct Probe {
    loads: AtomicU64,
    live: AtomicU64,
}

/// Test double: canned completions; `Drop` decrements `live` — that drop is
/// the observable "model memory freed".
#[derive(Debug)]
struct CountingLlm {
    probe: Arc<Probe>,
}

impl Drop for CountingLlm {
    fn drop(&mut self) {
        self.probe.live.fetch_sub(1, Ordering::Relaxed);
    }
}

impl LlmPipeline for CountingLlm {
    fn complete(&self, _req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            backend_id: "counting-llm".to_owned(),
            text: "hello residency".to_owned(),
            tokens: 2,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn complete_stream(
        &self,
        _req: &WorkerChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        sink("hello ");
        sink("residency");
        Ok(ChatResponse {
            backend_id: "counting-llm".to_owned(),
            text: "hello residency".to_owned(),
            tokens: 2,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn backend_id(&self) -> &'static str {
        "counting-llm"
    }
}

/// Constructor seam: counts constructions.
fn counting_loader(probe: &Arc<Probe>) -> LlmLoader {
    let probe = Arc::clone(probe);
    Arc::new(move |_slug: &str| {
        probe.loads.fetch_add(1, Ordering::Relaxed);
        probe.live.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(CountingLlm {
            probe: Arc::clone(&probe),
        }) as Arc<dyn LlmPipeline>)
    })
}

/// Swap settings from the pieces the tests vary.
fn settings(ttl: Duration, max_loaded: usize, pinned: &[&str]) -> SwapSettings {
    SwapSettings {
        llm_ttl: ttl,
        llm_max_loaded: max_loaded,
        max_ttl: Duration::from_hours(24),
        pinned: pinned.iter().map(|slug| (*slug).to_owned()).collect(),
    }
}

/// Catalog entries with distinct served `id` (alias) and registry `slug`.
fn entry(id: &str, slug: &str, kind: ModelKind) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: slug.to_owned(),
        kind,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities: honkbox_api_types::ModelCapabilities::for_kind(kind),
        category: "root".to_owned(),
        path: "/dev/null-fake-weights".to_owned(),
        bytes: 1,
        metadata: None,
    }
}

fn catalog_with(entries: &[ModelCatalogEntry]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries.to_vec(),
    }
}

fn residency_app(catalog: ModelCatalog, swap_settings: SwapSettings) -> Router {
    let probe = Arc::new(Probe::default());
    residency_app_with(
        catalog,
        SwapManager::new(swap_settings, counting_loader(&probe)),
    )
}

fn residency_app_with(catalog: ModelCatalog, manager: SwapManager) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition), auth: AuthConfig::default(),
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
    sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
    sd_model: None,
    swap: Some(Arc::new(manager)),
    embed: Arc::new(swap::stub_embed_cache()),
    swap_max_ttl: Duration::from_hours(24), data_dir: None, artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(None.clone(), crate::config::DEFAULT_ARTIFACT_MAX_BYTES)), vram_budget: None,
    model_entries: Vec::new(),
    downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")), })
}

/// `POST` a control route with no body; returns (status, body).
async fn post_control(app: Router, uri: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// `GET` a readout route; returns (status, parsed JSON body).
async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let parsed = serde_json::from_str(&body).unwrap_or_else(|err| panic!("{uri}: {err}: {body}"));
    (status, parsed)
}

/// `POST /v1/chat` to warm a resident entry; asserts the 200.
async fn warm(app: &Router, model: &str) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "warm {model}");
}

/// Explicit unload of a resident model evicts it, returns
/// `{unloaded:true, reason:"explicit"}`, and the pipeline's memory frees
/// (the cache `Arc` drop) once the request is done.
#[tokio::test]
async fn unload_resident_evicts_and_frees() {
    let _slot = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let app = residency_app_with(
        catalog_with(&[entry("llm:fake.gguf", "llm:fake.gguf", ModelKind::Llm)]),
        SwapManager::new(
            settings(Duration::from_secs(3600), 2, &[]),
            counting_loader(&probe),
        ),
    );
    warm(&app, "llm:fake.gguf").await;
    assert_eq!(probe.live.load(Ordering::Relaxed), 1, "resident after warm");

    let (status, body) = post_control(app.clone(), "/v1/models/llm:fake.gguf/unload").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["unloaded"], true);
    assert_eq!(parsed["slug"], "llm:fake.gguf");
    assert_eq!(parsed["reason"], "explicit");
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        0,
        "cache drop frees the pipeline"
    );

    // The warm set reads empty afterwards.
    let (status, running) = get_json(app, "/running").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(running["running"].as_array().map(Vec::len), Some(0));
}

/// A non-resident (or unknown) id is an idempotent 200
/// `{unloaded:false, reason:"not_loaded"}` — never a 404.
#[tokio::test]
async fn unload_not_resident_is_idempotent_200() {
    let _slot = SLOT_LOCK.lock().await;
    let app = residency_app(
        catalog_with(&[entry("llm:fake.gguf", "llm:fake.gguf", ModelKind::Llm)]),
        settings(Duration::from_secs(3600), 2, &[]),
    );
    for uri in [
        "/v1/models/llm:fake.gguf/unload",
        "/v1/models/llm:never-loaded.gguf/unload",
        "/api/models/unload/llm:fake.gguf",
    ] {
        let (status, body) = post_control(app.clone(), uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {body}");
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["unloaded"], false, "{uri}: {body}");
        assert_eq!(parsed["reason"], "not_loaded", "{uri}: {body}");
    }
}

/// llama-swap compat: an unknown model id is a 404 whose body contains the
/// literal `model not found` (loop-lore's classifier substring-matches it).
#[tokio::test]
async fn compat_unload_unknown_is_404_model_not_found() {
    let _slot = SLOT_LOCK.lock().await;
    let app = residency_app(
        catalog_with(&[entry("llm:fake.gguf", "llm:fake.gguf", ModelKind::Llm)]),
        settings(Duration::from_secs(3600), 2, &[]),
    );
    let (status, body) = post_control(app, "/api/models/unload/llm:nope.gguf").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        body.to_lowercase().contains("model not found"),
        "404 body must contain the classifier phrase: {body}"
    );
}

/// A pinned resident refuses explicit unload with 409 and stays resident;
/// the pinned-but-absent case is the idempotent no-op instead.
#[tokio::test]
async fn unload_pinned_is_409_and_stays_resident() {
    let _slot = SLOT_LOCK.lock().await;
    let app = residency_app(
        catalog_with(&[
            entry("llm:pinned.gguf", "llm:pinned.gguf", ModelKind::Llm),
            entry("llm:free.gguf", "llm:free.gguf", ModelKind::Llm),
        ]),
        settings(Duration::from_secs(3600), 4, &["llm:pinned.gguf"]),
    );
    warm(&app, "llm:pinned.gguf").await;
    let (status, body) = post_control(app.clone(), "/v1/models/llm:pinned.gguf/unload").await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");

    let (_, running) = get_json(app.clone(), "/running").await;
    let models: Vec<&str> = running["running"]
        .as_array()
        .expect("running array")
        .iter()
        .map(|r| r["model"].as_str().expect("model key"))
        .collect();
    assert_eq!(models, vec!["llm:pinned.gguf"], "pinned stays resident");

    // Pinned but never loaded: idempotent no-op, not a 409.
    let (status, body) = post_control(app, "/v1/models/llm:free.gguf/unload").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["unloaded"], false);
}

/// Unloading a model whose pipeline is held by an in-flight request does
/// not abort it: the holder's `Arc` clone keeps the pipeline alive, and it
/// still completes after eviction.
#[tokio::test]
async fn in_flight_pipeline_survives_eviction() {
    let _slot = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let manager = Arc::new(SwapManager::new(
        settings(Duration::from_secs(3600), 2, &[]),
        counting_loader(&probe),
    ));
    let pipe = manager.get("llm:a.gguf", None).await.unwrap();
    assert_eq!(
        manager.evict_by_slug("llm:a.gguf", metrics::UnloadReason::Explicit),
        EvictOutcome::Evicted
    );
    // The in-flight holder still completes.
    let req = WorkerChatRequest {
        model: "llm:a.gguf".to_owned(),
        messages: Vec::new(),
        max_tokens: None,
        temperature: None,
        seed: None,
        grammar: None,
        json_schema: None,
        json_object: false,
        tools_json: None,
        tool_choice_none: false,
        sampler: SamplerParams::default(),
    };
    let reply = pipe.complete(&req).expect("in-flight completion");
    assert_eq!(reply.backend_id, "counting-llm");
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        1,
        "the holder keeps the pipeline alive"
    );
    drop(pipe);
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        0,
        "last holder drop frees the model"
    );
    // And the manager can load it again on demand.
    let _reloaded = manager.get("llm:a.gguf", None).await.unwrap();
    assert_eq!(probe.loads.load(Ordering::Relaxed), 2);
}

/// `/running` carries exactly the load-bearing envelope: key `running`,
/// entry key `model`, `state: "running"`.
#[tokio::test]
async fn running_envelope_shape() {
    let _slot = SLOT_LOCK.lock().await;
    let app = residency_app(
        catalog_with(&[
            entry("llm:a.gguf", "llm:a.gguf", ModelKind::Llm),
            entry("llm:b.gguf", "llm:b.gguf", ModelKind::Llm),
        ]),
        settings(Duration::from_secs(3600), 4, &[]),
    );
    warm(&app, "llm:a.gguf").await;
    warm(&app, "llm:b.gguf").await;

    let (status, running) = get_json(app, "/running").await;
    assert_eq!(status, StatusCode::OK);
    let list = running["running"].as_array().expect("envelope key running");
    assert_eq!(list.len(), 2);
    let mut models: Vec<&str> = list
        .iter()
        .map(|r| {
            assert_eq!(r["state"], "running");
            r["model"].as_str().expect("entry key model")
        })
        .collect();
    models.sort_unstable();
    assert_eq!(models, vec!["llm:a.gguf", "llm:b.gguf"]);
}

/// `/v1/memory` lists the resident entries with `loaded_at`, `expires_at`,
/// `pinned`, `in_use`, plus capacity and GPU slot state; pinned entries
/// carry no `expires_at`.
#[tokio::test]
async fn memory_readout_fields() {
    let _slot = SLOT_LOCK.lock().await;
    let app = residency_app(
        catalog_with(&[
            entry("qwen3", "llm:qwen.gguf", ModelKind::Llm),
            entry("llm:pinned.gguf", "llm:pinned.gguf", ModelKind::Llm),
        ]),
        settings(Duration::from_secs(600), 4, &["llm:pinned.gguf"]),
    );
    warm(&app, "qwen3").await; // by alias
    warm(&app, "llm:pinned.gguf").await;

    let (status, memory) = get_json(app, "/v1/memory").await;
    assert_eq!(status, StatusCode::OK);
    let resident = memory["resident"].as_array().expect("resident array");
    assert_eq!(resident.len(), 2);

    let unpinned = resident
        .iter()
        .find(|r| r["slug"] == "llm:qwen.gguf")
        .expect("qwen resident");
    assert_eq!(unpinned["alias"], "qwen3", "served alias reported");
    assert_eq!(unpinned["kind"], "llm");
    assert!(unpinned["loaded_at"].as_u64().is_some_and(|t| t > 0));
    assert!(
        unpinned["expires_at"]
            .as_u64()
            .is_some_and(|e| e > unpinned["loaded_at"].as_u64().unwrap()),
        "expires_at inside the TTL window: {unpinned}"
    );
    assert_eq!(unpinned["pinned"], false);
    assert_eq!(unpinned["in_use"], false, "no request in flight");
    // REQ-12: estimated bytes from the registry scan (an estimate, not a
    // measurement) and the top-level sum.
    assert_eq!(unpinned["bytes"], 1, "registry bytes estimate reported");
    assert_eq!(
        memory["bytes_reserved"], 2,
        "sum of the resident entries' estimates"
    );

    let pinned = resident
        .iter()
        .find(|r| r["slug"] == "llm:pinned.gguf")
        .expect("pinned resident");
    assert_eq!(pinned["pinned"], true);
    assert!(
        pinned.get("expires_at").is_none(),
        "pinned has no TTL: {pinned}"
    );

    let capacity = &memory["capacity"];
    assert_eq!(capacity["llm_max_loaded"], 4);
    assert_eq!(capacity["llm_ttl_secs"], 600);
    assert_eq!(memory["gpu"]["slot_busy"], false);
}

/// Bulk compat unload evicts every resident non-pinned entry and skips
/// pinned ones; one model per call compat unload accepts aliases.
#[tokio::test]
async fn bulk_unload_skips_pinned() {
    let _slot = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let app = residency_app_with(
        catalog_with(&[
            entry("qwen3", "llm:qwen.gguf", ModelKind::Llm),
            entry("llm:free.gguf", "llm:free.gguf", ModelKind::Llm),
            entry("llm:pinned.gguf", "llm:pinned.gguf", ModelKind::Llm),
        ]),
        SwapManager::new(
            settings(Duration::from_secs(3600), 4, &["llm:pinned.gguf"]),
            counting_loader(&probe),
        ),
    );
    warm(&app, "qwen3").await;
    warm(&app, "llm:free.gguf").await;
    warm(&app, "llm:pinned.gguf").await;

    let (status, body) = post_control(app.clone(), "/api/models/unload").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["unloaded"], 2, "non-pinned only: {body}");
    assert_eq!(parsed["reason"], "explicit");

    let (_, running) = get_json(app.clone(), "/running").await;
    let models: Vec<&str> = running["running"]
        .as_array()
        .expect("running array")
        .iter()
        .map(|r| r["model"].as_str().expect("model key"))
        .collect();
    assert_eq!(models, vec!["llm:pinned.gguf"], "pinned survivor");

    // Single compat unload through the alias.
    let (status, body) = post_control(app, "/api/models/unload/llm:free.gguf").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["unloaded"], false, "already bulk-evicted: {body}");
}

/// REQ-6 (Option B): with the GPU slot held, `/v1/chat/stream` is a real
/// HTTP 429 with `Retry-After` — never a 200 with an error frame.
#[tokio::test]
async fn native_stream_busy_is_429() {
    let _slot = SLOT_LOCK.lock().await;
    let _held = honkbox_residency::global()
        .try_reserve()
        .expect("hold the slot");
    let app = residency_app(
        catalog_with(&[entry("llm:fake.gguf", "llm:fake.gguf", ModelKind::Llm)]),
        settings(Duration::from_secs(3600), 2, &[]),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "llm:fake.gguf",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("1")
    );
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.contains("event:"), "plain 429 body, no SSE: {body}");
}

/// REQ-6 (Option B): with the GPU slot held, `stream:true` on
/// `/v1/chat/completions` is a real HTTP 429 with the `OpenAI` envelope +
/// `Retry-After` — never a 200 whose only content is `event: error` +
/// `data: [DONE]` (loop-lore's `data:`-only parser would read that as a
/// clean empty completion).
#[tokio::test]
async fn completions_stream_busy_is_429_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let _held = honkbox_residency::global()
        .try_reserve()
        .expect("hold the slot");
    let app = residency_app(
        catalog_with(&[entry("llm:fake.gguf", "llm:fake.gguf", ModelKind::Llm)]),
        settings(Duration::from_secs(3600), 2, &[]),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "llm:fake.gguf",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("1")
    );
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.contains("event:"), "no SSE frames on a 429: {body}");
    let parsed: Value = serde_json::from_str(&body).expect("OpenAI envelope");
    assert_eq!(parsed["error"]["type"], "rate_limit_error");
    assert_eq!(parsed["error"]["code"], "gpu_slot_busy");
}

/// REQ-6: the busy-rejection counter renders the bounded `reason` grid; an
/// immediate rejection counts under `no_wait` (the only reachable reason
/// until the FIFO queue lands).
#[tokio::test]
async fn busy_rejection_reason_labels_render() {
    let _slot = SLOT_LOCK.lock().await;
    let _held = honkbox_residency::global()
        .try_reserve()
        .expect("hold the slot");
    let app = residency_app(
        catalog_with(&[entry("llm:fake.gguf", "llm:fake.gguf", ModelKind::Llm)]),
        settings(Duration::from_secs(3600), 2, &[]),
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "llm:fake.gguf",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains("honkbox_gpu_slot_busy_rejections_total{reason=\"no_wait\"} "),
        "reason series present: {body}"
    );
    for reason in ["queue_full", "queue_timeout"] {
        assert!(
            body.contains(&format!(
                "honkbox_gpu_slot_busy_rejections_total{{reason=\"{reason}\"}}"
            )),
            "reserved reason series renders: {body}"
        );
    }
}
