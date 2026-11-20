//! `POST /v1/embeddings` (REQ-4) integration tests over the stub backend.
//! Covers the acceptance checklist: deterministic and L2-normalized
//! vectors; batch order with `index` preservation; `usage.prompt_tokens`
//! from input tokens; unknown-model 404 envelope; chat-model refusal (an
//! explicit error, never garbage vectors); the operator
//! `capabilities.embeddings` override; and the reverse gate (an embeddings
//! model refused by `/v1/chat/completions`).
//!
//! Resource contract: the ledger's GPU slot is ONE process-global counter
//! (every kind counts against it, `capacities.gpu_slots = 1`), so the
//! `/v1/embeddings` requests in this binary are serialized on `SLOT_LOCK`
//! (taken by the `embed` helper) — a parallel test must not race the one
//! slot into a spurious 429.
//!
//! The chat-refusal test is the one request that takes NO lock: the route
//! gates it BEFORE it registers a job or calls `admission::acquire`, so it
//! never reaches the ledger. The test pins that invariant (a gated 400
//! carries no `X-Job-Id`), and that pin is what makes the missing lock
//! safe — if the route ever moved the gate after the acquire, the
//! assertion fails instead of the test racing the slot.
//!
//! No data dirs (`data_dir: None` → no bundle writes in this binary) and a
//! per-test catalog (`test_app`), so no model state is shared between tests.

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
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, app};

/// One fake model entry: `kind` decides the default capability set, an
/// operator `capabilities` override wins (REQ-2 semantics).
fn entry(id: &str, kind: ModelKind, embeddings: bool) -> ModelCatalogEntry {
    let mut capabilities = ModelCapabilities::for_kind(kind);
    capabilities.embeddings = embeddings;
    ModelCatalogEntry {
        modality: honkbox_api_types::ModelModalities::empty(),
        id: id.to_owned(),
        slug: id.to_owned(),
        kind,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities,
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

fn test_app(entries: Vec<ModelCatalogEntry>) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: entries,
        }))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
            ocr_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(honkbox_llm_worker::StubLlm::default()),
        stt: Arc::new(honkbox_audio_worker::StubStt),
        tts: Arc::new(honkbox_audio_worker::StubTts),
        sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
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
        auth: config::AuthConfig::default(),
    })
}

/// Serializes the process-global GPU slot across this binary's tests: the
/// stub embed window is tiny, but parallel tests can still race the one
/// slot into a spurious 429 (same pattern as `completions.rs`).
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// `POST /v1/embeddings`; returns (status, body).
async fn embed(app: Router, body: Value) -> (StatusCode, String) {
    let _slot = SLOT_LOCK.lock().await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn embed_body(model: &str, input: &Value) -> Value {
    json!({"model": model, "input": input})
}

/// The catalog the negative tests share: one chat model (embeddings off —
/// the `Llm` default), one embedding GGUF (the `Embedding` default).
fn default_catalog() -> Vec<ModelCatalogEntry> {
    vec![
        entry("llm:chat.gguf", ModelKind::Llm, false),
        entry("llm:embed.gguf", ModelKind::Embedding, true),
    ]
}

/// Acceptance: `data[0].embedding` is a finite vector, L2-normalized
/// (unit length), deterministic across identical requests, and the
/// envelope shape is exact (`object: "list"` / `"embedding"`).
#[tokio::test]
async fn embeddings_are_deterministic_and_normalized() {
    let body = embed_body("llm:embed.gguf", &json!("hello embeddings"));
    let (status, first) = embed(test_app(default_catalog()), body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let parsed: Value = serde_json::from_str(&first).expect("JSON body");
    assert_eq!(parsed["object"], "list");
    assert_eq!(parsed["model"], "llm:embed.gguf");
    let data = parsed["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let vector = data[0]["embedding"].as_array().expect("embedding array");
    assert!(!vector.is_empty(), "non-empty vector");
    let components: Vec<f64> = vector
        .iter()
        .map(|c| c.as_f64().expect("finite f32 component"))
        .collect();
    let norm: f64 = components.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "L2-normalized (norm = {norm}): {components:?}"
    );

    let (status, second) = embed(test_app(default_catalog()), body).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(first, second, "identical input ⇒ identical vector");
}

/// Acceptance: a batched input returns one entry per input, in order,
/// each with the correct `index`; batch entries equal the single-input
/// embeddings of the same strings (no cross-input leakage).
#[tokio::test]
async fn batch_preserves_order_and_index() {
    let app = test_app(default_catalog());
    let inputs = ["alpha query", "beta query", "gamma query"];
    let (status, body) = embed(app.clone(), embed_body("llm:embed.gguf", &json!(inputs))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let data = parsed["data"].as_array().expect("data array");
    assert_eq!(data.len(), inputs.len());
    for (index, item) in data.iter().enumerate() {
        assert_eq!(item["index"], index, "order = index: {item}");
    }
    let batched: Vec<Value> = data.iter().map(|item| item["embedding"].clone()).collect();
    for (index, input) in inputs.iter().enumerate() {
        let (_, single) = embed(app.clone(), embed_body("llm:embed.gguf", &json!(input))).await;
        let parsed: Value = serde_json::from_str(&single).expect("JSON body");
        assert_eq!(
            parsed["data"][0]["embedding"], batched[index],
            "batch[{index}] == single embed of {input:?}"
        );
    }
}

/// Acceptance: `usage.prompt_tokens` reflects the input token count (stub:
/// deterministic word count) and `total_tokens` mirrors it.
#[tokio::test]
async fn usage_counts_input_tokens() {
    let (status, body) = embed(
        test_app(default_catalog()),
        embed_body("llm:embed.gguf", &json!("one two three four five")),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["usage"]["prompt_tokens"], 5);
    assert_eq!(parsed["usage"]["total_tokens"], 5);
}

/// Acceptance: an unknown model is a 404 with an `OpenAI`-shaped body.
#[tokio::test]
async fn unknown_model_is_404_envelope() {
    let (status, body) = embed(
        test_app(default_catalog()),
        embed_body("llm:nope.gguf", &json!("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert_eq!(parsed["error"]["code"], "model_not_found");
}

/// Acceptance: a chat model (no `capabilities.embeddings`) is refused with
/// an explicit named error — never loaded, never garbage vectors.
#[tokio::test]
async fn chat_model_targeted_is_explicit_error() {
    let (status, body) = embed(
        test_app(default_catalog()),
        embed_body("llm:chat.gguf", &json!("hi")),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert_eq!(parsed["error"]["code"], "not_embeddings_capable");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("not embeddings-capable")),
        "{parsed}"
    );
}

/// Acceptance: an operator-declared `capabilities.embeddings = true`
/// override (REQ-2) makes the model loadable through this route even when
/// its on-disk kind is `llm`.
#[tokio::test]
async fn operator_embeddings_override_is_loadable() {
    let catalog = vec![entry("llm:declared-embed.gguf", ModelKind::Llm, true)];
    let (status, body) = embed(
        test_app(catalog),
        embed_body("llm:declared-embed.gguf", &json!("override")),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["data"][0]["index"], 0);
}

/// Acceptance (reverse gate): the embeddings-capable model is refused by
/// `/v1/chat/completions` with an explicit error, never garbage tokens.
#[tokio::test]
async fn embeddings_model_refused_by_chat_completions() {
    let app = test_app(default_catalog());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "llm:embed.gguf",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // The refusal is decided BEFORE the route registers a job or takes the
    // GPU slot: no `X-Job-Id` header means the request never reached the
    // ledger. This is why the request needs no slot lock — a gated 400 must
    // not burn the one shared slot (and must not race a parallel test).
    assert!(
        response.headers().get("x-job-id").is_none(),
        "a gated 400 must not reach the GPU slot"
    );
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert_eq!(parsed["error"]["code"], "embeddings_model_for_chat");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("embeddings model")),
        "{parsed}"
    );
}

/// Negative space: an empty `input` (and empty batch members) is a 400,
/// never an empty `data` array or a zero-dimension vector.
#[tokio::test]
async fn empty_input_is_400() {
    let app = test_app(default_catalog());
    for input in [json!(""), json!([]), json!(["", "ok"])] {
        let (status, body) = embed(app.clone(), embed_body("llm:embed.gguf", &input.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "input {input}: {body}");
        let parsed: Value = serde_json::from_str(&body).expect("envelope");
        assert_eq!(parsed["error"]["code"], "input_required");
    }
}

/// Negative space: a malformed body is a 400 `OpenAI` envelope.
#[tokio::test]
async fn malformed_body_is_400_envelope() {
    let (status, body) = embed(
        test_app(default_catalog()),
        json!({"model": "llm:embed.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
}
