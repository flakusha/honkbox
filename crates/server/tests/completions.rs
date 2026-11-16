//! `/v1/chat/completions` (REQ-1): `OpenAI` JSON + SSE translation over the
//! native chat pipeline. Covers the binding acceptance checklist: JSON
//! shape + usage, SSE frames + `[DONE]`, seed equality stream vs
//! non-stream, omitted-model 400, unknown-model 404 envelope, unknown-param
//! tolerance, `length` finish reason, 429 busy + `Retry-After` (both
//! routes), and the native route staying unchanged.

// No lib target on this binary crate, so include the modules directly.
// This binary exercises only the chat surfaces of `routes.rs`, so every
// included module carries `allow(dead_code)` for the unused rest.
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

use std::sync::{Arc, LazyLock, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's tests (the
/// busy-429 test holds it; the chat tests need it free).
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One fake LLM entry: served id + registry slug (identical here; alias
/// separation is covered by the bindings/REQ-2 tests).
fn entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
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
        swap: None,
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    })
}

/// `POST /v1/chat/completions` with a JSON body; returns (status,
/// content-type, body).
async fn completions(app: Router, body: Value) -> (StatusCode, String, String) {
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
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
    (
        status,
        content_type,
        String::from_utf8(bytes.to_vec()).unwrap(),
    )
}

fn completion_body(model: &str, extra: &Value) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    });
    for (key, value) in extra.as_object().into_iter().flatten() {
        body[key.as_str()] = value.clone();
    }
    body
}

/// Acceptance 133: `JSON.parse(body).choices[0].message.content` and
/// `.usage.total_tokens > 0`; exact `OpenAI` envelope keys; `finish_reason`.
#[tokio::test]
async fn completion_json_shape_and_usage() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, content_type, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["object"], "chat.completion");
    assert!(
        parsed["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("chatcmpl-")),
        "id: {parsed}"
    );
    assert_eq!(parsed["model"], "llm:fake.gguf");
    assert_eq!(parsed["choices"][0]["index"], 0);
    assert_eq!(parsed["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        parsed["choices"][0]["message"]["content"],
        "stub reply to: hi"
    );
    assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    assert_eq!(parsed["choices"].as_array().map(Vec::len), Some(1));
    let usage = &parsed["usage"];
    assert_eq!(usage["prompt_tokens"], 0); // the stub does not tokenize
    assert!(
        usage["completion_tokens"].as_u64().is_some_and(|n| n > 0),
        "usage: {usage}"
    );
    assert!(
        usage["total_tokens"].as_u64().is_some_and(|n| n > 0),
        "usage: {usage}"
    );
}

/// Acceptance 134: `text/event-stream`, every chunk has
/// `choices[0].delta`, the final chunk carries `finish_reason` + `usage`,
/// the body ends with a literal `data: [DONE]`, and no `event:` frame
/// appears on the success path.
#[tokio::test]
async fn completion_stream_frames_and_done() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, content_type, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({"stream": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("text/event-stream"),
        "{content_type}"
    );
    assert!(body.ends_with("data: [DONE]\n\n"), "tail: {body:?}");
    assert!(
        !body.contains("event:"),
        "no event frames on success: {body}"
    );

    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|payload| serde_json::from_str(payload).ok())
        .collect();
    assert!(!chunks.is_empty(), "no JSON chunks in: {body}");
    let last = chunks.last().expect("chunks");
    // Non-final frames: no usage yet, finish_reason still pending.
    for chunk in &chunks[..chunks.len() - 1] {
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert!(chunk["choices"][0]["delta"].is_object(), "chunk: {chunk}");
        assert!(chunk["usage"].is_null(), "usage only on the final frame");
        assert!(
            chunk["choices"][0]["finish_reason"].is_null(),
            "chunk: {chunk}"
        );
    }
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(
        !last["usage"].is_null(),
        "final frame carries usage: {last}"
    );
    // The first frame marks the assistant role.
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
}

/// Acceptance 135 + seed honored end-to-end: with the same seed, the
/// concatenated `delta.content` over the stream equals the non-streaming
/// content (stub worker: same deterministic text on both paths).
#[tokio::test]
async fn seed_stream_nonstream_equality() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    let (_, _, non_stream) = completions(
        app.clone(),
        completion_body("llm:fake.gguf", &json!({"seed": 42})),
    )
    .await;
    let (_, _, stream) = completions(
        app,
        completion_body("llm:fake.gguf", &json!({"seed": 42, "stream": true})),
    )
    .await;
    let content: Value = serde_json::from_str(&non_stream).expect("JSON body");
    let expected = content["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");

    let streamed: String = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(streamed, expected, "seed must make both paths agree");
}

/// Acceptance 137: an omitted model does not fail deserialization — it is
/// a 400 whose `OpenAI` envelope names the requirement.
#[tokio::test]
async fn omitted_model_is_400_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        json!({"messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert_eq!(parsed["error"]["code"], "model_required");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("model is required")),
        "{parsed}"
    );
}

/// Acceptance 138: an unknown model is 404 with an OpenAI-shaped body; a
/// known model is 200.
#[tokio::test]
async fn unknown_model_is_404_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:nope.gguf", &json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert_eq!(parsed["error"]["code"], "model_not_found");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("llm:nope.gguf")),
        "{parsed}"
    );

    let (status, _, _) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Undeclared `OpenAI` keys (`top_p`, `stop`) and arbitrary extras are
/// accepted silently.
#[tokio::test]
async fn unknown_params_are_tolerated() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"top_p": 0.9, "stop": ["\n"], "frequency_penalty": 0.1, "mystery": {"a": 1}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// `finish_reason` heuristic: generated tokens reaching the requested
/// `max_tokens` cap report `length`.
#[tokio::test]
async fn max_tokens_reports_length() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({"max_tokens": 1})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["choices"][0]["finish_reason"], "length");
}

/// Acceptance 140: both chat routes share the one GPU slot — with the slot
/// held, each busy response is 429 with a `Retry-After` hint (the
/// completions route's body is the `OpenAI` envelope; the native route keeps
/// its plain-text body).
#[tokio::test]
async fn busy_slot_429_with_retry_after_on_both_routes() {
    let _slot = SLOT_LOCK.lock().await;
    let _held = honkbox_residency::global()
        .try_reserve()
        .expect("hold the slot");
    let app = test_app(catalog_with(&["llm:fake.gguf"]));

    let (status, _, body) =
        completions(app.clone(), completion_body("llm:fake.gguf", &json!({}))).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "rate_limit_error");
    assert_eq!(parsed["error"]["code"], "gpu_slot_busy");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "llm:fake.gguf", "messages": [{"role": "user", "content": "hi"}]})
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
    assert!(!String::from_utf8_lossy(&bytes).contains("choices"));
}

/// The native route is untouched: same response shape, no `OpenAI` keys.
#[tokio::test]
async fn native_route_shape_unchanged() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "llm:fake.gguf", "messages": [{"role": "user", "content": "hi"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).expect("native JSON");
    for key in ["backend_id", "text", "tokens", "millis"] {
        assert!(parsed.get(key).is_some(), "native key {key}: {parsed}");
    }
    assert!(parsed.get("choices").is_none(), "no OpenAI shape: {parsed}");
    assert!(parsed.get("usage").is_none(), "no OpenAI shape: {parsed}");
}
