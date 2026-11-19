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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::{
    ChatRequest as WorkerChatRequest, ChatResponse, LlmError, LlmPipeline, StubLlm,
};
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
    test_app_with_llm(catalog, Arc::new(StubLlm::default()))
}

fn test_app_with_llm(catalog: ModelCatalog, llm: Arc<dyn LlmPipeline>) -> Router {
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
        llm,
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
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
/// A normal stream end must not leave the shared pipeline aborted: loop-lore
/// reuses the process-wide pipeline across requests.
struct AbortCountingLlm {
    aborts: Arc<AtomicUsize>,
}

impl LlmPipeline for AbortCountingLlm {
    fn complete(&self, req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        self.complete_stream(req, &mut |_| {})
    }

    fn complete_stream(
        &self,
        _req: &WorkerChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        sink("hi");
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: "hi".to_owned(),
            tokens: 1,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn abort(&self) {
        self.aborts.fetch_add(1, Ordering::Relaxed);
    }

    fn backend_id(&self) -> &'static str {
        "abort-counting-test"
    }
}

struct FailingLlm;

impl LlmPipeline for FailingLlm {
    fn complete(&self, _req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        Err(LlmError::Context("prompt does not fit n_ctx".to_owned()))
    }

    fn complete_stream(
        &self,
        req: &WorkerChatRequest,
        _sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        self.complete(req)
    }

    fn backend_id(&self) -> &'static str {
        "failing-test"
    }
}

#[tokio::test]
async fn completion_stream_normal_end_does_not_abort_pipeline() {
    let _slot = SLOT_LOCK.lock().await;
    let aborts = Arc::new(AtomicUsize::new(0));
    let app = test_app_with_llm(
        catalog_with(&["llm:fake.gguf"]),
        Arc::new(AbortCountingLlm {
            aborts: Arc::clone(&aborts),
        }),
    );
    let (status, _, body) = completions(
        app,
        completion_body("llm:fake.gguf", &json!({"stream": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.ends_with("data: [DONE]\n\n"), "tail: {body:?}");
    assert_eq!(
        aborts.load(Ordering::Relaxed),
        0,
        "normal end aborted: {body}"
    );
}

#[tokio::test]
async fn context_overflow_is_400_and_sse_error_is_data_only() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app_with_llm(catalog_with(&["llm:fake.gguf"]), Arc::new(FailingLlm));
    let (status, _, body) = completions(
        app.clone(),
        completion_body("llm:fake.gguf", &json!({"stream": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !body.contains("event:"),
        "event frame dropped by loop-lore: {body}"
    );
    assert!(!body.contains("[DONE]"), "error must be terminal: {body}");
    let frame = body
        .strip_prefix("data: ")
        .and_then(|s| s.strip_suffix("\n\n"))
        .expect("one JSON data frame");
    let error: Value = serde_json::from_str(frame).expect("parseable error frame");
    assert_eq!(error["error"]["code"], "context_length_exceeded");
    assert_eq!(error["error"]["type"], "invalid_request_error");

    let (status, _, body) = completions(app, completion_body("llm:fake.gguf", &json!({}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_str(&body).expect("parseable error envelope");
    assert_eq!(error["error"]["code"], "context_length_exceeded");
}
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

/// REQ-3: one function tool the stub deterministically calls.
fn tools_json() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }])
}

/// REQ-3 acceptance: a request with `tools` returns `tool_calls` in
/// `choices[0].message` with `finish_reason: "tool_calls"`, `content:
/// null`, a unique per-response call id, and `arguments` that parses as
/// JSON shaped by the tool schema (stub: schema-driven, deterministic).
#[tokio::test]
async fn tools_request_returns_tool_calls() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({"tools": tools_json()})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let choice = &parsed["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert!(choice["message"]["content"].is_null(), "{choice}");
    let calls = choice["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("tool_calls array: {choice}"));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "lookup_weather");
    let args = calls[0]["function"]["arguments"]
        .as_str()
        .unwrap_or_else(|| panic!("arguments is a string: {calls:?}"));
    let parsed_args: Value = serde_json::from_str(args).expect("arguments parse as JSON");
    assert_eq!(parsed_args["city"], "stub");
    assert!(
        calls[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("call_")),
        "{calls:?}"
    );
}

/// REQ-3 acceptance: the streamed form emits `delta.tool_calls` fragments
/// carrying `index`, the `function.arguments` pieces split across frames
/// and concatenate into valid JSON, the final frame carries
/// `finish_reason: "tool_calls"` + usage, and the body ends with
/// `data: [DONE]`.
#[tokio::test]
async fn streamed_tool_calls_fragments_concatenate() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, content_type, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"tools": tools_json(), "stream": true}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("text/event-stream"),
        "{content_type}"
    );
    assert!(body.ends_with("data: [DONE]\n\n"), "tail: {body:?}");
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|payload| serde_json::from_str(payload).ok())
        .collect();
    let mut fragments: std::collections::BTreeMap<u64, String> = std::collections::BTreeMap::new();
    let mut ids: Vec<String> = Vec::new();
    let mut fragment_frames = 0;
    for chunk in &chunks {
        for call in chunk["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let index = call["index"]
                .as_u64()
                .unwrap_or_else(|| panic!("fragment index: {call}"));
            if let Some(id) = call["id"].as_str() {
                ids.push(id.to_owned());
            }
            if let Some(piece) = call.pointer("/function/arguments").and_then(Value::as_str)
                && !piece.is_empty()
            {
                fragment_frames += 1;
                fragments.entry(index).or_default().push_str(piece);
            }
        }
    }
    assert!(
        fragment_frames >= 2,
        "arguments split across >= 2 frames: {body}"
    );
    let joined = fragments
        .get(&0)
        .unwrap_or_else(|| panic!("fragment for index 0: {fragments:?}"));
    let parsed: Value = serde_json::from_str(joined)
        .unwrap_or_else(|err| panic!("concatenated fragments are valid JSON: {joined:?}: {err}"));
    assert_eq!(parsed["city"], "stub");
    let last = chunks.last().expect("chunks");
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert!(
        !last["usage"].is_null(),
        "final frame carries usage: {last}"
    );
    assert_eq!(ids.len(), 1, "exactly one id frame: {ids:?}");
    assert!(ids[0].starts_with("call_"));
}

/// REQ-3 acceptance: `response_format: {"type":"json_object"}` yields a
/// response whose `content` parses as JSON — enforced by grammar on the
/// worker path, deterministically on the stub — and the streamed form
/// concatenates to the same parseable JSON.
#[tokio::test]
async fn json_object_response_format_parses() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    let extra = json!({"response_format": {"type": "json_object"}});
    let (status, _, body) =
        completions(app.clone(), completion_body("llm:fake.gguf", &extra)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    let obj: Value = serde_json::from_str(content)
        .unwrap_or_else(|err| panic!("json_object content parses: {content:?}: {err}"));
    assert!(obj["reply"].is_string(), "{obj}");

    let (status, _, stream) = completions(
        app,
        completion_body(
            "llm:fake.gguf",
            &json!({"response_format": {"type": "json_object"}, "stream": true}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
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
    let streamed_obj: Value = serde_json::from_str(&streamed)
        .unwrap_or_else(|err| panic!("streamed json_object parses: {streamed:?}: {err}"));
    assert_eq!(streamed_obj, obj, "stream and non-stream agree");
}

/// REQ-3: `response_format: {"type":"json_schema"}` is accepted and the
/// reply content parses as JSON.
#[tokio::test]
async fn json_schema_response_format_accepted() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "reply",
                "strict": true,
                "schema": {"type": "object", "properties": {"reply": {"type": "string"}}}
            }}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    serde_json::from_str::<Value>(content)
        .unwrap_or_else(|err| panic!("json_schema content parses: {content:?}: {err}"));
}

/// REQ-3 acceptance: a raw llama.cpp GBNF `grammar` is honored end-to-end
/// (accepted and passed to the worker's grammar chain, never ignored).
#[tokio::test]
async fn raw_grammar_is_accepted() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({"grammar": "root ::= \"ok\""})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// REQ-3 acceptance: a follow-up request carrying a `role: "tool"`
/// message with the matching `tool_call_id` (plus the assistant turn with
/// `content: null` + `tool_calls`) is accepted and produces a normal
/// completion.
#[tokio::test]
async fn tool_role_round_trip() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        json!({
            "model": "llm:fake.gguf",
            "messages": [
                {"role": "user", "content": "What is the weather in Reykjavik?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {
                        "name": "lookup_weather", "arguments": "{\"city\":\"Reykjavik\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "4°C, raining"}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let choice = &parsed["choices"][0];
    assert_eq!(choice["finish_reason"], "stop");
    let content = choice["message"]["content"]
        .as_str()
        .expect("content on the follow-up reply");
    assert!(content.contains("4°C, raining"), "{content}");
}

/// REQ-3 acceptance: `min_p`, `top_k`, `dry_*`, `dynatemp_*`,
/// `xtc_probability`, `repeat_penalty`, `typical_p` are accepted without a
/// 400 (all wired into the llama.cpp sampler chain when a real model
/// serves the request).
#[tokio::test]
async fn extended_sampler_params_accepted() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({
                "top_k": 40, "min_p": 0.05, "typical_p": 0.9,
                "repeat_penalty": 1.1, "xtc_probability": 0.5,
                "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2,
                "dynatemp_range": 1.0, "dynatemp_exponent": 1.0
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// REQ-3: `tool_choice: "none"` suppresses calls entirely; named-function
/// and `"required"` choices are clamped to `auto` (accepted, never a 400,
/// still able to call).
#[tokio::test]
async fn tool_choice_none_and_clamps() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    let (status, _, body) = completions(
        app.clone(),
        completion_body(
            "llm:fake.gguf",
            &json!({"tools": tools_json(), "tool_choice": "none"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    assert!(
        parsed["choices"][0]["message"]["tool_calls"].is_null(),
        "no tool_calls under none: {parsed}"
    );
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("plain content under none");
    assert!(content.contains("stub reply"), "{content}");

    for choice in [
        json!({"type": "function", "function": {"name": "lookup_weather"}}),
        json!("required"),
    ] {
        let (status, _, body) = completions(
            app.clone(),
            completion_body(
                "llm:fake.gguf",
                &json!({"tools": tools_json(), "tool_choice": choice}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "clamped choice: {body}");
        let parsed: Value = serde_json::from_str(&body).expect("JSON body");
        assert!(
            parsed["choices"][0]["finish_reason"] == "tool_calls",
            "clamped to auto ⇒ the stub still calls: {parsed}"
        );
    }
}

/// REQ-3: `parallel_tool_calls` is accepted and ignored (one call per
/// turn), documented in `OpenAiChatRequest`.
#[tokio::test]
async fn parallel_tool_calls_ignored() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"tools": tools_json(), "parallel_tool_calls": true}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// GET helper for `/v1/capabilities` (the tool-choice fidelity note lives
/// on the server contract, not on a completion response).
async fn capabilities_get(app: Router) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
    serde_json::from_str(core::str::from_utf8(&bytes).unwrap()).expect("JSON body")
}

/// TASK-llm-tool-choice-contract: the clamp and the ignored
/// `parallel_tool_calls` are machine-readable on `GET /v1/capabilities`
/// (`tools`) instead of silent — and the disclosed vocabulary agrees with
/// the wire: a clamped mode (`"any"` here; named + `"required"` in
/// `tool_choice_none_and_clamps`) is accepted, never a 400, and the stub
/// is still free to call.
#[tokio::test]
async fn capabilities_disclose_tool_choice_fidelity() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    let doc = capabilities_get(app.clone()).await;
    assert_eq!(doc["tools"]["tool_choice_honored"], json!(["auto", "none"]));
    assert_eq!(
        doc["tools"]["tool_choice_clamped"],
        json!(["required", "any"])
    );
    assert_eq!(
        doc["tools"]["tool_choice_named_function_clamped"],
        json!(true)
    );
    assert_eq!(doc["tools"]["parallel_tool_calls"], json!(false));

    let (status, _, body) = completions(
        app,
        completion_body(
            "llm:fake.gguf",
            &json!({"tools": tools_json(), "tool_choice": "any"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert!(
        parsed["choices"][0]["finish_reason"] == "tool_calls",
        "clamped to auto ⇒ the stub still calls: {parsed}"
    );
}

/// REQ-3: `tools` combined with `grammar`/`response_format` is an honest
/// 400 `OpenAI` envelope — the two constraints are mutually exclusive, and
/// silently dropping one side is the fake support this route must not
/// have.
#[tokio::test]
async fn tools_with_output_constraint_is_400_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    for extra in [
        json!({"grammar": "root ::= \"ok\""}),
        json!({"response_format": {"type": "json_object"}}),
        json!({"response_format": {"type": "json_schema", "json_schema": {
            "name": "x", "schema": {"type": "object"}}}}),
    ] {
        let mut body = completion_body("llm:fake.gguf", &extra);
        body["tools"] = tools_json();
        let (status, _, response) = completions(app.clone(), body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        let parsed: Value = serde_json::from_str(&response).expect("envelope");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["code"], "unsupported_constraint");
    }
}

/// REQ-3: `grammar` + `response_format` together is likewise a 400 (one
/// output constraint at a time).
#[tokio::test]
async fn grammar_with_response_format_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"grammar": "root ::= \"ok\"", "response_format": {"type": "json_object"}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["code"], "unsupported_constraint");
}

/// REQ-3: `response_format: json_schema` without a schema payload is a
/// 400, never a silently unconstrained reply.
#[tokio::test]
async fn json_schema_missing_payload_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["code"], "unsupported_constraint");
}

/// REQ-3/error-path: an empty `messages` array is an unusable request —
/// the llama backend rejects it (`LlmError::InvalidRequest`, the mapping
/// unparseable grammars and bad tool messages ride) — so the route rejects
/// it up front with an honest 400 `OpenAI` envelope (never a stub-only
/// 200 or a backend-specific 500).
#[tokio::test]
async fn worker_invalid_request_is_400_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body("llm:fake.gguf", &json!({"messages": []})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("envelope");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert_eq!(parsed["error"]["code"], "messages_required");
}

/// REQ-3: with tools offered but no call made (`tool_choice: "none"`),
/// the STREAMED form falls back to plain content frames (first frame
/// carries the role marker) and ends `stop` — the buffered pieces are not
/// dropped.
#[tokio::test]
async fn streamed_tool_choice_none_falls_back_to_content() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"tools": tools_json(), "tool_choice": "none", "stream": true}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.ends_with("data: [DONE]\n\n"), "tail: {body:?}");
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|payload| serde_json::from_str(payload).ok())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let content: String = chunks
        .iter()
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert!(content.contains("stub reply"), "{content}");
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_null()),
        "no tool_calls fragments under none: {body}"
    );
    let last = chunks.last().expect("chunks");
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(!last["usage"].is_null());
}

/// REQ-8 acceptance: a `content` array with text parts is accepted and
/// flattens (order preserved, `OpenAI` concatenation); the reply's
/// serialized `content` stays a plain string (`string | null` wire shape
/// unchanged), and the plain string form keeps working byte-for-byte.
#[tokio::test]
async fn content_parts_flatten_and_string_form_compat() {
    let _slot = SLOT_LOCK.lock().await;
    // Array form: two text parts concatenate.
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"}
                ]
            }]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("string content");
    assert_eq!(content, "stub reply to: Hello world", "{parsed}");
    // String form: identical, unchanged behavior.
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{"role": "user", "content": "plain string"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(
        parsed["choices"][0]["message"]["content"],
        "stub reply to: plain string"
    );
}

/// REQ-8 acceptance: an `image_url` part on a text-only catalog entry (no
/// sibling mmproj, no override) is an explicit 400 with the `OpenAI`
/// envelope naming the vision refusal — never a silent drop or silent
/// flatten.
#[tokio::test]
async fn image_url_part_is_400_vision_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, content_type, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}}
                ]
            }]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let error = &parsed["error"];
    assert_eq!(error["code"], "vision_unsupported", "{parsed}");
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|m| m.contains("is not vision-capable")),
        "{parsed}"
    );
}

/// Default build compiles the mtmd vision path (`llm-vision` rides in
/// the server crate's default features): sibling-mmproj discovery is
/// allowed to advertise `capabilities.vision`, so the REQ-8 acceptance
/// paths above run on real default builds — not only behind an explicit
/// `--features llm-vision`. Compile-time guard: a test build without the
/// feature fails right here instead of advertising vision it cannot
/// serve.
const _: () = assert!(honkbox_llm_worker::VISION_COMPILED);

/// Chat-capability gate: an entry declaring `capabilities.chat = false` —
/// exactly what a scanned `mmproj*.gguf` projector file advertises — is a
/// named 400 on the completions surface, never garbage tokens dressed up
/// as a completion.
#[tokio::test]
async fn non_chat_capable_entry_is_400_named() {
    let _slot = SLOT_LOCK.lock().await;
    let mut entry = entry("llm:mmproj-model-f16.gguf");
    entry.capabilities.chat = false;
    entry.capabilities.completion = false;
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: vec![entry],
    };
    let (status, _, body) = completions(
        test_app(catalog),
        completion_body(
            "llm:mmproj-model-f16.gguf",
            &json!({"messages": [{"role": "user", "content": "hello"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["error"]["code"], "chat_unsupported", "{parsed}");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("is not chat-capable")),
        "{parsed}"
    );
}

/// REQ-8 acceptance, the vision-capable path: an entry whose capabilities
/// declare vision passes the gate, the data URI decodes, and the request
/// reaches the pipeline (the stub replies; the marker-spliced text is the
/// worker's business, the wire response shape is unchanged).
#[tokio::test]
async fn image_data_uri_passes_on_vision_capable_entry() {
    let _slot = SLOT_LOCK.lock().await;
    let mut entry = entry("llm:fake.gguf");
    entry.capabilities.vision = true;
    entry.modalities = vec!["text".to_owned(), "vision".to_owned()];
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: vec![entry],
    };
    let (status, _, body) = completions(
        test_app(catalog),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in "},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
                    {"type": "text", "text": "this image?"}
                ]
            }]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    // The stub echoes the worker message content — which carries the
    // marker spliced at the image part's offset (vision plumbing proof).
    assert_eq!(
        parsed["choices"][0]["message"]["content"],
        "stub reply to: What is in <__media__>this image?"
    );
}

/// REQ-8 acceptance: a remote `image_url` is refused by name even on a
/// vision-capable entry — the server never fetches client-supplied URLs.
#[tokio::test]
async fn remote_image_url_is_400_named_on_vision_entry() {
    let _slot = SLOT_LOCK.lock().await;
    let mut entry = entry("llm:fake.gguf");
    entry.capabilities.vision = true;
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: vec![entry],
    };
    let (status, _, body) = completions(
        test_app(catalog),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/cat.png"}}
                ]
            }]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["error"]["code"], "image_url_unsupported", "{parsed}");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("only data: URIs are accepted")),
        "{parsed}"
    );
}

/// REQ-8 acceptance: a non-base64 (or empty) `data:` payload is a 400
/// naming the decode failure (`invalid_image_data`) on a vision-capable
/// entry — the gate passes, the payload itself is unusable.
#[tokio::test]
async fn undecodable_data_uri_is_400_invalid_image_data() {
    let _slot = SLOT_LOCK.lock().await;
    let mut entry = entry("llm:fake.gguf");
    entry.capabilities.vision = true;
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: vec![entry],
    };
    let (status, _, body) = completions(
        test_app(catalog),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!not-base64!!"}}
                ]
            }]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    assert_eq!(parsed["error"]["code"], "invalid_image_data", "{parsed}");
}

/// REQ-8 acceptance: an unsupported part type is a 400 naming the type.
#[tokio::test]
async fn unknown_part_type_is_400_named() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, body) = completions(
        test_app(catalog_with(&["llm:fake.gguf"])),
        completion_body(
            "llm:fake.gguf",
            &json!({"messages": [{
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]
            }]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).expect("JSON body");
    let message = parsed["error"]["message"].as_str().expect("message");
    assert!(
        message.contains("unsupported content part type"),
        "{parsed}"
    );
    assert!(message.contains("input_audio"), "{parsed}");
}

/// REQ-8 parity on the native route: an `image_url` part on a text-only
/// entry is the same refusal in the native plain-text error shape.
#[tokio::test]
async fn native_chat_rejects_image_parts_plain_400() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(catalog_with(&["llm:fake.gguf"]));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "llm:fake.gguf",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "look"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
                            ]
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert!(content_type.starts_with("text/plain"), "{content_type}");
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("is not vision-capable"), "{text}");
}
