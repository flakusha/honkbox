//! Bounded admission queue (REQ-6 Option A) over the single GPU slot: a
//! busy slot grants ONE bounded wait (`[server] queue_wait_secs`), a
//! further concurrent request gets the immediate `queue_full` 429, an
//! exceeded wait gets the `queue_timeout` 429 — both with the shared
//! `Retry-After` hint — and an idle slot still takes the fast path with no
//! queue delay. The queue bound is process config installed at serve
//! startup; these tests drive `admission::configure` directly and
//! serialize on the slot lock.

// No lib target on this binary crate, so include the modules directly.
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
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

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

/// Serializes the process-global GPU slot (and the admission global this
/// binary reconfigures) across the tests.
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Park/notify pair the "generation" waits on until the test says go.
struct Gate {
    open: Mutex<bool>,
    signal: Condvar,
}

impl Gate {
    fn new() -> Self {
        Self {
            open: Mutex::new(false),
            signal: Condvar::new(),
        }
    }

    fn open(&self) {
        let mut open = self.open.lock().expect("gate lock");
        *open = true;
        self.signal.notify_all();
    }

    fn wait(&self) {
        let mut open = self.open.lock().expect("gate lock");
        while !*open {
            open = self.signal.wait(open).expect("gate wait");
        }
    }
}

/// Chat double: the FIRST call parks on the gate (a long "generation"
/// holding the slot), every later call returns immediately — so the queued
/// second request completes fast once the first is released. `parked` is
/// set on entry to the gate wait, so it doubles as the test's "slot is
/// busy" signal.
struct ParkOnceLlm {
    gate: Arc<Gate>,
    parked: AtomicBool,
}

impl LlmPipeline for ParkOnceLlm {
    fn complete(&self, _req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        if !self.parked.swap(true, Ordering::SeqCst) {
            self.gate.wait();
        }
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: "ok".to_owned(),
            tokens: 1,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn complete_stream(
        &self,
        _req: &WorkerChatRequest,
        _sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: String::new(),
            tokens: 0,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn backend_id(&self) -> &'static str {
        "park-once-test"
    }
}

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

fn app_with_llm(llm: Arc<dyn LlmPipeline>) -> Router {
    app(AppConfig {
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: vec![entry("llm:park.gguf")],
        }))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: Some("llm:park.gguf".to_owned()),
        stt_model: None,
        tts_model: None,
        sd_model: None,
        llm,
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// `POST /v1/chat`; returns (status, headers, body text).
async fn chat(app: Router, body: Value) -> (StatusCode, axum::http::HeaderMap, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, headers, String::from_utf8(bytes.to_vec()).unwrap())
}

fn chat_body() -> Value {
    json!({
        "model": "llm:park.gguf",
        "messages": [{"role": "user", "content": "hi"}]
    })
}

/// Wait (bounded) until the first chat has actually parked inside the slot.
/// The parked flag is set only on entry to the gate wait, so it doubles as
/// "slot is busy".
async fn wait_parked(parked: &AtomicBool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !parked.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "first request never parked");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// REQ-6 acceptance: a busy slot QUEUES the next request (it completes 200
/// once the slot frees within the window), a third concurrent request is
/// rejected immediately with the queue-full 429 + `Retry-After`.
#[tokio::test]
async fn busy_slot_queues_one_then_rejects_the_rest() {
    let _slot = SLOT_LOCK.lock().await;
    admission::configure(Duration::from_secs(10));

    let gate = Arc::new(Gate::new());
    let llm = Arc::new(ParkOnceLlm {
        gate: Arc::clone(&gate),
        parked: AtomicBool::new(false),
    });
    let app = app_with_llm(llm.clone());

    // A: takes the slot and parks.
    let first = tokio::spawn(chat(app.clone(), chat_body()));
    wait_parked(&llm.parked).await;

    // B: busy → queues (one waiter allowed).
    let second = tokio::spawn(chat(app.clone(), chat_body()));
    tokio::time::sleep(Duration::from_millis(50)).await;

    // C: the queue is full → immediate 429, no waiting.
    let started = Instant::now();
    let (status, headers, body) = chat(app.clone(), chat_body()).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "queue_full must be immediate"
    );
    assert!(body.contains("admission queue is full"), "{body}");
    assert_eq!(
        headers.get("retry-after").map(|v| v.to_str().unwrap()),
        Some("1")
    );

    // Release A: B (the queued waiter) acquires the freed slot → 200.
    gate.open();
    let (status_a, _, _) = tokio::time::timeout(Duration::from_secs(5), first)
        .await
        .expect("first completes")
        .unwrap();
    assert_eq!(status_a, StatusCode::OK);
    let (status_b, _, body_b) = tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .expect("queued request acquires within the window")
        .unwrap();
    assert_eq!(status_b, StatusCode::OK, "{body_b}");

    admission::configure(Duration::ZERO);
}

/// REQ-6 acceptance: the single waiter gives up at its bound → 429 naming
/// the wait, `Retry-After` present; the waiter slot is released, so the
/// NEXT busy request queues again instead of seeing `queue_full`.
#[tokio::test]
async fn queue_wait_times_out_with_retry_after_then_slot_reopens() {
    let _slot = SLOT_LOCK.lock().await;
    admission::configure(Duration::from_millis(150));

    let gate = Arc::new(Gate::new());
    let llm = Arc::new(ParkOnceLlm {
        gate: Arc::clone(&gate),
        parked: AtomicBool::new(false),
    });
    let app = app_with_llm(llm.clone());

    let first = tokio::spawn(chat(app.clone(), chat_body()));
    wait_parked(&llm.parked).await;

    // B: queues, then times out at ~150ms.
    let started = Instant::now();
    let (status, headers, body) = chat(app.clone(), chat_body()).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(140) && waited < Duration::from_secs(2),
        "timeout must honor the bound, took {waited:?}"
    );
    assert!(body.contains("gave up waiting"), "{body}");
    assert_eq!(
        headers.get("retry-after").map(|v| v.to_str().unwrap()),
        Some("1")
    );

    // The waiter slot is free again: the next busy request QUEUES (and
    // times out too — the slot is still held) rather than queue_full.
    let (status_c, _, body_c) = chat(app.clone(), chat_body()).await;
    assert_eq!(status_c, StatusCode::TOO_MANY_REQUESTS);
    assert!(body_c.contains("gave up waiting"), "{body_c}");

    gate.open();
    let (status_a, _, _) = tokio::time::timeout(Duration::from_secs(5), first)
        .await
        .expect("first completes")
        .unwrap();
    assert_eq!(status_a, StatusCode::OK);

    admission::configure(Duration::ZERO);
}

/// The fast path is untouched: an idle slot with a queue configured never
/// waits behind admission machinery.
#[tokio::test]
async fn idle_slot_takes_the_fast_path() {
    let _slot = SLOT_LOCK.lock().await;
    admission::configure(Duration::from_secs(30));

    let app = app_with_llm(Arc::new(StubLlm::default()));
    let started = Instant::now();
    let (status, _, _) = chat(app, chat_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "idle slot must not wait, took {:?}",
        started.elapsed()
    );

    admission::configure(Duration::ZERO);
}
