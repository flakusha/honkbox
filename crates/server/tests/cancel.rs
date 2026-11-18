//! SSE disconnect cancellation (NEW-3 groundwork): dropping the stream body
//! marks the request cancelled, frees the GPU slot at the worker's next token
//! boundary, and calls the `LlmPipeline::abort` hook so the synchronous
//! generation stops at its next internal checkpoint instead of running to
//! completion.

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
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;
// Included only so `routes.rs`'s `crate::swap` paths resolve in this test
// binary; the swap module itself is not exercised here.
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// Serializes the residency-slot tests: they share the process-global GPU
/// slot, and cargo runs test fns in parallel threads. Async mutex so the
/// guard may be held across `.await`.
static SLOT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::{ChatRequest as WorkerChatRequest, ChatResponse, LlmError, LlmPipeline};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

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

/// Test double: first token immediately, then parks on the gate (a long
/// "generation"); after it opens, pushes another token — the cancel boundary
/// the server reacts to — and signals the test before finishing.
struct GatedLlm {
    started: std::sync::mpsc::Sender<()>,
    gate: Arc<Gate>,
    resumed: std::sync::mpsc::Sender<()>,
}

impl LlmPipeline for GatedLlm {
    fn complete(&self, _req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: "gated".to_owned(),
            tokens: 1,
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
        sink("one"); // delivered while the client is still there
        let _ = self.started.send(());
        self.gate.wait(); // "generation" in flight until the test says go
        sink("two"); // boundary: server frees the slot here when cancelled
        let _ = self.resumed.send(());
        sink("three");
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: "onetwothree".to_owned(),
            tokens: 3,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn backend_id(&self) -> &'static str {
        "gated-test"
    }
}

/// Abort-aware double: `abort` sets a flag and opens the gate (only exit
/// path from the parked "generation"). After resuming, the pipeline checks
/// the flag: aborted → signals `"aborted"` and returns a partial `Ok`;
/// otherwise it signals `"drained"` (proof the abort actually reached the
/// pipeline rather than the gate timing out).
struct AbortGateLlm {
    started: std::sync::mpsc::Sender<()>,
    gate: Arc<Gate>,
    finished: std::sync::mpsc::Sender<&'static str>,
    aborted: std::sync::atomic::AtomicBool,
}

impl LlmPipeline for AbortGateLlm {
    fn complete(&self, _req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: "gated".to_owned(),
            tokens: 1,
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
        sink("one");
        let _ = self.started.send(());
        self.gate.wait(); // only `abort` opens this
        let aborted = self.aborted.load(std::sync::atomic::Ordering::Relaxed);
        let _ = self
            .finished
            .send(if aborted { "aborted" } else { "drained" });
        if aborted {
            // Partial result as `Ok` — the caller asked for the cancel.
            return Ok(ChatResponse {
                backend_id: self.backend_id().to_owned(),
                text: "one".to_owned(),
                tokens: 1,
                prompt_tokens: 0,
                tool_calls: Vec::new(),
                millis: 0,
            });
        }
        sink("two");
        sink("three");
        Ok(ChatResponse {
            backend_id: self.backend_id().to_owned(),
            text: "onetwothree".to_owned(),
            tokens: 3,
            prompt_tokens: 0,
            tool_calls: Vec::new(),
            millis: 0,
        })
    }

    fn abort(&self) {
        self.aborted
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.gate.open();
    }

    fn backend_id(&self) -> &'static str {
        "abort-gate-test"
    }
}

#[tokio::test]
async fn dropped_stream_body_frees_gpu_slot() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let (started_tx, started_rx): (_, Receiver<()>) = channel();
    let (resumed_tx, resumed_rx): (_, Receiver<()>) = channel();
    let gate = Arc::new(Gate::new());
    let app = app(AppConfig {
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: Vec::new(),
        }))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(GatedLlm {
            started: started_tx,
            gate: Arc::clone(&gate),
            resumed: resumed_tx,
        }),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // The worker reserved the slot before its first token.
    started_rx.recv().expect("pipeline started");
    assert!(
        honkbox_residency::global().try_reserve().is_err(),
        "slot busy while the stream is parked mid-generation"
    );

    // Client disconnect: drop the SSE body, then let the pipeline resume.
    drop(response);
    gate.open();
    resumed_rx
        .recv()
        .expect("pipeline resumed past the cancel boundary");

    // The slot freed at the token boundary even though the pipeline call is
    // still finishing on its blocking thread (the honest stop-at-completion
    // FFI semantics).
    let probe = honkbox_residency::global()
        .try_reserve()
        .expect("slot freed at the cancel boundary");
    drop(probe);
}

/// Cancel during generation reaches the FFI: dropping the SSE body calls
/// `LlmPipeline::abort`, which un-parks the blocking pipeline mid-generation
/// — the worker returns promptly with a partial result instead of draining
/// the remaining tokens, and the GPU slot is freed.
#[tokio::test]
async fn dropped_stream_body_aborts_pipeline_mid_generation() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let (started_tx, started_rx): (_, Receiver<()>) = channel();
    let (finished_tx, finished_rx): (_, Receiver<&'static str>) = channel();
    let gate = Arc::new(Gate::new());
    let llm = Arc::new(AbortGateLlm {
        started: started_tx,
        gate: Arc::clone(&gate),
        finished: finished_tx,
        aborted: std::sync::atomic::AtomicBool::new(false),
    });
    let app = app(AppConfig {
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: Vec::new(),
        }))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::clone(&llm) as Arc<dyn LlmPipeline>,
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Generation is parked mid-flight holding the slot.
    started_rx.recv().expect("pipeline started");
    assert!(
        honkbox_residency::global().try_reserve().is_err(),
        "slot busy while parked mid-generation"
    );

    // Client disconnect: dropping the body calls `pipe.abort()`, which is the
    // only thing that opens the gate — the pipeline exits before "two" and
    // "three" are ever generated.
    drop(response);
    assert_eq!(
        finished_rx.recv().expect("pipeline finished"),
        "aborted",
        "abort hook must stop generation mid-flight"
    );

    // The worker task has fully returned (recv above only completes when
    // `complete_stream` returned), so the slot must be free again. The lease
    // is freed by the server just *after* `complete_stream` returns — a hair
    // after the pipeline signals us — so retry until it lands (bounded spin;
    // in practice the first attempt succeeds).
    let mut probe = None;
    for _ in 0..10_000 {
        if let Ok(lease) = honkbox_residency::global().try_reserve() {
            probe = Some(lease);
            break;
        }
        tokio::task::yield_now().await;
    }
    drop(probe.expect("slot freed after aborted generation returned"));
}
