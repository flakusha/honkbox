//! SSE disconnect cancellation (NEW-3 groundwork): dropping the stream body
//! marks the request cancelled and the worker frees the GPU slot at its next
//! token boundary, while the synchronous FFI call itself runs to completion
//! (the `LlmPipeline` trait has no abort hook yet — that seam is noted, not
//! built: it would touch `crates/llm-worker`, out of scope here).

// No lib target on this binary crate, so include the modules directly.
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;

use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::{ChatRequest as WorkerChatRequest, ChatResponse, LlmError, LlmPipeline};
use tower::ServiceExt as _;

use routes::{AppConfig, app};

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
            millis: 0,
        })
    }

    fn backend_id(&self) -> &'static str {
        "gated-test"
    }
}

#[tokio::test]
async fn dropped_stream_body_frees_gpu_slot() {
    let (started_tx, started_rx): (_, Receiver<()>) = channel();
    let (resumed_tx, resumed_rx): (_, Receiver<()>) = channel();
    let gate = Arc::new(Gate::new());
    let app = app(AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: Vec::new(),
        }))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
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
