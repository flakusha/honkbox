//! In-flight job registry + cooperative cancel (NEW-3): GPU routes stamp
//! `X-Job-Id`, `GET /v1/jobs` lists live work (kind/model/interruptible),
//! `POST /v1/jobs/{id}/cancel` marks the token — honest about whether it
//! can stop anything: multi-image generations abort BETWEEN images (409
//! `job_cancelled`), single-image/single-shot jobs run to completion, and
//! unknown ids are a plain 404.

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
use base64::Engine as _;
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's tests.
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

/// SD double: every generate call signals the test, parks on the gate (one
/// "diffusion run" in flight), then returns a 1x1 image.
struct GatedSd {
    gate: Arc<Gate>,
    calls: std::sync::atomic::AtomicUsize,
    entered: AtomicBool,
}

impl honkbox_sd_core::SdPipeline for GatedSd {
    fn generate(
        &self,
        _req: &honkbox_sd_core::Txt2ImgRequest,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.store(true, Ordering::SeqCst);
        self.gate.wait();
        Ok(honkbox_sd_core::GeneratedImage {
            width: 1,
            height: 1,
            rgba: vec![0, 0, 0, 255],
        })
    }

    fn backend_id(&self) -> &'static str {
        "gated-sd-test"
    }
}

fn test_app(sd: Arc<dyn honkbox_sd_core::SdPipeline>) -> Router {
    app(AppConfig {
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
        sd_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

async fn send(app: Router, request: Request<Body>) -> (StatusCode, axum::http::HeaderMap, String) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    (status, headers, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn images(app: Router, body: Value) -> (StatusCode, axum::http::HeaderMap, String) {
    send(
        app,
        Request::builder()
            .method("POST")
            .uri("/v1/images/generations")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

async fn jobs_list(app: Router) -> Value {
    let (_, _, body) = send(
        app,
        Request::builder()
            .uri("/v1/jobs")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    serde_json::from_str(&body).unwrap_or_else(|err| panic!("jobs list not JSON: {err}: {body}"))
}

async fn cancel_job(app: Router, id: &str) -> (StatusCode, axum::http::HeaderMap, String) {
    send(
        app,
        Request::builder()
            .method("POST")
            .uri(format!("/v1/jobs/{id}/cancel"))
            .body(Body::empty())
            .unwrap(),
    )
    .await
}

/// Wait (bounded) until the SD double has entered its first generate call
/// (= the job holds the slot mid-flight).
async fn wait_entered(entered: &AtomicBool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !entered.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "generation never entered");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Registry liveness: while a multi-image generation is parked, the job is
/// listed (kind `images`, interruptible), cancel marks it honestly
/// (`cancelled: true`), the generation aborts BETWEEN images with the 409
/// `job_cancelled` envelope, and the registry entry is gone afterwards.
#[tokio::test]
async fn multi_image_cancel_aborts_between_images() {
    let _slot = SLOT_LOCK.lock().await;
    let gate = Arc::new(Gate::new());
    let sd = Arc::new(GatedSd {
        gate: Arc::clone(&gate),
        calls: std::sync::atomic::AtomicUsize::new(0),
        entered: AtomicBool::new(false),
    });
    let app = test_app(sd.clone());

    let generation = tokio::spawn(images(
        app.clone(),
        json!({"prompt": "a honk", "n": 3, "size": "64x64"}),
    ));
    wait_entered(&sd.entered).await;

    // In-flight: the registry lists exactly the images job, interruptible.
    let listed = jobs_list(app.clone()).await;
    assert_eq!(listed["object"], "list");
    let rows = listed["data"].as_array().expect("data array");
    assert_eq!(rows.len(), 1, "one in-flight job: {listed}");
    assert_eq!(rows[0]["kind"], "images");
    assert_eq!(rows[0]["interruptible"], true);
    let id = rows[0]["id"].as_str().expect("job id").to_owned();
    assert!(id.starts_with("job_"), "{id}");
    assert!(rows[0]["started_at_ms"].as_u64().is_some());

    // Cancel: found + interruptible → honest true.
    let (status, _, body) = cancel_job(app.clone(), &id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["cancelled"], true);

    // Let the in-flight image finish; the loop checks the token before the
    // next one and aborts the remaining two.
    gate.open();
    let (status, _, body) = tokio::time::timeout(Duration::from_secs(5), generation)
        .await
        .expect("generation finishes")
        .unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "job_cancelled");
    assert!(body.contains("of 3 image(s) generated"), "{body}");
    // Exactly ONE image was generated before the abort.
    assert_eq!(sd.calls.load(Ordering::SeqCst), 1);

    // The registry entry is gone once the request ended.
    let listed = jobs_list(app.clone()).await;
    assert_eq!(listed["data"].as_array().unwrap().len(), 0);
    // Cancelling the finished id is now a plain 404.
    let (status, _, body) = cancel_job(app, &id).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

/// Honesty for non-interruptible jobs: a single-image generation is marked
/// by a cancel (`cancelled: false`) but still completes 200 with its image;
/// the response carries the `X-Job-Id` header.
#[tokio::test]
async fn single_image_cancel_is_honest_and_job_completes() {
    let _slot = SLOT_LOCK.lock().await;
    let gate = Arc::new(Gate::new());
    let sd = Arc::new(GatedSd {
        gate: Arc::clone(&gate),
        calls: std::sync::atomic::AtomicUsize::new(0),
        entered: AtomicBool::new(false),
    });
    let app = test_app(sd.clone());

    let generation = tokio::spawn(images(
        app.clone(),
        json!({"prompt": "a honk", "n": 1, "size": "64x64"}),
    ));
    wait_entered(&sd.entered).await;

    let listed = jobs_list(app.clone()).await;
    let row = &listed["data"].as_array().unwrap()[0];
    assert_eq!(row["interruptible"], false);
    let id = row["id"].as_str().expect("job id").to_owned();

    let (status, _, body) = cancel_job(app.clone(), &id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        parsed["cancelled"], false,
        "single FFI call cannot stop: {body}"
    );

    gate.open();
    let (status, headers, body) = tokio::time::timeout(Duration::from_secs(5), generation)
        .await
        .expect("generation finishes")
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    let job_header = headers
        .get("x-job-id")
        .map(|v| v.to_str().unwrap().to_owned())
        .expect("X-Job-Id stamped");
    assert_eq!(job_header, id);
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 1);
    // Decodes to a real PNG (the test double returns a 1x1 RGBA image).
    let b64 = parsed["data"][0]["b64_json"].as_str().unwrap();
    let png = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"), "not a PNG");
}

/// Unknown ids are 404 both before anything runs and for garbage input.
#[tokio::test]
async fn cancel_unknown_job_is_404() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(Arc::new(honkbox_sd_core::StubPipeline));

    let (status, _, body) = cancel_job(app.clone(), "job_00000000000000000000000000").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(body.contains("unknown job id"), "{body}");

    // The empty registry lists nothing.
    let listed = jobs_list(app).await;
    assert_eq!(listed["data"].as_array().unwrap().len(), 0);
}
