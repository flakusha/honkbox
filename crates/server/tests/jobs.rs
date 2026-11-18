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
use base64::Engine as _;
use honkbox_api_types::{ModelCatalog, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
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
    test_app_with_entries(sd, Vec::new())
}

fn test_app_with_entries(
    sd: Arc<dyn honkbox_sd_core::SdPipeline>,
    entries: Vec<honkbox_api_types::ModelCatalogEntry>,
) -> Router {
    app(AppConfig {
        auth: AuthConfig::default(),
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

/// `DELETE /v1/jobs/{id}` is the REST-shaped spelling of the cancel: same
/// honest `{cancelled}` contract, and it actually stops an interruptible
/// multi-image generation (between images); a finished id is 404 again.
#[tokio::test]
async fn delete_job_alias_cancels_in_flight() {
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
    let listed = jobs_list(app.clone()).await;
    let id = listed["data"].as_array().unwrap()[0]["id"]
        .as_str()
        .expect("job id")
        .to_owned();

    let (status, _, body) = send(
        app.clone(),
        Request::builder()
            .method("DELETE")
            .uri(format!("/v1/jobs/{id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["cancelled"], true, "{body}");

    gate.open();
    let (status, _, body) = tokio::time::timeout(Duration::from_secs(5), generation)
        .await
        .expect("generation finishes")
        .unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "job_cancelled");
    assert_eq!(sd.calls.load(Ordering::SeqCst), 1);

    // The id is gone once the request ended.
    let (status, _, body) = send(
        app,
        Request::builder()
            .method("DELETE")
            .uri(format!("/v1/jobs/{id}"))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

/// DELETE on an unknown id is the same plain 404 as the cancel endpoint.
#[tokio::test]
async fn delete_unknown_job_is_404() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app(Arc::new(honkbox_sd_core::StubPipeline));
    let (status, _, body) = send(
        app,
        Request::builder()
            .method("DELETE")
            .uri("/v1/jobs/job_00000000000000000000000000")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(body.contains("unknown job id"), "{body}");
}

/// Parse one complete SSE frame text into (event name, parsed JSON data);
/// `None` for comment/keep-alive frames without data.
fn parse_frame(frame: &str) -> Option<(String, Value)> {
    // Last occurrence wins, matching the SSE spec's field semantics (a
    // frame carries one `event:` and one `data:` in practice).
    let event = frame
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .next_back()
        .map(str::to_owned);
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .next_back()?;
    let parsed: Value = serde_json::from_str(data)
        .unwrap_or_else(|err| panic!("frame data not JSON: {err}: {data}"));
    Some((event.unwrap_or_default(), parsed))
}

/// The next complete SSE frame (through its blank line). Chunks may split
/// frames, so bytes accumulate until a blank line shows up.
async fn next_frame(stream: &mut axum::body::BodyDataStream) -> (String, Value) {
    let mut buffer = String::new();
    loop {
        let chunk = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("sse frame within 5s")
            .expect("stream alive")
            .expect("frame bytes");
        buffer.push_str(std::str::from_utf8(&chunk).expect("utf8"));
        while let Some(pos) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..pos + 2).collect();
            if let Some(parsed) = parse_frame(&frame) {
                return parsed;
            }
        }
    }
}

/// Read frames until one matches (event name + job id). Other tests in
/// this binary run concurrently against the SAME global registry, so the
/// stream legitimately carries foreign jobs' events — filtering by id is
/// what keeps the assertion meaningful.
async fn frame_for(stream: &mut axum::body::BodyDataStream, event: &str, id: &str) -> Value {
    loop {
        let (name, data) = next_frame(stream).await;
        if name == event && data["id"].as_str() == Some(id) {
            return data;
        }
    }
}

/// `GET /v1/events` streams the lifecycle of a real in-flight generation:
/// registered (kind/model/interruptible mirror `GET /v1/jobs`) → cancelled
/// (a cancel was REQUESTED) → finished (`cancelled: true`, the token was
/// marked).
#[tokio::test]
async fn events_stream_delivers_the_lifecycle() {
    let _slot = SLOT_LOCK.lock().await;
    let gate = Arc::new(Gate::new());
    let sd = Arc::new(GatedSd {
        gate: Arc::clone(&gate),
        calls: std::sync::atomic::AtomicUsize::new(0),
        entered: AtomicBool::new(false),
    });
    let app = test_app(sd.clone());

    // Subscribe FIRST so the registered event cannot race the stream.
    let sse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sse.status(), StatusCode::OK);
    assert_eq!(
        sse.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream"),
    );
    let mut stream = sse.into_body().into_data_stream();

    let generation = tokio::spawn(images(
        app.clone(),
        json!({"prompt": "a honk", "n": 3, "size": "64x64"}),
    ));
    wait_entered(&sd.entered).await;
    let listed = jobs_list(app.clone()).await;
    let row = &listed["data"].as_array().unwrap()[0];
    let id = row["id"].as_str().expect("job id").to_owned();

    let registered = frame_for(&mut stream, "job_registered", &id).await;
    assert_eq!(registered["kind"], "images");
    assert_eq!(registered["interruptible"], true);
    assert_eq!(registered["cancelled"], false);
    assert_eq!(registered["event"], "job_registered");
    assert!(registered["started_at_ms"].as_u64().is_some());

    let (status, _, body) = cancel_job(app.clone(), &id).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let cancelled = frame_for(&mut stream, "job_cancelled", &id).await;
    assert_eq!(cancelled["cancelled"], true);

    gate.open();
    let (status, _, body) = tokio::time::timeout(Duration::from_secs(5), generation)
        .await
        .expect("generation finishes")
        .unwrap();
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let finished = frame_for(&mut stream, "job_finished", &id).await;
    assert_eq!(finished["cancelled"], true);
    assert_eq!(finished["started_at_ms"], registered["started_at_ms"]);
}

/// A job that ends without a cancel finishes with `cancelled: false` —
/// the flag is "a cancel had been requested", never an outcome claim.
#[tokio::test]
async fn events_stream_reports_plain_finish() {
    let _slot = SLOT_LOCK.lock().await;
    let gate = Arc::new(Gate::new());
    let sd = Arc::new(GatedSd {
        gate: Arc::clone(&gate),
        calls: std::sync::atomic::AtomicUsize::new(0),
        entered: AtomicBool::new(false),
    });
    let app = test_app(sd.clone());

    let sse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut stream = sse.into_body().into_data_stream();

    let generation = tokio::spawn(images(
        app.clone(),
        json!({"prompt": "a honk", "n": 1, "size": "64x64"}),
    ));
    wait_entered(&sd.entered).await;
    let listed = jobs_list(app.clone()).await;
    let id = listed["data"].as_array().unwrap()[0]["id"]
        .as_str()
        .expect("job id")
        .to_owned();

    let registered = frame_for(&mut stream, "job_registered", &id).await;
    assert_eq!(registered["interruptible"], false);

    gate.open();
    let (status, _, body) = tokio::time::timeout(Duration::from_secs(5), generation)
        .await
        .expect("generation finishes")
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    let finished = frame_for(&mut stream, "job_finished", &id).await;
    assert_eq!(finished["cancelled"], false);
}

/// `iVBORw0KGgo=` is base64 of the 8-byte PNG magic — enough for the
/// magic-sniffing `control_image` check (full decoding is not the route's
/// job at validation time).
const CONTROL_PNG_B64: &str = "iVBORw0KGgo=";

/// One `controlnet`-kind catalog entry for the app catalog.
fn controlnet_entry() -> honkbox_api_types::ModelCatalogEntry {
    honkbox_api_types::ModelCatalogEntry {
        id: "comfy:controlnet/c.safetensors".to_owned(),
        slug: "comfy:controlnet/c.safetensors".to_owned(),
        kind: ModelKind::ControlNet,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities: honkbox_api_types::ModelCapabilities::for_kind(ModelKind::ControlNet),
        category: "controlnet".to_owned(),
        path: "/models/controlnet/c.safetensors".to_owned(),
        bytes: 100,
        metadata: None,
    }
}

/// Every rejection path of the controlnet pair is a precise 400, checked
/// in order: missing half of the pair, undecodable base64, non-PNG bytes,
/// unknown slug. None of them may reach the GPU pipeline.
#[tokio::test]
async fn controlnet_rejections_are_precise_400s() {
    let _slot = SLOT_LOCK.lock().await;
    let gate = Arc::new(Gate::new());
    let sd = Arc::new(GatedSd {
        gate: Arc::clone(&gate),
        calls: std::sync::atomic::AtomicUsize::new(0),
        entered: AtomicBool::new(false),
    });
    let app = test_app_with_entries(sd.clone(), vec![controlnet_entry()]);

    // Slug without the conditioning image.
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "controlnet": "comfy:controlnet/c.safetensors"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "controlnet_pair");

    // Conditioning image without the slug.
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "control_image": CONTROL_PNG_B64}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "controlnet_pair");

    // Undecodable base64 (both halves present).
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": "not base64!!"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_control_image");

    // Decodable bytes that are not a PNG.
    let not_png = base64::engine::general_purpose::STANDARD.encode(b"hello");
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": not_png}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_control_image");

    // Known-good image, but the slug is not a controlnet catalog entry.
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "controlnet": "comfy:controlnet/missing.safetensors",
               "control_image": CONTROL_PNG_B64}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "unknown_controlnet");

    // The GPU pipeline was never entered: every rejection is pre-slot.
    assert_eq!(sd.calls.load(Ordering::SeqCst), 0);
}

/// SD double carrying the real backend's id: captures each request for
/// assertions and returns a 1x1 image without touching any weights. This
/// is the honest stand-in for "the configured pipeline can apply control
/// inputs" — the route threads the resolved `ControlNet` pair through the
/// seam and the backend contract does the rest.
struct CapturingSd {
    captured: Mutex<Option<honkbox_sd_core::Txt2ImgRequest>>,
}

impl honkbox_sd_core::SdPipeline for CapturingSd {
    fn generate(
        &self,
        req: &honkbox_sd_core::Txt2ImgRequest,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        *self.captured.lock().expect("capture lock") = Some(req.clone());
        Ok(honkbox_sd_core::GeneratedImage {
            width: 1,
            height: 1,
            rgba: vec![0, 0, 0, 255],
        })
    }

    fn backend_id(&self) -> &'static str {
        honkbox_sd_core::SD_CPP_BACKEND_ID
    }
}

/// Fully valid controlnet inputs are accepted-and-validated, then honestly
/// refused with 501 `controlnet_unsupported` — before any GPU work, and
/// never silently ignored. Whitespace-only fields are NOT a request: the
/// request proceeds as plain generation.
#[tokio::test]
async fn controlnet_valid_inputs_answer_501_and_empties_fall_through() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app_with_entries(
        Arc::new(honkbox_sd_core::StubPipeline),
        vec![controlnet_entry()],
    );

    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "controlnet_unsupported");
    let message = parsed["error"]["message"].as_str().expect("message");
    assert!(message.contains("comfy:controlnet/c.safetensors"), "{body}");

    // Whitespace-only controlnet fields are not a request: plain render.
    let (status, _, body) = images(
        app,
        json!({"prompt": "a honk", "size": "64x64", "controlnet": "   "}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 1);
}

/// A valid pair on a pipeline that can apply control (backend id `sd-cpp-*`)
/// rides the generation: the resolved weights path, the decoded PNG bytes,
/// and the optional strength reach the seam. No model file involved — the
/// double records the request instead of rendering.
#[tokio::test]
async fn controlnet_valid_pair_is_threaded_into_the_sd_cpp_pipeline() {
    let _slot = SLOT_LOCK.lock().await;
    let sd = Arc::new(CapturingSd {
        captured: Mutex::new(None),
    });
    let app = test_app_with_entries(sd.clone(), vec![controlnet_entry()]);

    let (status, _, body) = images(
        app,
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64,
               "controlnet_strength": 0.5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 1);

    let captured = sd
        .captured
        .lock()
        .expect("capture lock")
        .take()
        .expect("request");
    assert_eq!(
        captured.control_net.as_deref(),
        Some(std::path::Path::new("/models/controlnet/c.safetensors")),
    );
    assert_eq!(
        captured.control_image.as_deref(),
        Some(CONTROL_PNG_B64_AS_BYTES)
    );
    assert_eq!(captured.controlnet_strength, Some(0.5));
}

/// Omitted `controlnet_strength` rides as `None` (the sd-core vendor
/// default 0.9 applies there), and the out-of-range override is a named
/// 400 — rejected, not clamped — before any GPU work.
#[tokio::test]
async fn controlnet_strength_default_and_rejection() {
    let _slot = SLOT_LOCK.lock().await;
    let sd = Arc::new(CapturingSd {
        captured: Mutex::new(None),
    });
    let app = test_app_with_entries(sd.clone(), vec![controlnet_entry()]);

    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let captured = sd
        .captured
        .lock()
        .expect("capture lock")
        .take()
        .expect("request");
    assert_eq!(captured.controlnet_strength, None);

    let (status, _, body) = images(
        app,
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64,
               "controlnet_strength": 1.5}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_controlnet_strength");
    assert!(sd.captured.lock().expect("capture lock").is_none());
}

/// A named `control_type` rides the resolved pair into the pipeline: the
/// captured request carries a NON-zero engine type (what the image depicts)
/// next to the weights path — the typed ControlNet execution contract. The
/// engine's alias table accepts aliases (`lineart` → canny) and rejects
/// unknown names with a named 400 before any GPU work; a stray
/// `control_type` without a pair is refused like a lone pair half.
#[tokio::test]
async fn controlnet_type_execution_contract() {
    let _slot = SLOT_LOCK.lock().await;
    let sd = Arc::new(CapturingSd {
        captured: Mutex::new(None),
    });
    let app = test_app_with_entries(sd.clone(), vec![controlnet_entry()]);

    let canny = honkbox_sd_core::control_type_from_name("canny").expect("engine name");
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64,
               "control_type": "canny"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let captured = sd
        .captured
        .lock()
        .expect("capture lock")
        .take()
        .expect("request");
    assert_eq!(captured.control_type, Some(canny));
    assert_ne!(
        captured.control_type,
        Some(honkbox_sd_core::sd_control_type_t::SD_CONTROL_TYPE_NONE)
    );
    assert_eq!(
        captured.control_net.as_deref(),
        Some(std::path::Path::new("/models/controlnet/c.safetensors"))
    );

    // The engine's alias resolves onto the same conditioning type.
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64,
               "control_type": "lineart"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let captured = sd
        .captured
        .lock()
        .expect("capture lock")
        .take()
        .expect("request");
    assert_eq!(captured.control_type, Some(canny));

    // An unknown type name is a named 400, refused pre-slot.
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64,
               "control_type": "bogus"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "unknown_control_type");
    assert!(sd.captured.lock().expect("capture lock").is_none());

    // The name match is the engine's own (case-sensitive).
    let (status, _, body) = images(
        app.clone(),
        json!({"prompt": "a honk", "size": "64x64",
               "controlnet": "comfy:controlnet/c.safetensors",
               "control_image": CONTROL_PNG_B64,
               "control_type": "CANNY"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "unknown_control_type");

    // A stray `control_type` without a pair is refused like a lone half.
    let (status, _, body) = images(
        app,
        json!({"prompt": "a honk", "size": "64x64", "control_type": "canny"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["code"], "controlnet_pair");
    // No request ever reached the pipeline across the refusal branches.
    assert!(sd.captured.lock().expect("capture lock").is_none());
}

/// The exact bytes behind [`CONTROL_PNG_B64`] (the 8-byte PNG magic).
const CONTROL_PNG_B64_AS_BYTES: &[u8] = b"\x89PNG\r\n\x1a\n";
