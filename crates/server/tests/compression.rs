//! Response compression (FEAT-response-compression): gzip/brotli negotiate
//! on ordinary JSON routes with byte-identical decompressed bodies, the
//! 1 KiB size floor leaves small bodies alone, and BOTH SSE routes
//! (`/v1/events`, `/v1/chat/completions` with `stream: true`) are excluded
//! — no `content-encoding`, and frames stream incrementally (the `/v1/events`
//! test holds a gated generation open and reads a lifecycle frame BEFORE
//! the generation finishes, which is the unbuffered-streaming proof; the
//! chat stream shares the exact same content-type exclusion, asserted by
//! headers here).
//!
//! Resource contract: every test builds its own app (port 0, stub
//! backends, `data_dir: None` → bundle writes are skipped, so NO disk
//! writes anywhere in this binary); SSE frames are matched by event name
//! AND job `kind` because the lifecycle registry is process-global and
//! the gated-generation + chat-stream tests run concurrently.
//!
//! `SLOT_LOCK` IS needed: the ledger's GPU slot is ONE process-global
//! counter shared by every kind (`residency::Ledger::committed` is
//! process-wide, `capacities.gpu_slots = 1`), so the parked SD generation
//! and the LLM chat test contend for the same slot — the chat test lost
//! the race and got a 429 `gpu slot busy`. The two job-running tests
//! serialize on the lock; the plain JSON-route tests never take a slot.

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

use std::io::Read as _;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::{Condvar, LazyLock, Mutex, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's job-running
/// tests (the ledger counts every kind against one shared slot).
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Park/notify pair the "generation" waits on until the test says go (the
/// same double as `jobs.rs`).
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

    fn is_open(&self) -> bool {
        *self.open.lock().expect("gate lock")
    }
}

/// SD double: parks on the gate so the test can observe the SSE stream
/// WHILE a generation is genuinely in flight.
struct GatedSd {
    gate: Arc<Gate>,
}

impl honkbox_sd_core::SdPipeline for GatedSd {
    fn generate(
        &self,
        _req: &honkbox_sd_core::Txt2ImgRequest,
        _cancel: &AtomicBool,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
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

fn llm_entry(id: &str) -> ModelCatalogEntry {
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

fn catalog_with(entries: &[ModelCatalogEntry]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries.to_vec(),
    }
}

fn app_with(catalog: ModelCatalog, sd: Arc<dyn honkbox_sd_core::SdPipeline>) -> Router {
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
    sd_model: None,
    llm: Arc::new(StubLlm::default()),
    stt: Arc::new(StubStt),
    tts: Arc::new(StubTts),
    sd,
    swap: None,
    embed: Arc::new(swap::stub_embed_cache()),
    swap_max_ttl: Duration::from_hours(24), data_dir: None, artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(None.clone(), crate::config::DEFAULT_ARTIFACT_MAX_BYTES)), vram_budget: None,
    model_entries: Vec::new(),
    downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")), })
}

fn plain_app() -> Router {
    app_with(
        catalog_with(&[llm_entry("llm:fake.gguf")]),
        Arc::new(honkbox_sd_core::StubPipeline),
    )
}

/// Raw-body GET: (status, headers, body bytes).
async fn get_bytes(
    app: Router,
    uri: &str,
    accept_encoding: Option<&str>,
) -> (StatusCode, axum::http::HeaderMap, Bytes) {
    let mut builder = Request::builder().uri(uri);
    if let Some(encoding) = accept_encoding {
        builder = builder.header("accept-encoding", encoding);
    }
    let response = app
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    (status, headers, bytes)
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = flate2::read::GzDecoder::new(bytes);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).unwrap_or_else(|err| {
        panic!(
            "body decompresses as gzip: {err} (magic: {:?})",
            &bytes[..2]
        )
    });
    out
}

fn header<'a>(headers: &'a axum::http::HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

/// A large JSON route compresses on request and decompresses to the
/// byte-identical body the same route serves uncompressed.
#[tokio::test]
async fn gzip_roundtrips_byte_identical() {
    let app = plain_app();
    let (status, headers, compressed) =
        get_bytes(app.clone(), "/v1/openapi.json", Some("gzip")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        header(&headers, "content-encoding"),
        Some("gzip"),
        "negotiated gzip"
    );
    let (_, _, plain) = get_bytes(app, "/v1/openapi.json", None).await;
    let decompressed = gunzip(&compressed);
    assert_eq!(decompressed, plain.to_vec(), "byte-identical bodies");
    assert!(
        compressed.len() < plain.len(),
        "gzip actually shrank the body"
    );
}

/// Brotli negotiation works too (decompression proof is gzip's job above;
/// this pins the negotiated header).
#[tokio::test]
async fn brotli_negotiates() {
    let (_, headers, body) = get_bytes(plain_app(), "/v1/openapi.json", Some("br")).await;
    assert_eq!(
        header(&headers, "content-encoding"),
        Some("br"),
        "negotiated brotli"
    );
    assert!(!body.is_empty());
}

/// Below the 1 KiB floor: no compression (and no double-encode surprises).
#[tokio::test]
async fn small_body_not_compressed() {
    let (status, headers, _body) = get_bytes(plain_app(), "/health", Some("gzip, br")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        header(&headers, "content-encoding"),
        None,
        "small body passes through"
    );
    assert_eq!(header(&headers, "content-type"), Some("application/json"));
}

/// An encoding this build does not ship is ignored (no header, plain body).
#[tokio::test]
async fn unsupported_encoding_passthrough() {
    let (_, headers, _) = get_bytes(plain_app(), "/v1/openapi.json", Some("zstd")).await;
    assert_eq!(
        header(&headers, "content-encoding"),
        None,
        "zstd not compiled in"
    );
}

/// SSE exclusion + unbuffered streaming: with `Accept-Encoding: gzip` the
/// events stream is NOT compressed, and a lifecycle frame arrives WHILE a
/// generation is parked mid-flight — proof the compression layer does not
/// buffer the stream.
#[tokio::test]
async fn events_stream_excluded_and_unbuffered() {
    let _slot = SLOT_LOCK.lock().await;
    let gate = Arc::new(Gate::new());
    let app = app_with(
        catalog_with(&[]),
        Arc::new(GatedSd {
            gate: Arc::clone(&gate),
        }),
    );
    // Open the SSE stream FIRST (gzip requested).
    let sse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .header("accept-encoding", "gzip")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sse.status(), StatusCode::OK);
    let sse_headers = sse.headers().clone();
    assert_eq!(
        header(&sse_headers, "content-type"),
        Some("text/event-stream"),
        "{sse_headers:?}"
    );
    assert_eq!(
        header(&sse_headers, "content-encoding"),
        None,
        "SSE is excluded from compression"
    );
    let mut stream = sse.into_body().into_data_stream();

    // Fire a generation that parks mid-flight, then read its
    // `job_registered` frame BEFORE opening the gate.
    let generation = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"prompt": "a honk", "size": "64x64"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    let registered = frame_for_id(&mut stream, "job_registered", "images").await;
    assert!(
        !gate.is_open(),
        "frame arrived while generation still parked"
    );
    assert_eq!(registered["kind"], "images", "{registered}");
    gate.open();
    let response = generation.await.expect("generation joins");
    assert_eq!(response.status(), StatusCode::OK);
}

/// Parse one complete SSE frame text into (event name, parsed JSON data).
fn parse_frame(frame: &str) -> Option<(String, Value)> {
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

/// The next SSE frame whose event matches AND whose `kind` names this
/// binary's job family (other tests in this binary run concurrently
/// against the SAME global registry, so name-only matching would race
/// foreign jobs' frames).
async fn frame_for_id(stream: &mut axum::body::BodyDataStream, event: &str, kind: &str) -> Value {
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
            if let Some((name, data)) = parse_frame(&frame)
                && name == event
                && data["kind"].as_str() == Some(kind)
            {
                return data;
            }
        }
    }
}

/// The other SSE surface: `POST /v1/chat/completions` with `stream: true`
/// is also excluded from compression (same `text/event-stream` predicate
/// arm; the unbuffered transport proof is the `/v1/events` test above).
#[tokio::test]
async fn chat_stream_excluded_from_compression() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, headers, body) = raw_post(
        plain_app(),
        "/v1/chat/completions",
        json!({
            "model": "llm:fake.gguf",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }),
        Some("gzip"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(header(&headers, "content-type"), Some("text/event-stream"));
    assert_eq!(
        header(&headers, "content-encoding"),
        None,
        "stream stays uncompressed"
    );
    assert!(body.ends_with(b"data: [DONE]\n\n"), "stream completed");
}

/// Raw-body POST helper.
async fn raw_post(
    app: Router,
    uri: &str,
    body: Value,
    accept_encoding: Option<&str>,
) -> (StatusCode, axum::http::HeaderMap, Bytes) {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(encoding) = accept_encoding {
        builder = builder.header("accept-encoding", encoding);
    }
    let response = app
        .oneshot(builder.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    (status, headers, bytes)
}
