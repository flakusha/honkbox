//! Binary artifact transport (TASK-binary-artifact-api): `POST /v1/artifacts`
//! (single-shot + chunked/resumable multipart) and
//! `GET /v1/artifacts/{id}/bytes` (+`HEAD`, single `Range`, `If-Match` /
//! `If-Range`, named 413/416/412 envelopes) against per-test tmp data dirs.
//!
//! Resource contract: every test builds its own app on its own data dir —
//! no shared registry, no GPU slot (artifacts are pure disk IO) — so tests
//! run in any order and in parallel. The chunk-cap test allocates an 8 MiB
//! body (the real cap; there is no knob to shrink it).

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

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use flate2::read::GzDecoder;
use honkbox_api_types::{ArtifactInfo, ModelCatalog, ModelCatalogEntry};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// One test-scoped data dir (unique per test name + pid; pre-cleaned).
fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-artifacts-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test data dir");
    dir
}

fn catalog_with(entries: &[ModelCatalogEntry]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries.to_vec(),
    }
}

fn app_with(data_dir: Option<PathBuf>, max_artifact_bytes: u64) -> Router {
    app(AppConfig {
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog_with(&[])))),
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
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: data_dir.clone(),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        artifacts: Arc::new(artifacts::Registry::new(data_dir, max_artifact_bytes)),
    })
}

fn default_app(name: &str) -> (Router, PathBuf) {
    let dir = tmp_dir(name);
    let router = app_with(Some(dir.clone()), crate::config::DEFAULT_ARTIFACT_MAX_BYTES);
    (router, dir)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// One multipart form part carrying one chunk (optional explicit index).
struct Chunk {
    index: Option<u32>,
    data: Vec<u8>,
}

fn chunk(index: u32, data: &[u8]) -> Chunk {
    Chunk {
        index: Some(index),
        data: data.to_vec(),
    }
}

/// Encode chunks into a multipart/form-data body; returns the
/// content-type value (boundary included) and the encoded body.
fn multipart_body(parts: &[Chunk]) -> (String, Vec<u8>) {
    let boundary = "honkbox-artifact-test";
    let mut body = Vec::new();
    for one in parts {
        body.extend_from_slice(
            format!("--{boundary}\r\ncontent-disposition: form-data; name=\"chunk\"\r\n")
                .as_bytes(),
        );
        if let Some(index) = one.index {
            body.extend_from_slice(format!("x-chunk-index: {index}\r\n").as_bytes());
        }
        body.extend_from_slice(b"content-type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(&one.data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// `POST /v1/artifacts` (multipart); returns (status, parsed JSON body).
async fn post_chunks(
    app: Router,
    headers: &[(&str, String)],
    parts: &[Chunk],
) -> (StatusCode, Value) {
    let (content_type, body) = multipart_body(parts);
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/artifacts")
        .header("content-type", content_type);
    for (name, value) in headers {
        builder = builder.header(*name, value);
    }
    let response = app
        .oneshot(builder.body(Body::from(body)).expect("request"))
        .await
        .expect("oneshot");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body");
    let parsed =
        serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("response not JSON: {err}"));
    (status, parsed)
}

/// Raw-body request against the artifacts routes; returns everything.
async fn raw_request(
    app: Router,
    method: &str,
    uri: &str,
    headers: &[(&str, String)],
    body: Vec<u8>,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        builder = builder.header(*name, value);
    }
    let response = app
        .oneshot(builder.body(Body::from(body)).expect("request"))
        .await
        .expect("oneshot");
    let status = response.status();
    let response_headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .expect("body")
        .to_vec();
    (status, response_headers, bytes)
}

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn error_code(body: &Value) -> Option<&str> {
    body.get("error")
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = GzDecoder::new(bytes);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .unwrap_or_else(|err| panic!("gzip decode: {err}"));
    out
}

/// Upload one single-shot artifact; returns the descriptor.
async fn upload_single_shot(app: Router, content_type: &str, data: &[u8]) -> ArtifactInfo {
    let (status, _, body) = raw_request(
        app,
        "POST",
        "/v1/artifacts",
        &[
            ("content-type", "application/octet-stream".to_owned()),
            ("x-artifact-content-type", content_type.to_owned()),
        ],
        data.to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "single-shot upload: {body:?}");
    serde_json::from_slice(&body).expect("artifact info json")
}

/// Single-shot round trip: 201 descriptor with the true content digest,
/// then GET/HEAD byte-identical with the house headers.
#[tokio::test]
async fn single_shot_roundtrip() {
    let (app, dir) = default_app("single-shot");
    let data: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
    let info = upload_single_shot(app.clone(), "application/x-honkbox", &data).await;
    assert_eq!(info.sha256, sha256_hex(&data));
    assert_eq!(info.id, info.sha256);
    assert_eq!(info.size_bytes, data.len() as u64);
    assert_eq!(info.content_type, "application/x-honkbox");

    let (status, headers, body) = raw_request(
        app.clone(),
        "GET",
        &format!("/v1/artifacts/{}/bytes", info.id),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, data);
    assert_eq!(
        header(&headers, "etag"),
        Some(format!("\"{}\"", info.sha256).as_str())
    );
    assert_eq!(header(&headers, "accept-ranges"), Some("bytes"));
    assert_eq!(
        header(&headers, "content-type"),
        Some("application/x-honkbox")
    );
    assert_eq!(header(&headers, "content-length"), Some("2048"));

    let (status, head_headers, body) = raw_request(
        app,
        "HEAD",
        &format!("/v1/artifacts/{}/bytes", info.id),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_empty(), "HEAD must not carry a body");
    assert_eq!(header(&head_headers, "content-length"), Some("2048"));
    assert_eq!(
        header(&head_headers, "etag"),
        Some(format!("\"{}\"", info.sha256).as_str())
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Chunked upload interrupted after two chunks, resumed with an idempotent
/// duplicate and the final chunk; the manifest's digest matches an
/// independently computed sha256 of the concatenation.
#[tokio::test]
async fn chunked_upload_interrupts_and_resumes() {
    let (app, dir) = default_app("resume");
    let c0 = b"alpha-alpha-alpha".to_vec();
    let c1 = b"beta-beta-beta-beta".to_vec();
    let c2 = b"gamma-gamma".to_vec();

    // Interrupt: the client stops after two chunks.
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "3".to_owned())],
        &[chunk(0, &c0), chunk(1, &c1)],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "chunks 0+1: {manifest}");
    assert_eq!(manifest["status"], "uploading");
    let upload_id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();
    assert_eq!(manifest["received"], serde_json::json!([0, 1]));
    assert_eq!(manifest["chunk_count"], 3);
    assert!(manifest.get("artifact").is_none());

    // Resume: an idempotent duplicate of chunk 1 (same bytes → 200, no
    // state change), then the missing final chunk.
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-upload-id", upload_id.clone())],
        &[chunk(1, &c1)],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "duplicate chunk: {manifest}");
    assert_eq!(manifest["status"], "uploading");
    assert_eq!(manifest["received"], serde_json::json!([0, 1]));

    let (status, manifest) =
        post_chunks(app.clone(), &[("x-upload-id", upload_id)], &[chunk(2, &c2)]).await;
    assert_eq!(status, StatusCode::OK, "final chunk: {manifest}");
    assert_eq!(manifest["status"], "complete");
    let artifact = manifest["artifact"].as_object().expect("artifact present");
    let whole: Vec<u8> = [c0.as_slice(), c1.as_slice(), c2.as_slice()].concat();
    assert_eq!(artifact["sha256"], sha256_hex(&whole).as_str());
    assert_eq!(artifact["size_bytes"], whole.len() as u64);
    assert_eq!(artifact["chunk_count"], 3);
    let id = artifact["id"].as_str().expect("artifact id").to_owned();

    let (status, _, body) = raw_request(
        app,
        "GET",
        &format!("/v1/artifacts/{id}/bytes"),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, whole, "reassembled bytes");
    let _ = std::fs::remove_dir_all(dir);
}

/// Chunks arriving out of order (explicit indices, plus one untagged part
/// that must fill the lowest hole) produce the same artifact as in-order
/// upload — insertion-order independence.
#[tokio::test]
async fn chunk_order_is_irrelevant() {
    let (app, dir) = default_app("out-of-order");
    let parts = [
        b"aaa".to_vec(),
        b"bbbbb".to_vec(),
        b"cc".to_vec(),
        b"dddddd".to_vec(),
    ];
    let whole: Vec<u8> = parts.iter().flatten().copied().collect();

    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "4".to_owned())],
        &[chunk(2, &parts[2]), chunk(0, &parts[0])],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{manifest}");
    let upload_id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();

    // One untagged part fills the lowest hole (1); then the last hole (3).
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-upload-id", upload_id.clone())],
        &[Chunk {
            index: None,
            data: parts[1].clone(),
        }],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{manifest}");
    assert_eq!(manifest["received"], serde_json::json!([0, 1, 2]));

    let (status, manifest) =
        post_chunks(app, &[("x-upload-id", upload_id)], &[chunk(3, &parts[3])]).await;
    assert_eq!(status, StatusCode::OK, "{manifest}");
    assert_eq!(manifest["status"], "complete");
    assert_eq!(
        manifest["artifact"]["sha256"],
        sha256_hex(&whole).as_str(),
        "order-independent digest"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The same index with different bytes is a named 409, not a silent
/// overwrite; identical bytes are idempotent 200.
#[tokio::test]
async fn conflicting_chunk_is_409() {
    let (app, dir) = default_app("conflict");
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "2".to_owned())],
        &[chunk(0, b"original-bytes")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let upload_id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();

    let (status, body) = post_chunks(
        app.clone(),
        &[("x-upload-id", upload_id.clone())],
        &[chunk(0, b"different-bytes")],
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(error_code(&body), Some("upload_conflict"));

    // Idempotent duplicate stays 200 after the conflict.
    let (status, manifest) = post_chunks(
        app,
        &[("x-upload-id", upload_id)],
        &[chunk(0, b"original-bytes")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(manifest["status"], "uploading");
    let _ = std::fs::remove_dir_all(dir);
}

/// Unknown or malformed upload ids are a named 404.
#[tokio::test]
async fn unknown_upload_id_is_404() {
    let (app, dir) = default_app("unknown-upload");
    let (status, body) = post_chunks(
        app,
        &[("x-upload-id", "01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned())],
        &[chunk(0, b"x")],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), Some("upload_not_found"));
    let _ = std::fs::remove_dir_all(dir);
}

/// A chunk above the 8 MiB per-chunk cap is a named 413 (reject-not-clamp).
#[tokio::test]
async fn oversized_chunk_is_named_413() {
    let (app, dir) = default_app("chunk-cap");
    let oversized =
        vec![0u8; usize::try_from(artifacts::CHUNK_MAX_BYTES).expect("chunk cap fits usize") + 1];
    let (status, body) = post_chunks(
        app,
        &[("x-chunk-count", "1".to_owned())],
        &[chunk(0, &oversized)],
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(error_code(&body), Some("chunk_too_large"));
    let _ = std::fs::remove_dir_all(dir);
}

/// The configurable total cap rejects both a raw single-shot body (with a
/// declared Content-Length, pre-stream) and a chunked total (mid-stream),
/// each with the named `artifact_too_large` 413.
#[tokio::test]
async fn artifact_cap_is_enforced_named() {
    let dir = tmp_dir("artifact-cap");
    let app = app_with(Some(dir.clone()), 1024);

    // Single-shot, declared length: rejected before a byte is streamed.
    let (status, _, body) = raw_request(
        app.clone(),
        "POST",
        "/v1/artifacts",
        &[
            ("content-type", "application/octet-stream".to_owned()),
            ("content-length", "2048".to_owned()),
        ],
        vec![7u8; 2048],
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    let parsed: Value = serde_json::from_slice(&body).expect("envelope");
    assert_eq!(error_code(&parsed), Some("artifact_too_large"));

    // Chunked: the second chunk pushes the total over the cap mid-stream.
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "2".to_owned())],
        &[chunk(0, &vec![1u8; 900])],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{manifest}");
    let upload_id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();
    let (status, body) = post_chunks(
        app,
        &[("x-upload-id", upload_id)],
        &[chunk(1, &vec![2u8; 400])],
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(error_code(&body), Some("artifact_too_large"));

    // Nothing was stored: a fresh app over the same dir has no blob for
    // the rejected chunk's digest.
    let (status, _, _) = raw_request(
        app_with(Some(dir.clone()), 1024),
        "GET",
        &format!("/v1/artifacts/{}/bytes", sha256_hex(&vec![1u8; 900])),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "rejected upload stored nothing"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Ranged GETs: exact slices, correct `Content-Range`/`Content-Length`,
/// open-ended and suffix forms.
#[tokio::test]
async fn ranged_gets_return_exact_slices() {
    let (app, dir) = default_app("ranges");
    let data: Vec<u8> = (0..1024u32).map(|i| (i % 253) as u8).collect();
    let info = upload_single_shot(app.clone(), "application/octet-stream", &data).await;
    let uri = format!("/v1/artifacts/{}/bytes", info.id);

    let (status, headers, body) = raw_request(
        app.clone(),
        "GET",
        &uri,
        &[("range", "bytes=100-199".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        header(&headers, "content-range"),
        Some("bytes 100-199/1024")
    );
    assert_eq!(header(&headers, "content-length"), Some("100"));
    assert_eq!(body, &data[100..200]);

    // Open-ended and suffix forms.
    let (status, headers, body) = raw_request(
        app.clone(),
        "GET",
        &uri,
        &[("range", "bytes=900-".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        header(&headers, "content-range"),
        Some("bytes 900-1023/1024")
    );
    assert_eq!(body, &data[900..]);

    let (status, headers, body) = raw_request(
        app,
        "GET",
        &uri,
        &[("range", "bytes=-24".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        header(&headers, "content-range"),
        Some("bytes 1000-1023/1024")
    );
    assert_eq!(body, &data[1000..]);
    let _ = std::fs::remove_dir_all(dir);
}

/// Bogus, multi-, and unsatisfiable ranges are the named honest 416 with
/// `Content-Range: bytes */<size>`.
#[tokio::test]
async fn bad_ranges_are_416() {
    let (app, dir) = default_app("bad-ranges");
    let data = vec![9u8; 1024];
    let info = upload_single_shot(app.clone(), "application/octet-stream", &data).await;
    let uri = format!("/v1/artifacts/{}/bytes", info.id);
    for (range, why) in [
        ("bytes=500-100", "start exceeds end"),
        ("bytes=0-1,3-4", "multi-range"),
        ("bytes=2000-3000", "unsatisfiable"),
        ("items=0-1", "non-bytes unit"),
        ("bytes=abc-def", "malformed"),
        ("bytes=-0", "empty suffix"),
    ] {
        let (status, headers, body) = raw_request(
            app.clone(),
            "GET",
            &uri,
            &[("range", range.to_owned())],
            Vec::new(),
        )
        .await;
        assert_eq!(status, StatusCode::RANGE_NOT_SATISFIABLE, "{why}: {range}");
        let parsed: Value = serde_json::from_slice(&body).expect("envelope");
        assert_eq!(error_code(&parsed), Some("invalid_range"), "{range}");
        assert_eq!(
            header(&headers, "content-range"),
            Some("bytes */1024"),
            "{range}"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// `If-Match` with a stale `ETag` is the named 412; the current `ETag` passes;
/// `If-Range` with a stale `ETag` degrades to a full 200 (RFC 7233).
#[tokio::test]
async fn if_match_and_if_range() {
    let (app, dir) = default_app("preconditions");
    let data: Vec<u8> = (0..512u32).map(|i| (i % 61) as u8).collect();
    let info = upload_single_shot(app.clone(), "application/octet-stream", &data).await;
    let uri = format!("/v1/artifacts/{}/bytes", info.id);
    let etag = format!("\"{}\"", info.sha256);
    let stale = "\"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef\"".to_owned();

    // Stale If-Match → named 412.
    let (status, _, body) = raw_request(
        app.clone(),
        "GET",
        &uri,
        &[("if-match", stale.clone())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
    let parsed: Value = serde_json::from_slice(&body).expect("envelope");
    assert_eq!(error_code(&parsed), Some("precondition_failed"));

    // Current If-Match passes (etag forms: quoted, unquoted, `*`).
    for value in [etag.clone(), info.sha256.clone(), "*".to_owned()] {
        let (status, headers, body) = raw_request(
            app.clone(),
            "GET",
            &uri,
            &[
                ("if-match", value.clone()),
                ("range", "bytes=0-9".to_owned()),
            ],
            Vec::new(),
        )
        .await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT, "if-match {value}");
        assert_eq!(body, &data[..10]);
        assert_eq!(header(&headers, "etag"), Some(etag.as_str()));
    }

    // Stale If-Range ignores the Range and serves the full 200.
    let (status, headers, body) = raw_request(
        app.clone(),
        "GET",
        &uri,
        &[("if-range", stale), ("range", "bytes=0-9".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(header(&headers, "content-range"), None);
    assert_eq!(body, data);

    // Matching If-Range keeps the 206.
    let (status, _, body) = raw_request(
        app,
        "GET",
        &uri,
        &[("if-range", etag), ("range", "bytes=0-9".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(body, &data[..10]);
    let _ = std::fs::remove_dir_all(dir);
}

/// Unknown artifact ids — well-formed-but-absent and malformed — are the
/// named 404. (Multi-segment or empty ids can't reach the route at all:
/// they fall through to the UI fallback by routing, never the handler.)
#[tokio::test]
async fn unknown_artifacts_are_404() {
    let (app, dir) = default_app("unknown-artifact");
    for (id, envelope) in [
        (sha256_hex(b"never uploaded"), false),
        ("zz".to_owned(), true),
        ("%2e%2e".to_owned(), true),
        (format!("{}zz", "a".repeat(62)), true),
    ] {
        let (status, _, body) = raw_request(
            app.clone(),
            "GET",
            &format!("/v1/artifacts/{id}/bytes"),
            &[],
            Vec::new(),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "id {id:?}");
        if envelope {
            let parsed: Value = serde_json::from_slice(&body).expect("envelope");
            assert_eq!(error_code(&parsed), Some("artifact_not_found"), "{id:?}");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The declared content type rides the upload and is served back; HEAD
/// keeps it too.
#[tokio::test]
async fn content_type_roundtrip() {
    let (app, dir) = default_app("content-type");
    let info = upload_single_shot(app.clone(), "image/png", b"fakepng").await;
    let (status, headers, _) = raw_request(
        app,
        "HEAD",
        &format!("/v1/artifacts/{}/bytes", info.id),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(header(&headers, "content-type"), Some("image/png"));
    let _ = std::fs::remove_dir_all(dir);
}

/// Compression interaction (the critical case): a LARGE full GET of a
/// compressible artifact compresses on negotiation, while a byte-range
/// response arrives UNCOMPRESSED and byte-identical — the router predicate
/// skips every response carrying `Content-Range` (same mechanism that
/// excludes SSE by content type), so range math and resume offsets stay
/// honest.
#[tokio::test]
async fn range_responses_stay_uncompressed_full_get_may_compress() {
    let (app, dir) = default_app("compression");
    let data = vec![b'h'; 4096]; // compressible, > 1 KiB floor
    let info = upload_single_shot(app.clone(), "text/plain", &data).await;
    let uri = format!("/v1/artifacts/{}/bytes", info.id);

    // Full GET MAY compress: negotiated gzip arrives encoded and decodes
    // byte-identical.
    let (status, headers, body) = raw_request(
        app.clone(),
        "GET",
        &uri,
        &[("accept-encoding", "gzip".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(header(&headers, "content-encoding"), Some("gzip"));
    assert_eq!(gunzip(&body), data);

    // Range GET must NOT compress: identity body, no content-encoding.
    let (status, headers, body) = raw_request(
        app,
        "GET",
        &uri,
        &[
            ("accept-encoding", "gzip".to_owned()),
            ("range", "bytes=100-199".to_owned()),
        ],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        header(&headers, "content-encoding"),
        None,
        "206 must be plain"
    );
    assert_eq!(
        header(&headers, "content-range"),
        Some("bytes 100-199/4096")
    );
    assert_eq!(body, &data[100..200], "byte-identical slice");
    let _ = std::fs::remove_dir_all(dir);
}

/// An upload survives a server restart: the on-disk chunk manifest lets a
/// fresh Registry (new app instance over the same data dir) finish an
/// upload started by the old one.
#[tokio::test]
async fn resume_survives_server_restart() {
    let dir = tmp_dir("restart");
    let app_before = app_with(Some(dir.clone()), crate::config::DEFAULT_ARTIFACT_MAX_BYTES);
    let c0 = b"before-restart".to_vec();
    let c1 = b"after-restart".to_vec();
    let (status, manifest) = post_chunks(
        app_before,
        &[("x-chunk-count", "2".to_owned())],
        &[chunk(0, &c0)],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{manifest}");
    let upload_id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();
    // `app_before` was consumed by the request — the new Registry below IS
    // the restart (its in-memory upload map starts empty; only the on-disk
    // manifest knows about chunk 0).
    let app_after = app_with(Some(dir.clone()), crate::config::DEFAULT_ARTIFACT_MAX_BYTES);
    let (status, manifest) = post_chunks(
        app_after.clone(),
        &[("x-upload-id", upload_id)],
        &[chunk(1, &c1)],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "resumed after restart: {manifest}");
    assert_eq!(manifest["status"], "complete");

    let whole: Vec<u8> = [c0.as_slice(), c1.as_slice()].concat();
    let id = manifest["artifact"]["id"]
        .as_str()
        .expect("artifact id")
        .to_owned();
    let (status, _, body) = raw_request(
        app_after,
        "GET",
        &format!("/v1/artifacts/{id}/bytes"),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, whole);
    let _ = std::fs::remove_dir_all(dir);
}

/// Chunk-count mismatches are conflicts, and a missing count on a new
/// chunked upload is a named 400.
#[tokio::test]
async fn chunk_count_contract() {
    let (app, dir) = default_app("count-contract");
    // New upload without X-Chunk-Count → named 400.
    let (status, body) = post_chunks(app.clone(), &[], &[chunk(0, b"x")]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("invalid_chunk_count"));

    // Declared count is immutable across the session.
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "2".to_owned())],
        &[chunk(0, b"x")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let upload_id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();
    let (status, body) = post_chunks(
        app,
        &[
            ("x-upload-id", upload_id),
            ("x-chunk-count", "5".to_owned()),
        ],
        &[chunk(1, b"y")],
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(error_code(&body), Some("upload_conflict"));
    let _ = std::fs::remove_dir_all(dir);
}

/// An index beyond the declared count is a named 400 (never an out-of-bounds
/// write into the chunk registry).
#[tokio::test]
async fn out_of_bounds_chunk_index_is_400() {
    let (app, dir) = default_app("index-bounds");
    let (status, body) = post_chunks(
        app,
        &[("x-chunk-count", "2".to_owned())],
        &[chunk(5, b"beyond")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("invalid_chunk_index"));
    let _ = std::fs::remove_dir_all(dir);
}

/// Active/declared content types are refused on upload, and every stored
/// byte response forces download + nosniff so a bad type cannot execute in
/// the server origin (S1).
#[tokio::test]
async fn active_content_types_rejected_and_get_forces_download() {
    let (app, dir) = default_app("xss-hardening");
    for bad in [
        "text/html",
        "TEXT/HTML",
        "image/svg+xml",
        "text/javascript",
        "application/javascript",
        "text/xml",
    ] {
        let (status, _, body) = raw_request(
            app.clone(),
            "POST",
            "/v1/artifacts",
            &[
                ("content-type", "application/octet-stream".to_owned()),
                ("x-artifact-content-type", bad.to_owned()),
            ],
            b"<script>alert(1)</script>".to_vec(),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "type {bad}: {}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(
            error_code(&serde_json::from_slice::<Value>(&body).expect("json")),
            Some("invalid_content_type")
        );
    }

    let info = upload_single_shot(app.clone(), "text/plain", b"hello").await;
    let uri = format!("/v1/artifacts/{}/bytes", info.id);
    let (status, headers, _) = raw_request(app.clone(), "GET", &uri, &[], Vec::new()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(header(&headers, "content-disposition"), Some("attachment"));
    assert_eq!(header(&headers, "x-content-type-options"), Some("nosniff"));

    // Ranged responses carry the stored bytes too: same hardening.
    let (status, headers, _) = raw_request(
        app,
        "GET",
        &uri,
        &[("range", "bytes=0-1".to_owned())],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(header(&headers, "content-disposition"), Some("attachment"));
    assert_eq!(header(&headers, "x-content-type-options"), Some("nosniff"));
    let _ = std::fs::remove_dir_all(dir);
}

/// Storage errors log the full path server-side but return a fixed message
/// with no absolute `data_dir` path (S2).
#[tokio::test]
async fn storage_error_hides_data_dir_path() {
    let (app, dir) = default_app("storage-error-redaction");
    let data = b"known payload for a deterministic digest";
    let sha = sha256_hex(data);
    // Make the sidecar's atomic-write .part path unwritable so the upload
    // fails in write_json_atomic, whose error text embeds the absolute path.
    let sidecar_part = dir
        .join("artifacts")
        .join("blobs")
        .join(&sha[..2])
        .join(format!("{sha}.json.part"));
    std::fs::create_dir_all(&sidecar_part).expect("create blocking directory");

    let (status, _, body) = raw_request(
        app,
        "POST",
        "/v1/artifacts",
        &[("content-type", "application/octet-stream".to_owned())],
        data.to_vec(),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let json: Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(json["error"]["code"], "storage_unavailable");
    assert_eq!(json["error"]["message"], "artifact storage failure");
    let text = String::from_utf8_lossy(&body);
    assert!(
        !text.contains(dir.to_string_lossy().as_ref()),
        "data dir leaked: {text}"
    );
    assert!(!text.contains(sha.as_str()), "internal path leaked: {text}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A fresh session that fails (no parts, or a rejected part) leaves no
/// directory behind; the map entry is dropped too (S3).
#[tokio::test]
async fn failed_new_upload_leaves_no_session() {
    let (app, dir) = default_app("abandoned-upload");
    let uploads = dir.join("artifacts").join("uploads");

    let (status, body) = post_chunks(app.clone(), &[("x-chunk-count", "2".to_owned())], &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("invalid_chunk_count"));
    assert!(
        std::fs::read_dir(&uploads)
            .expect("uploads dir")
            .next()
            .is_none()
    );

    let empty = Chunk {
        index: None,
        data: Vec::new(),
    };
    let (status, body) = post_chunks(app, &[("x-chunk-count", "1".to_owned())], &[empty]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("empty_chunk"));
    assert!(
        std::fs::read_dir(&uploads)
            .expect("uploads dir")
            .next()
            .is_none()
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A bad chunk on a RESUMED request must not destroy the session: the client
/// still owns the upload id and can retry (S3).
#[tokio::test]
async fn failed_resumed_upload_stays_resumable() {
    let (app, dir) = default_app("resume-after-error");
    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "2".to_owned())],
        &[chunk(0, b"first")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();

    let (status, body) = post_chunks(
        app.clone(),
        &[("x-upload-id", id.clone())],
        &[Chunk {
            index: None,
            data: Vec::new(),
        }],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("empty_chunk"));

    let (status, manifest) = post_chunks(app, &[("x-upload-id", id)], &[chunk(1, b"second")]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(manifest["status"], "complete");
    assert_eq!(manifest["artifact"]["size_bytes"], 11);
    let _ = std::fs::remove_dir_all(dir);
}

/// Chunk count and parts-per-request are bounded, and zero-byte parts are
/// rejected instead of consuming a manifest slot (S4).
#[tokio::test]
async fn chunk_and_part_bounds_are_enforced() {
    let (app, dir) = default_app("bounds");
    // MAX_CHUNK_COUNT is 10_000 in artifacts.rs.
    let (status, body) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "10001".to_owned())],
        &[chunk(0, b"x")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("invalid_chunk_count"));

    let parts: Vec<Chunk> = (0..1025).map(|i| chunk(i, b"x")).collect();
    let (status, body) = post_chunks(app, &[("x-chunk-count", "1025".to_owned())], &parts).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), Some("too_many_parts"));
    let _ = std::fs::remove_dir_all(dir);
}

/// Untagged parts fill the lowest free index across resumed requests; the
/// running byte total is incremented (not re-summed) on each register (S4).
#[tokio::test]
async fn untagged_chunks_fill_lowest_index_across_resume() {
    let (app, dir) = default_app("lowest-free");
    let untagged = |data: &'static [u8]| Chunk {
        index: None,
        data: data.to_vec(),
    };

    let (status, manifest) = post_chunks(
        app.clone(),
        &[("x-chunk-count", "3".to_owned())],
        &[untagged(b"a")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let id = manifest["upload_id"]
        .as_str()
        .expect("upload id")
        .to_owned();
    let (status, _) = post_chunks(
        app.clone(),
        &[("x-upload-id", id.clone())],
        &[untagged(b"b")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, manifest) = post_chunks(app, &[("x-upload-id", id)], &[untagged(b"c")]).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(manifest["status"], "complete");
    let expected = sha256_hex(b"abc");
    assert_eq!(manifest["artifact"]["id"].as_str(), Some(expected.as_str()));
    let _ = std::fs::remove_dir_all(dir);
}

/// A weak validator never matches the strong `ETag` (S6).
#[tokio::test]
async fn weak_if_match_is_rejected() {
    let (app, dir) = default_app("weak-if-match");
    let info = upload_single_shot(app.clone(), "application/octet-stream", b"payload").await;
    let uri = format!("/v1/artifacts/{}/bytes", info.id);

    let (status, _, body) = raw_request(
        app.clone(),
        "GET",
        &uri,
        &[("if-match", format!("W/\"{}\"", info.sha256))],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
    let json: Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(error_code(&json), Some("precondition_failed"));

    let (status, _, _) = raw_request(
        app,
        "GET",
        &uri,
        &[("if-match", format!("\"{}\"", info.sha256))],
        Vec::new(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let _ = std::fs::remove_dir_all(dir);
}
