//! `POST /v1/image-ops` (epic-editor-integration; tickets
//! TASK-image-ops-command-dto and TASK-image-ops-job-route): the bounded
//! command DTO (closed op vocabulary, `deny_unknown_fields`,
//! reject-not-clamp bounds), the `request_id` idempotency contract (same
//! key + same command replays the first job byte-identically without
//! re-executing; same key + different command is a named 400), the
//! deterministic pixel pipeline (Lanczos3, fixed encoders), bundle writing
//! with the sidecar records, artifact round-trips, and the job lifecycle
//! on the registry + `/v1/events`.
//!
//! No `SLOT_LOCK` here: image ops take the ledger's CPU-only `rag` bucket
//! (all-zero demand, unmanaged budgets by default), so admission never
//! contends — with itself or with the GPU slot other binaries serialize.
//! `request_id`s are unique per test because the idempotency ledger is a
//! process-global shared by this binary's parallel tests.
//!
//! `OP_LOCK` IS needed: the JOB registry is process-global too, and every
//! image-op request registers a job there (`GET /v1/jobs` lists them all).
//! Two image-op tests in parallel see each other's jobs — the cancel test
//! would even cancel a FOREIGN job. One image-op request at a time keeps
//! the registry view unambiguous.

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

use std::sync::{Arc, RwLock};
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
use sha2::Digest as _;
use tokio_stream::StreamExt as _;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Per-test data dir (the runner pattern from `edits.rs`): unique ULID
/// suffix, deterministic teardown at the end of the test.
fn fresh_data_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-image-ops-{tag}-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).expect("create data dir");
    dir
}

fn app_with(data_dir: Option<std::path::PathBuf>) -> Router {
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
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// A real `width`×`height` opaque-white PNG via the in-tree encoder (the
/// same container the API itself emits).
fn png_bytes(width: u32, height: u32) -> Vec<u8> {
    honkbox_sd_core::encode_image(
        honkbox_sd_core::GeneratedImage {
            width,
            height,
            rgba: vec![255; (width * height * 4) as usize],
        },
        honkbox_sd_core::OutputFormat::Png,
    )
    .expect("png encodes")
}

fn png_data_uri(width: u32, height: u32) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png_bytes(width, height))
    )
}

/// Serializes the image-op requests of this binary: the job registry is
/// PROCESS-GLOBAL, so two concurrent image-op requests would list each
/// other's jobs on `GET /v1/jobs` (and the cancel test would cancel a
/// foreign job). One request at a time keeps the registry unambiguous.
static OP_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Submit one image-op request; returns (status, headers, parsed JSON).
/// Takes `OP_LOCK`; a caller that already holds it uses `post_image_ops`.
async fn image_ops(app: Router, body: Value) -> (StatusCode, axum::http::HeaderMap, Value) {
    let _op = OP_LOCK.lock().await;
    post_image_ops(app, body).await
}

/// The raw image-op request; the caller owns `OP_LOCK`.
async fn post_image_ops(app: Router, body: Value) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/image-ops")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .unwrap();
    let parsed =
        serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("response not JSON: {err}"));
    (status, headers, parsed)
}

fn resize_request(request_id: &str, width: u32, height: u32, inputs: &Value) -> Value {
    json!({
        "command": {"resize": {"width": width, "height": height}},
        "inputs": inputs,
        "request_id": request_id,
    })
}

fn one_input(width: u32, height: u32) -> Value {
    json!([{"data_uri": png_data_uri(width, height)}])
}

/// Decode a response `b64_json` entry and return the magic-checked PNG
/// bytes plus the IHDR (width, height).
fn decoded_png(entry: &Value) -> (Vec<u8>, u32, u32) {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(entry["b64_json"].as_str().expect("b64_json string"))
        .expect("valid base64");
    assert_eq!(
        &bytes[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        "png magic"
    );
    assert_eq!(&bytes[12..16], b"IHDR");
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    (bytes, width, height)
}

/// Read and parse one JSON file from a bundle dir.
fn bundle_json(dir: &std::path::Path, name: &str) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(dir.join(name))
            .unwrap_or_else(|err| panic!("read {name} under {}: {err}", dir.display())),
    )
    .unwrap_or_else(|err| panic!("{name} not JSON: {err}"))
}

/// The single bundle dir under `<data_dir>/jobs/` (fails on 0 or 2+).
fn sole_bundle(data_dir: &std::path::Path) -> std::path::PathBuf {
    let mut entries = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let first = entries.next().expect("exactly one bundle").unwrap();
    assert!(
        entries.next().is_none(),
        "expected exactly one bundle dir, found more"
    );
    first.path()
}

/// Happy path: resize 4x4 → 2x3, exact dimensions, deterministic digest,
/// full bundle on disk with every sidecar record the ticket names.
#[tokio::test]
async fn resize_produces_exact_dimensions_and_bundle() {
    let data_dir = fresh_data_dir("resize");
    let (status, headers, body) = image_ops(
        app_with(Some(data_dir.clone())),
        resize_request("resize-bundle-1", 2, 3, &one_input(4, 4)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        headers
            .get("x-job-id")
            .is_some_and(|v| v.to_str().unwrap().starts_with("job_")),
        "X-Job-Id header: {headers:?}"
    );
    assert_eq!(body["request_id"], "resize-bundle-1", "{body}");
    assert_eq!(body["outputs"].as_array().unwrap().len(), 1);
    let (bytes, width, height) = decoded_png(&body["outputs"][0]);
    assert_eq!((width, height), (2, 3), "resized dimensions");
    // Response digest matches the bytes (editor-side determinism check).
    assert_eq!(
        body["outputs"][0]["sha256"],
        hex(&sha2::Sha256::digest(&bytes)),
        "{body}"
    );
    assert_eq!(body["outputs"][0]["format"], "png");

    // Bundle: layout + every sidecar record (command hash, input/output
    // hashes, seed, tool version, attempts, outcome).
    let dir = sole_bundle(&data_dir);
    assert_eq!(
        body["bundle"].as_str().unwrap(),
        dir.file_name().unwrap().to_string_lossy().as_ref()
    );
    for expected in ["command.json", "ops.json", "op_00.png", "run.log"] {
        assert!(dir.join(expected).exists(), "missing {expected}");
    }
    let command = bundle_json(&dir, "command.json");
    assert_eq!(command["op"], "resize", "{command}");
    assert_eq!(command["output_format"], "png", "{command}");
    assert_eq!(command["request_id"], "resize-bundle-1", "{command}");
    let raw_command = std::fs::read_to_string(dir.join("command.json")).unwrap();
    assert!(
        !raw_command.contains("base64"),
        "sidecar never carries payload bytes: {raw_command}"
    );
    let ops = bundle_json(&dir, "ops.json");
    assert_eq!(ops["op"], "resize", "{ops}");
    assert_eq!(
        ops["command_sha256"].as_str().unwrap().len(),
        64,
        "command hash is sha256 hex: {ops}"
    );
    assert_eq!(ops["inputs"][0]["source"], "data_uri", "{ops}");
    assert_eq!(
        ops["inputs"][0]["sha256"].as_str().unwrap().len(),
        64,
        "input digest recorded: {ops}"
    );
    assert_eq!(ops["outputs"][0]["file"], "op_00.png", "{ops}");
    assert_eq!(ops["outputs"][0]["sha256"], body["outputs"][0]["sha256"]);
    assert!(ops["seed"].is_null(), "pixel ops are seedless: {ops}");
    assert_eq!(ops["tool"]["name"], "honkbox-image-ops", "{ops}");
    assert_eq!(ops["tool"]["version"], env!("CARGO_PKG_VERSION"), "{ops}");
    assert_eq!(ops["attempts"], 1, "{ops}");
    assert_eq!(ops["outcome"], "done", "{ops}");
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Determinism: identical commands under DIFFERENT request ids produce
/// byte-identical outputs (fixed resample + fixed encoders, no timestamps).
#[tokio::test]
async fn identical_commands_produce_identical_bytes() {
    let app = app_with(None);
    let (_, _, first) = image_ops(
        app.clone(),
        resize_request("determinism-a", 3, 3, &one_input(4, 4)),
    )
    .await;
    let (_, _, second) =
        image_ops(app, resize_request("determinism-b", 3, 3, &one_input(4, 4))).await;
    assert_eq!(
        first["outputs"][0]["sha256"],
        second["outputs"][0]["sha256"]
    );
    // Different request ids → different jobs (no accidental dedup).
    assert_ne!(first["job_id"], second["job_id"]);
}

/// Idempotency: the same `request_id` + same command replays the first
/// response byte-identically (same job id, same bundle, same `created`) and
/// executes exactly once — one bundle dir on disk.
#[tokio::test]
async fn idempotent_retry_replays_first_response() {
    let data_dir = fresh_data_dir("idem");
    let app = app_with(Some(data_dir.clone()));
    let request = resize_request("idem-1", 2, 2, &one_input(4, 4));
    let (status, _, first) = image_ops(app.clone(), request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (replay_status, _, replay) = image_ops(app, request).await;
    assert_eq!(replay_status, StatusCode::OK, "{replay}");
    assert_eq!(replay, first, "replay is byte-identical (parsed)");
    assert_eq!(replay["job_id"], first["job_id"], "same job id");
    assert_eq!(replay["bundle"], first["bundle"], "same bundle");
    assert_eq!(
        sole_bundle(&data_dir)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .as_ref(),
        first["bundle"].as_str().unwrap(),
        "exactly one bundle was written"
    );
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Same `request_id`, different command: a named 400 conflict, and the
/// conflicting command did NOT execute (the original bundle is untouched).
#[tokio::test]
async fn request_id_reuse_with_different_command_is_400() {
    let data_dir = fresh_data_dir("conflict");
    let app = app_with(Some(data_dir.clone()));
    let (status, _, first) = image_ops(
        app.clone(),
        resize_request("conflict-1", 2, 2, &one_input(4, 4)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (status, _, conflict) = image_ops(
        app,
        json!({
            "command": "grayscale",
            "inputs": one_input(4, 4),
            "request_id": "conflict-1",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{conflict}");
    assert_eq!(
        conflict["error"]["code"], "image_op_request_id_conflict",
        "{conflict}"
    );
    assert!(sole_bundle(&data_dir).is_dir());
    std::fs::remove_dir_all(&data_dir).ok();
}

/// The DTO is closed: unknown op tags, unknown top-level fields, and
/// unknown fields inside op params are all named 400s (a missing required
/// key like `request_id` is the same named rejection family).
#[tokio::test]
async fn unknown_fields_and_ops_are_named_400s() {
    let app = app_with(None);
    let cases = [
        (
            json!({
                "command": {"sharpen": {"amount": 1}},
                "inputs": one_input(4, 4),
                "request_id": "unknown-op",
            }),
            "unknown op tag",
        ),
        (
            json!({
                "command": {"resize": {"width": 2, "height": 2}},
                "inputs": one_input(4, 4),
                "request_id": "unknown-field",
                "priority": 5,
            }),
            "unknown top-level field",
        ),
        (
            json!({
                "command": {"resize": {"width": 2, "height": 2, "filter": "nearest"}},
                "inputs": one_input(4, 4),
                "request_id": "unknown-param",
            }),
            "unknown field inside params",
        ),
        (
            json!({
                "command": {"rotate": {"degrees": 90}},
                "inputs": one_input(4, 4),
            }),
            "missing request_id",
        ),
    ];
    for (body, what) in cases {
        let (status, _, parsed) = image_ops(app.clone(), body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{what}: {parsed}");
        assert_eq!(
            parsed["error"]["code"], "invalid_image_op_request",
            "{what}: {parsed}"
        );
    }
}

/// Reject-not-clamp on every numeric param: zero-size edges, edges over the
/// 8192 cap, targets over the pixel cap, non-quarter rotations, empty crop
/// rects, and bad JPEG quality values.
#[tokio::test]
async fn out_of_range_params_are_rejected_not_clamped() {
    let app = app_with(None);
    let cases = [
        (
            json!({
                "command": {"resize": {"width": 0, "height": 4}},
                "inputs": one_input(4, 4),
                "request_id": "range-zero",
            }),
            "zero width",
        ),
        (
            json!({
                "command": {"resize": {"width": 8193, "height": 1}},
                "inputs": one_input(4, 4),
                "request_id": "range-edge",
            }),
            "edge over 8192",
        ),
        (
            json!({
                "command": {"resize": {"width": 8192, "height": 8192}},
                "inputs": one_input(4, 4),
                "request_id": "range-pixels",
            }),
            "target over the pixel cap",
        ),
        (
            json!({
                "command": {"rotate": {"degrees": 45}},
                "inputs": one_input(4, 4),
                "request_id": "range-rotate",
            }),
            "non-quarter rotation",
        ),
        (
            json!({
                "command": {"crop": {"x": 0, "y": 0, "width": 0, "height": 2}},
                "inputs": one_input(4, 4),
                "request_id": "range-crop",
            }),
            "empty crop",
        ),
        (
            json!({
                "command": "format",
                "inputs": one_input(4, 4),
                "request_id": "range-quality",
                "output_format": "jpeg",
                "output_compression": 101,
            }),
            "jpeg quality over 100",
        ),
        (
            json!({
                "command": "format",
                "inputs": one_input(4, 4),
                "request_id": "range-quality-fraction",
                "output_format": "jpeg",
                "output_compression": 3.5,
            }),
            "fractional jpeg quality",
        ),
    ];
    for (body, what) in cases {
        let (status, _, parsed) = image_ops(app.clone(), body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{what}: {parsed}");
        let code = parsed["error"]["code"].as_str().unwrap_or_default();
        assert!(
            code == "invalid_image_op_params" || code == "invalid_output_compression",
            "{what}: {parsed}"
        );
        let message = parsed["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("rejected")
                || message.contains("not clamped")
                || message.contains("must be"),
            "{what}: named message: {parsed}"
        );
    }
}

/// Out-of-bounds crop rects are a named 400 against the REAL pixels —
/// never clamped to the image edge.
#[tokio::test]
async fn crop_out_of_bounds_is_400() {
    let app = app_with(None);
    let (status, _, parsed) = image_ops(
        app,
        json!({
            "command": {"crop": {"x": 3, "y": 0, "width": 2, "height": 4}},
            "inputs": one_input(4, 4),
            "request_id": "crop-bounds",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{parsed}");
    assert_eq!(parsed["error"]["code"], "invalid_crop_rect", "{parsed}");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("4x4 input")),
        "{parsed}"
    );
}

/// The in-vocabulary ops run end-to-end with the fixed determinism
/// contract: 90° rotation swaps the IHDR dimensions, grayscale converts,
/// `format` re-encodes into the requested container (JPEG/WebP magic).
#[tokio::test]
async fn rotate_grayscale_and_format_ops_run() {
    let app = app_with(None);
    // Rotate 90 on a 4x2 input → 2x4 output.
    let (status, _, body) = image_ops(
        app.clone(),
        json!({
            "command": {"rotate": {"degrees": 90}},
            "inputs": one_input(4, 2),
            "request_id": "rotate-1",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, width, height) = decoded_png(&body["outputs"][0]);
    assert_eq!((width, height), (2, 4), "90° swap");

    // Grayscale still emits a valid PNG.
    let (status, _, body) = image_ops(
        app.clone(),
        json!({
            "command": "grayscale",
            "inputs": one_input(4, 4),
            "request_id": "gray-1",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    decoded_png(&body["outputs"][0]);

    // `format` with output_format=jpeg: JPEG magic, format field says jpeg.
    let (status, _, body) = image_ops(
        app.clone(),
        json!({
            "command": "format",
            "inputs": one_input(4, 4),
            "request_id": "fmt-jpeg",
            "output_format": "jpeg",
            "output_compression": 85,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jpeg = base64::engine::general_purpose::STANDARD
        .decode(body["outputs"][0]["b64_json"].as_str().unwrap())
        .unwrap();
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "jpeg magic");
    assert_eq!(body["outputs"][0]["format"], "jpeg");

    // …and webp.
    let (status, _, body) = image_ops(
        app,
        json!({
            "command": "format",
            "inputs": one_input(4, 4),
            "request_id": "fmt-webp",
            "output_format": "webp",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let webp = base64::engine::general_purpose::STANDARD
        .decode(body["outputs"][0]["b64_json"].as_str().unwrap())
        .unwrap();
    assert_eq!(&webp[..4], b"RIFF", "webp container");
    assert_eq!(&webp[8..12], b"WEBP", "webp magic");
    assert_eq!(body["outputs"][0]["format"], "webp");
}

/// Batch: one output per input, in order; 9 inputs is a named 400.
#[tokio::test]
async fn batch_one_output_per_input_and_cap_enforced() {
    let app = app_with(None);
    let (status, _, body) = image_ops(
        app.clone(),
        resize_request(
            "batch-ok",
            2,
            2,
            &json!([
                {"data_uri": png_data_uri(4, 4)},
                {"data_uri": png_data_uri(6, 6)},
                {"data_uri": png_data_uri(8, 8)},
            ]),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let outputs = body["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 3);
    for (i, expected) in [(2, 2), (2, 2), (2, 2)].iter().enumerate() {
        let (_, width, height) = decoded_png(&outputs[i]);
        assert_eq!((width, height), *expected, "output {i}");
    }

    let (status, _, body) = image_ops(
        app,
        resize_request(
            "batch-over",
            2,
            2,
            &json!(
                (0..9)
                    .map(|_| json!({"data_uri": png_data_uri(4, 4)}))
                    .collect::<Vec<_>>()
            ),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_image_op_inputs", "{body}");
}

/// Malformed inputs are named 400s: paths, non-image MIME, missing
/// `;base64`, invalid base64 payloads, garbage bytes, path-flavored
/// variants, and the empty batch.
#[tokio::test]
async fn malformed_inputs_are_named_400s() {
    let app = app_with(None);
    let bad_inputs = [
        (json!([{"data_uri": "/etc/passwd"}]), "path, not a data URI"),
        (
            json!([{"data_uri": "data:image/gif;base64,R0lGOD"}]),
            "gif MIME",
        ),
        (
            json!([{"data_uri": "data:image/png,R0lGOD"}]),
            "missing ;base64",
        ),
        (
            json!([{"data_uri": "data:image/png;base64,not-base64!!"}]),
            "invalid base64",
        ),
        (
            json!([{"data_uri": "data:image/jpeg;base64,eHl6"}]),
            "declared jpeg, actual garbage",
        ),
        (json!([{"file": "/etc/passwd"}]), "file path variant"),
        (json!([]), "empty batch"),
    ];
    for (inputs, what) in bad_inputs {
        let (status, _, parsed) =
            image_ops(app.clone(), resize_request("malformed", 2, 2, &inputs)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{what}: {parsed}");
        let code = parsed["error"]["code"].as_str().unwrap_or_default();
        assert!(
            code == "invalid_image_op_inputs" || code == "invalid_image_op_request",
            "{what}: {parsed}"
        );
    }
}

/// Artifact round-trip: an ops result feeds a second op by bundle+index;
/// the sidecar records the artifact provenance. Unknown bundles and
/// path-flavored ids are a named 400 (no filesystem probing).
#[tokio::test]
async fn artifact_round_trip_and_unknown_bundle() {
    let data_dir = fresh_data_dir("artifact");
    let app = app_with(Some(data_dir.clone()));
    let (status, _, first) = image_ops(
        app.clone(),
        resize_request("artifact-1", 3, 3, &one_input(4, 4)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let bundle = first["bundle"].as_str().unwrap().to_owned();

    let (status, _, second) = image_ops(
        app.clone(),
        json!({
            "command": "grayscale",
            "inputs": [{"artifact": {"bundle": bundle, "index": 0}}],
            "request_id": "artifact-2",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    // The second bundle's sidecar records the artifact provenance…
    let second_bundle = second["bundle"].as_str().unwrap().to_owned();
    assert_ne!(second_bundle, bundle);
    let ops = bundle_json(&data_dir.join("jobs").join(&second_bundle), "ops.json");
    assert_eq!(
        ops["inputs"][0]["source"],
        format!("artifact:{bundle}/0"),
        "{ops}"
    );

    // …and unknown bundles / bogus ids never touch the filesystem. The
    // `deny_unknown_fields` artifact shape rejects the `path` key outright.
    for bad in [
        json!([{"artifact": {"bundle": "not-a-ulid", "index": 0}}]),
        json!([{"artifact": {"bundle": "01ARZ3NDEKTSV4RRFFQ69G5FAV", "index": 0}}]),
        json!([{"artifact": {"bundle": bundle, "index": 7}}]),
        json!([{"artifact": {"bundle": bundle, "index": 0, "path": "../x"}}]),
    ] {
        let (status, _, parsed) = image_ops(
            app.clone(),
            json!({
                "command": "grayscale",
                "inputs": bad,
                "request_id": "artifact-bad",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {parsed}");
    }
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Lifecycle: while the batch runs the job is registry-visible (kind
/// `image_ops`, honest `interruptible`), cancel between units is honored —
/// either the 409 `job_cancelled` envelope, or — if the batch already
/// finished — the completed 200; both outcomes are contract-valid, so the
/// test never flakes. A cancelled batch writes no bundle.
#[tokio::test]
async fn cancel_between_units_is_honored() {
    // Hold `OP_LOCK` for the whole test: the registry below is
    // let _op = OP_LOCK.lock().await; // TEMP repro: lock removed
    // request below runs lock-free because this test owns the lock.
    let _op = OP_LOCK.lock().await;
    let data_dir = fresh_data_dir("cancel");
    let app = app_with(Some(data_dir.clone()));
    // 8 × 512² resizes: a real processing window, still fast in CI.
    let request = resize_request(
        "cancel-1",
        1024,
        1024,
        &json!(
            (0..8)
                .map(|_| json!({"data_uri": png_data_uri(512, 512)}))
                .collect::<Vec<_>>()
        ),
    );
    let work = tokio::spawn(post_image_ops(app.clone(), request));
    // Poll the registry for the live job; the batch is either caught
    // mid-flight or already completed — both accepted. The only
    // `image_ops` job that can be live is this test's own.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut cancelled_id: Option<String> = None;
    while Instant::now() < deadline {
        let (_, _, jobs) = send_get(app.clone(), "/v1/jobs").await;
        for job in jobs["data"].as_array().cloned().unwrap_or_default() {
            if job["kind"] == "image_ops" {
                assert_eq!(job["interruptible"], true, "cancel contract: {job}");
                cancelled_id = Some(job["id"].as_str().unwrap().to_owned());
                break;
            }
        }
        if cancelled_id.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let bundles_before = bundle_count(&data_dir);
    if let Some(id) = cancelled_id {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/jobs/{id}/cancel"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let cancel_body_bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let cancel_json: Value = serde_json::from_slice(&cancel_body_bytes).unwrap();
        assert_eq!(cancel_json["cancelled"], true, "{cancel_json}");
    }
    let (status, _, body) = work.await.expect("op task joins");
    match status {
        StatusCode::CONFLICT => {
            assert_eq!(body["error"]["code"], "job_cancelled", "{body}");
            assert_eq!(
                bundle_count(&data_dir),
                bundles_before,
                "cancelled batch writes no bundle"
            );
        }
        StatusCode::OK => {
            assert_eq!(body["request_id"], "cancel-1", "{body}");
        }
        other => panic!("unexpected status {other}: {body}"),
    }
    std::fs::remove_dir_all(&data_dir).ok();
}

fn bundle_count(data_dir: &std::path::Path) -> usize {
    std::fs::read_dir(data_dir.join("jobs")).map_or(0, std::iter::Iterator::count)
}

/// GET helper (used for `/v1/jobs` polling and `/v1/capabilities`).
async fn send_get(app: Router, uri: &str) -> (StatusCode, axum::http::HeaderMap, Value) {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    let parsed =
        serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("response not JSON: {err}"));
    (status, headers, parsed)
}

/// Parse one complete SSE frame text into (event name, parsed JSON data);
/// `None` for comment/keep-alive frames without data (same shape as
/// `jobs.rs`).
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

/// The next complete SSE frame (through its blank line).
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

/// Read frames until one matches (event name + job id).
async fn frame_for(stream: &mut axum::body::BodyDataStream, event: &str, id: &str) -> Value {
    loop {
        let (name, data) = next_frame(stream).await;
        if name == event && data["id"].as_str() == Some(id) {
            return data;
        }
    }
}

/// `GET /v1/events` carries the image-ops lifecycle: `job_registered` and
/// `job_finished` frames with `kind: "image_ops"` for the op job.
#[tokio::test]
async fn events_stream_carries_image_ops_lifecycle() {
    let app = app_with(None);
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
    let mut stream = sse.into_body().into_data_stream();

    let (status, headers, body) =
        image_ops(app, resize_request("events-1", 2, 2, &one_input(4, 4))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let job_id = headers
        .get("x-job-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();

    let registered = frame_for(&mut stream, "job_registered", &job_id).await;
    assert_eq!(registered["kind"], "image_ops", "{registered}");
    let finished = frame_for(&mut stream, "job_finished", &job_id).await;
    assert_eq!(finished["kind"], "image_ops", "{finished}");
}

/// The capabilities advertisement lists the closed op vocabulary (the
/// editor discovers what it can call — no probing).
#[tokio::test]
async fn capabilities_list_supported_ops() {
    let (_, _, doc) = send_get(app_with(None), "/v1/capabilities").await;
    assert_eq!(
        doc["image_ops"]["ops"],
        json!(["resize", "crop", "rotate", "grayscale", "format"]),
        "{doc}"
    );
}

/// Lowercase-hex helper (mirrors the server's digest spelling).
fn hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
