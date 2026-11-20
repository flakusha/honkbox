//! `POST /v1/jobs/blender` (FEAT-blender-job-route): the typed route's
//! contract. The request shape is checked BEFORE the store, an unstored
//! mesh is a 404, and a store-backed request without an installed Blender
//! is a named 501 that leaves NO job in the registry (a rejected request
//! never enters it).
//!
//! The end-to-end bake needs a real Blender; CI has no engine, so what CI
//! proves is the gate: the route refuses honestly instead of pretending.

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
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// One test-scoped data dir (unique per test name + pid; pre-cleaned).
fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-blender-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test data dir");
    dir
}

/// The app under test: no GPU slot, no engine, a per-test data dir.
fn app_with(data_dir: Option<PathBuf>) -> Router {
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
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: data_dir.clone(),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        artifacts: Arc::new(artifacts::Registry::new(
            data_dir,
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
    })
}

/// The typed request for the mesh with the given digest.
fn job_request(sha: &str) -> Value {
    json!({
        "input": {"asset_id": sha, "format": "glb", "sha256": sha},
        "stages": [{"op": "export_glb"}],
        "params": {"seed": 7},
        "engine": {
            "blender_version": "4.5.15",
            "script_version": honkbox_external_tools::recipe_sha256(),
        },
        "output": {"format": "glb", "max_bytes": 1_048_576},
        "manifest": 1,
    })
}

/// POST the request; returns the status, the decoded body and the headers.
async fn post_job(app: Router, body: Value) -> (StatusCode, Value, HeaderMap) {
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/jobs/blender")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value, headers)
}

/// Upload one artifact; returns its sha256 (the artifact id).
async fn upload(app: Router, bytes: &[u8]) -> String {
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/artifacts")
                .method("POST")
                .header("content-type", "application/octet-stream")
                .header("x-artifact-content-type", "model/gltf-binary")
                .body(Body::from(bytes.to_vec()))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::CREATED, "artifact upload");
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let info: Value = serde_json::from_slice(&bytes).expect("artifact info");
    info["sha256"].as_str().expect("sha256").to_owned()
}

/// GET a JSON document from the app.
async fn get_json(app: Router, uri: &str) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK, "GET {uri}");
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json")
}

/// The request shape is checked BEFORE the store: an unsupported manifest
/// version is a named 400 even when the mesh is not stored.
#[tokio::test]
async fn the_request_shape_is_checked_before_the_store() {
    let app = app_with(Some(tmp_dir("shape")));
    let mut request = job_request(&"a".repeat(64));
    request["manifest"] = json!(2);
    let (status, body, headers) = post_job(app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "blender_manifest_version");
    assert!(
        headers.get("x-job-id").is_none(),
        "refused before the job: the request never reached the slot"
    );
}

/// A mesh that is not in the store is a 404, not a 400.
#[tokio::test]
async fn an_unstored_mesh_is_a_404() {
    let app = app_with(Some(tmp_dir("missing")));
    let (status, body, headers) = post_job(app, job_request(&"a".repeat(64))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "blender_input_not_found");
    assert!(
        headers.get("x-job-id").is_none(),
        "refused before the job: the request never reached the slot"
    );
}

/// The artifact id IS the content digest: a caller naming another digest
/// asks for bytes the store does not hold.
#[tokio::test]
async fn a_digest_mismatch_is_a_400() {
    let app = app_with(Some(tmp_dir("digest")));
    let mut request = job_request(&"a".repeat(64));
    request["input"]["sha256"] = json!("c".repeat(64));
    let (status, body, headers) = post_job(app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "blender_input_sha256_mismatch");
    assert!(
        headers.get("x-job-id").is_none(),
        "refused before the job: the request never reached the slot"
    );
}

/// Without an installed Blender the route answers a named 501 and
/// registers NO job: a refused request never enters the registry.
#[tokio::test]
async fn without_an_engine_the_route_is_a_named_501() {
    let app = app_with(Some(tmp_dir("engine")));
    let sha = upload(app.clone(), b"glTF fake").await;
    let (status, body, headers) = post_job(app.clone(), job_request(&sha)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["code"], "blender_unavailable");
    assert!(
        headers.get("x-job-id").is_none(),
        "refused before the job: the request never reached the slot"
    );

    let jobs = get_json(app, "/v1/jobs").await;
    assert_eq!(jobs["data"].as_array().map(Vec::len), Some(0));
}

/// The `OpenAPI` document serves the route.
#[tokio::test]
async fn the_openapi_document_serves_the_route() {
    let app = app_with(Some(tmp_dir("openapi")));
    let doc = get_json(app, "/v1/openapi.json").await;
    assert!(
        doc["paths"]["/v1/jobs/blender"].is_object(),
        "the path is served"
    );
}

/// The capability document agrees with the route: the flag is `false`
/// without an engine, and the blender queue row is CALLABLE (the route
/// exists), so it is no longer reserved capacity.
#[tokio::test]
async fn the_capability_document_agrees_with_the_route() {
    let app = app_with(Some(tmp_dir("caps")));
    let doc = get_json(app, "/v1/capabilities").await;
    assert_eq!(doc["features"]["blender"], false);
    let kinds = doc["queue"]["kinds"].as_array().expect("queue kinds");
    let blender = kinds
        .iter()
        .find(|row| row["kind"] == "blender")
        .expect("the blender row");
    assert_eq!(blender["reserved"], false);
}
