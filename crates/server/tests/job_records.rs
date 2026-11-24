//! Read-only generation record API contract: sidecar fields, newest-first
//! paging, reject-not-clamp limits, traversal-safe ids, and no state changes.

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

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::{Router, body::Body, http::Request, http::StatusCode};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

const ID_A: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
const ID_B: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAW";
const ID_C: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAX";
const ID_D: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAY";

fn tmp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("honkbox-job-records-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test data dir");
    dir
}

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
            ocr_dirs: Vec::new(),
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
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
    })
}

fn bundle(root: &Path, id: &str, prompt: &str, seed: u64, with_hash: bool) {
    let dir = root.join("jobs").join(id);
    std::fs::create_dir_all(&dir).expect("bundle dir");
    let sidecar = json!({
        "prompt": prompt,
        "seed": seed,
        "model_slug": "comfy:checkpoints/test.safetensors",
        "model_sha256": if with_hash { Some("abc123") } else { None },
        "engine_version": if with_hash { Some("sdcpp-1.2") } else { None },
        "inputs_sha256": format!("inputs-{seed}"),
    });
    std::fs::write(dir.join("final_00.json"), sidecar.to_string()).expect("sidecar");
}

async fn get(app: Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let response = app
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, body.to_vec())
}

fn json_body(body: &[u8]) -> Value {
    serde_json::from_slice(body).expect("json response")
}

fn error_code(body: &[u8]) -> String {
    json_body(body)["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

#[tokio::test]
async fn single_record_reads_sidecar_without_mutating_it() {
    let root = tmp_dir("single");
    bundle(&root, ID_A, "a cat", 42, true);
    let sidecar = root.join("jobs").join(ID_A).join("final_00.json");
    let before = std::fs::read(&sidecar).expect("sidecar bytes");
    let app = app_with(Some(root.clone()));

    let (status, body) = get(app.clone(), &format!("/v1/jobs/records/{ID_A}")).await;
    assert_eq!(status, StatusCode::OK);
    let record = json_body(&body);
    assert_eq!(record["id"], ID_A);
    assert_eq!(record["prompt"], "a cat");
    assert_eq!(record["seed"], 42);
    assert_eq!(record["model_slug"], "comfy:checkpoints/test.safetensors");
    assert_eq!(record["model_sha256"], "abc123");
    assert_eq!(record["engine_version"], "sdcpp-1.2");
    assert_eq!(record["inputs_sha256"], "inputs-42");
    assert_eq!(std::fs::read(&sidecar).expect("sidecar bytes"), before);

    let text = String::from_utf8_lossy(&body);
    assert!(!text.contains(root.to_string_lossy().as_ref()));

    let (status, body) = get(app.clone(), "/v1/jobs/records/01ARZ3NDEKTSV4RRFFQ69G5FAZ").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "generation_record_not_found");

    let (status, body) = get(app.clone(), "/v1/jobs/records/not-a-ulid").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_job_id");

    // Separators and traversal never reach the filesystem: `validate_id`
    // rejects any id that is not a bare 26-character ULID.
}

#[tokio::test]
async fn list_pages_newest_first_with_bounded_limits() {
    let root = tmp_dir("list");
    bundle(&root, ID_A, "first", 1, false);
    bundle(&root, ID_B, "second", 2, false);
    bundle(&root, ID_C, "third", 3, false);
    let app = app_with(Some(root));

    let (status, body) = get(app.clone(), "/v1/jobs/records").await;
    assert_eq!(status, StatusCode::OK);
    let page = json_body(&body);
    assert_eq!(page["object"], "list");
    assert_eq!(page["has_more"], false);
    assert_eq!(page["data"][0]["id"], ID_C);
    assert_eq!(page["data"][1]["id"], ID_B);
    assert_eq!(page["data"][2]["id"], ID_A);
    assert!(page["data"][0]["model_sha256"].is_null());
    assert!(page["data"][0]["engine_version"].is_null());

    let (_, body) = get(app.clone(), "/v1/jobs/records?limit=1").await;
    let page = json_body(&body);
    assert_eq!(page["data"].as_array().map_or(0, Vec::len), 1);
    assert_eq!(page["data"][0]["id"], ID_C);
    assert_eq!(page["has_more"], true);

    let (_, body) = get(app.clone(), "/v1/jobs/records?limit=2&offset=1").await;
    let page = json_body(&body);
    assert_eq!(page["data"][0]["id"], ID_B);
    assert_eq!(page["data"][1]["id"], ID_A);
    assert_eq!(page["has_more"], false);

    for uri in ["/v1/jobs/records?limit=0", "/v1/jobs/records?limit=101"] {
        let (status, body) = get(app.clone(), uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        assert_eq!(error_code(&body), "invalid_limit");
    }
}

#[tokio::test]
async fn malformed_and_foreign_dirs_are_skipped() {
    let root = tmp_dir("skip");
    bundle(&root, ID_A, "kept", 7, true);
    let dir = root.join("jobs");
    std::fs::create_dir_all(dir.join("not-a-ulid")).expect("foreign dir");
    std::fs::write(dir.join("not-a-ulid/final_00.json"), "{}").expect("foreign sidecar");
    std::fs::create_dir_all(dir.join(ID_D)).expect("partial bundle");
    std::fs::write(dir.join(ID_D).join("final_00.json"), "not json").expect("broken sidecar");
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.join(ID_A), dir.join("01ARZ3NDEKTSV4RRFFQ69G5FAZ"))
        .expect("symlink bundle");
    let app = app_with(Some(root));

    let (status, body) = get(app, "/v1/jobs/records").await;
    assert_eq!(status, StatusCode::OK);
    let page = json_body(&body);
    assert_eq!(page["data"].as_array().map_or(0, Vec::len), 1);
    assert_eq!(page["data"][0]["id"], ID_A);
}

#[tokio::test]
async fn no_data_dir_degrades_honestly() {
    let app = app_with(None);
    let (status, body) = get(app.clone(), "/v1/jobs/records").await;
    assert_eq!(status, StatusCode::OK);
    let page = json_body(&body);
    assert_eq!(page["data"].as_array().map_or(0, Vec::len), 0);
    assert_eq!(page["has_more"], false);

    let (status, body) = get(app, &format!("/v1/jobs/records/{ID_A}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "generation_record_not_found");
}
