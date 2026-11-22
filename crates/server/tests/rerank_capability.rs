//! `features.rerank` across the three Python/allowlist states.
//!
//! `runtime_config::FACTS` is a first-wins `OnceLock`, so each state must own
//! its process: the one test re-executes this binary with
//! `HONKBOX_RERANK_CONTRACT_STATE` set, and every child configures its own
//! facts before driving the real capabilities document and rerank route.

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

use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

const STATE_ENV: &str = "HONKBOX_RERANK_CONTRACT_STATE";
const TEST_NAME: &str = "rerank_capability_follows_python_and_allowlist";
const STATES: [&str; 3] = [
    "python-disabled",
    "python-enabled-no-slug",
    "python-enabled-with-slug",
];

fn app_with_no_catalog() -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
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
        llm: Arc::new(honkbox_llm_worker::StubLlm::default()),
        stt: Arc::new(honkbox_audio_worker::StubStt),
        tts: Arc::new(honkbox_audio_worker::StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        artifacts: Arc::new(crate::artifacts::Registry::new(
            None,
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// Publish exactly one facts snapshot for this child process. The allowlist
/// is the only config fact the rerank gate consults, so a minimal TOML file
/// keeps the three states independent of ambient environment knobs.
fn configure_for(state: &str) {
    let (enabled, allowlist) = match state {
        "python-disabled" => (false, "[]"),
        "python-enabled-no-slug" => (true, "[]"),
        "python-enabled-with-slug" => (true, "[\"ranker:stub\"]"),
        other => panic!("unknown rerank state {other}"),
    };
    let dir = std::env::temp_dir().join(format!(
        "honkbox-rerank-contract-{}-{}",
        std::process::id(),
        state
    ));
    std::fs::create_dir_all(&dir).expect("config dir");
    let path = dir.join("honkbox.toml");
    std::fs::write(
        &path,
        format!("[python]\nenabled = {enabled}\nmodel_allowlist = {allowlist}\n"),
    )
    .expect("write config");
    let loaded = config::load(Some(path), Some(dir.clone())).expect("load config");
    routes::runtime_config::configure(&loaded);
    // Facts are in memory now; the temp config dir must not leak per state.
    std::fs::remove_dir_all(&dir).ok();
}

async fn get_capabilities(app: &Router) -> Value {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("infallible");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("capabilities JSON")
}

async fn post_rerank(app: &Router) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"query": "q", "texts": ["alpha", "beta"]}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("infallible");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn run_state(state: &str) {
    configure_for(state);
    let app = app_with_no_catalog();
    let doc = get_capabilities(&app).await;

    match state {
        "python-disabled" => {
            assert_eq!(doc["features"]["python_sidecar"], false, "{doc}");
            assert_eq!(doc["features"]["rerank"], false, "{doc}");
        }
        "python-enabled-no-slug" => {
            assert_eq!(doc["features"]["python_sidecar"], true, "{doc}");
            assert_eq!(doc["features"]["rerank"], false, "{doc}");
            let (status, body) = post_rerank(&app).await;
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
            assert!(body.contains("no ranker model configured"), "{body}");
        }
        "python-enabled-with-slug" => {
            assert_eq!(doc["features"]["python_sidecar"], true, "{doc}");
            assert_eq!(doc["features"]["rerank"], true, "{doc}");
            let (status, body) = post_rerank(&app).await;
            // No sidecar venv exists in this harness, so a scoring result
            // cannot be produced here. Admission is proven by the absence
            // of the no-ranker 501: the request reached the sidecar seam.
            assert!(
                !body.contains("no ranker model configured"),
                "{status} {body}"
            );
            assert!(
                status.is_success() || body.contains("sidecar"),
                "{status} {body}"
            );
        }
        other => panic!("unknown rerank state {other}"),
    }
}

#[tokio::test]
async fn rerank_capability_follows_python_and_allowlist() {
    if let Ok(state) = std::env::var(STATE_ENV) {
        run_state(&state).await;
        return;
    }

    for state in STATES {
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .arg("--exact")
            .arg(TEST_NAME)
            .arg("--nocapture")
            .env(STATE_ENV, state)
            .output()
            .expect("spawn state child");
        assert!(
            output.status.success(),
            "rerank state {state} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
