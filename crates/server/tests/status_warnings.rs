//! `GET /v1/status/warnings`: bounded, stable config-warning document.

// No lib target on this binary crate, so include the modules directly.
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
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;
// Included only so `routes.rs`'s `crate::swap` paths resolve in this test
// binary; the swap module itself is not exercised here.
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use std::sync::{Arc, RwLock};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

fn test_app() -> Router {
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
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
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

/// The warning store is process-wide: clear it even if an assert panics.
struct RestoreWarnings;

impl Drop for RestoreWarnings {
    fn drop(&mut self) {
        routes::status::set_warnings_for_tests(Vec::new());
    }
}

/// Resource contract: this test owns the process-wide warning store inside
/// this test binary; `RestoreWarnings` clears it in `Drop`, including on panic.
/// No port, socket, or shared file is used.
#[tokio::test]
async fn status_warnings_document_is_stable_and_bounded() {
    let _restore = RestoreWarnings;
    let messages: Vec<String> = (0..70).map(|i| format!("warning {i}")).collect();
    routes::status::set_warnings_for_tests(messages.clone());

    let response = test_app()
        .oneshot(
            Request::builder()
                .uri("/v1/status/warnings")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .expect("content-type header")
        .to_owned();
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );

    let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let document: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
    assert_eq!(document["schema_version"], 1);
    let warnings = document["warnings"].as_array().expect("warnings array");
    assert!(warnings.len() <= 64, "warning list is unbounded");
    assert_eq!(warnings.len(), 64, "warning list must clamp to 64");
    for (row, message) in warnings.iter().zip(messages.iter().take(64)) {
        assert_eq!(row["code"].as_str(), Some("config"));
        assert_eq!(row["message"].as_str(), Some(message.as_str()));
        assert_eq!(row["source"].as_str(), Some("config"));
    }
}
