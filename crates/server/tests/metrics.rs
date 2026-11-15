//! `/metrics` exposition (NEW-9): expected families and lines are present
//! after a request.

// No lib target on this binary crate, so include the modules directly.
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;

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

use routes::{AppConfig, app};

fn test_app() -> Router {
    app(AppConfig {
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
        llm: Arc::new(StubLlm),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
    })
}

#[tokio::test]
async fn metrics_lines_present_after_request() {
    let app = test_app();

    // One counted request first.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
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
    assert!(content_type.starts_with("text/plain"), "{content_type}");

    let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("# HELP honkbox_build_info"));
    assert!(body.contains("honkbox_build_info{version=\""));
    assert!(body.contains("# TYPE honkbox_requests_total counter"));
    assert!(
        body.contains("honkbox_requests_total{route=\"/v1/models\",code=\"200\"}"),
        "requests line missing after a 200 on /v1/models: {body}"
    );
    assert!(body.contains("# TYPE honkbox_request_duration_seconds histogram"));
    assert!(body.contains("honkbox_request_duration_seconds_bucket{le=\"+Inf\"}"));
    assert!(body.contains("honkbox_request_duration_seconds_count"));
    assert!(body.contains("honkbox_request_duration_seconds_sum "));
    assert!(body.contains("# TYPE honkbox_gpu_slot_busy_rejections_total counter"));
    assert!(body.contains("honkbox_gpu_slot_busy_rejections_total "));
    assert!(body.ends_with('\n'));
}
