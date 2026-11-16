//! `/health` contract: 200 + `{"status":"ok"}`.

// No lib target on this binary crate, so include the routes module directly.
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;
// Included only so `routes.rs`'s `crate::swap` paths resolve in this test
// binary; the swap module itself is not exercised here.
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt as _;

use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use routes::{AppConfig, app};
use std::sync::{Arc, RwLock};

/// Release-shaped app (embedded bundle; `web-dist/` is empty at init).
#[tokio::test]
async fn health_returns_ok() {
    let response = app(AppConfig {
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
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        swap: None,
    })
    .oneshot(
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    assert_eq!(body.as_ref(), br#"{"status":"ok"}"#);
}
