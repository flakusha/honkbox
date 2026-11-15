//! `/health` contract: 200 + `{"status":"ok"}`.

// No lib target on this binary crate, so include the routes module directly.
#[path = "../src/routes.rs"]
mod routes;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt as _;

use routes::{AppConfig, app};
use sd_rip_audio_worker::{StubStt, StubTts};
use sd_rip_llm_worker::StubLlm;
use std::sync::Arc;

/// Release-shaped app (embedded bundle; `web-dist/` is empty at init).
#[tokio::test]
async fn health_returns_ok() {
    let response = app(AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        llm: Arc::new(StubLlm),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
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
