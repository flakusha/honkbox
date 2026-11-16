//! `/health` contract: 200 + `"{"status":"ok"}` JSON by default, a
//! literal `OK` body for text/plain-preferring probes (REQ-2).

// No lib target on this binary crate, so include the routes module directly.
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
#[path = "../src/jobs.rs"]
mod jobs;
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

fn test_app() -> axum::Router {
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
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    })
}

/// GET `/health` with the given `Accept` value (`None` = no header);
/// returns (status, content-type, body).
async fn health_with_accept(accept: Option<&str>) -> (StatusCode, String, String) {
    let mut builder = Request::builder().uri("/health");
    if let Some(accept) = accept {
        builder = builder.header("accept", accept);
    }
    let response = test_app()
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    (
        status,
        content_type,
        String::from_utf8_lossy(&body).into_owned(),
    )
}

/// Release-shaped default: JSON `Health` (no Accept header at all).
#[tokio::test]
async fn health_returns_ok() {
    let (status, content_type, body) = health_with_accept(None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    assert_eq!(body, r#"{"status":"ok"}"#);
}

/// Explicit JSON preference keeps the JSON body.
#[tokio::test]
async fn health_json_accept_keeps_json() {
    let (status, content_type, body) = health_with_accept(Some("application/json")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    assert_eq!(body, r#"{"status":"ok"}"#);

    // Wildcards and unrelated types never flip the default.
    for accept in ["*/*", "text/*", "text/html", "application/xml, text/html"] {
        let (_, content_type, body) = health_with_accept(Some(accept)).await;
        assert!(content_type.starts_with("application/json"), "{accept}");
        assert_eq!(body, r#"{"status":"ok"}"#);
    }
}

/// A text/plain preference gets the literal `OK` body (uptime probes).
#[tokio::test]
async fn health_text_plain_prefers_ok_body() {
    for accept in [
        "text/plain",
        "text/plain;charset=utf-8",
        "text/html;q=0.1, text/plain;q=0.9",
    ] {
        let (status, content_type, body) = health_with_accept(Some(accept)).await;
        assert_eq!(status, StatusCode::OK, "{accept}");
        assert!(
            content_type.starts_with("text/plain"),
            "{accept}: {content_type}"
        );
        assert_eq!(body, "OK", "{accept}");
    }

    // JSON outranks plain when its q-value wins.
    let (_, content_type, body) =
        health_with_accept(Some("text/plain;q=0.5, application/json;q=1.0")).await;
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    assert_eq!(body, r#"{"status":"ok"}"#);
}
