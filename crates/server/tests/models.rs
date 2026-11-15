//! `/v1/models` catalog + per-request `model` validation (P0 of the
//! llama-swap parity plan).

// No lib target on this binary crate, so include the routes module directly.
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::json;
use std::sync::{Arc, LazyLock, RwLock};
use tower::ServiceExt as _;

use routes::{AppConfig, app};

/// Serializes GPU-slot-reserving tests within this binary (see `chat`).
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One fake LLM entry per slug; enough shape for the validation checks.
fn catalog_with(slugs: &[&str]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: slugs
            .iter()
            .map(|slug| ModelCatalogEntry {
                id: (*slug).to_owned(),
                kind: ModelKind::Llm,
                category: "root".to_owned(),
                path: format!("/models/{slug}"),
                bytes: 1,
                metadata: None,
            })
            .collect(),
    }
}

fn test_app_with_roots(catalog: ModelCatalog, roots: honkbox_registry::ModelRoots) -> Router {
    app(AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
        roots,
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(StubLlm),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
    })
}

fn test_app(catalog: ModelCatalog, llm_model: Option<String>) -> Router {
    app(AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
        },
        llm_model,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(StubLlm),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
    })
}

/// `POST /v1/chat` with a JSON body carrying `model`; returns status + body.
async fn chat(app: Router, model: &str) -> (StatusCode, String) {
    // The GPU slot is process-global; cargo runs this binary's tests in
    // parallel threads, so serialize every slot-reserving request here or
    // one chat test's reserve would 429 another's (resource contract: this
    // mutex owns the slot for the whole models.rs test binary). Async-aware
    // Mutex: the guard is intentionally held across `.await`.
    let _slot = SLOT_LOCK.lock().await;
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn models_list_shape() {
    let response = test_app(catalog_with(&["llm:a.gguf"]), None)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(
        body.as_ref(),
        br#"{"object":"list","data":[{"id":"llm:a.gguf","kind":"llm","category":"root","path":"/models/llm:a.gguf","bytes":1}]}"#
    );
}

#[tokio::test]
async fn unknown_model_is_400() {
    let (status, body) = chat(
        test_app(catalog_with(&["llm:a.gguf"]), None),
        "llm:nope.gguf",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("unknown model"), "body: {body}");
}

#[tokio::test]
async fn catalog_slug_and_empty_model_are_ok() {
    // A catalog slug passes (the startup pipeline still serves it; P1 swaps).
    let (status, _) = chat(test_app(catalog_with(&["llm:a.gguf"]), None), "llm:a.gguf").await;
    assert_eq!(status, StatusCode::OK);
    // Empty model keeps the startup-configured model, as before P0.
    let (status, _) = chat(test_app(catalog_with(&["llm:a.gguf"]), None), "").await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn configured_model_accepted_without_catalog() {
    let (status, _) = chat(
        test_app(catalog_with(&[]), Some("llm:mine.gguf".to_owned())),
        "llm:mine.gguf",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // No catalog entries: only the configured slug is accepted.
    let (status, body) = chat(
        test_app(catalog_with(&[]), Some("llm:mine.gguf".to_owned())),
        "llm:other.gguf",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("unknown model"), "body: {body}");
}

#[tokio::test]
async fn transcribe_validates_model_first() {
    // No content-type: model validation runs before the audio dispatch, so
    // an unknown slug is 400 (not 415).
    let response = test_app(catalog_with(&["llm:a.gguf"]), None)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/transcribe?model=llm:nope.gguf")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("unknown model"));
}

#[tokio::test]
async fn refresh_rescans_roots_and_updates_snapshot() {
    let base = std::env::temp_dir().join(format!("honkbox-refresh-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let llm = base.join("llm");
    std::fs::create_dir_all(&llm).expect("create test llm root");
    let app = test_app_with_roots(
        catalog_with(&[]),
        honkbox_registry::ModelRoots {
            llm_dirs: vec![llm.clone()],
            comfyui_dirs: Vec::new(),
        },
    );

    // Empty before: nothing on disk yet.
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
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(body.as_ref(), br#"{"object":"list","data":[]}"#);

    // A model file appears under the configured root.
    std::fs::write(llm.join("new-model.gguf"), b"x").expect("write model file");

    // Refresh re-scans and returns the fresh body.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/refresh")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("llm:new-model.gguf"), "refresh body: {body}");

    // GET now serves the refreshed snapshot, and the slug validates.
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
    let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("llm:new-model.gguf"));
    let (status, _) = chat(app, "llm:new-model.gguf").await;
    assert_eq!(status, StatusCode::OK);

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn refresh_is_idempotent_on_missing_roots() {
    // A root that does not exist refreshes to an empty catalog, never an error.
    let app = test_app_with_roots(
        catalog_with(&["llm:stale.gguf"]),
        honkbox_registry::ModelRoots {
            llm_dirs: vec![
                std::env::temp_dir()
                    .join(format!("honkbox-refresh-missing-{}", std::process::id())),
            ],
            comfyui_dirs: Vec::new(),
        },
    );
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/models/refresh")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(body.as_ref(), br#"{"object":"list","data":[]}"#);
    }
}
