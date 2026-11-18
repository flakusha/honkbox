//! Bearer auth (REQ-7): 200/401/401 across right/missing/wrong keys with
//! the OpenAI-shaped `authentication_error` envelope, keyless `/health` +
//! `/metrics` probes, the exempt UI/static fallback, `auth = none` being a
//! no-op, env rotation between requests (key read at request time), and
//! the key value never leaking into responses.

// No lib target on this binary crate, so include the modules directly.
// This binary exercises only the auth surface, so every included module
// carries `allow(dead_code)` for the unused rest.
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
#[allow(dead_code)]
#[path = "../src/artifacts.rs"]
mod artifacts;
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::sync::{Arc, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::Value;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, AuthMode, app, set_env_var};

/// Per-test env-var NAMEs: `std::env` is process-global and tests run in
/// parallel threads, so every test owns a unique name (values never
/// collide; the unsafe blocks are single-variable set/remove).
const KEY_ENV: &str = "HONKBOX_TEST_AUTH_KEY";
const EMPTY_ENV: &str = "HONKBOX_TEST_AUTH_EMPTY_KEY";
const ROTATION_ENV: &str = "HONKBOX_TEST_AUTH_ROTATION_KEY";

fn bearer_app(auth: AuthConfig) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition), auth,
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
    sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
    sd_model: None,
    swap: None,
    embed: Arc::new(swap::stub_embed_cache()),
    swap_max_ttl: std::time::Duration::from_hours(24), data_dir: None, artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(None.clone(), crate::config::DEFAULT_ARTIFACT_MAX_BYTES)), vram_budget: None,
    model_entries: Vec::new(),
    downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")), })
}

fn bearer_auth() -> AuthConfig {
    AuthConfig {
        mode: AuthMode::Bearer,
        api_key_env: KEY_ENV.to_owned(),
    }
}

/// GET `path` with an optional Authorization header; returns (status, body).
async fn get(app: Router, path: &str, authorization: Option<&str>) -> (StatusCode, String) {
    let mut builder = Request::builder().uri(path);
    if let Some(value) = authorization {
        builder = builder.header("authorization", value);
    }
    let response = app
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 256 * 1024).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// The OpenAI-shaped 401 envelope contract: `{error:{message,type:
/// "authentication_error",param:null,code:"invalid_api_key"}}`.
fn assert_auth_error_envelope(body: &str) {
    let value: Value = serde_json::from_str(body).expect("401 body is JSON");
    let error = value.get("error").expect("error envelope");
    assert_eq!(error["type"], "authentication_error");
    assert_eq!(error["code"], "invalid_api_key");
    assert_eq!(error["param"], Value::Null);
    assert!(error["message"].as_str().is_some_and(|m| !m.is_empty()));
}

#[tokio::test]
async fn bearer_right_key_passes_wrong_and_missing_get_401() {
    // Per-request env seam: production reads `std::env` directly (the
    // process-global env is never mutated); tests inject through
    // `set_env_var` because the workspace forbids `unsafe_code`.
    set_env_var(KEY_ENV, Some("sekrit-honk-key"));
    let app = bearer_app(bearer_auth());

    // Wrong key: 401 + envelope; the key VALUE never appears anywhere.
    let (status, body) = get(app.clone(), "/v1/models", Some("Bearer nope")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_auth_error_envelope(&body);
    assert!(
        !body.contains("sekrit-honk-key"),
        "key leaked in body: {body}"
    );

    // Missing header: same 401 + envelope.
    let (status, body) = get(app.clone(), "/v1/models", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_auth_error_envelope(&body);

    // Malformed scheme (no Bearer prefix): 401 too.
    let (status, _) = get(app.clone(), "/v1/models", Some("sekrit-honk-key")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Right key: 200.
    let (status, _) = get(app.clone(), "/v1/models", Some("Bearer sekrit-honk-key")).await;
    assert_eq!(status, StatusCode::OK);

    set_env_var(KEY_ENV, None);
}

#[tokio::test]
async fn health_and_metrics_stay_keyless() {
    // Fail-closed baseline: env unset, so ANY bearer request would 401 —
    // these two must still succeed.
    let app = bearer_app(bearer_auth());
    let (status, _) = get(app.clone(), "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get(app.clone(), "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn ui_fallback_stays_keyless() {
    // The static fallback is structurally exempt: route_layer middleware
    // never runs for unmatched (fallback) requests.
    let app = bearer_app(bearer_auth());
    let (status, _) = get(app, "/some/ui/page", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn auth_none_needs_no_key() {
    let app = bearer_app(AuthConfig::default());
    let (status, _) = get(app, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn env_rotation_applies_without_rebuild() {
    set_env_var(ROTATION_ENV, Some("key-one"));
    let app = bearer_app(AuthConfig {
        mode: AuthMode::Bearer,
        api_key_env: ROTATION_ENV.to_owned(),
    });
    let (status, _) = get(app.clone(), "/v1/models", Some("Bearer key-one")).await;
    assert_eq!(status, StatusCode::OK);

    // Rotate: same app instance, new env value — the old key stops working.
    set_env_var(ROTATION_ENV, Some("key-two"));
    let (status, body) = get(app.clone(), "/v1/models", Some("Bearer key-one")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_auth_error_envelope(&body);
    let (status, _) = get(app.clone(), "/v1/models", Some("Bearer key-two")).await;
    assert_eq!(status, StatusCode::OK);

    // Unset env fails closed.
    set_env_var(ROTATION_ENV, None);
    let (status, body) = get(app, "/v1/models", Some("Bearer key-two")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_auth_error_envelope(&body);
}

#[tokio::test]
async fn empty_bearer_header_value_is_rejected() {
    set_env_var(EMPTY_ENV, Some("real-key"));
    let app = bearer_app(AuthConfig {
        mode: AuthMode::Bearer,
        api_key_env: EMPTY_ENV.to_owned(),
    });
    for header in ["Bearer ", "Bearer", "Bearer   "] {
        let (status, _) = get(app.clone(), "/v1/models", Some(header)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "header {header:?}");
    }
    set_env_var(EMPTY_ENV, None);
}
