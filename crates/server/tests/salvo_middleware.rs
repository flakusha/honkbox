//! Salvo middleware parity: bearer 401/200, probe and signed-content
//! exemptions, the 1 KiB compression floor, and SSE staying uncompressed.
//!
//! No lib target on this binary crate, so the modules under test are included
//! directly. Unique env-var names per test keep the process-global override
//! map parallel-safe.

// `#[path]` includes pull in whole source modules; untested items and
// salvo's generated `#[handler]` futures are dead/async-free in this binary.
#![allow(
    dead_code,
    clippy::unused_async_trait_impl,
    clippy::redundant_closure_for_method_calls
)]

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
#[path = "../src/salvo_app.rs"]
mod salvo_app;
#[allow(dead_code)]
#[path = "../src/salvo_middleware.rs"]
mod salvo_middleware;
#[allow(dead_code)]
#[path = "../src/salvo_routes/mod.rs"]
mod salvo_routes;
#[allow(dead_code)]
#[path = "../src/salvo_support.rs"]
mod salvo_support;
#[allow(dead_code)]
#[path = "../src/salvo_version.rs"]
mod salvo_version;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;

#[path = "support/listener.rs"]
mod listener;

use std::sync::{Arc, RwLock};

use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use listener::TestServer;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE};
use salvo::http::{HeaderValue, ResBody, StatusCode};
use salvo::prelude::*;
use serde_json::{Value, json};

use routes::{AppConfig, AuthConfig, AuthMode, set_env_var};

fn config(key_env: &str) -> Arc<AppConfig> {
    Arc::new(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig {
            mode: AuthMode::Bearer,
            api_key_env: key_env.to_owned(),
            jwt: crate::config::JwtConfig::default(),
        },
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
        artifacts: Arc::new(artifacts::Registry::new(
            None,
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

#[handler]
async fn small_json(res: &mut Response) {
    res.render(Json(json!({"ok": true})));
}

#[handler]
async fn large_json(res: &mut Response) {
    res.render(Json(json!({"data": "x".repeat(2048)})));
}

#[handler]
async fn sse(res: &mut Response) {
    res.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    res.body = ResBody::from(format!("data: {}\n\n", "x".repeat(2048)).into_bytes());
}

#[handler]
async fn probe(res: &mut Response) {
    res.render(Text::Plain("ok"));
}

#[handler]
async fn content(_req: &mut Request, res: &mut Response) {
    res.render(Text::Plain("signed"));
}

#[handler]
async fn ui(res: &mut Response) {
    res.render(Text::Plain("ui"));
}

/// API routes get the hoops; `/ui` is outside them, like axum's fallback.
fn router(key_env: &str) -> Router {
    let api = Router::new()
        .push(Router::with_path("health").get(probe))
        .push(Router::with_path("metrics").get(probe))
        .push(Router::with_path("v1/ping").get(small_json))
        .push(Router::with_path("v1/big").get(large_json))
        .push(Router::with_path("v1/events").get(sse))
        .push(Router::with_path("v1/jobs/{id}/content/{name}").get(content));
    let api = salvo_middleware::api_hoops(config(key_env))
        .into_iter()
        .fold(api, |router, hoop| router.hoop(hoop));
    Router::new()
        .push(api)
        .push(Router::with_path("ui").get(ui))
}

#[tokio::test]
async fn bearer_mode_401_without_key_200_with_env_key() {
    let key_env = "HONKBOX_SALVO_MW_BEARER_KEY";
    set_env_var(key_env, Some("right-key"));
    let server = TestServer::start(router(key_env)).await;

    let rejected = server.get("/v1/ping").await;
    assert_eq!(rejected.status, StatusCode::UNAUTHORIZED);
    let body: Value = rejected.json().unwrap();
    assert_eq!(body["error"]["type"], "authentication_error");
    assert_eq!(body["error"]["code"], "invalid_api_key");
    assert_eq!(body["error"]["message"], "Invalid API key provided.");

    let allowed = server
        .request(
            TestServer::client()
                .get(server.url("/v1/ping"))
                .header("authorization", "Bearer right-key"),
        )
        .await;
    assert_eq!(allowed.status, StatusCode::OK);
    set_env_var(key_env, None);
    server.shutdown().await;
}

#[tokio::test]
async fn health_and_metrics_probes_are_exempt() {
    let key_env = "HONKBOX_SALVO_MW_PROBE_KEY";
    set_env_var(key_env, Some("probe-key"));
    let server = TestServer::start(router(key_env)).await;

    for path in ["/health", "/metrics"] {
        let response = server.get(path).await;
        assert_eq!(response.status, StatusCode::OK, "probe {path}");
    }
    assert_eq!(server.get("/ui").await.status, StatusCode::OK);
    set_env_var(key_env, None);
    server.shutdown().await;
}

#[tokio::test]
async fn signed_content_query_bypasses_bearer() {
    let key_env = "HONKBOX_SALVO_MW_SIGNED_KEY";
    set_env_var(key_env, Some("signed-key"));
    let server = TestServer::start(router(key_env)).await;

    let signed = server
        .get("/v1/jobs/01J/content/out.png?expires=1&sig=deadbeef")
        .await;
    assert_eq!(signed.status, StatusCode::OK);
    // Substring junk is not a signature: the query-pair parser must reject it.
    let forged = server
        .get("/v1/jobs/01J/content/out.png?junk=expires=0&junk2=sig=0")
        .await;
    assert_eq!(forged.status, StatusCode::UNAUTHORIZED);
    set_env_var(key_env, None);
    server.shutdown().await;
}

#[tokio::test]
async fn compression_floor_small_plain_large_compressed() {
    let key_env = "HONKBOX_SALVO_MW_COMPRESSION_KEY";
    set_env_var(key_env, Some("compression-key"));
    let server = TestServer::start(router(key_env)).await;
    let accept = |path: &str| {
        TestServer::client()
            .get(server.url(path))
            .header(ACCEPT_ENCODING, "gzip")
            .header("authorization", "Bearer compression-key")
    };

    let small = server.request(accept("/v1/ping")).await;
    assert_eq!(small.status, StatusCode::OK);
    assert!(!small.headers.contains_key(CONTENT_ENCODING));

    let large = server.request(accept("/v1/big")).await;
    assert_eq!(large.status, StatusCode::OK);
    assert_eq!(
        large
            .headers
            .get(CONTENT_ENCODING)
            .and_then(|v| v.to_str().ok()),
        Some("gzip")
    );
    set_env_var(key_env, None);
    server.shutdown().await;
}

#[tokio::test]
async fn sse_response_has_no_content_encoding() {
    let key_env = "HONKBOX_SALVO_MW_SSE_KEY";
    set_env_var(key_env, Some("sse-key"));
    let server = TestServer::start(router(key_env)).await;

    let response = server
        .request(
            TestServer::client()
                .get(server.url("/v1/events"))
                .header(ACCEPT_ENCODING, "gzip")
                .header("authorization", "Bearer sse-key"),
        )
        .await;
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(
        response
            .headers
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    assert!(!response.headers.contains_key(CONTENT_ENCODING));
    set_env_var(key_env, None);
    server.shutdown().await;
}

/// The assembled salvo router serves the cutover surface with the same hoops
/// as the old axum stack: probes stay open, API routes need the bearer key.
#[tokio::test]
async fn assembled_salvo_router_serves_cutover_surface() {
    let key_env = "HONKBOX_SALVO_CUTOVER_KEY";
    set_env_var(key_env, Some("cutover-key"));
    let cfg = config(key_env);
    let server = TestServer::start(salvo_app::router(&cfg)).await;

    let health = server.get("/health").await;
    assert_eq!(health.status, StatusCode::OK);
    assert_eq!(health.json::<Value>().unwrap()["status"], "ok");

    let unauth = server.get("/v1/queue").await;
    assert_eq!(unauth.status, StatusCode::UNAUTHORIZED);

    let version = server
        .request(
            TestServer::client()
                .get(server.url("/api/version"))
                .bearer_auth("cutover-key"),
        )
        .await;
    assert_eq!(version.status, StatusCode::OK);
    assert_eq!(version.json::<Value>().unwrap()["name"], "honkbox");

    let queue = server
        .request(
            TestServer::client()
                .get(server.url("/v1/queue"))
                .bearer_auth("cutover-key"),
        )
        .await;
    assert_eq!(queue.status, StatusCode::OK);
    assert_eq!(queue.json::<Value>().unwrap()["queued"], 0);

    server.shutdown().await;
    set_env_var(key_env, None);
}

/// The salvo front serves the `/comfyui/start` alias with the same hoops:
/// bearer gate, then the honest 501 when no manager is installed.
#[tokio::test]
async fn salvo_comfyui_start_alias_matches_axum_branches() {
    let key_env = "HONKBOX_SALVO_COMFYUI_START_KEY";
    set_env_var(key_env, Some("comfyui-start-key"));
    let cfg = config(key_env);
    let server = TestServer::start(salvo_app::router(&cfg)).await;

    let denied = server
        .request(TestServer::client().post(server.url("/comfyui/start")))
        .await;
    assert_eq!(denied.status, StatusCode::UNAUTHORIZED);

    let down = server
        .request(
            TestServer::client()
                .post(server.url("/comfyui/start"))
                .bearer_auth("comfyui-start-key"),
        )
        .await;
    assert_eq!(down.status, StatusCode::NOT_IMPLEMENTED);
    assert!(down.text().contains("comfyui unavailable"));

    server.shutdown().await;
    set_env_var(key_env, None);
}
