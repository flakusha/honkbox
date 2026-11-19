//! `/v1/models` catalog + per-request `model` validation (P0 of the
//! llama-swap parity plan).

// No lib target on this binary crate, so include the routes module directly.
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
use honkbox_api_types::{ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::json;
use std::sync::{Arc, LazyLock, RwLock};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes GPU-slot-reserving tests within this binary (see `chat`).
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One fake LLM entry per slug; enough shape for the validation checks.
fn catalog_with(slugs: &[&str]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: slugs.iter().map(|slug| entry_with(slug, slug)).collect(),
    }
}

/// One fake LLM entry: served `id` (alias or slug) plus its registry slug.
#[allow(clippy::too_many_arguments)]
fn entry_with(id: &str, slug: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: slug.to_owned(),
        kind: ModelKind::Llm,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities: honkbox_api_types::ModelCapabilities::for_kind(ModelKind::Llm),
        category: "root".to_owned(),
        path: format!("/models/{slug}"),
        bytes: 1,
        metadata: None,
    }
}

fn test_app_with_roots(catalog: ModelCatalog, roots: honkbox_registry::ModelRoots) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
        roots,
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
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            None.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    })
}

fn test_app(catalog: ModelCatalog, llm_model: Option<String>) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            None.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
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
        br#"{"object":"list","data":[{"id":"llm:a.gguf","slug":"llm:a.gguf","kind":"llm","owned_by":"local","context_length":null,"max_output":null,"thinking":false,"tool_calling":false,"modalities":["text"],"capabilities":{"chat":true,"completion":true,"tool_calling":false,"vision":false,"embeddings":false},"category":"root","path":"/models/llm:a.gguf","bytes":1}]}"#
    );
}

/// REQ-2: an alias (`id` differing from the slug) and the raw registry slug
/// both dispatch; an unknown value is still the native 400.
#[tokio::test]
async fn alias_and_slug_both_resolve() {
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: vec![entry_with(
            "qwen3-8b",
            "llm:models--org--M/snapshots/sha/Q8_0.gguf",
        )],
    };
    let (status, _) = chat(test_app(catalog.clone(), None), "qwen3-8b").await;
    assert_eq!(status, StatusCode::OK);
    // The raw slug keeps working alongside the alias.
    let (status, _) = chat(
        test_app(catalog, None),
        "llm:models--org--M/snapshots/sha/Q8_0.gguf",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
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
    // The model gate runs BEFORE the route registers a job or takes the GPU
    // slot: a gated 400 carries no `X-Job-Id`, so the request never reached
    // the ledger and needs no slot lock (the GPU-slot audit requires that
    // pin).
    assert!(
        response.headers().get("x-job-id").is_none(),
        "a gated 400 must not reach the GPU slot"
    );
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("unknown model"));
}

#[tokio::test]
async fn refresh_rescans_roots_and_updates_snapshot() {
    // Keep the test's temp-file models out of the real/default cache DB.
    honkbox_registry::db::set_db_override(None);
    let base = std::env::temp_dir().join(format!("honkbox-refresh-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let llm = base.join("llm");
    std::fs::create_dir_all(&llm).expect("create test llm root");
    let app = test_app_with_roots(
        catalog_with(&[]),
        honkbox_registry::ModelRoots {
            llm_dirs: vec![llm.clone()],
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
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

/// Audio roots ride the same scan + refresh machinery: STT/TTS files under the
/// configured roots surface as `stt:`/`tts:` catalog entries with `stt`/`tts`
/// kinds after `POST /v1/models/refresh`.
#[tokio::test]
async fn refresh_picks_up_audio_roots() {
    // Keep the test's temp-file models out of the real/default cache DB.
    honkbox_registry::db::set_db_override(None);
    let base = std::env::temp_dir().join(format!("honkbox-audio-refresh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let stt = base.join("stt");
    let tts = base.join("tts");
    std::fs::create_dir_all(&stt).expect("create test stt root");
    std::fs::create_dir_all(&tts).expect("create test tts root");
    let app = test_app_with_roots(
        catalog_with(&[]),
        honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: vec![stt.clone()],
            tts_dirs: vec![tts.clone()],
        },
    );

    // Audio weights appear under the configured roots.
    std::fs::write(stt.join("whisper-base.bin"), b"x").expect("write stt model");
    std::fs::write(tts.join("kokoro.safetensors"), b"yy").expect("write tts model");

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
    assert!(
        body.contains("stt:whisper-base.bin"),
        "refresh body: {body}"
    );
    assert!(
        body.contains("tts:kokoro.safetensors"),
        "refresh body: {body}"
    );
    assert!(body.contains("\"kind\":\"stt\""), "refresh body: {body}");
    assert!(body.contains("\"kind\":\"tts\""), "refresh body: {body}");

    // GET serves the refreshed snapshot with the audio entries.
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("stt:whisper-base.bin"));
    assert!(body.contains("tts:kokoro.safetensors"));

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test]
async fn refresh_is_idempotent_on_missing_roots() {
    // Missing roots must refresh to an EMPTY catalog: no cache fallback and
    // no writes to the real/default cache DB.
    honkbox_registry::db::set_db_override(None);
    // A root that does not exist refreshes to an empty catalog, never an error.
    let app = test_app_with_roots(
        catalog_with(&["llm:stale.gguf"]),
        honkbox_registry::ModelRoots {
            llm_dirs: vec![
                std::env::temp_dir()
                    .join(format!("honkbox-refresh-missing-{}", std::process::id())),
            ],
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
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

/// A delete-test app: empty roots (the file being deleted is synthetic),
/// optional swap manager for the residency guard.
fn delete_app(catalog: ModelCatalog, swap: Option<Arc<swap::SwapManager>>) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
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
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            None.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// `DELETE /v1/models/{slug}` against `app`; returns status + body text.
async fn delete_model(app: Router, slug: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/models/{slug}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 16_384).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// One resident-only swap manager (seeded with a stub pipeline, loader
/// never invoked) for the 409 guardrail test.
fn seeded_swap(slug: &str) -> Arc<swap::SwapManager> {
    let manager = swap::SwapManager::new(
        swap::SwapSettings {
            llm_ttl: std::time::Duration::from_secs(60),
            llm_max_loaded: 1,
            max_ttl: std::time::Duration::from_secs(3600),
            pinned: std::collections::HashSet::new(),
        },
        Arc::new(
            |_: &str| -> Result<Arc<dyn honkbox_llm_worker::LlmPipeline>, swap::SwapError> {
                Err(swap::SwapError::Load("loader unused in test".to_owned()))
            },
        ),
    );
    manager.seed(slug, Arc::new(StubLlm::default()));
    Arc::new(manager)
}

/// Deletion success: the weight file is gone from disk, the response names
/// what was removed, and the catalog no longer carries the entry
/// (permanent — no trash, no undo).
#[tokio::test]
async fn delete_removes_file_and_catalog_entry() {
    let dir = std::env::temp_dir().join(format!("honkbox-delete-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let file = dir.join("gone.gguf");
    std::fs::write(&file, b"weights").expect("write weight file");

    let mut catalog = catalog_with(&["llm:gone.gguf", "llm:keep.gguf"]);
    catalog.data[0].path = file.display().to_string();
    let app = delete_app(catalog, None);

    let (status, body) = delete_model(app.clone(), "llm:gone.gguf").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!file.exists(), "weight file must be gone");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("delete body");
    assert_eq!(parsed["deleted"], true);
    assert_eq!(parsed["slug"], "llm:gone.gguf");
    assert_eq!(
        parsed["removed"][0].as_str().map(str::to_owned),
        Some(file.display().to_string()),
    );
    std::fs::remove_dir_all(&dir).ok();

    // Catalog entry dropped: the post-delete rescan over the empty roots
    // swapped the snapshot, so GET /v1/models no longer lists the slug.
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 16_384).await.unwrap();
    let list: serde_json::Value = serde_json::from_slice(&bytes).expect("catalog list");
    assert_eq!(list["data"].as_array().map(Vec::len), Some(0), "{list}");
}

/// Deleting an unknown slug is a 404 `OpenAI` envelope.
#[tokio::test]
async fn delete_unknown_slug_is_404() {
    let (status, body) = delete_model(
        delete_app(catalog_with(&["llm:a.gguf"]), None),
        "llm:nope.gguf",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unknown model")),
        "{parsed}"
    );
}

/// Deleting a resident model is refused with 409: unload first.
#[tokio::test]
async fn delete_resident_model_is_409() {
    let slug = "llm:resident.gguf";
    let dir = std::env::temp_dir().join(format!("honkbox-delete-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let file = dir.join("resident.gguf");
    std::fs::write(&file, b"weights").expect("write weight file");

    let mut catalog = catalog_with(&[slug]);
    catalog.data[0].path = file.display().to_string();
    let (status, body) = delete_model(delete_app(catalog, Some(seeded_swap(slug))), slug).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(file.exists(), "resident model's file must survive");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("resident")),
        "{parsed}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
