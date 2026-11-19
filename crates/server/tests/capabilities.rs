//! `GET /v1/capabilities` contract tests (FEAT-capabilities-api +
//! TASK-capabilities-contract-test). The document must agree with what the
//! server actually does: every `true` flag is backed by a working route, a
//! `false` flag by its absence, the format list by the compiled encoders,
//! the vision flag by the named 400 guard, and the queue knobs by the live
//! ledger/admission config. Also pinned: `/api/version` stays byte-
//! compatible (the new document is purely additive), and no user data
//! (catalog slugs, filesystem paths) ever reaches the document.
//!
//! Vision has no separate "second server" fixture: the assertions read the
//! compile probe (`VISION_COMPILED`) and hold under EITHER feature set —
//! run this suite once with default features and once with
//! `--no-default-features` to cover both builds.

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

use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::Engine as _;
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_external_tools::{Config as ProbeConfig, Registry, Status, ToolId, Unavailable};
use honkbox_llm_worker::StubLlm;
use honkbox_residency::ResourceKind;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, AuthMode, app};

/// Serializes process-global state across this binary's tests: admission/
/// ledger config and the external-tools registry cache. Tests that mutate
/// them (`queue_knobs_read_live_ledger_and_admission`, the blender-flag
/// probe-registry injection) hold the lock; tests that read the same
/// state take it too, so execution order never matters.
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One LLM catalog entry with a chosen vision flag.
fn llm_entry(id: &str, vision: bool) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: id.to_owned(),
        kind: ModelKind::Llm,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: if vision {
            vec!["text".to_owned(), "image".to_owned()]
        } else {
            vec!["text".to_owned()]
        },
        capabilities: ModelCapabilities {
            chat: true,
            completion: true,
            tool_calling: false,
            vision,
            embeddings: false,
        },
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

/// One SD catalog entry (registry-scan shape: id == slug here).
fn sd_entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: id.to_owned(),
        kind: ModelKind::Sd,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["image".to_owned()],
        capabilities: ModelCapabilities::for_kind(ModelKind::Sd),
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

/// One `ControlNet` catalog entry (conditioning input).
fn controlnet_entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: id.to_owned(),
        kind: ModelKind::ControlNet,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["image".to_owned()],
        capabilities: ModelCapabilities::for_kind(ModelKind::ControlNet),
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

fn catalog_with(entries: &[ModelCatalogEntry]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries.to_vec(),
    }
}

fn base_config(catalog: ModelCatalog, sd: Arc<dyn honkbox_sd_core::SdPipeline>) -> AppConfig {
    AppConfig {
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
        sd_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            None.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    }
}

fn app_with(catalog: ModelCatalog) -> Router {
    app(base_config(
        catalog,
        Arc::new(honkbox_sd_core::StubPipeline),
    ))
}

/// `SdPipeline` double answering the real sd-cpp backend tag (whatever a
/// given feature set's `SD_CPP_BACKEND_ID` is), without any generation
/// surface — the capabilities path must never generate.
#[derive(Debug)]
struct FakeSdCpp;

impl honkbox_sd_core::SdPipeline for FakeSdCpp {
    fn backend_id(&self) -> &'static str {
        honkbox_sd_core::SD_CPP_BACKEND_ID
    }

    fn generate(
        &self,
        _req: &honkbox_sd_core::Txt2ImgRequest,
        _cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        Err(honkbox_sd_core::SdError::Backend(
            "capabilities tests never generate".to_owned(),
        ))
    }
}

async fn get_json(app: &Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .expect("infallible");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

async fn get_capabilities(app: &Router) -> Value {
    let (status, body) = get_json(app, "/v1/capabilities").await;
    assert_eq!(status, StatusCode::OK, "capabilities status: {body}");
    body
}

/// Text bodies (named 501s) are deliberately not JSON.
async fn get_text(app: &Router, uri: &str) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .expect("infallible");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// 1x1 transparent PNG (smallest valid container bytes).
const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// Container magic check: `true` when the decoded bytes carry that
/// format's signature.
type MagicCheck = fn(&[u8]) -> bool;

/// Wire-contract version is the stable `"v1"` and server identity mirrors
/// `GET /api/version` (crate version, NOT a hand-picked number); the legacy
/// probe keeps its exact two-key shape (byte-compat).
#[tokio::test]
async fn api_version_is_the_wire_contract() {
    let app = app_with(catalog_with(&[]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["api_version"], "v1");
    assert_eq!(doc["server"]["name"], "honkbox");
    assert_eq!(doc["server"]["version"], env!("CARGO_PKG_VERSION"));
    let (_, version) = get_json(&app, "/api/version").await;
    let keys: Vec<&str> = version
        .as_object()
        .expect("version object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["name", "version"]);
    assert_eq!(version["name"], "honkbox");
    assert_eq!(version["version"], env!("CARGO_PKG_VERSION"));
}

/// The version policy is discoverable WITHOUT prose: the generated `OpenAPI`
/// `Capabilities` schema description carries the rule (bump on an
/// incompatible wire change, additive optional fields under the current
/// version), and `docs/README.md` mirrors it — the `include_str!` fails
/// the build outright if the policy section is deleted.
#[tokio::test]
async fn version_policy_is_discoverable_from_the_contract() {
    let app = app_with(catalog_with(&[]));
    let (_, openapi) = get_json(&app, "/v1/openapi.json").await;
    let description = openapi["components"]["schemas"]["Capabilities"]["description"]
        .as_str()
        .expect("Capabilities schema description");
    for needle in ["wire-contract version", "incompatibly", "additive"] {
        assert!(
            description.contains(needle),
            "policy marker {needle:?} missing from schema description"
        );
    }
    let readme = include_str!("../../../docs/README.md");
    for needle in [
        "## API version policy",
        "wire-contract version",
        "min_client_api_version",
    ] {
        assert!(
            readme.contains(needle),
            "docs/README.md policy missing {needle:?}"
        );
    }
}

/// The document never carries user data: with a catalog full of
/// user-visible slugs/paths, none of them (nor any filesystem path shape)
/// appears in the serialized body. Queue labels are the fixed kind
/// vocabulary only.
#[tokio::test]
async fn document_carries_no_user_data() {
    let app = app_with(catalog_with(&[
        llm_entry("llm:v.gguf", true),
        sd_entry("comfy:f.safetensors"),
        controlnet_entry("cn:my-secret-controlnet"),
    ]));
    let doc = get_capabilities(&app).await;
    let text = serde_json::to_string(&doc).unwrap();
    for needle in [
        "llm:v.gguf",
        "comfy:f.safetensors",
        "my-secret-controlnet",
        "slug",
        "/models/",
        "/home/",
        "\"path\"",
        "owned_by",
    ] {
        assert!(!text.contains(needle), "document leaked {needle:?}: {text}");
    }
    for row in doc["queue"]["kinds"].as_array().expect("kind rows") {
        let label = row["kind"].as_str().expect("kind label");
        assert!(
            matches!(
                label,
                "sd" | "llm" | "ocr" | "embed" | "rag" | "comfy" | "blender" | "3d"
            ),
            "unexpected kind label {label:?}"
        );
    }
}

/// The vision feature agrees with the compile probe AND the live catalog:
/// `built` is the same gate the chat route's named 400 uses; `models`
/// counts exactly the entries `/v1/models` advertises as vision-capable
/// (what the web UI's model picker consumes), refreshed with the catalog.
#[tokio::test]
async fn vision_feature_agrees_with_compile_and_catalog() {
    let app = app_with(catalog_with(&[llm_entry("llm:v.gguf", true)]));
    let doc = get_capabilities(&app).await;
    assert_eq!(
        doc["features"]["vision"]["built"],
        honkbox_llm_worker::VISION_COMPILED
    );
    assert_eq!(doc["features"]["vision"]["models"], 1);
    let (_, models) = get_json(&app, "/v1/models").await;
    assert_eq!(models["data"][0]["capabilities"]["vision"], true);

    let app = app_with(catalog_with(&[llm_entry("llm:text.gguf", false)]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["vision"]["models"], 0);
}

/// The vision CHAT gate agrees with the document: the gate keys on the
/// CATALOG entry's `capabilities.vision` — exactly the flag `vision.models`
/// counts — in every build. A vision-capable entry is served (the stub
/// serves in a text-only build too; a real pipeline fails loudly at model
/// load, operator overrides win); a vision-less entry answers the named
/// 400 — in a text-only build that is the shape of every scan-derived
/// entry, since derived vision collapses without the mtmd path.
#[tokio::test]
async fn vision_chat_gate_agrees_with_document() {
    // Chat POSTs contend for the process-global GPU slot with the other
    // POSTing tests in this binary; serialize under the slot lock.
    let _slot = SLOT_LOCK.lock().await;
    let request = |model: &str| {
        json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{TINY_PNG_B64}")}}
                    ]
                }
            ]
        })
    };

    // Vision-capable entry: counted, accepted in either build.
    let app = app_with(catalog_with(&[llm_entry("llm:v.gguf", true)]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["vision"]["models"], 1);
    assert_eq!(
        doc["features"]["vision"]["built"],
        honkbox_llm_worker::VISION_COMPILED
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/chat")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&request("llm:v.gguf")).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "vision-flagged entry serves: {bytes:?}"
    );

    // Vision-less entry: named 400 in either build.
    let app = app_with(catalog_with(&[llm_entry("llm:t.gguf", false)]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["vision"]["models"], 0);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/chat")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&request("llm:t.gguf")).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    // The native route keeps its plain-text errors (only /v1/chat/completions
    // carries the JSON envelope); the message names the vision gate.
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        text.contains("not vision-capable"),
        "vision 400 names the gate: {text}"
    );
}

/// `ControlNet` honesty: the flag is `true` only when the configured SD
/// pipeline applies control images (sd-cpp backend tag) AND the catalog
/// holds conditioning targets; the stub backend (which answers the named
/// 501 for valid pairs) must advertise `false` even with targets present.
#[tokio::test]
async fn controlnet_flag_agrees_with_backend_and_catalog() {
    let entries = [sd_entry("comfy:f.safetensors"), controlnet_entry("cn:xyz")];

    // Stub backend + a ControlNet entry: counted, availability false.
    let app = app_with(catalog_with(&entries));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["controlnet"]["models"], 1);
    assert_eq!(doc["features"]["controlnet"]["available"], false);

    // sd-cpp-tagged pipeline + a ControlNet entry: available.
    let router = routes::app(base_config(catalog_with(&entries), Arc::new(FakeSdCpp)));
    let doc = get_capabilities(&router).await;
    assert_eq!(doc["features"]["controlnet"]["models"], 1);
    assert_eq!(doc["features"]["controlnet"]["available"], true);

    // No targets: false even with the sd-cpp backend.
    let router = routes::app(base_config(
        catalog_with(&[sd_entry("comfy:f.safetensors")]),
        Arc::new(FakeSdCpp),
    ));
    let doc = get_capabilities(&router).await;
    assert_eq!(doc["features"]["controlnet"]["models"], 0);
    assert_eq!(doc["features"]["controlnet"]["available"], false);
}

/// Formats are derived from the compiled encoders (sd-core
/// `OutputFormat::ALL` + the JPEG-only compression knob) and the router
/// body limit; every advertised write format must round-trip through the
/// generations route with the stub pipeline and come back in that
/// container.
#[tokio::test]
async fn formats_agree_with_compiled_encoders() {
    // Generations POSTs contend for the process-global GPU slot with the
    // other POSTing tests; serialize under the slot lock.
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]));
    let doc = get_capabilities(&app).await;

    let write = doc["formats"]["write"].as_array().expect("write list");
    let expected: Vec<(&str, bool)> = honkbox_sd_core::OutputFormat::ALL
        .iter()
        .map(|format| (format.name(), format.compression_applies()))
        .collect();
    let actual: Vec<(&str, bool)> = write
        .iter()
        .map(|entry| {
            (
                entry["format"].as_str().expect("format name"),
                entry["compression"].as_bool().expect("compression flag"),
            )
        })
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(doc["formats"]["read"], json!(["png"]));
    assert_eq!(doc["formats"]["limits"]["output_compression"]["min"], 0);
    assert_eq!(doc["formats"]["limits"]["output_compression"]["max"], 100);
    assert_eq!(
        doc["formats"]["limits"]["output_compression"]["applies_to"],
        "jpeg"
    );
    assert_eq!(
        doc["formats"]["limits"]["body_bytes"],
        routes::AUDIO_BODY_LIMIT_BYTES
    );

    let magics: [(&str, MagicCheck); 3] = [
        ("png", |bytes| bytes.starts_with(&[0x89, b'P', b'N', b'G'])),
        ("jpeg", |bytes| bytes.starts_with(&[0xFF, 0xD8, 0xFF])),
        ("webp", |bytes| {
            bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
        }),
    ];
    for (format, magic) in magics {
        let request = json!({
            "model": "comfy:f.safetensors",
            "prompt": "a b c",
            "output_format": format
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/images/generations")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{format} round trip: {}",
            String::from_utf8_lossy(&bytes)
        );
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        let b64 = body["data"][0]["b64_json"].as_str().expect("b64_json");
        let image = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .expect("base64 image");
        assert!(magic(&image), "{format} magic mismatch");
    }
}

/// Every `true` structural flag is backed by a real route in the generated
/// `OpenAPI` document; `video` stays `false` with no video path; the
/// capabilities DTOs are registered in the schema list.
#[tokio::test]
async fn flags_cross_check_the_openapi_document() {
    let app = app_with(catalog_with(&[]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["jobs"], true);
    assert_eq!(doc["features"]["events"], true);
    assert_eq!(doc["features"]["lora"], true);
    assert_eq!(doc["features"]["video"], false);

    let (_, openapi) = get_json(&app, "/v1/openapi.json").await;
    let paths = openapi["paths"].as_object().expect("openapi paths");
    for required in [
        "/v1/jobs",
        "/v1/events",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/capabilities",
    ] {
        assert!(paths.contains_key(required), "openapi missing {required}");
    }
    assert!(
        !paths.keys().any(|path| path.starts_with("/v1/videos")),
        "video advertised without a video job class"
    );
    let schemas = openapi["components"]["schemas"]
        .as_object()
        .expect("openapi schemas");
    assert!(schemas.contains_key("Capabilities"));
    assert!(schemas.contains_key("QueueCapabilities"));
    assert!(schemas.contains_key("OutputFormatCapability"));
}

/// The auth mode reflects runtime config; the key VALUE never appears.
#[tokio::test]
async fn auth_mode_reflects_config() {
    let app = app_with(catalog_with(&[]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["auth"]["mode"], "none");

    let mut config = base_config(catalog_with(&[]), Arc::new(honkbox_sd_core::StubPipeline));
    config.auth = AuthConfig {
        mode: AuthMode::Bearer,
        api_key_env: "HONKBOX_CAPABILITIES_TEST_KEY".to_owned(),
    };
    // Unique env var name: no other test in this binary reads it.
    routes::set_env_var("HONKBOX_CAPABILITIES_TEST_KEY", Some("caps-test-key-value"));
    let router = routes::app(config);
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header("authorization", "Bearer caps-test-key-value")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let doc: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(doc["auth"]["mode"], "bearer");
    let text = serde_json::to_string(&doc).unwrap();
    assert!(!text.contains("caps-test-key-value"), "key value leaked");
}

/// Queue knobs are read LIVE from the admission config and the resource
/// ledger: per-kind caps + TTL, the wait bound, and the `Retry-After` hint
/// all match what a 429 client actually experiences.
#[tokio::test]
async fn queue_knobs_read_live_ledger_and_admission() {
    let _slot = SLOT_LOCK.lock().await;
    admission::configure(Duration::from_secs(7));
    let sd_slot = ResourceKind::ALL
        .iter()
        .position(|kind| *kind == ResourceKind::Sd)
        .expect("sd in ALL");
    let mut ledger_config = honkbox_residency::LedgerConfig::new();
    ledger_config.capacities.gpu_slots = 2;
    ledger_config.kinds[sd_slot].queue_cap = 3;
    ledger_config.kinds[sd_slot].ttl = Duration::from_secs(7);
    honkbox_residency::global().configure(ledger_config);

    let app = app_with(catalog_with(&[]));
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["queue"]["gpu_slots"], 2);
    assert_eq!(doc["queue"]["queue_wait_secs"], 7);
    assert_eq!(
        doc["queue"]["retry_after_secs"],
        admission::RETRY_AFTER_HINT_SECS
    );
    let kinds = doc["queue"]["kinds"].as_array().expect("kind rows");
    let sd_row = kinds
        .iter()
        .find(|row| row["kind"] == "sd")
        .expect("sd row");
    assert_eq!(sd_row["queue_cap"], 3);
    assert_eq!(sd_row["ttl_secs"], 7);
    let labels: Vec<&str> = kinds
        .iter()
        .map(|row| row["kind"].as_str().unwrap())
        .collect();
    assert_eq!(labels.len(), honkbox_residency::ResourceKind::ALL.len());

    // Reset the process globals for any remaining tests.
    admission::configure(Duration::ZERO);
    honkbox_residency::global().configure(honkbox_residency::LedgerConfig::new());
}

/// The queue rows agree with the MOUNTED routes: a kind whose capacity a
/// route admits is advertised as callable, a kind with no admitting route
/// is a RESERVED row (capacity only — the closed job registry has no job
/// of that kind). The test fails when a reserved row is advertised as
/// callable, and when a row claims a route the generated `OpenAPI`
/// document does not serve.
#[tokio::test]
async fn reserved_rows_agree_with_the_admitting_routes() {
    let app = app_with(catalog_with(&[]));
    let doc = get_capabilities(&app).await;
    let (_, openapi) = get_json(&app, "/v1/openapi.json").await;
    let paths = openapi["paths"].as_object().expect("openapi paths");
    let rows = doc["queue"]["kinds"].as_array().expect("kind rows");
    assert_eq!(rows.len(), ResourceKind::ALL.len());
    for row in rows {
        let label = row["kind"].as_str().expect("kind label");
        let reserved = row["reserved"].as_bool().expect("reserved flag");
        let kind = ResourceKind::from_label(label).expect("known kind");
        match routes::admitting_route(kind) {
            Some(path) => {
                assert!(
                    !reserved,
                    "{label} has the admitting route {path} but is advertised as reserved"
                );
                assert!(
                    paths.contains_key(path),
                    "{label} claims {path}, which the openapi document does not serve"
                );
            }
            None => assert!(
                reserved,
                "{label} has no admitting route but is advertised as callable"
            ),
        }
    }
}

/// With no startup probe installed, the capability document still reports
/// both known tools — honestly marked missing — in registry order.
#[tokio::test]
async fn capabilities_report_external_tools_without_probing() {
    let _slot = SLOT_LOCK.lock().await;
    let doc = get_capabilities(&app_with(catalog_with(&[]))).await;
    let tools = doc["external_tools"]["tools"]
        .as_array()
        .expect("external tools");
    let ids: Vec<&str> = tools
        .iter()
        .map(|tool| tool["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["blender", "comfyui"]);
    // No probed Blender → the feature flag is false (the same truth the
    // job route's named 501 enforces).
    assert_eq!(doc["features"]["blender"], false);
    for tool in tools {
        assert_eq!(tool["available"], false);
        assert_eq!(tool["unavailable"], "not installed");
        assert_eq!(&tool["capabilities"], &json!([]));
    }
}

/// One probed-or-missing tool status in registry order.
fn probe_status(id: ToolId, available: bool) -> Status {
    Status {
        id,
        version: available.then(|| "4.5.15".to_owned()),
        capabilities: Vec::new(),
        devices: Vec::new(),
        unavailable: (!available).then_some(Unavailable::NotInstalled),
    }
}

/// The `blender` feature flag agrees with the probe registry on BOTH
/// branches: no startup probe (or a probe that found nothing) → `false` —
/// the same truth the job route's named 501 enforces — and a probed,
/// available engine → `true`, reported with its version. The injection
/// swaps the shared registry cache, so it holds `SLOT_LOCK` and restores
/// the unprobed state before returning.
#[tokio::test]
async fn blender_flag_agrees_with_the_probe_registry() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[]));

    // False: this binary never ran the startup probe.
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["blender"], false);

    // Still false when a registry exists but its Blender probe found
    // nothing: presence without a proven engine is not availability.
    routes::set_registry_for_tests(Some(Registry::from_statuses(
        ProbeConfig::default(),
        vec![
            probe_status(ToolId::Blender, false),
            probe_status(ToolId::ComfyUi, false),
        ],
    )))
    .await;
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["blender"], false);

    // True: a probed, available engine.
    routes::set_registry_for_tests(Some(Registry::from_statuses(
        ProbeConfig::default(),
        vec![
            probe_status(ToolId::Blender, true),
            probe_status(ToolId::ComfyUi, false),
        ],
    )))
    .await;
    let doc = get_capabilities(&app).await;
    assert_eq!(doc["features"]["blender"], true);
    let tools = doc["external_tools"]["tools"].as_array().expect("tools");
    assert_eq!(tools[0]["available"], true);
    assert_eq!(tools[0]["version"], "4.5.15");

    // Restore the unprobed cache for the other tests in this binary.
    routes::set_registry_for_tests(None).await;
}

#[tokio::test]
async fn external_tool_routes_name_missing_tools() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[]));
    let (status, body) = get_json(&app, "/v1/external-tools").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);

    let (status, body) = get_text(&app, "/v1/external-tools/blender").await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(body.contains("blender"), "{body}");
    assert!(body.contains("not installed"), "{body}");

    let (status, _) = get_text(&app, "/v1/external-tools/unknown").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // No registry installed: reloading is a cheap no-op, never a probe.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/external-tools/reload")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("infallible");
    assert_eq!(response.status(), StatusCode::OK);
}
