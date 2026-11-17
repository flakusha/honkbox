//! `POST /v1/images/edits`: honkbox img2img/inpaint multipart shape over the
//! startup-configured SD pipeline (stub in tests). The stub honestly ignores
//! image inputs — pixels never depend on the upload — so these tests cover
//! the wire/validation paths: required `image` (PNG magic-sniffed), optional
//! `mask` (present/absent paths), `prompt`, `strength` range checks, `size`
//! handling, multipart parity of the shared recipe validation, unknown-model
//! 404, and the job bundle recording the denoise strength.

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
#[path = "../src/downloads.rs"]
mod downloads;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/jobs_registry.rs"]
mod jobs_registry;
#[allow(dead_code)]
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::sync::{Arc, LazyLock, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's tests.
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

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

fn catalog_with(entries: &[ModelCatalogEntry]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries.to_vec(),
    }
}

fn app_with(catalog: ModelCatalog, data_dir: Option<std::path::PathBuf>) -> Router {
    app(AppConfig {
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
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// One multipart form part.
struct Part {
    name: &'static str,
    filename: Option<&'static str>,
    content_type: Option<&'static str>,
    data: Vec<u8>,
}

fn part(name: &'static str, text: &str) -> Part {
    Part {
        name,
        filename: None,
        content_type: None,
        data: text.as_bytes().to_vec(),
    }
}

/// A file part carrying PNG bytes (the declared content type is advisory;
/// the handler magic-sniffs).
fn png_part(name: &'static str, bytes: Vec<u8>) -> Part {
    Part {
        name,
        filename: Some("input.png"),
        content_type: Some("image/png"),
        data: bytes,
    }
}

/// Encode parts into a multipart/form-data body (boundary included in the
/// returned content-type value).
fn multipart_body(parts: &[Part]) -> (String, Vec<u8>) {
    let boundary = "honkbox-edit-test";
    let mut body = Vec::new();
    for one in parts {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\ncontent-disposition: form-data; name=\"{}\"",
                one.name
            )
            .as_bytes(),
        );
        if let Some(filename) = one.filename {
            body.extend_from_slice(format!("; filename=\"{filename}\"\r\n").as_bytes());
        } else {
            body.extend_from_slice(b"\r\n");
        }
        if let Some(content_type) = one.content_type {
            body.extend_from_slice(format!("content-type: {content_type}\r\n").as_bytes());
        }
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(&one.data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// `POST /v1/images/edits`; returns (status, parsed JSON body).
async fn edits(app: Router, parts: &[Part]) -> (StatusCode, Value) {
    let (content_type, body) = multipart_body(parts);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header("content-type", content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .unwrap();
    let parsed =
        serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("response not JSON: {err}"));
    (status, parsed)
}

/// A real `width`×`height` RGBA PNG via the in-tree encoder (the same
/// container the API itself emits).
fn png_bytes(width: u32, height: u32) -> Vec<u8> {
    honkbox_sd_core::encode_image(
        honkbox_sd_core::GeneratedImage {
            width,
            height,
            rgba: vec![255; (width * height * 4) as usize],
        },
        honkbox_sd_core::OutputFormat::Png,
    )
    .expect("png encodes")
}

/// Decode one `b64_json` entry and assert the PNG magic + IHDR dimensions.
fn assert_png(entry: &Value, width: u32, height: u32) {
    use base64::Engine as _;
    let b64 = entry["b64_json"].as_str().expect("b64_json string");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("valid base64");
    assert_eq!(
        &bytes[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
    );
    assert_eq!(&bytes[12..16], b"IHDR");
    assert_eq!(u32::from_be_bytes(bytes[16..20].try_into().unwrap()), width);
    assert_eq!(
        u32::from_be_bytes(bytes[20..24].try_into().unwrap()),
        height
    );
}

/// Missing `image` part is an explicit 400 (`image_required`).
#[tokio::test]
async fn missing_image_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[part("prompt", "redo this")],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "image_required", "{body}");
}

/// Missing `prompt` is an explicit 400.
#[tokio::test]
async fn missing_prompt_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[png_part("image", png_bytes(4, 4))],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "prompt_required", "{body}");
}

/// Non-PNG `image` bytes (magic sniff) are a 400 — the declared content
/// type is not trusted.
#[tokio::test]
async fn non_png_image_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            Part {
                name: "image",
                filename: Some("input.png"),
                content_type: Some("image/png"),
                data: b"definitely not a png".to_vec(),
            },
            part("prompt", "redo this"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_image", "{body}");
}

/// A present-but-non-PNG `mask` is a 400 (`invalid_mask`).
#[tokio::test]
async fn non_png_mask_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            Part {
                name: "mask",
                filename: Some("mask.png"),
                content_type: Some("image/png"),
                data: vec![0x89, b'P', b'N', b'G', 0, 0, 0, 0],
            },
            part("prompt", "inpaint here"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_mask", "{body}");
}

/// `strength` outside `(0, 1]` is a 400 (`invalid_strength`); unparsable
/// text is the multipart-level 400 naming the field.
#[tokio::test]
async fn strength_out_of_range_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    for strength in ["0", "0.0", "1.5", "-0.5", "nan"] {
        let (status, body) = edits(
            app.clone(),
            &[
                png_part("image", png_bytes(4, 4)),
                part("prompt", "redo this"),
                part("strength", strength),
            ],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "strength {strength}: {body}"
        );
        assert_eq!(
            body["error"]["code"], "invalid_strength",
            "strength {strength}: {body}"
        );
    }
    let (status, body) = edits(
        app,
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("strength", "not-a-number"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("strength must be a number")),
        "{body}"
    );
}

/// Unknown model ids are a 404 (`model_not_found`), the same contract as
/// `/v1/images/generations`.
#[tokio::test]
async fn unknown_model_is_404() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[sd_entry("comfy:f.safetensors")]);
    let app = app_with(catalog, None);
    let (status, body) = edits(
        app,
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("model", "nope"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");
}

/// img2img happy path (mask absent): renders, and the bundle records the
/// default denoise strength (0.75) with no loras.
#[tokio::test]
async fn img2img_without_mask_records_default_strength() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-edit-test-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let (status, body) = edits(
        app_with(
            catalog_with(&[sd_entry("comfy:f.safetensors")]),
            Some(data_dir.clone()),
        ),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "make it blue"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // No `size` given: output dims default to the source PNG's IHDR dims.
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "{body}");
    assert_png(&data[0], 4, 4);
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.next().expect("one bundle").unwrap();
    let request: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("request.json")).unwrap())
            .unwrap();
    assert_eq!(request["strength"], 0.75, "{request}");
    assert_eq!(request["loras"], json!([]), "{request}");
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Inpaint happy path (mask present): renders and the bundle records the
/// explicitly requested strength.
#[tokio::test]
async fn inpaint_with_mask_records_explicit_strength() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-edit-mask-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let (status, body) = edits(
        app_with(
            catalog_with(&[sd_entry("comfy:f.safetensors")]),
            Some(data_dir.clone()),
        ),
        &[
            png_part("image", png_bytes(4, 4)),
            png_part("mask", png_bytes(4, 4)),
            part("prompt", "inpaint here"),
            part("strength", "0.42"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.next().expect("one bundle").unwrap();
    let request: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("request.json")).unwrap())
            .unwrap();
    // f32 strengths widen to f64 in the sidecar JSON: compare with a
    // tolerance, not equality.
    assert!(
        (request["strength"].as_f64().unwrap() - 0.42).abs() < 1e-6,
        "{request}"
    );
    std::fs::remove_dir_all(&data_dir).ok();
}

/// `size` overrides the source-image default (output dims follow it).
#[tokio::test]
async fn size_override_changes_output_dims() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("size", "8x16"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().expect("data array");
    assert_png(&data[0], 8, 16);
}

/// A bad `size` string is a 400 (`invalid_size`).
#[tokio::test]
async fn bad_size_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("size", "bogus"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_size", "{body}");
}

/// Multipart parity of the shared recipe validation: unknown sampler name
/// is the same 400 as on `/v1/images/generations`; a known one renders.
#[tokio::test]
async fn recipe_fields_parity() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    let (status, body) = edits(
        app.clone(),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("sampler", "dpm++2m"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = edits(
        app,
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("sampler", "nope"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unknown_sampler", "{body}");
}

/// Multipart parity of the named-recipe and steps validation: a valid
/// `recipe` + `steps` pair renders; an unknown recipe and an out-of-range
/// steps value are the same 400s as on `/v1/images/generations`.
#[tokio::test]
async fn recipe_and_steps_parity() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    let (status, body) = edits(
        app.clone(),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("recipe", "turbo"),
            part("steps", "6"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    for (field, value, code) in [
        ("recipe", "dreamweaver", "unknown_recipe"),
        ("steps", "0", "invalid_steps"),
        ("steps", "101", "invalid_steps"),
    ] {
        let (status, body) = edits(
            app.clone(),
            &[
                png_part("image", png_bytes(4, 4)),
                part("prompt", "redo this"),
                part(field, value),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}={value}: {body}");
        assert_eq!(body["error"]["code"], code, "{field}={value}: {body}");
    }
}
