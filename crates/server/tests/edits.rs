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
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{
    ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind, ModelMetadata,
    SafetensorsMetadata,
};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's tests.
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One SD catalog entry (registry-scan shape: id == slug here).
fn sd_entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        modality: honkbox_api_types::ModelModalities::empty(),
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

/// `ash-sd`-tagged double: deterministic stub pixels, but reports the backend
/// ID whose unsupported image inputs must be refused.
struct AshLikeSd;

impl honkbox_sd_core::SdPipeline for AshLikeSd {
    fn generate(
        &self,
        req: &honkbox_sd_core::Txt2ImgRequest,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        honkbox_sd_core::SdPipeline::generate(&honkbox_sd_core::StubPipeline, req, cancel)
    }

    fn backend_id(&self) -> &'static str {
        "ash-sd"
    }
}
fn app_with(catalog: ModelCatalog, data_dir: Option<std::path::PathBuf>) -> Router {
    app_with_sd(catalog, data_dir, Arc::new(honkbox_sd_core::StubPipeline))
}

#[allow(clippy::needless_pass_by_value)] // mirrors the other test helpers' owned-argument style
fn app_with_sd(
    catalog: ModelCatalog,
    data_dir: Option<std::path::PathBuf>,
    sd: Arc<dyn honkbox_sd_core::SdPipeline>,
) -> Router {
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
            ocr_dirs: Vec::new(),
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
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: data_dir.clone(),
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            data_dir.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
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
    edits_with_delivery(app, parts, None).await
}

/// Same request with an explicit `X-Honkbox-Delivery` value.
async fn edits_with_delivery(
    app: Router,
    parts: &[Part],
    delivery: Option<&str>,
) -> (StatusCode, Value) {
    let (content_type, body) = multipart_body(parts);
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/images/edits")
        .header("content-type", content_type);
    if let Some(delivery) = delivery {
        builder = builder.header("X-Honkbox-Delivery", delivery);
    }
    let response = app
        .oneshot(builder.body(Body::from(body)).unwrap())
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

/// A backend without img2img/inpaint must refuse the edit instead of quietly
/// generating a plain txt2img image from the source prompt.
#[tokio::test]
async fn ash_backend_refuses_image_inputs() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with_sd(
        catalog_with(&[sd_entry("comfy:f.safetensors")]),
        None,
        Arc::new(AshLikeSd),
    );
    for parts in [
        vec![png_part("image", png_bytes(4, 4)), part("prompt", "redo")],
        vec![
            png_part("image", png_bytes(4, 4)),
            png_part("mask", png_bytes(4, 4)),
            part("prompt", "inpaint"),
        ],
    ] {
        let (status, body) = edits(app.clone(), &parts).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
        assert_eq!(body["error"]["code"], "conditioning_unsupported", "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("init_image/mask_image")),
            "{body}"
        );
    }
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

/// A mask whose IHDR dims differ from the source is rejected before admission
/// rather than silently resized by the vendor.
#[tokio::test]
async fn mismatched_mask_dimensions_are_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            png_part("mask", png_bytes(8, 4)),
            part("prompt", "inpaint here"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "mask_dimensions_mismatch", "{body}");
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
    let sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("final_00.json")).unwrap())
            .unwrap();
    assert_eq!(sidecar["strength"], 0.75, "{sidecar}");
    assert_eq!(sidecar["mask_mode"], "replace", "{sidecar}");
    assert!(sidecar["mask_sha256"].is_null(), "{sidecar}");
    assert_eq!(sidecar["loras"], json!([]), "{sidecar}");
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Inpaint happy path (mask present): renders and the bundle records the
/// explicitly requested strength.
#[tokio::test]
async fn inpaint_with_mask_records_explicit_strength() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-edit-mask-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let mask = png_bytes(4, 4);
    let (status, body) = edits(
        app_with(
            catalog_with(&[sd_entry("comfy:f.safetensors")]),
            Some(data_dir.clone()),
        ),
        &[
            png_part("image", png_bytes(4, 4)),
            png_part("mask", mask.clone()),
            part("prompt", "inpaint here"),
            part("strength", "0.42"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.next().expect("one bundle").unwrap();
    let sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("final_00.json")).unwrap())
            .unwrap();
    // f32 strengths widen to f64 in the sidecar JSON: compare with a
    // tolerance, not equality.
    assert!(
        (sidecar["strength"].as_f64().unwrap() - 0.42).abs() < 1e-6,
        "{sidecar}"
    );
    assert_eq!(sidecar["mask_mode"], "inpaint", "{sidecar}");
    assert_eq!(
        sidecar["mask_sha256"],
        artifacts::hex_lower(&<sha2::Sha256 as sha2::Digest>::digest(&mask)),
        "{sidecar}"
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
/// Decode one `b64_json` entry to raw bytes.
fn decode_entry(entry: &Value) -> Vec<u8> {
    use base64::Engine as _;
    let b64 = entry["b64_json"].as_str().expect("b64_json string");
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("valid base64")
}

/// Parse the generation record from the first JPEG APP1 EXIF `UserComment`.
fn jpeg_exif_record(bytes: &[u8]) -> Value {
    assert_eq!(&bytes[..4], b"\xff\xd8\xff\xe1");
    let length = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
    let exif = &bytes[6..length + 4];
    let tiff = exif.strip_prefix(b"Exif\0\0").expect("EXIF prefix");
    assert_eq!(&tiff[..2], b"II");
    let ifd_offset = usize::try_from(u32::from_le_bytes(
        tiff[4..8].try_into().expect("IFD offset"),
    ))
    .expect("IFD offset fits usize");
    let count = usize::from(u16::from_le_bytes(
        tiff[ifd_offset..ifd_offset + 2]
            .try_into()
            .expect("IFD count"),
    ));
    let comment = (0..count)
        .find_map(|index| {
            let entry = ifd_offset + 2 + index * 12;
            if u16::from_le_bytes(tiff[entry..entry + 2].try_into().expect("tag")) != 0x9286 {
                return None;
            }
            let length = usize::try_from(u32::from_le_bytes(
                tiff[entry + 4..entry + 8]
                    .try_into()
                    .expect("comment count"),
            ))
            .expect("comment count fits usize");
            let offset = usize::try_from(u32::from_le_bytes(
                tiff[entry + 8..entry + 12]
                    .try_into()
                    .expect("comment offset"),
            ))
            .expect("comment offset fits usize");
            Some(&tiff[offset..offset + length])
        })
        .expect("EXIF UserComment");
    let json = comment
        .strip_prefix(b"UNICODE\0")
        .expect("Unicode UserComment");
    serde_json::from_slice(json).expect("generation record JSON")
}

/// `output_format=jpeg` returns JPEG bytes and lands a `final_00.jpg`
/// bundle whose sidecar `format` says `jpeg`.
#[tokio::test]
async fn jpeg_output_format_and_bundle_agree() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-edit-fmt-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let (status, body) = edits(
        app_with(
            catalog_with(&[sd_entry("comfy:f.safetensors")]),
            Some(data_dir.clone()),
        ),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("output_format", "jpeg"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(&decode_entry(&body["data"][0])[..3], &[0xFF, 0xD8, 0xFF]);
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.next().expect("one bundle").unwrap().path();
    let art = std::fs::read(dir.join("final_00.jpg")).expect("bundle artifact with jpg extension");
    assert_eq!(&art[..3], &[0xFF, 0xD8, 0xFF]);
    assert_eq!(
        decode_entry(&body["data"][0]),
        art,
        "bundle bytes are response bytes"
    );
    let embedded_record = jpeg_exif_record(&art);
    let sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("final_00.json")).unwrap()).unwrap();
    assert_eq!(sidecar["format"], "jpeg", "final_00.json");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("c2pa_manifest.json")).unwrap())
            .unwrap();
    let assertion = manifest["assertions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|assertion| assertion["label"] == "honkbox.generation")
        .expect("generation assertion");
    assert_eq!(
        embedded_record, assertion["data"],
        "JPEG EXIF and manifest carry the same generation record"
    );
    assert_eq!(embedded_record["record_type"], "honkbox.generation/1");
    assert_eq!(embedded_record["prompt"], "redo this");
    assert!(
        embedded_record["strength"].is_number(),
        "img2img strength recorded"
    );
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Multipart parity of the format/compression validation: an unknown
/// `output_format` and a non-numeric / out-of-range `output_compression`
/// are the same named 400s as on `/v1/images/generations`; a valid
/// compression value renders.
#[tokio::test]
async fn output_format_and_compression_parity() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    let (status, body) = edits(
        app.clone(),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("output_format", "jpeg"),
            part("output_compression", "50"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(&decode_entry(&body["data"][0])[..3], &[0xFF, 0xD8, 0xFF]);
    for (field, value, code) in [
        ("output_format", "gif", "unsupported_output_format"),
        ("output_format", "JPEG", "unsupported_output_format"),
        ("output_compression", "abc", "invalid_output_compression"),
        ("output_compression", "101", "invalid_output_compression"),
        ("output_compression", "-1", "invalid_output_compression"),
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

/// One `LoRA` catalog entry (registry-scan shape; `path` mirrors the
/// absolute weight-file path a real scan produces) — same fixture as the
/// `/v1/images/generations` tests.
fn lora_entry(id: &str, path: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        modality: honkbox_api_types::ModelModalities::empty(),
        kind: ModelKind::Lora,
        capabilities: ModelCapabilities::for_kind(ModelKind::Lora),
        modalities: vec!["image".to_owned()],
        category: "loras".to_owned(),
        path: path.to_owned(),
        ..sd_entry(id)
    }
}

fn tagged_entry(mut entry: ModelCatalogEntry, base: &str) -> ModelCatalogEntry {
    entry.metadata = Some(ModelMetadata::Safetensors(SafetensorsMetadata {
        tensor_count: 0,
        dtypes: Vec::new(),
        base_model: Some(base.to_owned()),
    }));
    entry
}

/// A multipart edits request with `loras` as a JSON array text part.
fn loras_part(value: &Value) -> Part {
    part("loras", &value.to_string())
}

/// An unknown `LoRA` slug — or a known non-`LoRA` slug — is a 400 naming the
/// offender: the edits route rides the same `resolve_loras` validation as
/// txt2img, with no edits-side duplicates.
#[tokio::test]
async fn edits_unknown_lora_slug_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let mut catalog = catalog_with(&[sd_entry("comfy:f.safetensors")]);
    catalog.data.push(lora_entry(
        "comfy:loras/detail.safetensors",
        "/models/loras/detail.safetensors",
    ));
    let app = app_with(catalog, None);
    for slug in ["comfy:loras/missing.safetensors", "comfy:f.safetensors"] {
        let (status, body) = edits(
            app.clone(),
            &[
                png_part("image", png_bytes(4, 4)),
                part("prompt", "redo this"),
                loras_part(&json!([{ "slug": slug }])),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "slug {slug}: {body}");
        assert_eq!(body["error"]["code"], "unknown_lora", "slug {slug}: {body}");
    }
}

/// A known SDXL `LoRA` on an SD1.5 checkpoint is refused before GPU work.
#[tokio::test]
async fn known_base_model_mismatch_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        tagged_entry(sd_entry("comfy:sd15.safetensors"), "sd15"),
        tagged_entry(
            lora_entry(
                "comfy:loras/sdxl.safetensors",
                "/models/loras/sdxl.safetensors",
            ),
            "sdxl",
        ),
    ]);
    let (status, body) = edits(
        app_with(catalog, None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("model", "comfy:sd15.safetensors"),
            part("prompt", "redo this"),
            loras_part(&json!([{ "slug": "comfy:loras/sdxl.safetensors" }])),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "base_model_mismatch", "{body}");
}

/// Matching tags pass and both tags land in the job sidecar for replay.
#[tokio::test]
async fn matching_base_model_tags_are_recorded_in_sidecar() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-edit-base-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let catalog = catalog_with(&[
        tagged_entry(sd_entry("comfy:sd15.safetensors"), "sd15"),
        tagged_entry(
            lora_entry(
                "comfy:loras/detail.safetensors",
                "/models/loras/detail.safetensors",
            ),
            "sd15",
        ),
    ]);
    let (status, body) = edits(
        app_with(catalog, Some(data_dir.clone())),
        &[
            png_part("image", png_bytes(4, 4)),
            part("model", "comfy:sd15.safetensors"),
            part("prompt", "redo this"),
            loras_part(&json!([{ "slug": "comfy:loras/detail.safetensors" }])),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.next().expect("one bundle").unwrap().path();
    let sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("final_00.json")).unwrap()).unwrap();
    assert_eq!(sidecar["checkpoint_base_model"], "sd15", "{sidecar}");
    assert_eq!(sidecar["loras"][0]["base_model"], "sd15", "{sidecar}");
    std::fs::remove_dir_all(&data_dir).ok();
}

/// More than two adapters per request is the documented `too_many_loras`
/// 400, on edits exactly as on generations.
#[tokio::test]
async fn edits_too_many_loras_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry("comfy:loras/a.safetensors", "/models/loras/a.safetensors"),
        lora_entry("comfy:loras/b.safetensors", "/models/loras/b.safetensors"),
    ]);
    let (status, body) = edits(
        app_with(catalog, None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            loras_part(&json!([
                {"slug": "comfy:loras/a.safetensors"},
                {"slug": "comfy:loras/b.safetensors"},
                {"slug": "comfy:loras/c.safetensors"}
            ])),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "too_many_loras", "{body}");
}

/// A `LoRA` strength outside `[-2.0, 2.0]` is a 400 on edits too —
/// rejected, never clamped (same `invalid_lora_strength` code).
#[tokio::test]
async fn edits_lora_strength_out_of_range_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry("comfy:loras/a.safetensors", "/models/loras/a.safetensors"),
    ]);
    let app = app_with(catalog, None);
    for strength in [5.0_f64, -3.0] {
        let (status, body) = edits(
            app.clone(),
            &[
                png_part("image", png_bytes(4, 4)),
                part("prompt", "redo this"),
                loras_part(&json!([
                    {"slug": "comfy:loras/a.safetensors", "strength": strength}
                ])),
            ],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "strength {strength}: {body}"
        );
        assert_eq!(body["error"]["code"], "invalid_lora_strength", "{body}");
    }
}

/// Adapters spanning more than one directory are the `mixed_lora_dirs` 400:
/// the vendor applies a single `LoRA` dir per generation.
#[tokio::test]
async fn edits_mixed_lora_dirs_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry(
            "comfy:loras/a.safetensors",
            "/roots/one/loras/a.safetensors",
        ),
        lora_entry(
            "comfy:loras/b.safetensors",
            "/roots/two/loras/b.safetensors",
        ),
    ]);
    let (status, body) = edits(
        app_with(catalog, None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            loras_part(&json!([
                {"slug": "comfy:loras/a.safetensors"},
                {"slug": "comfy:loras/b.safetensors"}
            ])),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "mixed_lora_dirs", "{body}");
}

/// A valid edits `LoRA` request renders (stub backend) with the advisory
/// license warning in the response, and the bundle sidecar records the
/// resolved slug + strength. Per-test temp job dir, removed afterwards.
#[tokio::test]
async fn edits_loras_warn_and_record() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-edit-lora-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry(
            "comfy:loras/detail.safetensors",
            "/models/loras/detail.safetensors",
        ),
    ]);
    let (status, body) = edits(
        app_with(catalog, Some(data_dir.clone())),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            loras_part(&json!([
                {"slug": "comfy:loras/detail.safetensors", "strength": 0.8}
            ])),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let warnings = body["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 1, "{body}");
    assert!(
        warnings[0]
            .as_str()
            .is_some_and(|w| w.contains("no license recorded")
                && w.contains("comfy:loras/detail.safetensors")),
        "{body}"
    );
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.next().expect("one bundle").unwrap().path();
    let sidecar: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("final_00.json")).unwrap()).unwrap();
    assert_eq!(
        sidecar["loras"][0]["slug"],
        "comfy:loras/detail.safetensors"
    );
    // f32 strengths widen to f64 in the sidecar JSON: compare with a
    // tolerance, not equality.
    assert!((sidecar["loras"][0]["strength"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Without `loras` the response omits the `warnings` key entirely
/// (skip-if-none), keeping edits byte-compatible with pre-`LoRA` consumers.
#[tokio::test]
async fn edits_warnings_absent_without_loras() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("warnings").is_none(), "{body}");
}

/// A malformed `loras` text part (not a JSON array of `{slug, strength}`)
/// is a 400 naming the field, never a silent ignore.
#[tokio::test]
async fn edits_malformed_loras_field_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry("comfy:loras/a.safetensors", "/models/loras/a.safetensors"),
    ]);
    let (status, body) = edits(
        app_with(catalog, None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            part("loras", "not-json"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("loras must be a JSON array")),
        "{body}"
    );
}

/// `GET /v1/jobs` while a job is registered (the guard drops when the
/// request ends, so an in-flight gate keeps the row observable).
async fn jobs_list(app: Router) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/jobs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Read SSE frames until the `job_result` for `job_id`; other tests in
/// this binary share the global registry, so frames are filtered by id.
async fn result_frame(stream: &mut axum::body::BodyDataStream, job_id: &str) -> Value {
    let mut buffer = String::new();
    loop {
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
            .await
            .expect("sse frame within 5s")
            .expect("stream alive")
            .expect("frame bytes");
        buffer.push_str(std::str::from_utf8(&chunk).expect("utf8"));
        while let Some(pos) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..pos + 2).collect();
            let Some(data) = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .next_back()
            else {
                continue;
            };
            let parsed: Value = serde_json::from_str(data).expect("frame data JSON");
            if frame.contains("event: job_result") && parsed["id"].as_str() == Some(job_id) {
                return parsed;
            }
        }
    }
}

/// Park/notify pair: `GatedSd` parks in `generate` until the test opens
/// the gate, holding the request (and its job registration) open.
struct Gate {
    open: Mutex<bool>,
    wake: Condvar,
}

impl Gate {
    fn new() -> Self {
        Self {
            open: Mutex::new(false),
            wake: Condvar::new(),
        }
    }

    fn open(&self) {
        *self.open.lock().expect("gate lock") = true;
        self.wake.notify_all();
    }

    fn wait(&self) {
        let mut open = self.open.lock().expect("gate lock");
        while !*open {
            open = self.wake.wait(open).expect("gate wait");
        }
    }
}

struct GatedSd {
    entered: Arc<AtomicBool>,
    gate: Arc<Gate>,
}

impl honkbox_sd_core::SdPipeline for GatedSd {
    fn generate(
        &self,
        _req: &honkbox_sd_core::Txt2ImgRequest,
        _cancel: &AtomicBool,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        self.entered.store(true, Ordering::SeqCst);
        self.gate.wait();
        Ok(honkbox_sd_core::GeneratedImage {
            width: 1,
            height: 1,
            rgba: vec![0, 0, 0, 255],
        })
    }

    fn backend_id(&self) -> &'static str {
        "edits-gate-test"
    }
}

async fn wait_entered(entered: &AtomicBool) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("edit generation entered");
}

/// Inline delivery with a mask stays byte-compatible: the only response
/// keys are `created`/`data` and `data[0].b64_json` is a real PNG.
#[tokio::test]
async fn inline_delivery_with_mask_keeps_base64_response() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits_with_delivery(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            png_part("mask", png_bytes(4, 4)),
            part("prompt", "inpaint here"),
        ],
        Some("inline"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "{body}");
    assert_png(&data[0], 4, 4);
    let mut keys: Vec<_> = body
        .as_object()
        .expect("response object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["created", "data"], "{body}");
}

/// Artifact delivery with a mask: the returned `job_id` is live while
/// registered, and the one `job_result` matches the returned artifact.
#[tokio::test]
async fn artifact_delivery_with_mask_registers_live_job_and_emits_result() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir =
        std::env::temp_dir().join(format!("honkbox-edit-artifact-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let gate = Arc::new(Gate::new());
    let entered = Arc::new(AtomicBool::new(false));
    let app = app_with_sd(
        catalog_with(&[sd_entry("comfy:f.safetensors")]),
        Some(data_dir.clone()),
        Arc::new(GatedSd {
            entered: Arc::clone(&entered),
            gate: Arc::clone(&gate),
        }),
    );
    // Subscribe first: registration/result events must not be missed.
    let sse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sse.status(), StatusCode::OK);
    let mut stream = sse.into_body().into_data_stream();
    let parts = [
        png_part("image", png_bytes(4, 4)),
        png_part("mask", png_bytes(4, 4)),
        part("prompt", "inpaint here"),
    ];
    let request_app = app.clone();
    let request =
        tokio::spawn(
            async move { edits_with_delivery(request_app, &parts, Some("artifact")).await },
        );
    wait_entered(&entered).await;
    let live = jobs_list(app.clone()).await;
    let rows = live["data"].as_array().expect("jobs array");
    assert!(!rows.is_empty(), "no live jobs: {live}");
    gate.open();
    let (status, body) = request.await.expect("edit finished");
    assert_eq!(status, StatusCode::OK, "{body}");
    let job_id = body["job_id"].as_str().expect("job id").to_owned();
    let row = rows
        .iter()
        .find(|row| row["id"].as_str() == Some(job_id.as_str()))
        .unwrap_or_else(|| panic!("job {job_id} not live: {live}"));
    assert_eq!(row["interruptible"], false, "{row}");
    assert_eq!(body["data"].as_array().expect("data array").len(), 0);
    assert!(
        body["bundle"].as_str().is_some_and(|dir| !dir.is_empty()),
        "{body}"
    );
    let artifact = &body["artifacts"][0];
    assert_eq!(artifact["job_id"], job_id, "{body}");
    assert_eq!(artifact["content_type"], "image/png", "{body}");
    assert!(
        artifact["size_bytes"].as_u64().is_some_and(|n| n > 0),
        "{body}"
    );
    let frame = result_frame(&mut stream, &job_id).await;
    assert_eq!(frame["event"], "job_result", "{frame}");
    assert_eq!(frame["artifact_id"], artifact["artifact_id"], "{frame}");
    assert_eq!(frame["sha256"], artifact["artifact_id"], "{frame}");
    assert_eq!(frame["size_bytes"], artifact["size_bytes"], "{frame}");
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Inline delivery with a `LoRA` keeps the license warning and base64 body,
/// with no artifact fields added.
#[tokio::test]
async fn inline_delivery_with_lora_keeps_warning_and_base64() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry(
            "comfy:loras/detail.safetensors",
            "/models/loras/detail.safetensors",
        ),
    ]);
    let (status, body) = edits_with_delivery(
        app_with(catalog, None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            loras_part(&json!([
                {"slug": "comfy:loras/detail.safetensors", "strength": 0.8}
            ])),
        ],
        Some("inline"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_png(&body["data"][0], 4, 4);
    assert_eq!(body["warnings"].as_array().expect("warnings").len(), 1);
    let mut keys: Vec<_> = body
        .as_object()
        .expect("response object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["created", "data", "warnings"], "{body}");
}

/// Artifact delivery with a `LoRA` keeps the warning and emits the same
/// artifact id as the response.
#[tokio::test]
async fn artifact_delivery_with_lora_emits_matching_result() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir =
        std::env::temp_dir().join(format!("honkbox-edit-lora-artifact-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry(
            "comfy:loras/detail.safetensors",
            "/models/loras/detail.safetensors",
        ),
    ]);
    let app = app_with(catalog, Some(data_dir.clone()));
    let sse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut stream = sse.into_body().into_data_stream();
    let (status, body) = edits_with_delivery(
        app,
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
            loras_part(&json!([
                {"slug": "comfy:loras/detail.safetensors", "strength": 0.8}
            ])),
        ],
        Some("artifact"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["warnings"].as_array().expect("warnings").len(), 1);
    let job_id = body["job_id"].as_str().expect("job id").to_owned();
    let artifact = &body["artifacts"][0];
    assert_eq!(artifact["job_id"], job_id, "{body}");
    assert!(
        body["bundle"].as_str().is_some_and(|dir| !dir.is_empty()),
        "{body}"
    );
    let frame = result_frame(&mut stream, &job_id).await;
    assert_eq!(frame["artifact_id"], artifact["artifact_id"], "{frame}");
    assert_eq!(frame["sha256"], artifact["artifact_id"], "{frame}");
    assert_eq!(frame["artifact_name"], "final_00.png", "{frame}");
    assert_eq!(frame["artifact_kind"], "image", "{frame}");
    assert_eq!(frame["artifact_index"], 0, "{frame}");
    assert_eq!(frame["artifact_primary"], true, "{frame}");
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Unknown delivery values are a named 400 before any GPU work.
#[tokio::test]
async fn unknown_delivery_header_is_named_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = edits_with_delivery(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        &[
            png_part("image", png_bytes(4, 4)),
            part("prompt", "redo this"),
        ],
        Some("sideways"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_delivery", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("X-Honkbox-Delivery")),
        "{body}"
    );
}
