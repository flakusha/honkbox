//! `POST /v1/recognize` + `features.recognition` contract tests (recognition
//! stage 1: TASK-recognition-provider-trait + TASK-barcode-decode). The
//! route is synchronous in stage 1 — barcode-class latency needs no job
//! registry entry; the OCR pipeline becomes job-based when FEAT-ocr-ingest
//! lands. Honesty is the contract: the capabilities flag, the route's 501
//! gate, and the compiled backend set are the SAME source (`kinds()`), and
//! QR round-trip proves the backend end to end on a generated fixture.

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

use std::sync::{Arc, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::Engine as _;
use honkbox_api_types::{ModelCatalog, ModelCatalogEntry};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use rxing::{BarcodeFormat, Writer};
use serde_json::{Value, json};
use std::io::Cursor;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

fn base_config() -> AppConfig {
    AppConfig {
        recognition: if honkbox_recognition_worker::BARCODE_COMPILED {
            Arc::new(honkbox_recognition_worker::BarcodeRecognition)
                as Arc<dyn honkbox_recognition_worker::RecognitionPipeline>
        } else {
            Arc::new(honkbox_recognition_worker::StubRecognition)
                as Arc<dyn honkbox_recognition_worker::RecognitionPipeline>
        },
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: Vec::<ModelCatalogEntry>::new(),
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
        sd_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        artifacts: Arc::new(artifacts::Registry::new(
            None,
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
    }
}

use std::time::Duration;

fn app_with() -> Router {
    app(base_config())
}

/// Post a JSON body and return (status, parsed body).
async fn post_json(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("infallible");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

/// Error envelope shape helper: `{error: {message, code, type}}`-style —
/// the code field must name the branch so clients branch on it, not prose.
fn error_code(body: &Value) -> String {
    body["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// Encode `contents` as a QR bit matrix and rasterize to PNG bytes (rxing
/// encodes; the workspace `image` rasterizes — no fixture file, and the
/// bytes exercise the full route decode path).
fn qr_data_uri(contents: &str) -> String {
    let matrix = rxing::MultiFormatWriter
        .encode(contents, &BarcodeFormat::QR_CODE, 0, 0)
        .expect("encode QR");
    let img = image::GrayImage::from_fn(matrix.width(), matrix.height(), |x, y| {
        if matrix.get(x, y) {
            image::Luma([0u8])
        } else {
            image::Luma([255u8])
        }
    });
    let mut png = Vec::new();
    image::DynamicImage::ImageLuma8(img)
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode PNG");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

/// QR round trip through the HTTP surface: payload, rxing engine pin,
/// corner-derived bbox, `confidence` absent (rxing has none).
#[tokio::test]
async fn barcode_round_trip_recovers_payload() {
    let payload = "https://honkbox.example/contract-test";
    let (status, body) = post_json(
        &app_with(),
        "/v1/recognize",
        json!({"kind": "barcode", "image": qr_data_uri(payload)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["kind"], "barcode");
    assert_eq!(body["engine"]["id"], "rxing");
    assert_eq!(body["engine"]["version"], "0.9");
    assert_eq!(body["engine"]["bbox_support"], true);
    assert_eq!(body["pages"][0]["text"], payload);
    assert_eq!(body["pages"][0]["regions"][0]["text"], payload);
    assert!(body["pages"][0]["regions"][0].get("confidence").is_none());
    let bbox = body["pages"][0]["regions"][0]["bbox"]
        .as_array()
        .expect("bbox");
    let x_min = bbox[0].as_f64().expect("x_min");
    let y_min = bbox[1].as_f64().expect("y_min");
    let x_max = bbox[2].as_f64().expect("x_max");
    let y_max = bbox[3].as_f64().expect("y_max");
    assert!(x_min < x_max && y_min < y_max, "bbox not a box: {bbox:?}");
}

/// A valid container with no barcode is the honest empty result (200,
/// `pages: []`) — never a 500.
#[tokio::test]
async fn no_barcode_is_a_clean_empty_result() {
    let img = image::GrayImage::from_pixel(64, 64, image::Luma([255]));
    let mut png = Vec::new();
    image::DynamicImage::ImageLuma8(img)
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode PNG");
    let (status, body) = post_json(
        &app_with(),
        "/v1/recognize",
        json!({"kind": "barcode", "image": format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png)
        )}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["pages"].as_array().expect("pages").len(), 0);
}

/// Named 400s: unknown kind is a serde-typed rejection; non-data-URI input,
/// undeclared base64, and empty payload each name their branch.
#[tokio::test]
async fn invalid_requests_carry_named_codes() {
    let app = app_with();
    // Unknown kind → typed enum parse rejection (named 400).
    let (status, body) = post_json(
        &app,
        "/v1/recognize",
        json!({"kind": "telepathy", "image": "data:image/png;base64,aGk="}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_recognize_request");
    // Unknown field → same rejection class (deny_unknown_fields).
    let (status, body) = post_json(
        &app,
        "/v1/recognize",
        json!({"kind": "barcode", "image": "data:image/png;base64,aGk=", "wat": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_recognize_request");
    // Non-data URI (a URL is NEVER fetched — no SSRF surface).
    let (status, body) = post_json(
        &app,
        "/v1/recognize",
        json!({"kind": "barcode", "image": "https://example.invalid/qr.png"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_recognize_input");
    // Undeclared base64.
    let (status, body) = post_json(
        &app,
        "/v1/recognize",
        json!({"kind": "barcode", "image": "data:image/png,not-base64-plain"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_recognize_input");
    // Unsupported container MIME.
    let (status, body) = post_json(
        &app,
        "/v1/recognize",
        json!({"kind": "barcode", "image": "data:image/bmp;base64,aGk="}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_recognize_input");
}

/// Garbage-in-container is a named 400, not a misleading "no barcode".
#[tokio::test]
async fn undecodable_container_is_invalid_input() {
    let (status, body) = post_json(
        &app_with(),
        "/v1/recognize",
        json!({"kind": "barcode", "image": format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode([0u8; 64])
        )}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "invalid_recognize_input");
}

/// OCR honesty: without a compiled OCR backend the request is the named 501
/// naming the serving backend — and the capabilities flag mirrors the same
/// `kinds()` source, so a client can trust either direction.
#[tokio::test]
async fn unserved_kind_is_named_501_matching_capabilities() {
    let app = app_with();
    let (status, body) = post_json(
        &app,
        "/v1/recognize",
        json!({"kind": "ocr", "image": "data:image/png;base64,aGk="}),
    )
    .await;
    if honkbox_recognition_worker::BARCODE_COMPILED {
        // rxing serves only barcode: OCR is the named 501.
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "body: {body}");
        assert_eq!(error_code(&body), "recognition_kind_unsupported");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("ocr")),
            "501 must name the kind: {body}"
        );
    }
    // The capabilities document agrees with the route's gate either way
    // (`recognition` sits at the document's top level — additive optional
    // block per the API version policy, like `image_ops`/`tools`).
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("infallible");
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let caps: Value = serde_json::from_slice(&bytes).unwrap();
    let recognition = &caps["recognition"];
    let barcode_flag = recognition["barcode"].as_bool().expect("barcode flag");
    let ocr_flag = recognition["ocr"].as_bool().expect("ocr flag");
    // Probe agreement: barcode flag == BARCODE_COMPILED (the route served
    // the QR round trip above under the same gate).
    assert_eq!(barcode_flag, honkbox_recognition_worker::BARCODE_COMPILED);
    // OCR honesty: the 501 above corresponds to a `false` flag (no OCR
    // backend exists yet — any `true` would be a half seam).
    assert!(!ocr_flag);
}
