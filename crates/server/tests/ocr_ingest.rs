//! `POST /v1/ocr/ingest` contract tests: text-layer PDF, image OCR,
//! cache replay, mid-document cancel, and the named 400/413/501/503 gates.
//! Fixtures are generated in-test; no host paths, no external process.

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

use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::Engine as _;
use honkbox_api_types::{ModelCatalog, ModelCatalogEntry, OcrIngestMode, OcrResult};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use honkbox_recognition_worker::{
    RecEngine, RecError, RecKind, RecPage, RecRegion, RecognitionPipeline, RecognizeOutput,
    RecognizeRequest,
};
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// The admission ledger is process-global with one OCR slot; serialize the
/// HTTP tests so a sibling job cannot turn another test's 202 into a 429.
static OCR_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// A result file lands just before the job guard drops; wait for the
/// registry to clear so the next test starts with a free slot.
fn wait_idle() {
    let deadline = Instant::now() + Duration::from_secs(5);
    while jobs_registry::global()
        .list()
        .iter()
        .any(|job| job.kind == "ocr")
    {
        assert!(Instant::now() < deadline, "OCR job never left the registry");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Deterministic in-process OCR backend: the real `ocrs` weights are not
/// checked in, so the route/job/cache/cancel path is exercised against a
/// pipeline with the same honesty contract (boxes yes, confidence no).
#[derive(Debug, Default)]
struct FakeOcr {
    cancelled: Arc<AtomicBool>,
}

impl RecognitionPipeline for FakeOcr {
    fn backend_id(&self) -> &'static str {
        "fake-ocr-test"
    }

    fn kinds(&self) -> &'static [RecKind] {
        &[RecKind::Ocr]
    }

    fn serves(&self, kind: RecKind) -> bool {
        kind == RecKind::Ocr
    }

    fn recognize(&self, req: &RecognizeRequest) -> Result<RecognizeOutput, RecError> {
        if req.image.is_empty() {
            return Err(RecError::Invalid("empty image".to_owned()));
        }
        Ok(RecognizeOutput {
            kind: RecKind::Ocr,
            engine: RecEngine {
                id: self.backend_id(),
                version: None,
                bbox_support: true,
            },
            pages: vec![RecPage {
                text: "FAKE OCR LINE".to_owned(),
                regions: vec![RecRegion {
                    bbox: [1.0, 2.0, 3.0, 4.0],
                    text: "FAKE OCR LINE".to_owned(),
                    confidence: None,
                }],
            }],
        })
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

fn temp_data_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-ocr-ingest-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp data dir");
    dir
}

fn base_config(data_dir: Option<PathBuf>, ocr: bool) -> AppConfig {
    let recognition: Arc<dyn RecognitionPipeline> = if ocr {
        Arc::new(FakeOcr::default())
    } else {
        Arc::new(honkbox_recognition_worker::StubRecognition)
    };
    AppConfig {
        recognition,
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(std::sync::RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: Vec::<ModelCatalogEntry>::new(),
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
        sd_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        artifacts: Arc::new(artifacts::Registry::new(
            None,
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
    }
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr/ingest")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("infallible");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn error_code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or_default()
}

/// Minimal one-page-per-string PDF with a real text layer. Hand-built so
/// the test never depends on a checked-in fixture or a foreign tool.
fn pdf_data_uri(pages: &[&str]) -> String {
    fn add_obj(out: &mut Vec<u8>, offsets: &mut Vec<usize>, num: usize, body: &str) {
        offsets.push(out.len());
        out.extend_from_slice(format!("{num} 0 obj\n{body}\nendobj\n").as_bytes());
    }
    let n = pages.len();
    let font_num = 3 + 2 * n;
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    add_obj(
        &mut out,
        &mut offsets,
        1,
        "<< /Type /Catalog /Pages 2 0 R >>",
    );
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + 2 * i)).collect();
    add_obj(
        &mut out,
        &mut offsets,
        2,
        &format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), n),
    );
    for (i, text) in pages.iter().enumerate() {
        let page_num = 3 + 2 * i;
        let content_num = page_num + 1;
        let escaped = text
            .replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)");
        let stream = format!("BT /F1 12 Tf 72 700 Td ({escaped}) Tj ET\n");
        add_obj(
            &mut out,
            &mut offsets,
            page_num,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_num} 0 R >> >> /Contents {content_num} 0 R >>"
            ),
        );
        add_obj(
            &mut out,
            &mut offsets,
            content_num,
            &format!(
                "<< /Length {} >>\nstream\n{}endstream",
                stream.len(),
                stream
            ),
        );
    }
    add_obj(
        &mut out,
        &mut offsets,
        font_num,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    );
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", font_num + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for offset in offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            font_num + 1,
            xref
        )
        .as_bytes(),
    );
    format!(
        "data:application/pdf;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(out)
    )
}

fn png_data_uri(fill: u8) -> String {
    let image = image::RgbImage::from_fn(8, 8, |_, _| image::Rgb([fill, fill, fill]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("encode png");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn bundles(data_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(data_dir.join("jobs")) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.join("ocr_result.json").exists())
        .collect();
    dirs.sort();
    dirs
}

fn wait_for_result(data_dir: &Path, before: &HashSet<PathBuf>) -> OcrResult {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(dir) = bundles(data_dir)
            .into_iter()
            .find(|dir| !before.contains(dir))
        {
            let bytes = std::fs::read(dir.join("ocr_result.json")).expect("read result");
            return serde_json::from_slice(&bytes).expect("parse result");
        }
        assert!(Instant::now() < deadline, "OCR job result never landed");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn request(uri: &str, mode: &str) -> Value {
    json!({"data_uri": uri, "mode": mode})
}

#[tokio::test]
async fn text_layer_pdf_is_extracted_without_boxes_or_confidence() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let dir = temp_data_dir("text-layer");
    let app = app(base_config(Some(dir.clone()), true));
    let uri = pdf_data_uri(&["Hello OCR Page"]);
    let (status, body) = post(&app, request(&uri, "text_layer")).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["status"], "queued");
    let result = wait_for_result(&dir, &HashSet::new());
    assert_eq!(result.page_count, 1);
    assert!(result.pages[0].text.contains("Hello OCR Page"));
    assert_eq!(result.pages[0].regions[0].bbox, None);
    assert_eq!(result.pages[0].regions[0].confidence, None);
    assert!(!result.engine.confidence_support);
    assert!(!result.engine.bbox_support);
    assert!(!result.cache_hit);
    assert_eq!(result.mode, OcrIngestMode::TextLayer);
}

#[tokio::test]
async fn image_ocr_happy_path_uses_real_boxes_and_no_confidence() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let dir = temp_data_dir("image");
    let app = app(base_config(Some(dir.clone()), true));
    let (status, body) = post(&app, request(&png_data_uri(7), "full_page")).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let result = wait_for_result(&dir, &HashSet::new());
    assert_eq!(result.page_count, 1);
    assert_eq!(result.pages[0].regions[0].bbox, Some([1.0, 2.0, 3.0, 4.0]));
    assert_eq!(result.pages[0].regions[0].confidence, None);
    assert!(result.engine.bbox_support);
    assert!(!result.engine.confidence_support);
    assert!(!result.cache_hit);
}

#[tokio::test]
async fn cache_hit_replays_the_result_without_a_second_run() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let dir = temp_data_dir("cache");
    let app = app(base_config(Some(dir.clone()), true));
    let body = request(&png_data_uri(31), "layout_regions");
    assert_eq!(post(&app, body.clone()).await.0, StatusCode::ACCEPTED);
    let first = wait_for_result(&dir, &HashSet::new());
    assert!(!first.cache_hit);
    let before: HashSet<PathBuf> = bundles(&dir).into_iter().collect();
    assert_eq!(post(&app, body).await.0, StatusCode::ACCEPTED);
    let replay = wait_for_result(&dir, &before);
    assert!(replay.cache_hit);
    assert_eq!(replay.pages, first.pages);
    assert_eq!(replay.engine, first.engine);
}

#[test]
fn cancel_stops_between_pages_and_reports_partial_progress() {
    let texts = vec!["p0".to_owned(), "p1".to_owned(), "p2".to_owned()];
    let (pages, cancelled) = routes::ocr_ingest::text_pages(texts, |i| i >= 2);
    assert!(cancelled);
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].text, "p0");
    assert_eq!(pages[1].index, 1);
    assert_eq!(pages[0].regions[0].text, "p0");
}

#[tokio::test]
async fn missing_weights_is_named_501_and_matches_capabilities() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let dir = temp_data_dir("no-weights");
    let app = app(base_config(Some(dir), false));
    let (status, body) = post(&app, request(&png_data_uri(11), "full_page")).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(error_code(&body), "ocr_engine_unavailable");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let caps: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(caps["recognition"]["ocr"], false);
}

#[tokio::test]
async fn pdf_raster_modes_are_named_501() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let dir = temp_data_dir("raster");
    let app = app(base_config(Some(dir), true));
    for mode in ["layout_regions", "full_page"] {
        let (status, body) = post(&app, request(&pdf_data_uri(&["x"]), mode)).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(error_code(&body), "ocr_pdf_raster_unavailable");
    }
}

#[tokio::test]
async fn path_free_and_bounded_inputs_are_named_errors() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let dir = temp_data_dir("errors");
    let app = app(base_config(Some(dir.clone()), true));
    let cases = vec![
        (
            json!({"data_uri": "/tmp/secret.png", "mode": "full_page"}),
            StatusCode::BAD_REQUEST,
            "invalid_ocr_input",
        ),
        (
            json!({"data_uri": "data:image/png;base64,!!!", "mode": "full_page"}),
            StatusCode::BAD_REQUEST,
            "invalid_ocr_input",
        ),
        (
            json!({"data_uri": png_data_uri(3), "mode": "sideways"}),
            StatusCode::BAD_REQUEST,
            "invalid_ocr_ingest_request",
        ),
        (
            json!({"data_uri": png_data_uri(3), "mode": "full_page", "dpi": 10}),
            StatusCode::BAD_REQUEST,
            "ocr_invalid_dpi",
        ),
        (
            json!({"data_uri": format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(vec![0u8; 16 * 1024 * 1024 + 1])), "mode": "full_page"}),
            StatusCode::PAYLOAD_TOO_LARGE,
            "ocr_input_too_large",
        ),
    ];
    for (body, status, code) in cases {
        let (got_status, got_body) = post(&app, body).await;
        assert_eq!(got_status, status, "{got_body}");
        assert_eq!(error_code(&got_body), code, "{got_body}");
    }
}

#[tokio::test]
async fn absent_data_dir_is_named_503() {
    let _lock = OCR_LOCK.lock().await;
    wait_idle();
    let app = app(base_config(None, true));
    let (status, body) = post(&app, request(&png_data_uri(5), "full_page")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error_code(&body), "ocr_storage_unavailable");
}
