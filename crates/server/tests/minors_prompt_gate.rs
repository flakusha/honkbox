//! Focused safety gate tests: the shared minors-facing prompt refusal on both
//! image routes, and the shared NSFW decision path (flag/block/off/501) with
//! its decision recorded in bundle sidecars and job-result frames.
//! Stub pipelines only, per-test temp data dirs, no ports or model downloads.

#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "../src/apps.rs"]
mod apps;
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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    response::Response,
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Pipeline that delegates to the stub but counts admissions, so a refused
/// prompt can be proven never to have reached generation.
struct CountingSd {
    calls: Arc<AtomicUsize>,
}

impl honkbox_sd_core::SdPipeline for CountingSd {
    fn generate(
        &self,
        req: &honkbox_sd_core::Txt2ImgRequest,
        cancel: &AtomicBool,
    ) -> Result<honkbox_sd_core::GeneratedImage, honkbox_sd_core::SdError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        honkbox_sd_core::StubPipeline.generate(req, cancel)
    }

    fn backend_id(&self) -> &'static str {
        "stub"
    }
}

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

fn catalog() -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: vec![sd_entry("comfy:checkpoints/fake.safetensors")],
    }
}

#[allow(clippy::needless_pass_by_value)]
fn app_with_sd(data_dir: Option<PathBuf>, sd: Arc<dyn honkbox_sd_core::SdPipeline>) -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog()))),
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
        artifacts: Arc::new(crate::artifacts::Registry::new(
            data_dir,
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// Fresh data dir per test; removed even on failure.
struct TempData(PathBuf);

impl TempData {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("honkbox-minors-gate-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&path).expect("temp data dir");
        Self(path)
    }
}

impl Drop for TempData {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Scope one NSFW policy to a test. Every image-route test takes a guard, so
/// policy-sensitive cases cannot race with benign controls.
fn facts(
    enabled: bool,
    policy: &str,
    threshold: f64,
    bin: Option<PathBuf>,
) -> routes::nsfw::test_support::Scoped {
    routes::nsfw::test_support::scoped(enabled, policy, threshold, bin)
}

/// Executable classifier fixture: the server invokes it as `bin <image>` and
/// it prints one bounded JSON object. Never a request-supplied threshold.
#[cfg(unix)]
fn classifier(dir: &Path, score: f64, classes: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let bin = dir.join(format!("nsfw-classifier-{}", ulid::Ulid::new()));
    let script = format!("#!/bin/sh\necho '{{\"score\":{score},\"classes\":{classes}}}'\n");
    std::fs::write(&bin, script).expect("classifier script");
    let mut perms = std::fs::metadata(&bin)
        .expect("classifier metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&bin, perms).expect("classifier executable");
    bin
}

async fn json_body(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .expect("body bytes");
    let text = String::from_utf8(bytes.to_vec()).expect("utf8 body");
    let parsed = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("response not JSON: {err}: {text}"));
    (status, parsed)
}

async fn generate(app: Router, prompt: &str, delivery: Option<&str>) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/images/generations")
        .header("content-type", "application/json");
    if let Some(delivery) = delivery {
        builder = builder.header("X-Honkbox-Delivery", delivery);
    }
    let response = app
        .oneshot(
            builder
                .body(Body::from(json!({ "prompt": prompt }).to_string()))
                .expect("request"),
        )
        .await
        .expect("oneshot");
    json_body(response).await
}

/// One multipart field (file or text).
struct Part {
    name: &'static str,
    filename: Option<&'static str>,
    content_type: Option<&'static str>,
    data: Vec<u8>,
}

fn text_part(name: &'static str, value: &str) -> Part {
    Part {
        name,
        filename: None,
        content_type: None,
        data: value.as_bytes().to_vec(),
    }
}

fn png_part(name: &'static str, bytes: Vec<u8>) -> Part {
    Part {
        name,
        filename: Some("input.png"),
        content_type: Some("image/png"),
        data: bytes,
    }
}

fn png_bytes() -> Vec<u8> {
    honkbox_sd_core::encode_image(
        honkbox_sd_core::GeneratedImage {
            width: 4,
            height: 4,
            rgba: vec![255; 4 * 4 * 4],
        },
        honkbox_sd_core::OutputFormat::Png,
    )
    .expect("png encodes")
}

fn multipart_body(parts: &[Part]) -> (String, Vec<u8>) {
    let boundary = "honkbox-minors-gate";
    let mut body = Vec::new();
    for part in parts {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\ncontent-disposition: form-data; name=\"{}\"",
                part.name
            )
            .as_bytes(),
        );
        if let Some(filename) = part.filename {
            body.extend_from_slice(format!("; filename=\"{filename}\"\r\n").as_bytes());
        } else {
            body.extend_from_slice(b"\r\n");
        }
        if let Some(content_type) = part.content_type {
            body.extend_from_slice(format!("content-type: {content_type}\r\n").as_bytes());
        }
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(&part.data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn edits(app: Router, prompt: &str, delivery: Option<&str>) -> (StatusCode, Value) {
    let parts = [png_part("image", png_bytes()), text_part("prompt", prompt)];
    let (content_type, body) = multipart_body(&parts);
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/images/edits")
        .header("content-type", content_type);
    if let Some(delivery) = delivery {
        builder = builder.header("X-Honkbox-Delivery", delivery);
    }
    let response = app
        .oneshot(builder.body(Body::from(body)).expect("request"))
        .await
        .expect("oneshot");
    json_body(response).await
}

fn assert_refused(status: StatusCode, body: &Value) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"],
        json!("minors_facing_prompt_refused"),
        "{body}"
    );
}

fn assert_blocked(status: StatusCode, body: &Value) {
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], json!("nsfw_blocked"), "{body}");
}

fn assert_succeeded(status: StatusCode, body: &Value) {
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().map(Vec::len), Some(1), "{body}");
}

/// Artifact mode answers descriptors, not inline `data`.
fn assert_artifact_succeeded(status: StatusCode, body: &Value) {
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["artifacts"].as_array().map(Vec::len),
        Some(1),
        "{body}"
    );
}

/// Every bundle written by the flag-policy calls carries the decision.
fn bundle_annotations(data_dir: &Path) -> Vec<Value> {
    let jobs_dir = data_dir.join("jobs");
    let mut annotations = Vec::new();
    for entry in std::fs::read_dir(&jobs_dir).expect("jobs dir") {
        let sidecar = entry.expect("job dir").path().join("final_00.json");
        let parsed: Value =
            serde_json::from_slice(&std::fs::read(sidecar).expect("bundle sidecar"))
                .expect("sidecar JSON");
        annotations.push(parsed["annotations"].clone());
    }
    annotations
}

/// Read SSE frames until an annotation-bearing `job_result` for `job_id`.
async fn annotation_frame(stream: &mut axum::body::BodyDataStream, job_id: &str) -> Value {
    let mut buffer = String::new();
    loop {
        let chunk = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("sse frame within 5s")
            .expect("stream alive")
            .expect("frame bytes");
        buffer.push_str(std::str::from_utf8(&chunk).expect("utf8 frame"));
        while let Some(pos) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..pos + 2).collect();
            let Some(data) = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .next_back()
            else {
                continue;
            };
            let parsed: Value = serde_json::from_str(data).expect("frame JSON");
            if frame.contains("event: job_result")
                && parsed["id"].as_str() == Some(job_id)
                && parsed["annotations"].is_array()
            {
                return parsed;
            }
        }
    }
}

#[tokio::test]
async fn minors_prompt_refused_on_both_routes_before_pipeline() {
    let _slot = SLOT_LOCK.lock().await;
    let _facts = facts(false, "flag", 0.8, None);
    let temp = TempData::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let app = app_with_sd(
        Some(temp.0.clone()),
        Arc::new(CountingSd {
            calls: Arc::clone(&calls),
        }),
    );
    for prompt in ["CHILD PORN", "UnderAge model", "csam", "an under-aged girl"] {
        let (status, body) = generate(app.clone(), prompt, None).await;
        assert_refused(status, &body);
        let (status, body) = edits(app.clone(), prompt, None).await;
        assert_refused(status, &body);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0, "pipeline must never run");
}

#[tokio::test]
async fn benign_control_generates_on_both_routes() {
    let _slot = SLOT_LOCK.lock().await;
    let _facts = facts(false, "flag", 0.8, None);
    let temp = TempData::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let app = app_with_sd(
        Some(temp.0.clone()),
        Arc::new(CountingSd {
            calls: Arc::clone(&calls),
        }),
    );
    // "child" as a plain word is NOT in the blocklist: whole-token phrases
    // keep this benign control working.
    let prompt = "a happy child playing in the park, childhood memories";
    let (status, body) = generate(app.clone(), prompt, None).await;
    assert_succeeded(status, &body);
    let (status, body) = edits(app.clone(), prompt, None).await;
    assert_succeeded(status, &body);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[cfg(unix)]
#[tokio::test]
async fn nsfw_flag_and_block_decisions_are_reported_on_both_routes() {
    let _slot = SLOT_LOCK.lock().await;
    let temp = TempData::new();
    let high = classifier(&temp.0, 0.95, "[\"nsfw\"]");
    let low = classifier(&temp.0, 0.1, "[]");
    let app = app_with_sd(
        Some(temp.0.clone()),
        Arc::new(honkbox_sd_core::StubPipeline),
    );
    {
        let _facts = facts(true, "flag", 0.8, Some(high.clone()));
        let (status, body) = generate(app.clone(), "benign prompt", None).await;
        assert_succeeded(status, &body);
        let (status, body) = edits(app.clone(), "benign prompt", None).await;
        assert_succeeded(status, &body);
    }
    let bundles = bundle_annotations(&temp.0);
    assert_eq!(bundles.len(), 2, "one bundle per inline image route");
    for annotations in &bundles {
        let annotations = annotations.as_array().expect("annotation array");
        assert_eq!(annotations.len(), 1, "{annotations:?}");
        assert_eq!(annotations[0]["name"], json!("nsfw"));
        assert_eq!(annotations[0]["value"]["score"], json!(0.95));
        assert_eq!(
            annotations[0]["value"]["model_version"],
            json!("test-nsfw-v1")
        );
    }
    {
        let _facts = facts(true, "block", 0.8, Some(high));
        // Unique prompts: inline responses with an identical request are
        // replayed from the generation cache, bypassing classification.
        let (status, body) = generate(app.clone(), "block-policy prompt", None).await;
        assert_blocked(status, &body);
        let (status, body) = edits(app.clone(), "block-policy prompt", None).await;
        assert_blocked(status, &body);
    }
    {
        let _facts = facts(true, "flag", 0.8, Some(low));
        let (status, body) = generate(app.clone(), "low-score prompt", None).await;
        assert_succeeded(status, &body);
        let (status, body) = edits(app, "low-score prompt", None).await;
        assert_succeeded(status, &body);
    }
}

#[tokio::test]
async fn nsfw_off_disabled_and_misconfigured_are_honest() {
    let _slot = SLOT_LOCK.lock().await;
    let temp = TempData::new();
    let app = app_with_sd(
        Some(temp.0.clone()),
        Arc::new(honkbox_sd_core::StubPipeline),
    );
    {
        // Engine disabled: no classifier is needed and no score is faked.
        let _facts = facts(false, "flag", 0.8, None);
        let (status, body) = generate(app.clone(), "disabled-engine prompt", None).await;
        assert_succeeded(status, &body);
        let (status, body) = edits(app.clone(), "disabled-engine prompt", None).await;
        assert_succeeded(status, &body);
    }
    {
        // Policy off: classification is skipped even with a binary present.
        let _facts = facts(true, "off", 0.8, None);
        let (status, body) = generate(app.clone(), "off-policy prompt", None).await;
        assert_succeeded(status, &body);
        let (status, body) = edits(app.clone(), "off-policy prompt", None).await;
        assert_succeeded(status, &body);
    }
    let no_bundles = bundle_annotations(&temp.0)
        .iter()
        .all(|annotations| annotations.as_array().is_some_and(Vec::is_empty));
    assert!(no_bundles, "off/disabled must not fabricate a score");
    {
        // Enabled policy with no binary is a named 501, never a zero score.
        let _facts = facts(true, "flag", 0.8, None);
        let (status, body) = generate(app.clone(), "no-classifier prompt", None).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
        assert_eq!(
            body["error"]["code"],
            json!("nsfw_classifier_unavailable"),
            "{body}"
        );
        let (status, body) = edits(app, "no-classifier prompt", None).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
        assert_eq!(
            body["error"]["code"],
            json!("nsfw_classifier_unavailable"),
            "{body}"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn nsfw_decision_reaches_job_result_on_both_routes() {
    let _slot = SLOT_LOCK.lock().await;
    let temp = TempData::new();
    let high = classifier(&temp.0, 0.95, "[\"nsfw\"]");
    let _facts = facts(true, "flag", 0.8, Some(high));
    let app = app_with_sd(
        Some(temp.0.clone()),
        Arc::new(honkbox_sd_core::StubPipeline),
    );
    let sse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/events")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("oneshot");
    assert_eq!(sse.status(), StatusCode::OK);
    let mut stream = sse.into_body().into_data_stream();
    let (status, body) = generate(app.clone(), "benign prompt", Some("artifact")).await;
    assert_artifact_succeeded(status, &body);
    let gen_job = body["job_id"]
        .as_str()
        .expect("generation job id")
        .to_owned();
    let (status, body) = edits(app, "benign prompt", Some("artifact")).await;
    assert_artifact_succeeded(status, &body);
    let edit_job = body["job_id"].as_str().expect("edit job id").to_owned();
    for job_id in [gen_job, edit_job] {
        let frame = annotation_frame(&mut stream, &job_id).await;
        let annotations = frame["annotations"].as_array().expect("annotations");
        assert_eq!(annotations.len(), 1, "{frame}");
        assert_eq!(annotations[0]["score"], json!(0.95));
        assert_eq!(annotations[0]["model_version"], json!("test-nsfw-v1"));
    }
}
