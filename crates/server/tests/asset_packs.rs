//! QAL pack guard + CC0 pack fetch (TASK-qal-guard, TASK-cc0-pack-fetch).
//! Local `TcpListener` fixture only: no external network, per-test temp dirs.

// No lib target on this binary crate, so include the modules directly.
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

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use routes::AppConfig;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt as _;

use catalog::asset_packs as packs;
use catalog::asset_packs::{FetchError, FetchJob};

// ---------------------------------------------------------------- fixture

/// Local one-shot HTTP fixture; `hits` counts GETs (zero on a refusal).
struct MockServer {
    base: String,
}

impl MockServer {
    fn new(body: Vec<u8>, hits: Option<Arc<AtomicUsize>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let base = format!("http://{}", listener.local_addr().expect("mock addr"));
        let body = Arc::new(body);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let body = Arc::clone(&body);
                let hits = hits.clone();
                std::thread::spawn(move || serve(stream, &body, hits.as_deref()));
            }
        });
        Self { base }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base, path.trim_start_matches('/'))
    }
}

fn serve(mut stream: TcpStream, body: &[u8], hits: Option<&AtomicUsize>) {
    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match stream.read(&mut byte) {
            Ok(0) | Err(_) => return,
            Ok(_) => {
                raw.push(byte[0]);
                if raw.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
        }
        if raw.len() > 64 * 1024 {
            return;
        }
    }
    if let Some(hits) = hits {
        hits.fetch_add(1, Ordering::SeqCst);
    }
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = stream.write_all(body);
}

/// Unique per-test temp dir under the system temp dir.
fn temp_root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "honkbox-packs-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp root");
    dir
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn app_config(root: &Path) -> AppConfig {
    AppConfig {
        auth: routes::AuthConfig::default(),
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(std::sync::RwLock::new(Arc::new(
            honkbox_api_types::ModelCatalog {
                object: "list".to_owned(),
                data: Vec::new(),
            },
        ))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(honkbox_llm_worker::StubLlm::default()),
        stt: Arc::new(honkbox_audio_worker::StubStt),
        tts: Arc::new(honkbox_audio_worker::StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: Some(root.to_path_buf()),
        artifacts: Arc::new(artifacts::Registry::new(
            Some(root.to_path_buf()),
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("manager")),
    }
}

async fn post_json(app: &Router, body: Value) -> Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/asset-packs/fetch")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("oneshot")
}

async fn body_json(response: Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json body")
}

fn bundle_inputs() -> jobs::BundleInputs {
    jobs::BundleInputs {
        model_slug: "comfy:checkpoints/x.safetensors".to_string(),
        prompt: "a red fox".to_string(),
        seed: 42,
        steps: 20,
        width: 16,
        height: 16,
        format: honkbox_sd_core::OutputFormat::Png,
        backend_id: "stub",
        model_sha256: None,
        engine_version: None,
        sampler: "euler".to_string(),
        scheduler: "normal".to_string(),
        cfg_scale: 7.0,
        negative_prompt: None,
        loras: Vec::new(),
        strength: None,
        recipe: None,
        hires: None,
        annotations: Vec::new(),
    }
}

// ---------------------------------------------------------------- tests

/// QAL packs are refused at enqueue with the named 412 before any bytes.
#[tokio::test]
async fn qal_pack_refused_at_enqueue_with_named_error() {
    let root = temp_root("qal-enqueue");
    let app = routes::app(app_config(&root));
    let response = post_json(
        &app,
        json!({
            "pack_id": "quaternius-ultimate-modular-nature-kit",
            "source_url": "http://127.0.0.1:9/never.zip",
            "artifact_id": "qal-refusal"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::PRECONDITION_FAILED);
    let body = body_json(response).await;
    assert_eq!(body["error"]["code"], "pack_license_restricted");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("quaternius-ultimate-modular-nature-kit"));
    assert!(message.contains("no redistribution as assets"));
    assert!(!root.join("asset-packs").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// CC0 fetch records a locally computed sha + curated license row, and the
/// row is consumable by both bundle manifest files.
#[tokio::test]
async fn cc0_fetch_records_sha_and_license_row() {
    let root = temp_root("cc0-fetch");
    let bytes = b"polyhaven fixture bytes v1".to_vec();
    let expected = sha256_hex(&bytes);
    let server = MockServer::new(bytes.clone(), None);
    let app = routes::app(app_config(&root));
    let response = post_json(
        &app,
        json!({
            "pack_id": "polyhaven-rocks",
            "source_url": server.url("/rocks.zip"),
            "artifact_id": "attribution-fixture"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let record = body_json(response).await;
    assert_eq!(record["pack_id"], "polyhaven-rocks");
    assert_eq!(record["license"], "CC0-1.0");
    assert_eq!(record["sha256"], expected);
    assert_eq!(record["bytes"], bytes.len() as u64);
    assert_eq!(record["source_url"], "https://polyhaven.com/a/rocks");
    assert_eq!(record["fetch_url"], server.url("/rocks.zip"));
    assert_eq!(record["attribution_text"], "Poly Haven — CC0-1.0");

    let packs_dir = root.join("asset-packs");
    assert_eq!(
        std::fs::read(packs_dir.join("attribution-fixture.bin")).expect("bytes"),
        bytes
    );
    let row: Value = serde_json::from_slice(
        &std::fs::read(packs_dir.join("attribution-fixture.license.json")).expect("row"),
    )
    .expect("row json");
    assert_eq!(row["sha256"], expected);
    assert_eq!(row["license"], "CC0-1.0");

    let dir = jobs::write_bundle(&root, &bundle_inputs(), b"image", 0).expect("bundle");
    let sidecar: Value =
        serde_json::from_slice(&std::fs::read(dir.join("final_00.json")).expect("sidecar"))
            .expect("sidecar json");
    assert_eq!(sidecar["pack_licenses"][0]["sha256"], expected);
    assert_eq!(sidecar["pack_licenses"][0]["license"], "CC0-1.0");
    let request: Value =
        serde_json::from_slice(&std::fs::read(dir.join("request.json")).expect("request"))
            .expect("request json");
    assert_eq!(request["pack_licenses"][0]["pack_id"], "polyhaven-rocks");
    std::fs::remove_dir_all(&root).ok();
}

/// A refused fetch never requests bytes and never leaves a `.part`.
#[tokio::test]
async fn refused_qal_fetch_leaves_no_partial_bytes() {
    let root = temp_root("qal-nobytes");
    let hits = Arc::new(AtomicUsize::new(0));
    let server = MockServer::new(b"qal bytes".to_vec(), Some(Arc::clone(&hits)));
    let job = FetchJob {
        pack_id: "quaternius-ultimate-space-kit".to_owned(),
        source_url: server.url("/kit.zip"),
        artifact_id: "qal".to_owned(),
        root: root.clone(),
    };
    match packs::fetch(&job).await.expect_err("QAL must refuse") {
        FetchError::Refused(refusal) => {
            assert_eq!(refusal.code(), "pack_license_restricted");
            assert!(refusal.message().contains("quaternius-ultimate-space-kit"));
            assert!(refusal.message().contains("no redistribution as assets"));
        }
        other => panic!("wrong error: {other}"),
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    assert!(!root.join("asset-packs").exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Filename fallback is a curated list match, never a fuzzy guess.
#[test]
fn filename_fallback_matches_curated_list_only() {
    assert!(packs::guard_pack("polyhaven-rocks").is_ok());
    let refusal = packs::guard_pack("UltimateModularNatureKit.zip")
        .expect_err("curated QAL filename must refuse");
    assert!(
        refusal
            .message()
            .contains("quaternius-ultimate-modular-nature-kit")
    );
    assert!(packs::guard_pack("some-other-kit.zip").is_err());
    assert!(packs::lookup("polyhaven").is_none());
    assert!(
        catalog::packs()
            .iter()
            .any(|pack| pack.license == "QAL-1.0")
    );
    assert!(
        catalog::packs()
            .iter()
            .any(|pack| pack.license == "CC0-1.0")
    );
}
