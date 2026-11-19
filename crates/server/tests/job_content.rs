//! Job content download contract: deterministic bundle ZIPs, the closed
//! single-file allowlist, named envelopes for bad ids/missing files/oversized
//! files, and no host paths in any response. Every test owns a temp data dir
//! and builds its own router; no fixed ports or shared state.

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

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use axum::{
    Router,
    body::Body,
    http::{HeaderMap, Request, StatusCode, header},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::Value;
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

const ID_A: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
const ID_B: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAW";

/// One test-scoped data dir (unique per test name + pid; pre-cleaned).
fn tmp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("honkbox-job-content-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test data dir");
    dir
}

fn app_with(data_dir: Option<PathBuf>) -> Router {
    app(AppConfig {
        auth: AuthConfig::default(),
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
        sd_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: data_dir.clone(),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        artifacts: Arc::new(artifacts::Registry::new(
            data_dir,
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
    })
}

/// Create one job bundle with the given top-level files.
fn bundle(root: &Path, id: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let dir = root.join("jobs").join(id);
    std::fs::create_dir_all(&dir).expect("bundle dir");
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).expect("bundle file");
    }
    dir
}

/// GET a route on the router; no network, no fixed port.
async fn get(app: Router, uri: &str) -> (StatusCode, HeaderMap, Vec<u8>) {
    let response = app
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let headers = response.headers().clone();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, headers, body.to_vec())
}

/// The named code from an `OpenAI` error envelope.
fn code(body: &[u8]) -> String {
    let value: Value = serde_json::from_slice(body).expect("json error envelope");
    value["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// No response may carry a host path (the temp root is the obvious one).
fn assert_no_host_path(body: &[u8], root: &Path) {
    let text = String::from_utf8_lossy(body);
    let root = root.to_string_lossy();
    assert!(!text.contains(root.as_ref()), "host path leaked: {text}");
}

#[tokio::test]
async fn zip_bytes_are_deterministic_across_runs() {
    let root_a = tmp_dir("zip-a");
    let root_b = tmp_dir("zip-b");
    let files: &[(&str, &[u8])] = &[
        ("output.glb", b"glb-bytes"),
        ("manifest.json", br#"{"manifest":1}"#),
        ("output_turntable.webp", b"webp-bytes"),
    ];
    bundle(&root_a, ID_A, files);
    // Same contents, different creation order, different mtimes, and a
    // nested directory that must be ignored instead of traversed.
    let dir_b = bundle(&root_b, ID_B, &[files[2], files[1], files[0]]);
    let nested = dir_b.join("nested");
    std::fs::create_dir_all(&nested).expect("nested dir");
    std::fs::write(nested.join("secret.txt"), b"do not include").expect("nested file");
    let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
    std::fs::File::open(dir_b.join("output.glb"))
        .expect("open file")
        .set_modified(old)
        .expect("set mtime");

    let (status, headers, first) = get(
        app_with(Some(root_a.clone())),
        &format!("/v1/jobs/{ID_A}/bundle.zip"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "application/zip");
    assert!(
        headers[header::CONTENT_DISPOSITION]
            .to_str()
            .expect("disposition")
            .contains("job-bundle.zip")
    );
    let (status, _, second) = get(
        app_with(Some(root_b.clone())),
        &format!("/v1/jobs/{ID_B}/bundle.zip"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        first, second,
        "identical bundles must produce identical ZIP bytes"
    );
    assert_no_host_path(&first, &root_a);
    assert_no_host_path(&second, &root_b);

    let mut archive = zip::ZipArchive::new(Cursor::new(first)).expect("zip archive");
    let names: Vec<String> = (0..archive.len())
        .map(|i| archive.by_index(i).expect("zip entry").name().to_owned())
        .collect();
    assert_eq!(
        names,
        vec!["manifest.json", "output.glb", "output_turntable.webp"]
    );
    for i in 0..archive.len() {
        let entry = archive.by_index(i).expect("zip entry");
        assert_eq!(entry.last_modified(), Some(zip::DateTime::default()));
    }
}

#[tokio::test]
async fn content_route_serves_only_allowlisted_names() {
    let root = tmp_dir("allowlist");
    bundle(
        &root,
        ID_A,
        &[
            ("manifest.json", br#"{"manifest":1}"#),
            ("output.glb", b"glb"),
            ("output_turntable.webp", b"webp"),
            ("secret.txt", b"secret"),
        ],
    );
    let app = app_with(Some(root.clone()));
    for (name, expected) in [
        ("manifest.json", br#"{"manifest":1}"#.as_slice()),
        ("output.glb", b"glb".as_slice()),
        ("output_turntable.webp", b"webp".as_slice()),
    ] {
        let (status, headers, body) =
            get(app.clone(), &format!("/v1/jobs/{ID_A}/content/{name}")).await;
        assert_eq!(status, StatusCode::OK, "allowlisted {name}");
        assert_eq!(body, expected);
        assert!(headers.contains_key(header::CONTENT_TYPE));
    }
    for name in ["secret.txt", "nested%2Fsecret.txt", "output.obj"] {
        let (status, _, body) = get(app.clone(), &format!("/v1/jobs/{ID_A}/content/{name}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "rejected {name}");
        assert_eq!(code(&body), "invalid_job_content_name");
        assert_no_host_path(&body, &root);
    }
}

#[tokio::test]
async fn invalid_ulid_is_a_named_400() {
    let root = tmp_dir("invalid-id");
    let app = app_with(Some(root.clone()));
    for uri in [
        "/v1/jobs/short/bundle.zip".to_owned(),
        format!("/v1/jobs/{}/bundle.zip", "x".repeat(27)),
        "/v1/jobs/not-a-ulid/content/manifest.json".to_owned(),
    ] {
        let (status, _, body) = get(app.clone(), &uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        assert_eq!(code(&body), "invalid_job_id");
        assert_no_host_path(&body, &root);
    }
}

#[tokio::test]
async fn missing_job_and_missing_data_dir_are_404() {
    let root = tmp_dir("missing-job");
    // A different job exists in the same data dir: the miss is per-job.
    bundle(&root, ID_A, &[("manifest.json", b"{}")]);
    let app = app_with(Some(root.clone()));
    for uri in [
        format!("/v1/jobs/{ID_B}/bundle.zip"),
        format!("/v1/jobs/{ID_B}/content/manifest.json"),
    ] {
        let (status, _, body) = get(app.clone(), &uri).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
        assert_eq!(code(&body), "job_content_not_found");
        assert_no_host_path(&body, &root);
    }
    let no_data = app_with(None);
    for uri in [
        format!("/v1/jobs/{ID_A}/bundle.zip"),
        format!("/v1/jobs/{ID_A}/content/manifest.json"),
    ] {
        let (status, _, body) = get(no_data.clone(), &uri).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(code(&body), "job_content_not_found");
    }
}

#[tokio::test]
async fn missing_file_is_404() {
    let root = tmp_dir("missing-file");
    bundle(&root, ID_A, &[("manifest.json", b"{}")]);
    let (status, _, body) = get(
        app_with(Some(root.clone())),
        &format!("/v1/jobs/{ID_A}/content/output.glb"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(code(&body), "job_content_not_found");
    assert_no_host_path(&body, &root);
}

#[tokio::test]
async fn oversized_file_is_413() {
    let root = tmp_dir("too-large");
    let dir = bundle(&root, ID_A, &[]);
    // Sparse: the cap is checked from metadata before any bytes are read.
    let file = std::fs::File::create(dir.join("output.glb")).expect("sparse file");
    file.set_len(routes::job_content::MAX_CONTENT_BYTES + 1)
        .expect("set sparse length");
    drop(file);
    let (status, _, body) = get(
        app_with(Some(root.clone())),
        &format!("/v1/jobs/{ID_A}/content/output.glb"),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(code(&body), "job_content_too_large");
    assert_no_host_path(&body, &root);
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_content_name_is_404() {
    let root = tmp_dir("symlink");
    let dir = bundle(&root, ID_A, &[]);
    let secret = root.join("secret.txt");
    std::fs::write(&secret, b"outside the bundle").expect("secret");
    std::os::unix::fs::symlink(&secret, dir.join("manifest.json")).expect("symlink");

    let (status, _, body) = get(
        app_with(Some(root.clone())),
        &format!("/v1/jobs/{ID_A}/content/manifest.json"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(!body.windows(6).any(|w| w == b"secret"));
}
