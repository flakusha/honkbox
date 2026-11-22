//! `POST /v1/fizgig/train` (FEAT-fizgig-lora-training-jobs) contract.
//! A fake pinned CLI stands in for Fizgig: the test proves argv construction,
//! the three-stage lifecycle, path confinement, bundle commit and cancel
//! without `PyTorch` or a GPU. Real CLI fidelity stays in the research note.

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

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app, set_env_var};

fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-fizgig-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test data dir");
    dir
}

fn app_with(data_dir: &Path) -> axum::Router {
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
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: Some(data_dir.to_path_buf()),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        artifacts: Arc::new(artifacts::Registry::new(
            Some(data_dir.to_path_buf()),
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
    })
}

/// Fake python: `--help` probes clean; each stage logs argv and the train
/// stage writes one `LoRA`. A delay on train makes cancel deterministic.
fn fake_cli(root: &Path) {
    let families = root.join("src/fizgig/families");
    std::fs::create_dir_all(&families).expect("families dir");
    std::fs::write(families.join("cache.py"), b"# pinned CLI marker\n").unwrap();
    std::fs::write(families.join("train.py"), b"# pinned CLI marker\n").unwrap();
    let python = root.join("venv/bin/python");
    std::fs::create_dir_all(python.parent().unwrap()).expect("venv dir");
    std::fs::write(
        &python,
        b"#!/bin/sh\ncase \" $* \" in\n  *\" --help \"*) echo 'usage: train.py'; exit 0;;\nesac\nout=''\ncfg=''\nname='model'\nprev=''\nfor arg in \"$@\"; do\n  case \"$prev\" in\n    --dataset_config) cfg=\"$arg\";;\n    --output_dir) out=\"$arg\";;\n    --output_name) name=\"$arg\";;\n  esac\n  prev=\"$arg\"\ndone\nif [ -z \"$out\" ] && [ -n \"$cfg\" ]; then\n  out=\"$(dirname \"$cfg\")/out\"\nfi\nif [ -n \"$out\" ]; then\n  echo \"stage $*\"\n  printf '%s\\n' \"$*\" >> \"$out/argv.log\"\n  case \" $* \" in\n    *\" --dit \"*) sleep 1; printf 'fake lora' > \"$out/$name-000001.safetensors\";;\n  esac\nfi\nexit 0\n",
    )
    .unwrap();
    std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

async fn get(app: &axum::Router, path: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

async fn wait_state(app: &axum::Router, id: &str, wanted: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let (_, status) = get(app, &format!("/v1/fizgig/jobs/{id}")).await;
        if status["state"] == wanted {
            return status;
        }
        assert_ne!(status["state"], "failed", "job failed: {status}");
        assert!(
            Instant::now() < deadline,
            "job never reached {wanted}: {status}"
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
}

// ponytail: one scenario owns the fake CLI, capability, both job lifecycles.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn fizgig_train_lifecycle_with_fake_cli() {
    let base = tmp_dir("e2e");
    let data_dir = base.join("data");
    let checkout = base.join("fizgig");
    let datasets = base.join("datasets");
    let models = base.join("models");
    for dir in [&data_dir, &datasets, &models] {
        std::fs::create_dir_all(dir).unwrap();
    }
    fake_cli(&checkout);
    let dataset = datasets.join("subject");
    std::fs::create_dir_all(&dataset).unwrap();
    std::fs::write(dataset.join("01.png"), b"png").unwrap();
    std::fs::write(dataset.join("01.txt"), b"a subject").unwrap();
    let dit = models.join("sdxl.safetensors");
    std::fs::write(&dit, b"checkpoint").unwrap();

    set_env_var("HONKBOX_FIZGIG_ENABLED", Some("1"));
    set_env_var("HONKBOX_FIZGIG_ROOT", Some(checkout.to_str().unwrap()));
    set_env_var("HONKBOX_FIZGIG_FAMILIES", Some("sdxl"));
    set_env_var("HONKBOX_FIZGIG_DATASETS", Some(datasets.to_str().unwrap()));
    set_env_var("HONKBOX_FIZGIG_MODELS", Some(models.to_str().unwrap()));
    set_env_var("HONKBOX_FIZGIG_TIMEOUT_SECS", Some("30"));
    set_env_var("HONKBOX_FIZGIG_GRACE_SECS", Some("1"));

    let app = app_with(&data_dir);
    let (_, caps) = get(&app, "/v1/capabilities").await;
    assert_eq!(
        caps["features"]["fizgig"], true,
        "probe must succeed: {caps}"
    );

    let valid = json!({
        "family": "sdxl",
        "dataset_dir": dataset.to_str().unwrap(),
        "dit": dit.to_str().unwrap(),
        "output_name": "my_subject",
        "epochs": 2,
        "rank": 8,
        "seed": 42,
    });
    let mut invalid = valid.clone();
    invalid["family"] = json!("../evil");
    let (status, _) = post(&app, "/v1/fizgig/train", invalid).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "family charset is validated"
    );

    let (status, accepted) = post(&app, "/v1/fizgig/train", valid.clone()).await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "valid job admitted: {accepted}"
    );
    let id = accepted["job_id"].as_str().unwrap().to_owned();
    let done = wait_state(&app, &id, "completed").await;
    assert_eq!(done["stage"], "done");
    assert_eq!(done["bundle"], format!("jobs/{id}"));

    let bundle = data_dir.join("jobs").join(&id);
    assert!(bundle.join("manifest.json").is_file());
    assert!(bundle.join("my_subject-000001.safetensors").is_file());
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["family"], "sdxl");
    assert_eq!(manifest["rank"], 8);
    assert_eq!(
        manifest["artifacts"][0]["sha256"].as_str().unwrap().len(),
        64
    );
    let argv =
        std::fs::read_to_string(data_dir.join("fizgig").join(&id).join("out/argv.log")).unwrap();
    assert!(
        argv.contains("--stage latents"),
        "latent cache stage: {argv}"
    );
    assert!(argv.contains("--stage text"), "text cache stage: {argv}");
    assert!(argv.contains("--network_dim 8"), "train stage: {argv}");
    let stdout = std::fs::read_to_string(bundle.join("train.stdout.log")).unwrap();
    assert!(stdout.contains("--network_dim 8"), "stage stdout: {stdout}");
    assert!(bundle.join("train.stderr.log").is_file());

    // Status flips before the RAII lease drops; give admission a tick.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (status, accepted) = post(&app, "/v1/fizgig/train", valid.clone()).await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "lease released after completion"
    );
    let cancel_id = accepted["job_id"].as_str().unwrap().to_owned();
    let (cancel_status, _) = post(
        &app,
        &format!("/v1/fizgig/jobs/{cancel_id}/cancel"),
        json!({}),
    )
    .await;
    assert_eq!(cancel_status, StatusCode::ACCEPTED);
    let cancelled = wait_state(&app, &cancel_id, "cancelled").await;
    assert_eq!(cancelled["job_id"], cancel_id);
    assert!(
        data_dir
            .join("fizgig")
            .join(&cancel_id)
            .join("out/.pause_requested")
            .is_file()
    );

    // A CLI that ignores the pause marker must still lose its process group.
    let python = checkout.join("venv/bin/python");
    std::fs::write(
        &python,
        b"#!/bin/sh\ncase \" $* \" in\n  *\" --help \"*) echo 'usage: train.py'; exit 0;;\nesac\necho \"stubborn $*\"\nsleep 30\n",
    )
    .unwrap();
    std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (status, accepted) = post(&app, "/v1/fizgig/train", valid.clone()).await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "stubborn job admitted: {accepted}"
    );
    let stubborn_id = accepted["job_id"].as_str().unwrap().to_owned();
    let _running = wait_state(&app, &stubborn_id, "running").await;
    let (cancel_status, _) = post(
        &app,
        &format!("/v1/fizgig/jobs/{stubborn_id}/cancel"),
        json!({}),
    )
    .await;
    assert_eq!(cancel_status, StatusCode::ACCEPTED);
    let hard = wait_state(&app, &stubborn_id, "cancelled").await;
    assert_eq!(hard["job_id"], stubborn_id);
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(
            data_dir
                .join("jobs")
                .join(&stubborn_id)
                .join("manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest["artifacts"].as_array().unwrap().len(),
        0,
        "hard stop keeps no LoRA"
    );
}
