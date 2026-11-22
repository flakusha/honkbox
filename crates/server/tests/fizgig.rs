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

// The training lease is process-global; serialize this test with other GPU tests.
static SLOT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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
    let script = br#"#!/bin/sh
case " $* " in
  *" --help "*) echo 'usage: train.py'; exit 0;;
esac
out=''
cfg=''
name='model'
prev=''
for arg in "$@"; do
  case "$prev" in
    --dataset_config) cfg="$arg";;
    --output_dir) out="$arg";;
    --output_name) name="$arg";;
  esac
  prev="$arg"
done
if [ -z "$out" ] && [ -n "$cfg" ]; then
  out="$(dirname "$cfg")/out"
fi
if [ -n "$out" ]; then
  echo "stage $*"
  printf '%s\n' "$*" >> "$out/argv.log"
  case " $* " in
    *" --dit "*)
      sleep 1
      i=1
      while [ "$i" -le 5 ]; do
        printf 'fake lora %s' "$i" > "$out/$name-00000$i.safetensors"
        i=$((i + 1))
      done
      printf 'fake final' > "$out/$name.safetensors"
      mkdir -p "$out/state"
      printf 'fake state' > "$out/state/optimizer.pt"
      ;;
  esac
fi
exit 0
"#;
    std::fs::write(&python, script).unwrap();
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

/// The probe is refreshed on the blocking pool, so capability flips are
/// observed asynchronously even in the test harness.
async fn wait_capability(app: &axum::Router, wanted: bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let (_, caps) = get(app, "/v1/capabilities").await;
        if caps["features"]["fizgig"] == wanted {
            return caps;
        }
        assert!(
            Instant::now() < deadline,
            "fizgig capability never reached {wanted}: {caps}"
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
}

// ponytail: one scenario owns the fake CLI, capability, both job lifecycles.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn fizgig_train_lifecycle_with_fake_cli() {
    let _slot = SLOT_LOCK.lock().await;
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
    let vae = models.join("flux.vae.safetensors");
    std::fs::write(&vae, b"vae").unwrap();
    let text_encoder = models.join("flux.te.safetensors");
    std::fs::write(&text_encoder, b"text encoder").unwrap();

    set_env_var("HONKBOX_FIZGIG_ENABLED", Some("1"));
    set_env_var("HONKBOX_FIZGIG_ROOT", Some(checkout.to_str().unwrap()));
    set_env_var("HONKBOX_FIZGIG_FAMILIES", Some("sdxl,flux"));
    set_env_var("HONKBOX_FIZGIG_DATASETS", Some(datasets.to_str().unwrap()));
    set_env_var("HONKBOX_FIZGIG_MODELS", Some(models.to_str().unwrap()));
    set_env_var("HONKBOX_FIZGIG_TIMEOUT_SECS", Some("30"));
    set_env_var("HONKBOX_FIZGIG_GRACE_SECS", Some("1"));
    set_env_var("HONKBOX_FIZGIG_PROBE_TTL_SECS", Some("1"));

    let app = app_with(&data_dir);
    let caps = wait_capability(&app, true).await;
    assert_eq!(caps["features"]["fizgig"], true, "probe: {caps}");

    // A failing probe flips the capability false, then retries and recovers
    // without a restart (TTL is 1s in this test).
    let python = checkout.join("venv/bin/python");
    let good_script = std::fs::read(&python).unwrap();
    std::fs::write(&python, b"#!/bin/sh\necho probe-down\nexit 1\n").unwrap();
    std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
    let _down = wait_capability(&app, false).await;
    std::fs::write(&python, &good_script).unwrap();
    std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
    let _up = wait_capability(&app, true).await;

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
    let (status, err) = post(&app, "/v1/fizgig/train", invalid).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(err["error"]["code"], "fizgig_request_invalid");

    // Non-SDXL families need explicit VAE + text encoder before admission.
    let mut flux = valid.clone();
    flux["family"] = json!("flux");
    let (status, err) = post(&app, "/v1/fizgig/train", flux.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "no fallback models: {err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("text_encoder")
    );
    flux["vae"] = json!(vae.to_str().unwrap());
    flux["text_encoder"] = json!(text_encoder.to_str().unwrap());
    let (status, accepted) = post(&app, "/v1/fizgig/train", flux).await;
    assert_eq!(status, StatusCode::ACCEPTED, "explicit models: {accepted}");
    let flux_id = accepted["job_id"].as_str().unwrap().to_owned();
    let flux_done = wait_state(&app, &flux_id, "completed").await;
    assert_eq!(flux_done["bundle"], format!("jobs/{flux_id}"));
    let flux_argv =
        std::fs::read_to_string(data_dir.join("fizgig").join(&flux_id).join("out/argv.log"))
            .unwrap();
    assert!(flux_argv.contains(&format!("--model {}", vae.display())));
    assert!(flux_argv.contains(&format!("--model {}", text_encoder.display())));
    let flux_manifest: Value = serde_json::from_slice(
        &std::fs::read(data_dir.join("jobs").join(&flux_id).join("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(flux_manifest["state_dir"], "out/state");

    // Every caller path is confined to its operator allowlist.
    let mut outside = valid.clone();
    outside["dataset_dir"] = json!(data_dir.to_str().unwrap());
    let (status, err) = post(&app, "/v1/fizgig/train", outside).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "dataset root: {err}");
    assert_eq!(err["error"]["code"], "fizgig_request_invalid");
    let outside_file = checkout.join("src/fizgig/families/train.py");
    let mut outside = valid.clone();
    outside["dit"] = json!(outside_file.to_str().unwrap());
    let (status, err) = post(&app, "/v1/fizgig/train", outside).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "dit root: {err}");
    assert_eq!(err["error"]["code"], "fizgig_request_invalid");
    let mut outside = valid.clone();
    outside["family"] = json!("flux");
    outside["vae"] = json!(outside_file.to_str().unwrap());
    outside["text_encoder"] = json!(text_encoder.to_str().unwrap());
    let (status, err) = post(&app, "/v1/fizgig/train", outside).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "vae root: {err}");
    let mut outside = valid.clone();
    outside["family"] = json!("flux");
    outside["vae"] = json!(vae.to_str().unwrap());
    outside["text_encoder"] = json!(outside_file.to_str().unwrap());
    let (status, err) = post(&app, "/v1/fizgig/train", outside).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "text encoder root: {err}");

    // Curated weight terms are enforced before admission. These file names
    // match the shipped restricted rows (gated/NC and territory-excluded).
    let gated_weights = models.join("ae.safetensors");
    std::fs::write(&gated_weights, b"gated").unwrap();
    let mut restricted = valid.clone();
    restricted["dit"] = json!(gated_weights.to_str().unwrap());
    let (status, err) = post(&app, "/v1/fizgig/train", restricted).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "gated: {err}");
    assert_eq!(err["error"]["code"], "model_license_gated");
    let territory_weights = models.join("model.safetensors");
    std::fs::write(&territory_weights, b"territory").unwrap();
    let mut restricted = valid.clone();
    restricted["dit"] = json!(territory_weights.to_str().unwrap());
    let (status, err) = post(&app, "/v1/fizgig/train", restricted).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "territory: {err}");
    assert_eq!(err["error"]["code"], "model_license_territory");

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
    assert!(bundle.join("my_subject.safetensors").is_file());
    for epoch in ["000003", "000004", "000005"] {
        assert!(
            bundle
                .join(format!("my_subject-{epoch}.safetensors"))
                .is_file()
        );
    }
    for epoch in ["000001", "000002"] {
        assert!(
            !bundle
                .join(format!("my_subject-{epoch}.safetensors"))
                .is_file()
        );
    }
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["family"], "sdxl");
    assert_eq!(manifest["rank"], 8);
    assert_eq!(manifest["artifacts"].as_array().unwrap().len(), 4);
    assert_eq!(manifest["state_dir"], "out/state");
    assert_eq!(
        manifest["artifacts"][0]["sha256"].as_str().unwrap().len(),
        64
    );

    // Terminal jobs reject cancel, and stay out of the generic job registry.
    let (status, err) = post(&app, &format!("/v1/fizgig/jobs/{id}/cancel"), json!({})).await;
    assert_eq!(status, StatusCode::CONFLICT, "late cancel: {err}");
    assert_eq!(err["error"]["code"], "fizgig_job_ended");
    let (_, listed) = get(&app, "/v1/jobs").await;
    assert!(
        !listed.to_string().contains(&id),
        "fizgig jobs use their own status route: {listed}"
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
