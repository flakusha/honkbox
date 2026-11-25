//! Ash promotion contract: CLI and HTTP selection share the one compile-
//! reachable `backends::make_sd` constructor, and the capability flag derives
//! from it. Run once with default features and once with `--features ash-sd`;
//! no test needs a GPU or a checkpoint.

#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "../src/artifacts.rs"]
mod artifacts;
#[allow(dead_code)]
#[path = "../src/backends.rs"]
mod backends;
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
#[path = "../src/llm_attached.rs"]
mod llm_attached;
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

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use clap::Parser;
use honkbox_api_types::ModelCatalog;
use routes::AppConfig;

/// Same loud-exit seam `backends` sees in the binary; these tests never reach
/// it (stub/sd-cpp construct without loading, Ash selection is parse-only).
fn or_exit<T, E: std::fmt::Display>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|e| panic!("or_exit: {e}"))
}

/// CLI and HTTP flags both use `SdBackend` as their clap value source.
#[derive(Debug, Parser)]
struct SdSelection {
    #[arg(long, value_enum)]
    backend: backends::SdBackend,
}

fn select(value: &str) -> Result<backends::SdBackend, clap::Error> {
    Ok(SdSelection::try_parse_from(["honkbox", "--backend", value])?.backend)
}

fn config() -> AppConfig {
    let catalog = ModelCatalog {
        object: "list".to_owned(),
        data: Vec::new(),
    };
    AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: routes::AuthConfig::default(),
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
        llm: Arc::new(honkbox_llm_worker::StubLlm::default()),
        stt: Arc::new(honkbox_audio_worker::StubStt),
        tts: Arc::new(honkbox_audio_worker::StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        artifacts: Arc::new(artifacts::Registry::new(
            None,
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    }
}

fn advertised_ash() -> bool {
    routes::capabilities::capabilities_document(&config())
        .features
        .experimental_ash_sd
}

#[test]
fn stub_and_sd_cpp_selection_stay_unchanged() {
    let resolve = |slug: &str| PathBuf::from("/models").join(slug);
    assert!(matches!(select("stub"), Ok(backends::SdBackend::Stub)));
    assert!(matches!(select("sd-cpp"), Ok(backends::SdBackend::SdCpp)));
    assert_eq!(
        backends::make_sd(backends::SdBackend::Stub, None, &resolve, None).backend_id(),
        "stub"
    );
    assert_eq!(
        backends::make_sd(backends::SdBackend::SdCpp, Some("sd"), &resolve, None).backend_id(),
        honkbox_sd_core::SD_CPP_BACKEND_ID
    );
}

#[cfg(feature = "ash-sd")]
#[test]
fn feature_on_selects_ash_and_advertises_it() {
    assert!(backends::ASH_SD_COMPILED);
    assert!(matches!(select("ash-sd"), Ok(backends::SdBackend::AshSd)));
    assert!(advertised_ash());
}

#[cfg(not(feature = "ash-sd"))]
#[test]
fn feature_off_rejects_ash_and_hides_it() {
    const {
        assert!(!backends::ASH_SD_COMPILED);
    };
    let err = select("ash-sd").expect_err("feature-off build must reject ash-sd");
    let message = err.to_string();
    assert!(message.contains("ash-sd"), "{message}");
    assert!(message.contains("invalid value"), "{message}");
    assert!(!advertised_ash());
}
