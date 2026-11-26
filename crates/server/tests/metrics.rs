//! `/metrics` exposition (NEW-9): expected families and lines are present
//! after a request.

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
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;
// Included only so `routes.rs`'s `crate::swap` paths resolve in this test
// binary; the swap module itself is not exercised here.
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::ModelCatalog;
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

fn test_app() -> Router {
    app(AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
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
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            None.clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    })
}

#[tokio::test]
async fn metrics_lines_present_after_request() {
    let app = test_app();

    // One counted request first.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .expect("content-type header")
        .to_owned();
    assert!(content_type.starts_with("text/plain"), "{content_type}");

    let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("# HELP honkbox_build_info"));
    assert!(body.contains("honkbox_build_info{version=\""));
    assert!(body.contains("# TYPE honkbox_requests_total counter"));
    assert!(
        body.contains("honkbox_requests_total{route=\"/v1/models\",code=\"200\"}"),
        "requests line missing after a 200 on /v1/models: {body}"
    );
    assert!(body.contains("# TYPE honkbox_request_duration_seconds histogram"));
    assert!(body.contains("honkbox_request_duration_seconds_bucket{le=\"+Inf\"}"));
    assert!(body.contains("honkbox_request_duration_seconds_count"));
    assert!(body.contains("honkbox_request_duration_seconds_sum "));
    assert!(body.contains("# TYPE honkbox_gpu_slot_busy_rejections_total counter"));
    assert!(
        body.contains("honkbox_gpu_slot_busy_rejections_total{reason=\"no_wait\"}"),
        "labeled busy series missing: {body}"
    );
    assert!(body.contains("honkbox_gpu_slot_busy_rejections_total{reason=\"queue_full\"}"));
    assert!(body.contains("honkbox_gpu_slot_busy_rejections_total{reason=\"queue_timeout\"}"));
    // GC reclaim families share the closed reason vocabulary; counters are
    // process-wide, so only the bounded label grid is asserted here.
    assert!(body.contains("# TYPE honkbox_gc_reclaimed_bytes_total counter"));
    assert!(body.contains("# TYPE honkbox_gc_entries_total counter"));
    for family in [
        "honkbox_gc_reclaimed_bytes_total",
        "honkbox_gc_entries_total",
    ] {
        for reason in ["expired", "size_cap", "part_sweep", "temp_sweep"] {
            assert!(
                body.contains(&format!("{family}{{reason=\"{reason}\"}} ")),
                "missing {family} series for {reason}: {body}"
            );
        }
    }
    // Kind-labeled swap families render their full bounded grid. Counts in
    // this process depend on which in-module unit tests ran first (the
    // statics are process-wide, see `metrics.rs` tests), so only presence +
    // untouched-slot zeros are asserted here; count behavior is covered in
    // tests/swap.rs `swap_events_emit_kind_metrics`.
    assert!(body.contains("# TYPE honkbox_model_loads_total counter"));
    assert!(body.contains("honkbox_model_loads_total{kind=\"llm\",reason=\"request\"}"));
    assert!(body.contains("# TYPE honkbox_model_unloads_total counter"));
    assert!(body.contains("honkbox_model_unloads_total{kind=\"llm\",reason=\"ttl\"}"));
    assert!(body.contains("# TYPE honkbox_swap_evictions_total counter"));
    assert!(body.contains("honkbox_swap_evictions_total{kind=\"llm\"}"));
    assert!(body.contains("# TYPE honkbox_model_load_duration_seconds histogram"));
    assert!(body.contains("honkbox_model_load_duration_seconds_bucket{kind=\"llm\",le=\"+Inf\"}"));
    assert!(body.contains("honkbox_model_load_duration_seconds_count{kind=\"llm\"}"));
    // Slots nothing in this process can touch stay at exact zero.
    assert!(body.contains("honkbox_model_loads_total{kind=\"vae\",reason=\"manual\"} 0"));
    assert!(body.contains("honkbox_swap_evictions_total{kind=\"vae\"} 0"));
    assert!(body.ends_with('\n'));
}

fn fresh_gc_root() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let unique = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "honkbox-metrics-gc-{}-{nanos}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir_all(&path).expect("unique gc root");
    path
}

fn metric_value(body: &str, family: &str, reason: &str) -> u64 {
    let prefix = format!("{family}{{reason=\"{reason}\"}} ");
    body.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing series: {prefix}"))
        .parse()
        .expect("numeric counter")
}

/// Deterministic teardown for the unique temp root, even if an assert panics.
struct CleanupGcRoot(PathBuf);

impl Drop for CleanupGcRoot {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// Resource contract: this test owns a unique temp dir keyed by PID, monotonic
/// timestamp, and atomic counter; `CleanupGcRoot` removes it on every exit.
/// Counters are process-wide, so assertions compare deltas, never totals.
#[test]
fn gc_reclaim_counters_increase_after_expired_sweep() {
    const ENTRIES: &str = "honkbox_gc_entries_total";
    const BYTES: &str = "honkbox_gc_reclaimed_bytes_total";
    let base = fresh_gc_root();
    let _cleanup = CleanupGcRoot(base.clone());
    let jobs_root = base.join("jobs");
    let old_job = jobs_root.join("01OLD");
    std::fs::create_dir_all(&old_job).expect("job dir");
    let blob = old_job.join("blob");
    std::fs::write(&blob, [0_u8; 32]).expect("blob");
    let old = SystemTime::now()
        .checked_sub(Duration::from_secs(7200))
        .expect("clock");
    let times = std::fs::FileTimes::new()
        .set_modified(old)
        .set_accessed(old);
    std::fs::File::options()
        .write(true)
        .open(&blob)
        .expect("open blob")
        .set_times(times)
        .expect("backdate blob");

    let before = metrics::exposition();
    let before_entries = metric_value(&before, ENTRIES, "expired");
    let before_bytes = metric_value(&before, BYTES, "expired");

    jobs::gc_jobs(Some(&jobs_root), Duration::from_secs(3600), u64::MAX);

    assert!(!old_job.exists(), "expired job dir must be swept");
    let after = metrics::exposition();
    assert!(
        metric_value(&after, ENTRIES, "expired") > before_entries,
        "expired entry counter did not rise: {after}"
    );
    assert!(
        metric_value(&after, BYTES, "expired") > before_bytes,
        "expired byte counter did not rise: {after}"
    );
}
