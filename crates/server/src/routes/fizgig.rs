//! `POST /v1/fizgig/train` — opt-in `LoRA` training through a pinned Fizgig CLI
//! (FEAT-fizgig-lora-training-jobs). No Python runtime is linked: one job is
//! one dataset TOML, three explicit-argv supervised stages and one
//! external-GPU lease. Config is env-only for this slice (no `config.rs`
//! change); a failing CLI probe is a named 501, never a silent fallback.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{FizgigTrainAccepted, FizgigTrainRequest, FizgigTrainStatus, OpenAiError};
use tokio_util::sync::CancellationToken;
use ulid::Ulid;

use super::AppConfig;
#[cfg(unix)]
use super::fizgig_run;
use super::openai_constraints::openai_error_response;
#[cfg(unix)]
use super::supervisor;

/// Stable marker: a capability gap (501), like the rank sidecar's prefix.
pub(super) const UNAVAILABLE_PREFIX: &str = "fizgig unavailable: ";
/// Default wall-clock bound per stage (24 h); training is a long job.
const DEFAULT_TIMEOUT_SECS: u64 = 86_400;
/// Upper bound on the per-stage timeout (7 d).
const MAX_TIMEOUT_SECS: u64 = 604_800;
/// Grace between the pause marker and the process-group kill (1 h ceiling).
const DEFAULT_GRACE_SECS: u64 = 30;
const MAX_GRACE_SECS: u64 = 3_600;
/// Terminal job records retained for polling; live jobs keep their records.
const MAX_JOB_RECORDS: usize = 200;
/// Default retry interval for a failed or stale probe.
const DEFAULT_PROBE_TTL_SECS: u64 = 60;
const MAX_PROBE_TTL_SECS: u64 = 3_600;

/// Resolved env-only backend settings. Paths are canonicalized at load so a
/// later request cannot escape an allowlisted root with `..`.
#[derive(Debug, Clone)]
pub(super) struct Config {
    enabled: bool,
    root: Option<PathBuf>,
    venv: Option<PathBuf>,
    families: Vec<String>,
    /// Families whose own checkpoint carries VAE and text encoder weights, so
    /// the cache stages may fall back to `--dit`. All others must supply both.
    checkpoint_models: Vec<String>,
    dataset_roots: Vec<PathBuf>,
    model_roots: Vec<PathBuf>,
    timeout: Duration,
    grace: Duration,
    probe_ttl: Duration,
}

impl Config {
    fn from_env() -> Self {
        let enabled = flag("HONKBOX_FIZGIG_ENABLED");
        let root = path("HONKBOX_FIZGIG_ROOT");
        let venv = path("HONKBOX_FIZGIG_VENV").or_else(|| root.clone().map(|p| p.join("venv")));
        Self {
            enabled,
            root,
            venv,
            families: list("HONKBOX_FIZGIG_FAMILIES"),
            checkpoint_models: {
                let configured = list("HONKBOX_FIZGIG_CHECKPOINT_MODELS");
                if configured.is_empty() {
                    vec!["sdxl".to_owned()]
                } else {
                    configured
                }
            },
            dataset_roots: roots("HONKBOX_FIZGIG_DATASETS"),
            model_roots: roots("HONKBOX_FIZGIG_MODELS"),
            timeout: Duration::from_secs(
                number("HONKBOX_FIZGIG_TIMEOUT_SECS")
                    .unwrap_or(DEFAULT_TIMEOUT_SECS)
                    .clamp(1, MAX_TIMEOUT_SECS),
            ),
            grace: Duration::from_secs(
                number("HONKBOX_FIZGIG_GRACE_SECS")
                    .unwrap_or(DEFAULT_GRACE_SECS)
                    .clamp(1, MAX_GRACE_SECS),
            ),
            probe_ttl: Duration::from_secs(
                number("HONKBOX_FIZGIG_PROBE_TTL_SECS")
                    .unwrap_or(DEFAULT_PROBE_TTL_SECS)
                    .clamp(1, MAX_PROBE_TTL_SECS),
            ),
        }
    }

    /// The reason this backend cannot serve a request, when there is one.
    fn gap(&self) -> Option<String> {
        if !self.enabled {
            return Some("disabled".to_owned());
        }
        if self.root.is_none() || self.venv.is_none() {
            return Some("no pinned checkout or venv configured".to_owned());
        }
        if self.families.is_empty() {
            return Some("no training family allowlisted".to_owned());
        }
        if self.dataset_roots.is_empty() || self.model_roots.is_empty() {
            return Some("no dataset/model roots configured".to_owned());
        }
        None
    }
}

static CONFIG: OnceLock<Config> = OnceLock::new();
static JOBS: LazyLock<Mutex<HashMap<String, LiveJob>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Probe cache: only a bounded-age result is trusted, and a failed probe is
/// retried after its TTL instead of being cached for the process lifetime.
struct ProbeCache {
    at: Option<Instant>,
    result: Result<(), String>,
}

static PROBE: LazyLock<Mutex<ProbeCache>> = LazyLock::new(|| {
    Mutex::new(ProbeCache {
        at: None,
        result: Err("not probed yet".to_owned()),
    })
});
static PROBE_BUSY: AtomicBool = AtomicBool::new(false);

fn config() -> &'static Config {
    CONFIG.get_or_init(Config::from_env)
}

/// Capability honesty: enabled AND pinned CLI present AND last probe passed.
/// The probe itself runs on the blocking pool; a stale result is refreshed in
/// the background and the previous verdict is served until it lands.
pub(super) fn available() -> bool {
    let cfg = config();
    if cfg.gap().is_some() {
        return false;
    }
    let cache = PROBE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let stale = cache.at.is_none_or(|at| at.elapsed() >= cfg.probe_ttl);
    if stale && !PROBE_BUSY.swap(true, Ordering::AcqRel) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let cfg = cfg.clone();
            handle.spawn(async move {
                let result = match tokio::task::spawn_blocking(move || probe(&cfg)).await {
                    Ok(result) => result,
                    Err(err) => Err(format!("probe task failed: {err}")),
                };
                let mut cache = PROBE
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                cache.at = Some(Instant::now());
                cache.result = result;
                PROBE_BUSY.store(false, Ordering::Release);
            });
        } else {
            // No runtime (sync caller): leave the previous verdict intact.
            PROBE_BUSY.store(false, Ordering::Release);
        }
    }
    cache.result.is_ok()
}

fn probe_error() -> String {
    PROBE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .result
        .as_ref()
        .map_or_else(std::clone::Clone::clone, |()| "probe passed".to_owned())
}

/// One accepted run's mutable, process-local status (not persisted).
#[derive(Debug, Clone)]
pub(super) struct LiveJob {
    pub state: String,
    pub stage: String,
    pub output_dir: PathBuf,
    pub bundle: Option<String>,
    pub error: Option<String>,
    pub cancel: CancellationToken,
}

/// Everything the runner needs; paths are already validated and owned.
#[derive(Debug, Clone)]
pub(super) struct Plan {
    pub family: String,
    pub checkpoint_embedded: bool,
    pub dataset_dir: PathBuf,
    pub dit: PathBuf,
    pub vae: Option<PathBuf>,
    pub text_encoder: Option<PathBuf>,
    pub output_name: String,
    pub epochs: u32,
    pub rank: u32,
    pub seed: u64,
    pub timeout: Duration,
    pub grace: Duration,
    pub python: PathBuf,
    pub cache_py: PathBuf,
    pub train_py: PathBuf,
    pub work_dir: PathBuf,
    pub output_dir: PathBuf,
}

/// `env_var` reads the test override map in harness builds and the real
/// process env in production, so the CLI paths stay testable without `unsafe`.
fn env(name: &str) -> Option<String> {
    super::middleware::env_var(name)
}

fn flag(name: &str) -> bool {
    env(name)
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

fn path(name: &str) -> Option<PathBuf> {
    env(name).and_then(|v| {
        let v = v.trim();
        (!v.is_empty()).then(|| PathBuf::from(v))
    })
}

fn list(name: &str) -> Vec<String> {
    env(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn roots(name: &str) -> Vec<PathBuf> {
    list(name)
        .iter()
        .filter_map(|s| Path::new(s).canonicalize().ok())
        .collect()
}

fn number(name: &str) -> Option<u64> {
    env(name).and_then(|v| v.trim().parse::<u64>().ok())
}

fn unavailable(detail: impl fmt::Display) -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        format!("{UNAVAILABLE_PREFIX}{detail}"),
        Some("fizgig_unavailable"),
    )
}

fn bad_request(message: impl fmt::Display) -> Response {
    openai_error_response(
        StatusCode::BAD_REQUEST,
        message.to_string(),
        Some("fizgig_request_invalid"),
    )
}

/// Probe the pinned CLI via the same supervised path a job uses. One bounded
/// `--help` call; no model files, CUDA or network involved.
#[cfg(unix)]
fn probe(cfg: &Config) -> Result<(), String> {
    let (Some(root), Some(venv)) = (&cfg.root, &cfg.venv) else {
        return Err("no pinned checkout or venv configured".to_owned());
    };
    let python = venv.join("bin").join("python");
    let train_py = root.join("src/fizgig/families/train.py");
    let cache_py = root.join("src/fizgig/families/cache.py");
    for file in [&python, &train_py, &cache_py] {
        if !file.is_file() {
            return Err(format!("missing pinned CLI file {}", file.display()));
        }
    }
    let dir = std::env::temp_dir().join(format!("honkbox-fizgig-probe-{}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create probe dir: {e}"))?;
    let spec = supervisor::Spec::new(
        &python,
        [train_py.display().to_string(), "--help".to_owned()],
        &dir,
        Duration::from_secs(30),
    );
    let result = supervisor::run(spec)
        .map_err(|e| format!("probe could not start: {e}"))
        .and_then(|out| match out.termination {
            supervisor::Termination::Exited(0) => Ok(()),
            other => Err(format!("probe {other:?}: {}", out.stderr)),
        });
    let _ = fs::remove_dir_all(&dir);
    result
}

#[cfg(not(unix))]
fn probe(_cfg: &Config) -> Result<(), String> {
    Err("process supervision requires unix".to_owned())
}

/// Confine a caller path to an allowlisted root; canonicalization resolves
/// symlinks and `..` before the prefix check.
fn confined(raw: &str, roots: &[PathBuf], expect_dir: bool) -> Result<PathBuf, String> {
    let path = Path::new(raw)
        .canonicalize()
        .map_err(|_| format!("path does not exist: {raw}"))?;
    if expect_dir && !path.is_dir() {
        return Err(format!("not a directory: {raw}"));
    }
    if !expect_dir && !path.is_file() {
        return Err(format!("not a file: {raw}"));
    }
    if !roots.iter().any(|root| path.starts_with(root)) {
        return Err(format!("path outside the configured root: {raw}"));
    }
    Ok(path)
}

fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Apply the shared curated-weight gate to every model a run would load.
/// Uncurated local files stay local-only (the gate's existing posture);
/// curated non-commercial, gated or territory-excluded rows refuse before a
/// lease, job record or subprocess exists.
#[allow(clippy::result_large_err)]
fn check_weight_license(family: &str, models: [(&str, Option<&Path>); 3]) -> Result<(), Response> {
    for (label, path) in models {
        let Some(path) = path else { continue };
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        let Some(record) = crate::catalog::licenses::record_for_weight_file(file_name) else {
            continue;
        };
        let slug = format!("fizgig:{family}:{label}");
        if let Err(refusal) =
            crate::catalog::license_gate::check(&slug, Some(record), crate::config::licenses())
        {
            return Err(openai_error_response(
                crate::catalog::license_gate::LicenseRefusal::status(),
                format!("fizgig weights refused: {}", refusal.message()),
                Some(refusal.code()),
            ));
        }
    }
    Ok(())
}

// Axum's Response is the route's error type; boxing it would add indirection
// to every rejected request without changing the wire contract.
#[allow(clippy::result_large_err)]
fn validate(
    req: &FizgigTrainRequest,
    cfg: &Config,
    data_dir: Option<&Path>,
    id: &str,
) -> Result<Plan, Response> {
    if let Some(gap) = cfg.gap() {
        return Err(unavailable(gap));
    }
    if !available() {
        return Err(unavailable(format!("CLI probe failed: {}", probe_error())));
    }
    let problem = if !safe_name(&req.output_name) {
        Some("output_name must be 1-64 letters, digits, `_` or `-`".to_owned())
    } else if !req
        .family
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b == b'_' || b.is_ascii_digit())
        || !cfg.families.iter().any(|f| f == &req.family)
    {
        Some(format!("family {:?} is not allowlisted", req.family))
    } else if !(1..=1000).contains(&req.epochs) || !(1..=256).contains(&req.rank) {
        Some("epochs must be 1-1000 and rank 1-256".to_owned())
    } else {
        None
    };
    if let Some(problem) = problem {
        return Err(bad_request(problem));
    }
    let resolve = |raw: &str, dir: bool| {
        confined(
            raw,
            if dir {
                &cfg.dataset_roots
            } else {
                &cfg.model_roots
            },
            dir,
        )
    };
    let dataset_dir = resolve(&req.dataset_dir, true).map_err(bad_request)?;
    let dit = resolve(&req.dit, false).map_err(bad_request)?;
    let vae = req
        .vae
        .as_deref()
        .map(|v| resolve(v, false))
        .transpose()
        .map_err(bad_request)?;
    let text_encoder = req
        .text_encoder
        .as_deref()
        .map(|v| resolve(v, false))
        .transpose()
        .map_err(bad_request)?;
    let checkpoint_embedded = cfg.checkpoint_models.contains(&req.family);
    if !checkpoint_embedded && (vae.is_none() || text_encoder.is_none()) {
        return Err(bad_request(format!(
            "family {:?} requires explicit vae and text_encoder (checkpoint-embedded families: {})",
            req.family,
            cfg.checkpoint_models.join(", ")
        )));
    }
    check_weight_license(
        &req.family,
        [
            ("dit", Some(dit.as_path())),
            ("vae", vae.as_deref()),
            ("text_encoder", text_encoder.as_deref()),
        ],
    )?;
    let Some(data_dir) = data_dir else {
        return Err(unavailable("no data dir: job bundles are disabled"));
    };
    let root = cfg.root.as_ref().expect("gap checked").clone();
    let venv = cfg.venv.as_ref().expect("gap checked").clone();
    let work_dir = data_dir.join("fizgig").join(id);
    Ok(Plan {
        family: req.family.clone(),
        checkpoint_embedded,
        dataset_dir,
        dit,
        vae,
        text_encoder,
        output_name: req.output_name.clone(),
        epochs: req.epochs,
        rank: req.rank,
        seed: req.seed,
        timeout: cfg.timeout,
        grace: cfg.grace,
        python: venv.join("bin").join("python"),
        cache_py: root.join("src/fizgig/families/cache.py"),
        train_py: root.join("src/fizgig/families/train.py"),
        output_dir: work_dir.join("out"),
        work_dir,
    })
}

/// Keep status history bounded. Terminal records are removed oldest-first;
/// live records stay until they end. ULIDs sort chronologically in `job_*`.
fn prune_jobs(jobs: &mut HashMap<String, LiveJob>) {
    if jobs.len() < MAX_JOB_RECORDS {
        return;
    }
    let mut terminal: Vec<String> = jobs
        .iter()
        .filter(|(_, job)| matches!(job.state.as_str(), "completed" | "cancelled" | "failed"))
        .map(|(id, _)| id.clone())
        .collect();
    terminal.sort_unstable();
    let remove = terminal.len().min(jobs.len() - MAX_JOB_RECORDS + 1);
    for id in terminal.into_iter().take(remove) {
        jobs.remove(&id);
    }
}

/// Admit one training run. Validation happens before any process starts;
/// admission and the lease belong to the spawned task so 202 is immediate.
#[utoipa::path(
    post,
    path = "/v1/fizgig/train",
    request_body = FizgigTrainRequest,
    responses(
        (status = 202, description = "Job admitted; poll the status route", body = FizgigTrainAccepted),
        (status = 400, description = "Invalid family, name, range or confined path", body = OpenAiError),
        (status = 412, description = "Curated model weights refused by license terms", body = OpenAiError),
        (status = 501, description = "Backend disabled, CLI probe failed, or no data dir", body = OpenAiError)
    )
)]
pub(super) async fn post_fizgig_train(
    State(app): State<AppConfig>,
    body: Result<Json<FizgigTrainRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return bad_request(format!("invalid request: {}", rejection.body_text()));
        }
    };
    let id = format!("job_{}", Ulid::new());
    let plan = match validate(&req, config(), app.data_dir.as_deref(), &id) {
        Ok(plan) => plan,
        Err(response) => return response,
    };
    // Refuse before inserting any record: a phantom `queued` job that can
    // never run is worse than a 501 on a build without process supervision.
    if !cfg!(unix) {
        return unavailable("process supervision requires unix");
    }
    if let Err(err) = fs::create_dir_all(&plan.output_dir) {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create job dir: {err}"),
            Some("fizgig_job_dir_failed"),
        );
    }
    let cancel = CancellationToken::new();
    {
        let mut jobs = JOBS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prune_jobs(&mut jobs);
        jobs.insert(
            id.clone(),
            LiveJob {
                state: "queued".to_owned(),
                stage: "none".to_owned(),
                output_dir: plan.output_dir.clone(),
                bundle: None,
                error: None,
                cancel: cancel.clone(),
            },
        );
    }
    #[cfg(unix)]
    tokio::spawn(fizgig_run::run_fizgig_job(plan, &JOBS, id.clone()));
    #[cfg(not(unix))]
    {
        let _ = (plan, &id);
        return unavailable("process supervision requires unix");
    }
    (
        StatusCode::ACCEPTED,
        Json(FizgigTrainAccepted {
            job_id: id,
            state: "queued".to_owned(),
        }),
    )
        .into_response()
}

#[utoipa::path(
    get,
    path = "/v1/fizgig/jobs/{id}",
    params(("id" = String, Path, description = "Job id from the accepted response")),
    responses(
        (status = 200, description = "Current job status", body = FizgigTrainStatus),
        (status = 404, description = "Unknown job id", body = OpenAiError)
    )
)]
pub(super) async fn get_fizgig_job(AxumPath(id): AxumPath<String>) -> Response {
    let jobs = JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match jobs.get(&id) {
        Some(job) => Json(FizgigTrainStatus {
            job_id: id,
            state: job.state.clone(),
            stage: job.stage.clone(),
            bundle: job.bundle.clone(),
            error: job.error.clone(),
        })
        .into_response(),
        None => openai_error_response(
            StatusCode::NOT_FOUND,
            "unknown fizgig job".to_owned(),
            Some("fizgig_job_unknown"),
        ),
    }
}

#[utoipa::path(
    post,
    path = "/v1/fizgig/jobs/{id}/cancel",
    params(("id" = String, Path, description = "Job id from the accepted response")),
    responses(
        (status = 202, description = "Pause requested; process group killed after grace if ignored"),
        (status = 404, description = "Unknown job id", body = OpenAiError),
        (status = 409, description = "Job already ended", body = OpenAiError)
    )
)]
pub(super) async fn cancel_fizgig_job(AxumPath(id): AxumPath<String>) -> Response {
    // Hold the lock across the marker write: a cancel that arrives after the
    // runner's terminal `set` must not leave a pause marker on a finished run.
    let mut jobs = JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(job) = jobs.get_mut(&id) else {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            "unknown fizgig job".to_owned(),
            Some("fizgig_job_unknown"),
        );
    };
    if !matches!(job.state.as_str(), "queued" | "running") {
        return openai_error_response(
            StatusCode::CONFLICT,
            "job already ended".to_owned(),
            Some("fizgig_job_ended"),
        );
    }
    if let Err(err) = fs::write(job.output_dir.join(".pause_requested"), b"") {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot request pause: {err}"),
            Some("fizgig_cancel_failed"),
        );
    }
    job.cancel.cancel();
    StatusCode::ACCEPTED.into_response()
}
