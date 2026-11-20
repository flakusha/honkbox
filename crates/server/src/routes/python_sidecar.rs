//! Off-by-default Python sidecar subprocess seam (TASK-python-sidecar-subprocess).
//!
//! One rank call = one per-request job dir, one explicit-argv child and one
//! GPU-slot lease held by the supervisor. No Python runtime in Rust, no new
//! dependency: `<venv>/bin/python -m honkbox_sidecar -- <job.json> <result.json>`.
//! Capability gaps carry `UNAVAILABLE_PREFIX` (named 501); other errors are 500.

use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use honkbox_residency::ResourceKind;
use serde::{Deserialize, Serialize};

use super::runtime_config::{python_allowlist, python_enabled, python_timeout_secs, python_venv};

#[cfg(unix)]
use super::supervisor::{self, Spec, Termination};

/// Stable marker: a rank error with this prefix is a capability gap (501).
pub(super) const UNAVAILABLE_PREFIX: &str = "python sidecar unavailable: ";

/// Makes per-job temp dirs unique inside one nanosecond tick.
static JOB_SEQ: AtomicU64 = AtomicU64::new(0);

/// A capability gap, named for the `/v1/capabilities` row that gates it.
fn unavailable(detail: impl fmt::Display) -> String {
    format!("{UNAVAILABLE_PREFIX}{detail}; see GET /v1/capabilities python")
}

/// The job file handed to `honkbox_sidecar`.
#[derive(Debug, Serialize)]
struct RankJob<'a> {
    op: &'static str,
    query: &'a str,
    texts: &'a [String],
    model: &'a str,
}

/// The result file `honkbox_sidecar` writes: one score per input text.
#[derive(Debug, Deserialize)]
struct RankResult {
    scores: Vec<f64>,
}

/// Score `texts` against `query` with the allowlisted `model`.
///
/// Scores come back in input order; the caller owns sorting and `top_n`.
/// No model process is shared and no state survives the call.
pub(super) async fn rank(query: &str, texts: &[String], model: &str) -> Result<Vec<f64>, String> {
    // Gate BEFORE any lease, directory or process exists.
    if !python_enabled() {
        return Err(unavailable("disabled"));
    }
    if !python_allowlist().iter().any(|allowed| allowed == model) {
        return Err(unavailable(format!("model {model} is not allowlisted")));
    }
    let Some(venv) = python_venv() else {
        return Err(unavailable("no sidecar venv configured"));
    };

    // Rerank is retrieval-side work (`Rag`); the sidecar may use a GPU in
    // its own process, but honkbox only owns the slot here.
    let lease = crate::admission::acquire(ResourceKind::Rag)
        .await
        .map_err(|(_, message)| message)?;

    let dir = job_dir()?;
    let result = run_job(&dir, &venv, query, texts, model, lease).await;
    let _ = fs::remove_dir_all(&dir);
    result
}

/// Create a fresh, private directory for this one call.
fn job_dir() -> Result<PathBuf, String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| format!("system clock is behind the epoch: {err}"))?
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "honkbox-sidecar-{}-{nanos}-{}",
        std::process::id(),
        JOB_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).map_err(|err| format!("cannot create sidecar job dir: {err}"))?;
    Ok(dir)
}

/// Write the job, run the child under the supervisor on the blocking pool
/// (`supervisor::run` polls synchronously), then read its scores.
#[cfg(unix)]
async fn run_job(
    dir: &std::path::Path,
    venv: &std::path::Path,
    query: &str,
    texts: &[String],
    model: &str,
    lease: crate::admission::GpuLease,
) -> Result<Vec<f64>, String> {
    let job_path = dir.join("job.json");
    let result_path = dir.join("result.json");
    let job = RankJob {
        op: "rank",
        query,
        texts,
        model,
    };
    let payload =
        serde_json::to_vec(&job).map_err(|err| format!("cannot encode sidecar job: {err}"))?;
    fs::write(&job_path, payload).map_err(|err| format!("cannot write sidecar job: {err}"))?;

    let program = venv.join("bin").join("python");
    let args = [
        "-m".to_owned(),
        "honkbox_sidecar".to_owned(),
        "--".to_owned(),
        job_path.display().to_string(),
        result_path.display().to_string(),
    ];
    let work_dir = dir.to_path_buf();
    let timeout = std::time::Duration::from_secs(python_timeout_secs());
    let outcome = tokio::task::spawn_blocking(move || {
        // Spec::new's default 64 KiB per-stream log cap applies untouched.
        let mut spec = Spec::new(program, args, work_dir, timeout);
        spec.lease = Some(lease);
        supervisor::run(spec)
    })
    .await
    .map_err(|err| format!("sidecar task failed: {err}"))?
    .map_err(|err| format!("sidecar could not start: {err}"))?;

    match outcome.termination {
        Termination::Exited(0) => {}
        Termination::Exited(code) => {
            return Err(format!(
                "sidecar exited with code {code}: {}",
                outcome.stderr
            ));
        }
        other => return Err(format!("sidecar {other:?}: {}", outcome.stderr)),
    }
    let bytes =
        fs::read(&result_path).map_err(|err| format!("cannot read sidecar result: {err}"))?;
    let result: RankResult =
        serde_json::from_slice(&bytes).map_err(|err| format!("invalid sidecar result: {err}"))?;
    if result.scores.len() != texts.len() {
        return Err(format!(
            "sidecar returned {} scores for {} texts",
            result.scores.len(),
            texts.len()
        ));
    }
    Ok(result.scores)
}

/// The supervisor is a unix mechanism (`routes::supervisor` is `cfg(unix)`).
#[cfg(not(unix))]
async fn run_job(
    _dir: &PathBuf,
    _venv: &std::path::Path,
    _query: &str,
    _texts: &[String],
    _model: &str,
    _lease: crate::admission::GpuLease,
) -> Result<Vec<f64>, String> {
    Err(unavailable("process supervision requires unix"))
}
