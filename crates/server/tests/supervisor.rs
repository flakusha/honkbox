#![cfg(unix)]

//! Contract tests for the external process supervisor
//! (`TASK-process-supervisor`).
//!
//! The supervisor owns the child lifecycle: bounded runs, process-group
//! teardown on every exit path, no orphans, the lease released when the
//! child is reaped, and the run's logs and artifacts collected.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use honkbox_residency::ResourceKind;
use tokio_util::sync::CancellationToken;

#[path = "../src/admission.rs"]
#[allow(dead_code)]
mod admission;
#[path = "../src/catalog.rs"]
#[allow(dead_code)]
mod catalog;
#[path = "../src/config.rs"]
#[allow(dead_code)]
mod config;
#[path = "../src/jobs.rs"]
#[allow(dead_code)]
mod jobs;
#[path = "../src/metrics.rs"]
#[allow(dead_code)]
mod metrics;
#[path = "../src/supervisor.rs"]
#[allow(dead_code)]
mod supervisor;

use supervisor::{Spec, Termination};

/// A per-test scratch directory (unique per test tag and process).
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-supervisor-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Whether the pid is still alive (procfs).
fn alive(pid: u32) -> bool {
    std::fs::metadata(format!("/proc/{pid}")).is_ok()
}

/// Whether the pid disappears within the budget: the group kill is delivered
/// asynchronously, so the check polls instead of sampling once.
fn gone_within(pid: u32, budget: Duration) -> bool {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if !alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    !alive(pid)
}

/// Cancel the run once the helper process exists: the orphan check needs
/// its pid.
fn cancel_once_helper_exists(dir: &Path, token: &CancellationToken) {
    let dir = dir.to_path_buf();
    let token = token.clone();
    std::thread::spawn(move || {
        let pid_file = dir.join("grandchild.pid");
        let deadline = Instant::now() + Duration::from_secs(10);
        while !pid_file.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        token.cancel();
    });
}

#[test]
fn a_failed_run_still_reports_its_logs() {
    let dir = scratch_dir("failed");
    let spec = Spec::new(
        "sh",
        ["-c", "echo out; echo err >&2; exit 3"],
        &dir,
        Duration::from_secs(30),
    );
    let outcome = supervisor::run(spec).expect("run");
    assert_eq!(outcome.termination, Termination::Exited(3));
    assert!(outcome.stdout.contains("out"), "stdout: {}", outcome.stdout);
    assert!(outcome.stderr.contains("err"), "stderr: {}", outcome.stderr);
}

#[test]
fn a_missing_program_is_a_spawn_error() {
    let dir = scratch_dir("missing");
    let missing = dir.join("no-such-binary");
    let spec = Spec::new(&missing, Vec::<String>::new(), &dir, Duration::from_secs(5));
    match supervisor::run(spec) {
        Err(supervisor::RunError::Spawn { program, .. }) => assert_eq!(program, missing),
        other => panic!("expected a spawn error, got {other:?}"),
    }
}

#[test]
fn the_timeout_kills_the_process_group() {
    let dir = scratch_dir("timeout");
    let spec = Spec::new("sh", ["-c", "sleep 30"], &dir, Duration::from_millis(300));
    let outcome = supervisor::run(spec).expect("run");
    assert_eq!(outcome.termination, Termination::TimedOut);
}

#[test]
fn a_cancel_kills_the_group_and_leaves_no_orphan() {
    let dir = scratch_dir("cancel");
    let token = CancellationToken::new();
    cancel_once_helper_exists(&dir, &token);
    let mut spec = Spec::new(
        "sh",
        ["-c", "sleep 30 & echo $! > grandchild.pid; wait"],
        &dir,
        Duration::from_secs(30),
    );
    spec.cancel = Some(token);
    let outcome = supervisor::run(spec).expect("run");
    assert_eq!(outcome.termination, Termination::Cancelled);
    let pid: u32 = std::fs::read_to_string(dir.join("grandchild.pid"))
        .expect("the helper pid")
        .trim()
        .parse()
        .expect("a pid");
    assert!(
        gone_within(pid, Duration::from_secs(5)),
        "the helper process survived the cancel"
    );
}

#[test]
fn the_rss_cap_kills_the_group() {
    let dir = scratch_dir("rss");
    let mut spec = Spec::new("sh", ["-c", "sleep 30"], &dir, Duration::from_secs(30));
    spec.rss_limit_bytes = Some(1);
    let outcome = supervisor::run(spec).expect("run");
    assert_eq!(outcome.termination, Termination::RssExceeded);
}

#[test]
fn artifacts_are_collected_by_glob() {
    let dir = scratch_dir("artifacts");
    let mut spec = Spec::new(
        "sh",
        [
            "-c",
            "mkdir -p out; echo a > out/a.png; echo b > out/b.png; echo c > out/c.txt",
        ],
        &dir,
        Duration::from_secs(30),
    );
    spec.artifacts = vec!["out/*.png".to_string()];
    let outcome = supervisor::run(spec).expect("run");
    let names: Vec<String> = outcome
        .artifacts
        .iter()
        .map(|path| {
            path.file_name()
                .expect("a file name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(names, vec!["a.png".to_string(), "b.png".to_string()]);
}

#[test]
fn captured_logs_are_capped_but_the_child_never_blocks() {
    let dir = scratch_dir("logcap");
    let spec = Spec::new(
        "sh",
        ["-c", "yes aaaaaaaa | head -c 100000"],
        &dir,
        Duration::from_secs(30),
    );
    let outcome = supervisor::run(spec).expect("run");
    assert!(outcome.stdout_truncated, "the log was not capped");
    assert_eq!(outcome.stdout.len(), 64 * 1024);
}

#[tokio::test]
async fn a_cancelled_job_releases_its_lease() {
    let dir = scratch_dir("lease");
    let lease = admission::acquire(ResourceKind::Blender)
        .await
        .expect("the lease");
    // The shared GPU slot is busy while the lease is held.
    assert!(
        honkbox_residency::global()
            .try_acquire(ResourceKind::Blender)
            .is_err()
    );

    let token = CancellationToken::new();
    let trigger = token.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        trigger.cancel();
    });
    let mut spec = Spec::new("sh", ["-c", "sleep 30"], &dir, Duration::from_secs(30));
    spec.cancel = Some(token);
    spec.lease = Some(lease);
    let outcome = supervisor::run(spec).expect("run");
    assert_eq!(outcome.termination, Termination::Cancelled);
    // The lease died with the run: the slot is free again.
    let released = honkbox_residency::global()
        .try_acquire(ResourceKind::Blender)
        .expect("the slot is free");
    drop(released);
}
