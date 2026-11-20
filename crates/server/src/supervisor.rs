//! Supervised execution of external engines (`TASK-process-supervisor`).
//!
//! One supervisor owns the whole child lifecycle for the engines honkbox
//! drives (the Blender CLI, a `ComfyUI` server, future tools): it starts the
//! process with EXPLICIT arguments, bounds the run (wall-clock timeout, RSS
//! cap, cancellation), tears the whole process GROUP down on every exit path
//! (success, failure, timeout, cancel) and collects the run's logs and
//! output artifacts.
//!
//! Teardown is process-group-wide on purpose: Blender spawns helpers, and a
//! cancel that only kills the direct child leaves orphans behind. The child
//! is started in its own process group (`CommandExt::process_group`) and the
//! group is killed through `rustix::process::kill_process_group` — the
//! workspace forbids `unsafe`, so the syscall is reached through rustix's
//! safe wrapper instead of a hand-rolled `libc` call.
//!
//! The GPU lease is handed IN and held for the child's whole lifetime: the
//! server's admission path owns the queueing, the supervisor owns the
//! reservation, and the lease is released (waking the queued waiters) the
//! moment the child is reaped — a cancelled job never keeps the slot.
//!
//! Deliberate ceilings:
//!
//! - The RSS cap watches the process-group LEADER's resident set (procfs),
//!   not a cgroup total. ponytail: per-leader RSS, not cgroup v2 accounting
//!   — move the child into a delegated cgroup with `memory.max` when the
//!   host grants one.
//! - Process groups are a unix mechanism. ponytail: unix-only — Windows
//!   needs job objects for the same guarantee.
//!
//! The module is unix-only: `main.rs` declares it behind `#[cfg(unix)]`.

use std::fmt;
use std::io::Read;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use rustix::process::{Pid, Signal};
use tokio_util::sync::CancellationToken;

use crate::admission::GpuLease;

/// The cap on the captured bytes per stream when the caller sets none.
const DEFAULT_LOG_BYTES: usize = 64 * 1024;

/// How often the supervisor re-checks the child (cancel, timeout, RSS).
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// How long the supervisor waits for a log reader to finish after the child
/// is reaped. The group is killed on every path, so the pipes close at once;
/// the bound only covers a helper that escaped the group.
const LOG_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// One supervised run: what to execute and how it is bounded.
#[derive(Debug)]
pub struct Spec {
    /// The program to run (a path, or a name resolved on `PATH`).
    pub program: PathBuf,
    /// The arguments, passed EXPLICITLY: the supervisor never builds a
    /// shell string and never guesses tool flags (thread caps, device
    /// selection) — those belong to the caller.
    pub args: Vec<String>,
    /// The child's working directory (the caller's per-job dir).
    pub work_dir: PathBuf,
    /// The wall-clock bound; the process group is killed when it expires.
    pub timeout: Duration,
    /// Optional resident-memory cap in bytes; the group is killed when the
    /// group leader's RSS exceeds it. Needs procfs (Linux).
    pub rss_limit_bytes: Option<u64>,
    /// Optional cancellation token; the group is killed when it is set.
    pub cancel: Option<CancellationToken>,
    /// The glob patterns (relative to `work_dir`) of the artifacts to
    /// collect after the run.
    pub artifacts: Vec<String>,
    /// The cap on the captured bytes per stream.
    pub log_bytes: usize,
    /// The ledger lease to hold for the child's whole lifetime (the GPU
    /// lease of GPU-backed work). It is released — and the queued waiters
    /// woken — the moment the child is reaped.
    pub lease: Option<GpuLease>,
}

impl Spec {
    /// A spec for one run: the program, its EXPLICIT arguments, the working
    /// directory and the wall-clock bound. The log cap defaults to 64 KiB per
    /// stream; the RSS cap, the cancel token, the artifact patterns and the
    /// lease start empty.
    pub fn new(
        program: impl Into<PathBuf>,
        args: impl IntoIterator<Item = impl Into<String>>,
        work_dir: impl Into<PathBuf>,
        timeout: Duration,
    ) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            work_dir: work_dir.into(),
            timeout,
            rss_limit_bytes: None,
            cancel: None,
            artifacts: Vec::new(),
            log_bytes: DEFAULT_LOG_BYTES,
            lease: None,
        }
    }
}

/// Why a supervised run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The child exited on its own with this exit code.
    Exited(i32),
    /// The child was killed by a signal (the signal number).
    Signalled(i32),
    /// The process group was killed after the timeout expired.
    TimedOut,
    /// The process group was killed after the cancel token was set.
    Cancelled,
    /// The process group was killed after the RSS cap was exceeded.
    RssExceeded,
}

/// What a supervised run produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Why the run ended.
    pub termination: Termination,
    /// The child's stdout (capped at `Spec::log_bytes`).
    pub stdout: String,
    /// The child's stderr (capped at `Spec::log_bytes`).
    // Read by the blender/python routes and the supervisor contract test;
    // harnesses that include only `routes.rs` never read it.
    #[allow(dead_code)]
    pub stderr: String,
    /// `true` when stdout hit the cap (the rest was drained, not kept).
    pub stdout_truncated: bool,
    /// `true` when stderr hit the cap.
    // Same harness pattern as `stderr` above.
    #[allow(dead_code)]
    pub stderr_truncated: bool,
    /// The artifacts matched by `Spec::artifacts`, sorted and deduplicated.
    // The blender caller reads its artifacts from the recipe manifest (it
    // needs their kinds and hashes), so nothing reads this list there.
    #[allow(dead_code)]
    pub artifacts: Vec<PathBuf>,
}

/// Why a supervised run could not happen.
#[derive(Debug)]
pub enum RunError {
    /// The program could not be started (missing binary, bad work dir).
    Spawn {
        /// The program that could not be started.
        program: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The child could not be waited for.
    Wait {
        /// The underlying I/O error.
        source: std::io::Error,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { program, source } => {
                write!(f, "cannot start {}: {source}", program.display())
            }
            Self::Wait { source } => write!(f, "cannot wait for the child: {source}"),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Wait { source } => Some(source),
        }
    }
}

/// Run one supervised job to completion.
///
/// The child is started in its own process group, bounded by the spec's
/// timeout, RSS cap and cancel token, and reaped before this returns. The
/// spec's lease (when set) is held for the child's whole lifetime and
/// released when the child is reaped.
pub fn run(spec: Spec) -> Result<Outcome, RunError> {
    let Spec {
        program,
        args,
        work_dir,
        timeout,
        rss_limit_bytes,
        cancel,
        artifacts,
        log_bytes,
        lease,
    } = spec;

    let mut command = Command::new(&program);
    command
        .args(&args)
        .current_dir(&work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group: the teardown kills the whole tree, not just the
    // direct child.
    command.process_group(0);

    let mut child = command.spawn().map_err(|source| RunError::Spawn {
        program: program.clone(),
        source,
    })?;
    let group = Pid::from_child(&child);
    let pid = child.id();
    let stdout = child.stdout.take().map(|pipe| reader(pipe, log_bytes));
    let stderr = child.stderr.take().map(|pipe| reader(pipe, log_bytes));

    let deadline = Instant::now() + timeout;
    let mut killed_for: Option<Termination> = None;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(source) => return Err(RunError::Wait { source }),
        }
        let cause = if cancel.as_ref().is_some_and(CancellationToken::is_cancelled) {
            Some(Termination::Cancelled)
        } else if Instant::now() >= deadline {
            Some(Termination::TimedOut)
        } else if rss_limit_bytes.is_some_and(|limit| rss_bytes(pid).is_some_and(|rss| rss > limit))
        {
            Some(Termination::RssExceeded)
        } else {
            None
        };
        if let Some(cause) = cause {
            kill_group(group);
            killed_for = Some(cause);
            break child.wait().map_err(|source| RunError::Wait { source })?;
        }
        std::thread::sleep(POLL_INTERVAL);
    };

    // Deterministic teardown on EVERY path: a helper that outlived the
    // child dies with the group, and the log pipes close with it.
    kill_group(group);
    // The lease is released here — after the child is reaped, before the
    // caller sees the outcome.
    drop(lease);

    let (stdout, stdout_truncated) = drain_log(stdout);
    let (stderr, stderr_truncated) = drain_log(stderr);
    Ok(Outcome {
        termination: killed_for.unwrap_or_else(|| termination_of(status)),
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        artifacts: collect_artifacts(&work_dir, &artifacts),
    })
}

/// Why the child ended, when it ended on its own.
fn termination_of(status: ExitStatus) -> Termination {
    match status.code() {
        Some(code) => Termination::Exited(code),
        // Unix: a signalled child has no exit code, the signal is the reason.
        None => Termination::Signalled(status.signal().unwrap_or(0)),
    }
}

/// Kill the whole process group (SIGKILL): the child and every helper it
/// spawned. rustix is the safe door to the syscall — the workspace forbids
/// `unsafe`.
fn kill_group(group: Pid) {
    let _ = rustix::process::kill_process_group(group, Signal::KILL);
}

/// The group leader's resident memory in bytes (procfs). `None` when the
/// host has no procfs: the cap cannot be enforced there.
fn rss_bytes(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib.saturating_mul(1024))
}

/// Read a pipe on its own thread, so the child never blocks on a full pipe.
fn reader(pipe: impl Read + Send + 'static, cap: usize) -> Receiver<(String, bool)> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(drain(pipe, cap));
    });
    rx
}

/// Drain a pipe to EOF, keeping at most `cap` bytes (the rest is read and
/// dropped so the child never blocks on a full pipe).
fn drain(mut pipe: impl Read, cap: usize) -> (String, bool) {
    let mut kept: Vec<u8> = Vec::new();
    let mut truncated = false;
    let mut buf = [0_u8; 8192];
    loop {
        match pipe.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                if kept.len() < cap {
                    let room = cap - kept.len();
                    let take = room.min(read);
                    kept.extend_from_slice(&buf[..take]);
                    truncated |= take < read;
                } else {
                    truncated = true;
                }
            }
        }
    }
    (String::from_utf8_lossy(&kept).into_owned(), truncated)
}

/// Wait for a log reader, bounded: a helper that escaped the process group
/// can hold the pipe open, and a stuck log must not hang the job.
fn drain_log(reader: Option<Receiver<(String, bool)>>) -> (String, bool) {
    reader.map_or_else(Default::default, |reader| {
        reader.recv_timeout(LOG_DRAIN_TIMEOUT).unwrap_or_default()
    })
}

/// The files matching the caller's glob patterns (relative to `work_dir`),
/// sorted and deduplicated.
fn collect_artifacts(work_dir: &Path, patterns: &[String]) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = patterns
        .iter()
        .flat_map(
            |pattern| match glob::glob(&work_dir.join(pattern).to_string_lossy()) {
                Ok(paths) => paths.filter_map(Result::ok).collect::<Vec<_>>(),
                Err(_) => Vec::new(),
            },
        )
        .collect();
    found.sort();
    found.dedup();
    found
}
