//! Unified logging: human-readable events to stderr, structured JSONL to a file.
//!
//! Stdout stays reserved for program output (`openapi` JSON).
//!
//! Rotation: a log file over [`LOG_ROTATE_BYTES`] is renamed to
//! `<name>.log.1` (single backup, overwriting the previous `.1`) — once at
//! startup, and again mid-run whenever a write would cross the cap. Chosen
//! over `tracing-appender` daily rolling to keep one predictable file and no
//! extra dependency surface; the boring size-cap + one-backup scheme matches
//! the single-user local-first deployment.

use std::fs::{self, File};
use std::io::{IsTerminal as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tracing_subscriber::{
    EnvFilter, fmt, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
};

/// Log rotation cap: files over this many bytes rotate to `<name>.log.1`
/// (10 MiB), checked at startup and on every write that would cross the cap.
pub const LOG_ROTATE_BYTES: u64 = 10 * 1024 * 1024;

/// Default log file: `<data_dir>/logs/honkbox.log` (derived from the
/// resolved data dir; `None` when no data dir could be resolved).
fn default_log_file(data_dir: Option<&Path>) -> Option<PathBuf> {
    data_dir.map(|base| base.join("logs").join("honkbox.log"))
}

/// Resolve the log path: explicit arg beats `HONKBOX_LOG_FILE` beats the
/// data-dir default. This is the single owner of the env override (clap
/// leaves it alone so `--log-file` always wins); an empty value counts as
/// unset.
fn resolve_log_file(arg: Option<PathBuf>, data_dir: Option<&Path>) -> Option<PathBuf> {
    if arg.is_some() {
        return arg;
    }
    if let Ok(env) = std::env::var("HONKBOX_LOG_FILE")
        && !env.is_empty()
    {
        return Some(PathBuf::from(env));
    }
    default_log_file(data_dir)
}

/// Install the global subscriber. Never fails: a bad log file warns to stderr
/// and serving continues stderr-only. Called after `config::load` so the
/// file-refined data dir applies to the log path too.
pub fn init(log_file: Option<PathBuf>, data_dir: Option<&Path>) {
    // `RUST_LOG` wins when set; otherwise a target-scoped baseline (the
    // registry scan is chatty at `info`).
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,honkbox_registry=warn"));
    // ANSI only for a real terminal; the file layer below is always plain.
    let stderr = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal());
    match resolve_log_file(log_file, data_dir) {
        Some(path) => match RotatingLogFile::open(&path, LOG_ROTATE_BYTES) {
            Some(file) => {
                let file_layer = fmt::layer()
                    .json()
                    .with_writer(Mutex::new(file))
                    .with_ansi(false);
                tracing_subscriber::registry()
                    .with(filter)
                    .with(stderr)
                    .with(file_layer)
                    .init();
            }
            None => {
                tracing_subscriber::registry()
                    .with(filter)
                    .with(stderr)
                    .init();
            }
        },
        None => {
            tracing_subscriber::registry()
                .with(filter)
                .with(stderr)
                .init();
        }
    }
}

/// Append-only log writer that keeps the one-file-plus-one-backup contract
/// mid-run: a write that would cross `cap` rotates first (best-effort; on
/// failure the write still lands in the current file and rotation is
/// retried on the next crossing write). The outer `Mutex` in `init`
/// serializes events, so no inner locking is needed.
struct RotatingLogFile {
    path: PathBuf,
    cap: u64,
    file: File,
    written: u64,
}

impl RotatingLogFile {
    /// Create parent dirs, run the init-time rotation, and open (append)
    /// the log file; `None` after a stderr warning.
    fn open(path: &Path, cap: u64) -> Option<Self> {
        if let Some(parent) = path.parent()
            && let Err(e) = fs::create_dir_all(parent)
        {
            eprintln!("warn: cannot create log dir {}: {e}", parent.display());
            return None;
        }
        rotate_if_large(path, cap);
        match fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => {
                let written = file.metadata().map_or(0, |m| m.len());
                Some(Self {
                    path: path.to_owned(),
                    cap,
                    file,
                    written,
                })
            }
            Err(e) => {
                eprintln!(
                    "warn: cannot open log file {}: {e}; continuing stderr-only",
                    path.display()
                );
                None
            }
        }
    }

    /// Rename `path` to `<path>.1` (overwriting the previous backup) and
    /// reopen fresh; best-effort, never fails the write.
    fn rotate(&mut self) {
        let mut backup = self.path.clone().into_os_string();
        backup.push(".1");
        let backup = PathBuf::from(backup);
        let _ = self.file.flush();
        let _ = fs::remove_file(&backup);
        if fs::rename(&self.path, &backup).is_ok()
            && let Ok(file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
        {
            self.file = file;
            self.written = 0;
        }
    }
}

impl std::io::Write for RotatingLogFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.written + buf.len() as u64 > self.cap {
            self.rotate();
        }
        let n = self.file.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// Rotate `path` to `<path>.1` (single backup, previous `.1` overwritten)
/// when it exists and exceeds `cap`; best-effort, never fails startup.
fn rotate_if_large(path: &Path, cap: u64) {
    let over = fs::metadata(path).is_ok_and(|m| m.len() > cap);
    if !over {
        return;
    }
    let mut backup = path.to_owned().into_os_string();
    backup.push(".1");
    let backup = PathBuf::from(backup);
    let _ = fs::remove_file(&backup);
    match fs::rename(path, &backup) {
        Ok(()) => eprintln!(
            "info: rotated {} to {} (over {} bytes)",
            path.display(),
            backup.display(),
            cap
        ),
        Err(e) => eprintln!(
            "warn: cannot rotate {} to {}: {e}; appending anyway",
            path.display(),
            backup.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prod naming: `<path>.1` (os-string push, not an extension swap).
    fn backup_of(path: &Path) -> PathBuf {
        let mut b = path.to_owned().into_os_string();
        b.push(".1");
        PathBuf::from(b)
    }

    fn temp_log(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("honkbox-log-rotate-{}-{tag}", std::process::id()))
    }

    #[test]
    fn small_file_not_rotated() {
        let path = temp_log("small");
        fs::write(&path, b"tiny").expect("write log");
        rotate_if_large(&path, 1024);
        assert!(path.exists(), "log kept");
        assert!(!backup_of(&path).exists(), "no backup");
        fs::remove_file(&path).ok();
    }

    #[test]
    fn large_file_rotates_to_single_backup() {
        let path = temp_log("large");
        fs::write(&path, [0u8; 2048]).expect("write log");
        // Pre-seed an old backup: it must be overwritten, not merged.
        fs::write(backup_of(&path), b"old-backup").expect("seed backup");
        rotate_if_large(&path, 1024);
        assert!(!path.exists(), "oversized log rotated away");
        let backup = fs::read(backup_of(&path)).expect("backup");
        assert_eq!(backup.len(), 2048, "backup holds the rotated log");
        fs::remove_file(backup_of(&path)).ok();
    }

    #[test]
    fn missing_file_is_noop() {
        let path = temp_log("missing");
        let _ = fs::remove_file(&path);
        rotate_if_large(&path, 0);
        assert!(!path.exists());
    }

    #[test]
    fn writer_rotates_when_crossing_cap() {
        let path = temp_log("writer");
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(backup_of(&path));
        let mut w = RotatingLogFile::open(&path, 16).expect("open log");
        w.write_all(b"0123456789").expect("write");
        // This write crosses the 16-byte cap: the first 10 bytes move to
        // the backup, the new bytes land in a fresh file.
        w.write_all(b"abcdefghij").expect("write");
        w.flush().expect("flush");
        assert_eq!(fs::read(backup_of(&path)).expect("backup").len(), 10);
        assert_eq!(fs::read(&path).expect("current").len(), 10);
        fs::remove_file(&path).ok();
        fs::remove_file(backup_of(&path)).ok();
    }

    /// Log path derivation: arg > env-independent default under the data
    /// dir; no data dir → `None` (stderr-only).
    #[test]
    fn log_file_resolves_under_data_dir() {
        let base = std::env::temp_dir().join(format!("honkbox-log-base-{}", std::process::id()));
        assert_eq!(
            resolve_log_file(Some(PathBuf::from("/explicit/log")), Some(&base)),
            Some(PathBuf::from("/explicit/log"))
        );
        assert_eq!(
            resolve_log_file(None, Some(&base)),
            Some(base.join("logs/honkbox.log"))
        );
        assert_eq!(resolve_log_file(None, None), None);
    }
}
