//! Unified logging: human-readable events to stderr, structured JSONL to a file.
//!
//! Stdout stays reserved for program output (`openapi` JSON). The file writer is
//! blocking at init; switch to `tracing-appender` (non-blocking) in M2 if log
//! volume ever matters.
//!
//! Rotation: at startup, a log file over [`LOG_ROTATE_BYTES`] is renamed to
//! `<name>.log.1` (single backup, overwriting the previous `.1`). Chosen over
//! `tracing-appender` daily rolling to keep one predictable file and no extra
//! dependency surface; the boring size-cap + one-backup scheme matches the
//! single-user local-first deployment.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tracing_subscriber::{
    EnvFilter, fmt, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
};

/// Log rotation cap: files over this many bytes rotate to `<name>.log.1` at
/// startup (10 MiB).
pub const LOG_ROTATE_BYTES: u64 = 10 * 1024 * 1024;

/// Default log file: `<data_dir>/logs/honkbox.log` (derived from the
/// resolved data dir; `None` when no data dir could be resolved).
fn default_log_file(data_dir: Option<&Path>) -> Option<PathBuf> {
    data_dir.map(|base| base.join("logs").join("honkbox.log"))
}

/// Resolve the log path: explicit arg beats `HONKBOX_LOG_FILE` beats the
/// data-dir default.
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
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let stderr = fmt::layer().with_writer(std::io::stderr);
    match resolve_log_file(log_file, data_dir) {
        Some(path) => match open_log_file(&path) {
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

/// Create parent dirs and open (append) the log file; `None` after a stderr warning.
fn open_log_file(path: &std::path::Path) -> Option<File> {
    if let Some(parent) = path.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        eprintln!("warn: cannot create log dir {}: {e}", parent.display());
        return None;
    }
    rotate_if_large(path, LOG_ROTATE_BYTES);
    match fs::OpenOptions::new().create(true).append(true).open(path) {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!(
                "warn: cannot open log file {}: {e}; continuing stderr-only",
                path.display()
            );
            None
        }
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
