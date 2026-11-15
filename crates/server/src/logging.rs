//! Unified logging: human-readable events to stderr, structured JSONL to a file.
//!
//! Stdout stays reserved for program output (`openapi` JSON). The file writer is
//! blocking at init; switch to `tracing-appender` (non-blocking) in M2 if log
//! volume ever matters.

use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Mutex;

use tracing::warn;

use tracing_subscriber::{
    EnvFilter, fmt, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
};

/// Default log file: `~/.honkbox/logs/honkbox.log` (user-kept logs under the
/// data dir). ONE-RELEASE compat: the old `~/.sd-rip/logs/sd-rip.log` is used
/// once (with a `warn!`) when the new file is absent but the old one exists.
fn default_log_file() -> Option<PathBuf> {
    let home = home::home_dir()?;
    let new = home.join(".honkbox").join("logs").join("honkbox.log");
    if !new.exists() {
        let old = home.join(".sd-rip").join("logs").join("sd-rip.log");
        if old.exists() {
            warn!(
                "{} is deprecated; move it to {} (one-release compat: the old path will stop working)",
                old.display(),
                new.display()
            );
            return Some(old);
        }
    }
    Some(new)
}

/// Resolve the log path: explicit arg beats `HONKBOX_LOG_FILE` beats default.
/// ONE-RELEASE compat: `SD_RIP_LOG_FILE` is honored once (with a `warn!`)
/// when the new var is absent.
fn resolve_log_file(arg: Option<PathBuf>) -> Option<PathBuf> {
    if arg.is_some() {
        return arg;
    }
    if let Ok(env) = std::env::var("HONKBOX_LOG_FILE")
        && !env.is_empty()
    {
        return Some(PathBuf::from(env));
    }
    if let Ok(env) = std::env::var("SD_RIP_LOG_FILE")
        && !env.is_empty()
    {
        warn!(
            "SD_RIP_LOG_FILE is deprecated; use HONKBOX_LOG_FILE (one-release compat: the old name will stop working)"
        );
        return Some(PathBuf::from(env));
    }
    default_log_file()
}

/// Install the global subscriber. Never fails: a bad log file warns to stderr
/// and serving continues stderr-only.
pub fn init(log_file: Option<PathBuf>) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let stderr = fmt::layer().with_writer(std::io::stderr);
    match resolve_log_file(log_file) {
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
