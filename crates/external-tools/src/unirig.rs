//! UniRig auto-rig probe (TASK-unirig-probe).
//!
//! Upstream UniRig is MIT; loop-lore's ComfyUI wrapper is GPL-3.0, so honkbox
//! drives only a direct, operator-supplied binary. It is opt-in via
//! `HONKBOX_UNIRIG` (default `unirig` on `PATH`), never linked or vendored, and
//! never runs mesh fixtures at discovery time. The probe proves the binary
//! can answer a version request; bone plausibility and zero unweighted
//! vertices belong to a later job, not to a startup probe.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{Capability, Status, ToolId, Unavailable, probe_version, status_from_probe};

/// Environment variable naming the rigger binary (operator-supplied path).
pub const UNIRIG_ENV: &str = "HONKBOX_UNIRIG";

/// Binary used when [`UNIRIG_ENV`] is unset.
pub const DEFAULT_UNIRIG: &str = "unirig";

/// The configured rigger binary (never a value from a request).
#[must_use]
pub fn binary() -> PathBuf {
    std::env::var_os(UNIRIG_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_UNIRIG))
}

/// Probe the rigger's version.
pub fn probe(binary: &Path, timeout: Duration) -> Result<(String, Vec<Capability>), Unavailable> {
    let version = probe_version(binary, timeout)?;
    Ok((version, vec![Capability::AutoRig]))
}

/// 501-ready status: absent, broken, or available with its version.
#[must_use]
pub fn status(binary: &Path, timeout: Duration) -> Status {
    status_from_probe(ToolId::Unirig, probe(binary, timeout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "honkbox-unirig-{}-{}-{tag}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[cfg(unix)]
    fn fake_binary(tag: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = scratch(tag);
        let path = dir.join("unirig");
        let staged = dir.join("unirig.staged");
        std::fs::write(&staged, script).expect("fake binary");
        let mut perms = std::fs::metadata(&staged).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged, perms).expect("chmod");
        std::fs::rename(staged, &path).expect("rename");
        path
    }

    #[test]
    fn a_missing_rigger_is_a_named_not_installed_status() {
        let bin = scratch("missing").join("no-such-unirig");
        let status = status(&bin, Duration::from_secs(2));
        assert_eq!(status.id, ToolId::Unirig);
        assert_eq!(status.unavailable, Some(Unavailable::NotInstalled));
        assert!(status.capabilities.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_versioned_rigger_is_available() {
        let bin = fake_binary("versioned", "#!/bin/sh\necho 'UniRig 0.3.1'\n");
        let status = status(&bin, Duration::from_secs(2));
        assert!(status.is_available(), "{status:?}");
        assert_eq!(status.version.as_deref(), Some("0.3.1"));
        assert_eq!(status.capabilities, vec![Capability::AutoRig]);
    }

    #[cfg(unix)]
    #[test]
    fn a_broken_rigger_is_a_probe_failed_status() {
        let bin = fake_binary(
            "broken",
            "#!/bin/sh\necho 'unirig: bad weights' >&2\nexit 5\n",
        );
        let status = status(&bin, Duration::from_secs(2));
        assert!(
            matches!(status.unavailable, Some(Unavailable::ProbeFailed(_))),
            "{status:?}"
        );
    }
}
