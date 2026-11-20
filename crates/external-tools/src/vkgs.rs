//! Prebuilt `vk_gaussian_splatting` renderer probe (TASK-splat-render-vkgs-probe).
//!
//! The binary is opt-in: [`binary`] reads `HONKBOX_VKGS` and falls back to
//! `vk_gaussian_splatting` on `PATH`. Nothing is linked, downloaded or rendered
//! at discovery time — the probe only proves the executable can answer a
//! version request and records the sequencefile contract a job driver must
//! use. GPU work (RADV vs NVIDIA pipelines, one screenshot from a `.ply`)
//! stays a job concern, never a startup probe.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{Capability, Status, ToolId, Unavailable, probe_version, status_from_probe};

/// Environment variable naming the renderer binary (operator-supplied path).
pub const VKGS_ENV: &str = "HONKBOX_VKGS";

/// Binary used when [`VKGS_ENV`] is unset.
pub const DEFAULT_VKGS: &str = "vk_gaussian_splatting";

/// The headless automation contract exposed as a capability detail.
///
/// A job writes a `.cfg` sequencefile with `SEQUENCE` blocks, selects a camera
/// preset and pipeline with `--activateCameraPreset` / `--pipeline`, and asks
/// for one frame with `--screenshot`. The probe records this contract; it does
/// not claim a render succeeded on this box.
pub const SEQUENCEFILE_CONTRACT: &str = "sequencefile .cfg: SEQUENCE blocks; \
    --activateCameraPreset <name>; --pipeline <name>; --screenshot <path>; \
    one headless frame from a .ply; no GPU render exercised by the probe";

/// The configured renderer binary (never a value from a request).
#[must_use]
pub fn binary() -> PathBuf {
    std::env::var_os(VKGS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VKGS))
}

/// Probe the binary's version and record what a renderer job can drive.
pub fn probe(binary: &Path, timeout: Duration) -> Result<(String, Vec<Capability>), Unavailable> {
    let version = probe_version(binary, timeout)?;
    Ok((
        version,
        vec![
            Capability::SplatRender,
            Capability::Sequencefile,
            Capability::detail(SEQUENCEFILE_CONTRACT),
        ],
    ))
}

/// 501-ready status for the renderer: absent, broken, or available with its
/// version and the sequencefile contract.
#[must_use]
pub fn status(binary: &Path, timeout: Duration) -> Status {
    status_from_probe(ToolId::SplatRenderer, probe(binary, timeout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    /// Unique per-test scratch directory; nothing is shared between tests.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "honkbox-vkgs-{}-{}-{tag}",
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
        let path = dir.join("vkgs");
        let staged = dir.join("vkgs.staged");
        std::fs::write(&staged, script).expect("fake binary");
        let mut perms = std::fs::metadata(&staged).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged, perms).expect("chmod");
        std::fs::rename(staged, &path).expect("rename");
        path
    }

    #[cfg(unix)]
    #[test]
    fn a_versioned_binary_reports_the_sequencefile_contract() {
        let bin = fake_binary(
            "versioned",
            "#!/bin/sh\necho 'vk_gaussian_splatting 1.4.0'\n",
        );
        let status = status(&bin, Duration::from_secs(2));
        assert!(status.is_available(), "{status:?}");
        assert_eq!(status.version.as_deref(), Some("1.4.0"));
        assert!(status.capabilities.contains(&Capability::SplatRender));
        assert!(status.capabilities.contains(&Capability::Sequencefile));
        let details: Vec<&str> = status.capabilities.iter().map(Capability::as_str).collect();
        let contract = details
            .iter()
            .find(|detail| detail.contains("SEQUENCE"))
            .expect("contract detail");
        for flag in ["--activateCameraPreset", "--pipeline", "--screenshot"] {
            assert!(contract.contains(flag), "{contract}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn help_output_is_an_honest_version_fallback() {
        let bin = fake_binary(
            "help-only",
            "#!/bin/sh\nif [ \"$1\" = \"--help\" ]; then echo 'vk_gaussian_splatting 0.9'; exit 0; fi\nexit 2\n",
        );
        let version = probe(&bin, Duration::from_secs(2)).expect("probe").0;
        assert_eq!(version, "0.9");
    }

    #[test]
    fn a_missing_binary_is_a_named_not_installed_status() {
        let bin = scratch("missing").join("no-such-vkgs");
        let status = status(&bin, Duration::from_secs(2));
        assert_eq!(status.id, ToolId::SplatRenderer);
        assert_eq!(status.unavailable, Some(Unavailable::NotInstalled));
        assert!(status.version.is_none());
        assert!(status.capabilities.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_broken_binary_is_a_probe_failed_status() {
        let bin = fake_binary("broken", "#!/bin/sh\necho 'vk: no device' >&2\nexit 3\n");
        let status = status(&bin, Duration::from_secs(2));
        assert!(
            matches!(status.unavailable, Some(Unavailable::ProbeFailed(_))),
            "{status:?}"
        );
    }
}
