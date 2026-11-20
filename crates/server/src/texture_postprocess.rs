//! OIIO texture post-process capability (`TASK-texture-postprocess-wiring`).
//!
//! One probe, frozen: the first call probes `oiiotool` and `maketx` and every
//! later call reads that answer. The binaries come from
//! `HONKBOX_OIIO_BIN` / `HONKBOX_MAKETX_BIN` (default: the names on `PATH`).
//!
//! The probe result is a version token or a presence flag — never a path and
//! never a secret — so the capability row and the `501` detail can be served
//! without leaking the host layout.

// The capability row is consumed by routes/capabilities.rs.

use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;

use honkbox_api_types::TexturePostprocessCapability;
use honkbox_external_tools::{OiioTool, Unavailable, probe_texture_tools};

/// How long a version probe may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// The frozen probe outcome: the probed toolchain, or why it is unusable.
static PROBE: LazyLock<Result<OiioTool, Unavailable>> =
    LazyLock::new(|| probe_texture_tools(&oiiotool_bin(), &maketx_bin(), PROBE_TIMEOUT));

/// The `oiiotool` binary: `HONKBOX_OIIO_BIN`, else the name on `PATH`.
fn oiiotool_bin() -> PathBuf {
    std::env::var_os("HONKBOX_OIIO_BIN").map_or_else(|| PathBuf::from("oiiotool"), PathBuf::from)
}

/// The `maketx` binary: `HONKBOX_MAKETX_BIN`, else the name on `PATH`.
fn maketx_bin() -> PathBuf {
    std::env::var_os("HONKBOX_MAKETX_BIN").map_or_else(|| PathBuf::from("maketx"), PathBuf::from)
}

/// The capability row: available only when BOTH binaries passed their probe.
pub(crate) fn capability() -> TexturePostprocessCapability {
    match &*PROBE {
        Ok(tool) => TexturePostprocessCapability {
            available: true,
            oiiotool_version: Some(tool.oiiotool_version.clone()),
            maketx_version: Some(tool.maketx_version.clone()),
        },
        Err(_) => TexturePostprocessCapability {
            available: false,
            oiiotool_version: None,
            maketx_version: None,
        },
    }
}

/// The `501` detail: why the texture toolchain cannot serve work.
///
/// The reason is the tool's own verdict (not installed vs. probe failed) and
/// never a binary path.
pub(crate) fn unavailable_error() -> String {
    match &*PROBE {
        Ok(_) => "texture post-process tools are available".to_string(),
        Err(Unavailable::NotInstalled) => {
            "texture post-process tools are not installed (oiiotool/maketx)".to_string()
        }
        Err(Unavailable::ProbeFailed(_)) => {
            "texture post-process tools failed their version probe".to_string()
        }
    }
}

/// The binaries the route drives, in the order the toolchain expects them.
pub(crate) fn binaries() -> (PathBuf, PathBuf) {
    (oiiotool_bin(), maketx_bin())
}

/// How long one post-process subprocess may take. A version probe is near
/// instant, but tiling a real texture is not.
pub(crate) const RUN_TIMEOUT: Duration = Duration::from_secs(600);
