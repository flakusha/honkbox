//! Blender add-on and operator probe (`TASK-blender-addon-capability-probe`).
//!
//! The probe asks Blender which add-ons are installed and which operators can
//! run headless, and classifies the add-ons into tiers. The Blender side only
//! reports facts; the tiers and the headless verdict are decided here, so both
//! are unit-tested without a Blender install.
//!
//! Operators are never executed - only `poll()` is consulted, which has no
//! side effects. Executing an operator can write files, and a probe must not.
//!
//! Size: the payload is one JSON line, bounded by the installed add-ons; it
//! is read whole and never truncated. No cap is declared: the bytes come
//! from the local Blender install, not from user-supplied input.
//!
//! Extensibility: the capability vocabulary this feeds is OPEN
//! (`addon:<id>`, `op:<id>`), so a new kind is additive - there is no closed
//! enum to widen, and the `api_version` contract does not change.

use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::{Unavailable, run_tool};

/// Marker the add-on probe prints before its JSON payload.
const ADDON_MARKER: &str = "HONKBOX_ADDONS=";

/// The probe script, executed by `blender -b -noaudio --python-expr`.
const ADDON_PROBE_SCRIPT: &str = include_str!("blender_addons.py");

/// Operators honkbox may offer as headless jobs (the verified baseline).
///
/// `object.quadriflow` is listed on purpose: Blender 4.5 dropped it, and the
/// probe has to report that honestly instead of advertising a missing
/// operator.
const BASELINE_OPERATORS: &[&str] = &[
    "object.quadriflow",
    "object.voxel_remesh",
    "object.modifier_add",
    "object.modifier_apply",
    "object.bake",
    "uv.smart_project",
    "uv.unwrap",
];

/// License tokens that mark a third-party add-on as free/open source.
///
/// A third-party add-on without a recognized token stays detect-only: honkbox
/// never runs an add-on whose license it cannot read.
const FREE_LICENSE_TOKENS: &[&str] = &[
    "apache",
    "artistic",
    "bsd",
    "cc0",
    "gpl",
    "isc",
    "lgpl",
    "mit",
    "mpl",
    "unlicense",
    "zlib",
];

/// How an installed add-on may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddonTier {
    /// Ships with Blender itself.
    BuiltIn,
    /// Third-party add-on with a recognized free/open-source license.
    FreeOss,
    /// Third-party add-on without a proven free license: detect only.
    PaidDetectOnly,
}

impl AddonTier {
    /// Stable lowercase identifier (never user data).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BuiltIn => "built_in",
            Self::FreeOss => "free_oss",
            Self::PaidDetectOnly => "paid_detect_only",
        }
    }

    /// Whether honkbox may offer the add-on as a callable capability.
    ///
    /// A third-party add-on whose license honkbox cannot read stays
    /// detect-only: it is recorded, never run.
    pub const fn is_callable(self) -> bool {
        !matches!(self, Self::PaidDetectOnly)
    }
}

/// One installed Blender add-on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddonInfo {
    /// Stable add-on id (the extension id, or the module name).
    pub id: String,
    /// Version the add-on declared.
    pub version: Option<String>,
    /// License the add-on declared.
    pub license: Option<String>,
    /// Tier the probe classified it as.
    pub tier: AddonTier,
    /// Whether the add-on is enabled in the probed profile.
    pub enabled: bool,
}

/// One probed Blender operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorInfo {
    /// Operator id (`"uv.smart_project"`).
    pub id: String,
    /// The operator exists in this build and its poll passes headless.
    pub headless: bool,
}

/// What the add-on/operator probe learned about one Blender install.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlenderAddons {
    /// Every installed add-on, sorted by id.
    pub addons: Vec<AddonInfo>,
    /// Every probed operator, sorted by id.
    pub operators: Vec<OperatorInfo>,
}

/// Probe the installed Blender add-ons and the baseline operators.
///
/// The probe runs with the user's Blender profile (no `--factory-startup`),
/// so the reported enabled state is the one a Blender job would get.
pub fn probe(bin: &Path, timeout: Duration) -> Result<BlenderAddons, Unavailable> {
    let script = format!(
        "HONKBOX_CANDIDATES = {}\n{ADDON_PROBE_SCRIPT}",
        serde_json::to_string(BASELINE_OPERATORS).unwrap_or_else(|_| "[]".to_string())
    );
    let output = run_tool(bin, &["-b", "-noaudio", "--python-expr", &script], timeout)?;
    parse(&output)
        .ok_or_else(|| Unavailable::ProbeFailed("unreadable add-on probe output".to_string()))
}

/// One add-on as the Blender side reports it.
#[derive(Debug, Deserialize)]
struct RawAddon {
    /// Add-on id.
    id: String,
    /// Path of the add-on module.
    #[serde(default)]
    file: String,
    /// Declared version.
    #[serde(default)]
    version: Option<String>,
    /// Declared license.
    #[serde(default)]
    license: Option<String>,
    /// Whether the add-on is enabled.
    #[serde(default)]
    enabled: bool,
}

/// One operator as the Blender side reports it.
#[derive(Debug, Deserialize)]
struct RawOperator {
    /// Operator id.
    id: String,
    /// Whether the operator exists in this build.
    #[serde(default)]
    available: bool,
    /// Whether its poll passes with an active object.
    #[serde(default)]
    object_poll: bool,
    /// Whether its poll passes in edit mode.
    #[serde(default)]
    edit_poll: bool,
}

/// The whole probe payload.
#[derive(Debug, Deserialize)]
struct RawProbe {
    /// Bundled scripts root; built-in add-ons live below it.
    scripts: String,
    /// Installed add-ons.
    addons: Vec<RawAddon>,
    /// Probed operators.
    operators: Vec<RawOperator>,
}

/// Parse the add-on probe payload out of the Blender output.
fn parse(output: &str) -> Option<BlenderAddons> {
    let line = output
        .lines()
        .find(|line| line.trim_start().starts_with(ADDON_MARKER))?;
    let (_, payload) = line.trim_start().split_once(ADDON_MARKER)?;
    let raw: RawProbe = serde_json::from_str(payload.trim()).ok()?;
    let RawProbe {
        scripts,
        addons,
        operators,
    } = raw;
    Some(BlenderAddons {
        addons: addons
            .into_iter()
            .map(|addon| classify(addon, &scripts))
            .collect(),
        operators: operators.into_iter().map(classify_operator).collect(),
    })
}

/// Classify one installed add-on into a tier.
fn classify(addon: RawAddon, scripts_root: &str) -> AddonInfo {
    let tier = if is_builtin(&addon.file, scripts_root) {
        AddonTier::BuiltIn
    } else if is_free_license(addon.license.as_deref()) {
        AddonTier::FreeOss
    } else {
        AddonTier::PaidDetectOnly
    };
    AddonInfo {
        id: addon.id,
        version: addon.version,
        license: addon.license,
        tier,
        enabled: addon.enabled,
    }
}

/// Whether the add-on ships with Blender itself.
fn is_builtin(file: &str, scripts_root: &str) -> bool {
    if file.is_empty() || scripts_root.is_empty() {
        return false;
    }
    Path::new(file).starts_with(Path::new(scripts_root))
}

/// Whether the declared license is a recognized free/open-source license.
fn is_free_license(license: Option<&str>) -> bool {
    let Some(license) = license else {
        return false;
    };
    let tokens: Vec<String> = license
        .to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect();
    FREE_LICENSE_TOKENS
        .iter()
        .any(|token| tokens.iter().any(|seen| seen == token))
}

/// Whether the operator exists in this build and can run headless.
fn classify_operator(operator: RawOperator) -> OperatorInfo {
    OperatorInfo {
        id: operator.id,
        headless: operator.available && (operator.object_poll || operator.edit_poll),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// One probe payload with a built-in, a free-OSS and a paid add-on.
    const PAYLOAD: &str = concat!(
        r#"{"scripts":"/opt/blender/4.5/scripts","addons":["#,
        r#"{"id":"cycles","file":"/opt/blender/4.5/scripts/addons_core/cycles/__init__.py","version":"4.5.0","license":null,"enabled":true},"#,
        r#"{"id":"honkbox_fake","file":"/home/u/.config/blender/4.5/scripts/addons/honkbox_fake/__init__.py","version":"1.2.3","license":"GPL-3.0-or-later","enabled":false},"#,
        r#"{"id":"paid_tool","file":"/home/u/.config/blender/4.5/scripts/addons/paid_tool/__init__.py","version":"2.0","license":"Commercial","enabled":true}],"#,
        r#""operators":[{"id":"uv.smart_project","available":true,"object_poll":false,"edit_poll":true},"#,
        r#"{"id":"view3d.view_axis","available":true,"object_poll":false,"edit_poll":false},"#,
        r#"{"id":"object.quadriflow","available":false,"object_poll":false,"edit_poll":false}]}"#,
    );

    /// The probe output as Blender prints it.
    fn output(payload: &str) -> String {
        format!("Blender 4.5.15 LTS\n{ADDON_MARKER}{payload}\nBlender quit\n")
    }

    #[test]
    fn addons_are_classified_into_tiers() {
        let probe = parse(&output(PAYLOAD)).expect("probe payload");
        let tiers: Vec<(&str, AddonTier)> = probe
            .addons
            .iter()
            .map(|addon| (addon.id.as_str(), addon.tier))
            .collect();
        assert_eq!(
            tiers,
            vec![
                ("cycles", AddonTier::BuiltIn),
                ("honkbox_fake", AddonTier::FreeOss),
                ("paid_tool", AddonTier::PaidDetectOnly),
            ]
        );
    }

    #[test]
    fn addon_versions_and_licenses_are_recorded() {
        let probe = parse(&output(PAYLOAD)).expect("probe payload");
        let fake = probe
            .addons
            .iter()
            .find(|addon| addon.id == "honkbox_fake")
            .expect("fake addon");
        assert_eq!(fake.version.as_deref(), Some("1.2.3"));
        assert_eq!(fake.license.as_deref(), Some("GPL-3.0-or-later"));
        assert!(!fake.enabled);
    }

    #[test]
    fn operators_need_a_passing_poll() {
        let probe = parse(&output(PAYLOAD)).expect("probe payload");
        let headless: Vec<&str> = probe
            .operators
            .iter()
            .filter(|operator| operator.headless)
            .map(|operator| operator.id.as_str())
            .collect();
        assert_eq!(headless, vec!["uv.smart_project"]);
    }

    #[test]
    fn an_unreadable_payload_is_not_a_probe_result() {
        assert!(parse("Blender 4.5.15 LTS\nno payload here\n").is_none());
    }

    #[test]
    fn free_licenses_are_recognized_by_token() {
        assert!(is_free_license(Some("MIT")));
        assert!(is_free_license(Some("SPDX:GPL-3.0-or-later")));
        assert!(is_free_license(Some("CC0-1.0")));
        assert!(!is_free_license(Some("Commercial")));
        assert!(!is_free_license(None));
    }

    /// A fake Blender that answers the version probe and prints one add-on
    /// payload: an enabled free-OSS add-on, a disabled paid one, a headless
    /// operator and a UI-only one.
    #[cfg(unix)]
    const FAKE_BLENDER: &str = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Blender 4.2.1 LTS"
  exit 0
fi
echo 'HONKBOX_ADDONS={"scripts":"/opt/blender/scripts","addons":[{"id":"honkbox_fake","file":"/home/u/blender/addons/honkbox_fake/__init__.py","version":"1.2.3","license":"GPL-3.0-or-later","enabled":true},{"id":"paid_tool","file":"/home/u/blender/addons/paid_tool/__init__.py","version":"2.0","license":"Commercial","enabled":false}],"operators":[{"id":"uv.smart_project","available":true,"object_poll":false,"edit_poll":true},{"id":"view3d.view_axis","available":true,"object_poll":false,"edit_poll":false}]}'
"#;

    /// Write an executable fake Blender and return its path.
    #[cfg(unix)]
    fn write_fake_blender(_tag: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = crate::scratch_dir("ext-addons").expect("scratch dir");
        let path = dir.join("blender");
        std::fs::write(&path, script).expect("fake blender");
        let mut perms = std::fs::metadata(&path).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod");
        path
    }

    #[cfg(unix)]
    #[test]
    fn probed_addons_and_operators_become_capabilities() {
        let bin = write_fake_blender("blender-addons", FAKE_BLENDER);
        let config = crate::Config {
            blender_path: Some(bin),
            comfyui_url: Some("http://127.0.0.1:1".to_string()),
            llama_server_url: None,
            timeout: Duration::from_secs(5),
        };
        let (status, addons) = crate::probe_blender(&config);
        let capabilities: Vec<&str> = status
            .capabilities
            .iter()
            .map(crate::Capability::as_str)
            .collect();
        assert!(capabilities.contains(&"addon:honkbox_fake"));
        assert!(!capabilities.contains(&"addon:paid_tool"));
        assert!(capabilities.contains(&"op:uv.smart_project"));
        assert!(!capabilities.contains(&"op:view3d.view_axis"));
        let tiers: Vec<AddonTier> = addons.addons.iter().map(|addon| addon.tier).collect();
        assert_eq!(tiers, vec![AddonTier::FreeOss, AddonTier::PaidDetectOnly]);
    }

    /// The real probe against the local Blender install.
    #[test]
    #[ignore = "needs a local Blender install"]
    fn the_local_blender_reports_addons_and_operators() {
        let found = probe(Path::new("blender"), Duration::from_secs(120)).expect("blender probe");
        assert_ne!(found.addons.len(), 0);
        assert!(found.operators.iter().any(|operator| operator.headless));
        // Blender's own add-ons must classify as built-in: the capability
        // filter drops every add-on whose license honkbox cannot read.
        let cycles = found
            .addons
            .iter()
            .find(|addon| addon.id == "cycles")
            .expect("the cycles add-on");
        assert_eq!(cycles.tier, AddonTier::BuiltIn);
        // The whole advertisement path on the real install.
        let config = crate::Config {
            blender_path: Some(PathBuf::from("blender")),
            comfyui_url: Some("http://127.0.0.1:1".to_string()),
            llama_server_url: None,
            timeout: Duration::from_secs(120),
        };
        let (status, _) = crate::probe_blender(&config);
        let advertised: Vec<&str> = status
            .capabilities
            .iter()
            .map(crate::Capability::as_str)
            .collect();
        assert!(advertised.contains(&"addon:cycles"), "{advertised:?}");
        assert!(
            advertised.contains(&"op:uv.smart_project"),
            "{advertised:?}"
        );
    }
}
