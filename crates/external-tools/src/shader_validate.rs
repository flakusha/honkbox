//! Headless shader compile-check hook (TASK-shader-validate-hook).
//!
//! The validator is an external binary, opt-in via `HONKBOX_SHADER_VALIDATOR`
//! (default `godot` on `PATH`). It is never linked or vendored. The probe
//! proves the binary answers a version request; the validate call hands it one
//! staged shader file and reads diagnostics from stdout. Godot-family binaries
//! get a throwaway headless project plus a one-shot `GDScript` that actually
//! compiles the shader through `RenderingServer`; a bare positional shader path
//! only prints the engine banner and never validates anything.
//!
//! - empty stdout or `OK` lines ⇒ the shader compiles;
//! - `path:line: message` or Godot's `SHADER ERROR` + `E <n>->` ⇒ named issue.
//!
//! A nonzero exit WITH diagnostics is a shader verdict (compilers do this);
//! a silent nonzero exit is a tool failure. No shader text is
//! stored in this crate: callers stage it in a per-request temp directory.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{
    Capability, Status, ToolId, Unavailable, probe_version, run_tool_output, status_from_probe,
};

/// Environment variable naming the validator binary (operator-supplied path).
pub const SHADER_VALIDATOR_ENV: &str = "HONKBOX_SHADER_VALIDATOR";

/// Binary used when [`SHADER_VALIDATOR_ENV`] is unset.
pub const DEFAULT_VALIDATOR: &str = "godot";

/// One compile failure reported by the validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderIssue {
    /// Shader file name as the validator reported it (never a host path).
    pub file: String,
    /// Source line, when the validator named one.
    pub line: Option<u32>,
    /// Diagnostic text.
    pub message: String,
}

/// Diagnostics from one validation run. An empty list means the shader
/// compiled; a failure always carries at least one issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderReport {
    /// Compile failures, in the order the validator printed them.
    pub issues: Vec<ShaderIssue>,
}

impl ShaderReport {
    /// Whether the shader compiled.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

/// The configured validator binary (never a value from a request).
#[must_use]
pub fn binary() -> PathBuf {
    std::env::var_os(SHADER_VALIDATOR_ENV)
        .map_or_else(|| PathBuf::from(DEFAULT_VALIDATOR), PathBuf::from)
}

/// Probe the validator's version.
pub fn probe(binary: &Path, timeout: Duration) -> Result<(String, Vec<Capability>), Unavailable> {
    let version = probe_version(binary, timeout)?;
    Ok((version, vec![Capability::ShaderValidate]))
}

/// 501-ready status: absent, broken, or available with its version.
#[must_use]
pub fn status(binary: &Path, timeout: Duration) -> Status {
    status_from_probe(ToolId::ShaderValidator, probe(binary, timeout))
}

/// Compile-check one staged shader file.
///
/// The argument is a path this crate's caller staged; no network and no
/// engine project state is involved. A missing or failing binary comes back as
/// [`Unavailable`], so the HTTP layer can answer a named 501/500 instead of a
/// silent skip. Godot gets a throwaway project: its bare positional argument
/// only prints the engine banner and never compiles the shader.
pub fn validate(
    binary: &Path,
    shader: &Path,
    timeout: Duration,
) -> Result<ShaderReport, Unavailable> {
    if is_godot(binary) {
        return validate_godot(binary, shader, timeout);
    }
    let path = shader
        .to_str()
        .ok_or_else(|| Unavailable::ProbeFailed("staged shader path is not UTF-8".to_owned()))?;
    let output = run_tool_output(binary, &[path], timeout)?;
    let diagnostics = if output.stdout.trim().is_empty() {
        output.stderr.as_str()
    } else {
        output.stdout.as_str()
    };
    let issues: Vec<ShaderIssue> = diagnostics
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != "OK")
        .map(parse_issue)
        .collect();
    if issues.is_empty() && !output.success && diagnostics.trim().is_empty() {
        return Err(Unavailable::ProbeFailed(output.status));
    }
    Ok(ShaderReport { issues })
}

fn is_godot(binary: &Path) -> bool {
    binary
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "godot" || name.starts_with("godot-"))
}

/// Godot only compiles a shader when a rendering server is told to set its
/// code. Stage a minimal project + one-shot script, run it headless, and remove
/// the project on every path. The staged shader's own name is reported to the
/// caller; Godot's diagnostics name the engine's shader slot instead.
fn validate_godot(
    binary: &Path,
    shader: &Path,
    timeout: Duration,
) -> Result<ShaderReport, Unavailable> {
    const PROJECT: &str = "config_version=5\n\n[application]\nconfig/name=\"ShaderProbe\"\n";
    const SCRIPT: &str = "extends SceneTree\n\nfunc _init():\n\tvar source = load(\"res://shader.gdshader\")\n\tvar shader = RenderingServer.shader_create()\n\tRenderingServer.shader_set_code(shader, source.code)\n\tRenderingServer.free_rid(shader)\n\tquit()\n";

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let project = std::env::temp_dir().join(format!(
        "honkbox-godot-shader-{}-{stamp}",
        std::process::id()
    ));
    let result = (|| {
        std::fs::create_dir_all(&project)
            .map_err(|err| Unavailable::ProbeFailed(format!("stage godot project: {err}")))?;
        std::fs::copy(shader, project.join("shader.gdshader"))
            .map_err(|err| Unavailable::ProbeFailed(format!("stage godot shader: {err}")))?;
        std::fs::write(project.join("project.godot"), PROJECT)
            .map_err(|err| Unavailable::ProbeFailed(format!("stage godot config: {err}")))?;
        std::fs::write(project.join("validate.gd"), SCRIPT)
            .map_err(|err| Unavailable::ProbeFailed(format!("stage godot script: {err}")))?;
        let project_dir = project.to_str().ok_or_else(|| {
            Unavailable::ProbeFailed("godot project path is not UTF-8".to_owned())
        })?;
        let script = project.join("validate.gd");
        let script_path = script
            .to_str()
            .ok_or_else(|| Unavailable::ProbeFailed("godot script path is not UTF-8".to_owned()))?;
        let output = run_tool_output(
            binary,
            &["--headless", "--path", project_dir, "--script", script_path],
            timeout,
        )?;
        let diagnostics = if output.stdout.trim().is_empty() {
            output.stderr.as_str()
        } else {
            output.stdout.as_str()
        };
        let file = shader
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("shader.gdshader");
        let issues = godot_issues(diagnostics, file);
        if issues.is_empty() && !output.success && diagnostics.trim().is_empty() {
            return Err(Unavailable::ProbeFailed(output.status));
        }
        Ok(ShaderReport { issues })
    })();
    let _ = std::fs::remove_dir_all(&project);
    result
}

/// Godot prints an `E   <line>->` marker before each `SHADER ERROR:` message.
fn godot_issues(diagnostics: &str, file: &str) -> Vec<ShaderIssue> {
    let mut issues = Vec::new();
    let mut pending_line = None;
    for line in diagnostics.lines() {
        let line = line.trim();
        if let Some((before, _)) = line.split_once("->")
            && let Some(number) = before.strip_prefix('E')
            && let Ok(parsed) = number.trim().parse::<u32>()
        {
            pending_line = Some(parsed);
        }
        if let Some(message) = line.strip_prefix("SHADER ERROR:") {
            issues.push(ShaderIssue {
                file: file.to_owned(),
                line: pending_line,
                message: message.trim().to_owned(),
            });
            pending_line = None;
        }
    }
    if issues.is_empty() && diagnostics.contains("Shader compilation failed") {
        issues.push(ShaderIssue {
            file: file.to_owned(),
            line: pending_line,
            message: "Shader compilation failed".to_owned(),
        });
    }
    issues
}

/// Parse `path:line: message`; a line without those separators is still a
/// named issue (message only), never silently dropped.
fn parse_issue(line: &str) -> ShaderIssue {
    let (file, rest) = line.split_once(':').unwrap_or(("", line));
    let (number, message) = match rest.split_once(':') {
        Some((number, message)) => (number.trim().parse::<u32>().ok(), message.trim()),
        None => (None, rest.trim()),
    };
    ShaderIssue {
        file: file.to_owned(),
        line: number,
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "honkbox-shader-validate-{}-{}-{tag}",
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
        let path = dir.join("validator");
        let staged = dir.join("validator.staged");
        std::fs::write(&staged, script).expect("fake validator");
        let mut perms = std::fs::metadata(&staged).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged, perms).expect("chmod");
        std::fs::rename(staged, &path).expect("rename");
        path
    }

    #[test]
    fn a_missing_validator_is_a_named_not_installed_status() {
        let bin = scratch("missing").join("no-such-validator");
        let status = status(&bin, Duration::from_secs(2));
        assert_eq!(status.id, ToolId::ShaderValidator);
        assert_eq!(status.unavailable, Some(Unavailable::NotInstalled));
        assert_eq!(status.capabilities, Vec::new());
    }

    #[cfg(unix)]
    #[test]
    fn a_versioned_validator_is_available() {
        let bin = fake_binary(
            "versioned",
            "#!/bin/sh\necho 'Godot Engine v4.2.1.stable.official'\n",
        );
        let status = status(&bin, Duration::from_secs(2));
        assert!(status.is_available(), "{status:?}");
        assert_eq!(status.version.as_deref(), Some("v4.2.1.stable.official"));
        assert_eq!(status.capabilities, vec![Capability::ShaderValidate]);
    }

    #[cfg(unix)]
    #[test]
    fn a_broken_validator_is_a_probe_failed_status() {
        let bin = fake_binary("broken", "#!/bin/sh\necho 'bad validator' >&2\nexit 4\n");
        let status = status(&bin, Duration::from_secs(2));
        assert!(
            matches!(status.unavailable, Some(Unavailable::ProbeFailed(_))),
            "{status:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn validation_names_the_shader_and_line() {
        let bin = fake_binary(
            "diagnostics",
            "#!/bin/sh\nprintf '%s\\n' 'hero.gdshader:3: unexpected token' 'other.gdshader:17: bad uniform'\n",
        );
        let shader = scratch("diagnostics").join("hero.gdshader");
        std::fs::write(&shader, "shader_type spatial;\n").expect("staged shader");
        let report = validate(&bin, &shader, Duration::from_secs(2)).expect("validate");
        assert!(!report.is_valid());
        assert_eq!(report.issues.len(), 2);
        assert_eq!(report.issues[0].file, "hero.gdshader");
        assert_eq!(report.issues[0].line, Some(3));
        assert_eq!(report.issues[0].message, "unexpected token");
        assert_eq!(report.issues[1].line, Some(17));
    }

    #[cfg(unix)]
    #[test]
    fn a_compiler_style_nonzero_exit_with_diagnostics_is_a_verdict() {
        let bin = fake_binary(
            "nonzero",
            "#!/bin/sh\nprintf '%s\\n' 'hero.gdshader:9: bad type' >&2\nexit 1\n",
        );
        let shader = scratch("nonzero").join("hero.gdshader");
        std::fs::write(&shader, "shader_type spatial;\n").expect("staged shader");
        let report = validate(&bin, &shader, Duration::from_secs(2)).expect("validate");
        assert!(!report.is_valid());
        assert_eq!(report.issues[0].line, Some(9));
    }

    #[cfg(unix)]
    #[test]
    fn a_clean_validation_has_no_issues() {
        let bin = fake_binary("clean", "#!/bin/sh\necho OK\n");
        let shader = scratch("clean").join("hero.gdshader");
        std::fs::write(&shader, "shader_type spatial;\n").expect("staged shader");
        let report = validate(&bin, &shader, Duration::from_secs(2)).expect("validate");
        assert!(report.is_valid());
        assert_eq!(report.issues, Vec::new());
    }

    #[cfg(unix)]
    #[test]
    fn godot_validation_is_headless() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = scratch("headless");
        let args_log = dir.join("args");
        let bin = dir.join("godot");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\ntest -f \"$3/project.godot\" || exit 9\ntest -f \"$3/shader.gdshader\" || exit 8\ntest -f \"$5\" || exit 7\necho OK\n",
            args_log.display()
        );
        std::fs::write(&bin, script).expect("fake godot");
        let mut perms = std::fs::metadata(&bin).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bin, perms).expect("chmod");
        let shader = dir.join("hero.gdshader");
        std::fs::write(&shader, "shader_type spatial;\n").expect("staged shader");

        let report = validate(&bin, &shader, Duration::from_secs(2)).expect("validate");
        assert!(report.is_valid());
        let args = std::fs::read_to_string(&args_log).expect("args log");
        let lines: Vec<_> = args.lines().collect();
        assert_eq!(lines.first(), Some(&"--headless"));
        assert_eq!(lines.get(1), Some(&"--path"));
        assert!(
            lines
                .get(2)
                .is_some_and(|path| path.contains("honkbox-godot-shader-")),
            "{lines:?}"
        );
        assert_eq!(lines.get(3), Some(&"--script"));
    }

    #[test]
    fn godot_diagnostics_name_the_shader_and_line() {
        let diagnostics = "--Main Shader--\nE   5-> }\nSHADER ERROR: Expected a ';'.\n";
        let issues = godot_issues(diagnostics, "hero.gdshader");
        assert_eq!(
            issues,
            vec![ShaderIssue {
                file: "hero.gdshader".to_owned(),
                line: Some(5),
                message: "Expected a ';'.".to_owned(),
            }]
        );
    }
}
