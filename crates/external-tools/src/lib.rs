//! External tool registry (`TASK-external-tool-registry`).
//!
//! Discovers the external engines honkbox can drive — the Blender CLI and a
//! `ComfyUI` server — once, caches the result, and re-probes only on an
//! explicit reload. Nothing here links an engine into the workspace and
//! nothing here starts an engine: this crate only *discovers*.
//!
//! Discovery is honest by construction: a missing tool reports
//! [`Unavailable::NotInstalled`], a present-but-misbehaving tool reports
//! [`Unavailable::ProbeFailed`] with the reason, and a capability is claimed
//! only after the probe that proves it succeeded. The server surface (DTO in
//! `crates/api-types`, the capabilities block, the 501 for a missing tool) is
//! wired by a follow-up hunk; this crate is the discovery half.
//!
//! Probes are startup/reload work, never per-request work: the Blender probe
//! spawns a process and the `ComfyUI` probe is an HTTP call, both bounded by
//! [`Config::timeout`].

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Default `ComfyUI` base URL (loopback — honkbox is local-first).
pub const DEFAULT_COMFYUI_URL: &str = "http://127.0.0.1:8188";

/// Marker the Blender Cycles probe prints before its JSON payload.
const CYCLES_MARKER: &str = "HONKBOX_CYCLES=";

/// Python expression that makes Blender print its Cycles device list.
const CYCLES_PROBE_EXPR: &str = "import bpy,json;p=bpy.context.preferences.addons['cycles'].preferences;p.get_devices();print('HONKBOX_CYCLES='+json.dumps([{'name':d.name,'type':d.type} for d in p.devices]))";

/// Closed vocabulary of external engines honkbox can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolId {
    /// Blender, driven headless through its command line.
    Blender,
    /// `ComfyUI`, driven through its HTTP API.
    ComfyUi,
}

impl ToolId {
    /// Every known tool, in a stable order.
    pub const ALL: [ToolId; 2] = [ToolId::Blender, ToolId::ComfyUi];

    /// Stable lowercase identifier (never user data).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Blender => "blender",
            Self::ComfyUi => "comfyui",
        }
    }
}

impl fmt::Display for ToolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A capability a tool is known to provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Headless rendering.
    Render,
    /// The Cycles add-on with a probed device list.
    Cycles,
    /// Workflow execution.
    Workflow,
}

impl Capability {
    /// Stable lowercase identifier (never user data).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Cycles => "cycles",
            Self::Workflow => "workflow",
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a tool cannot serve work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unavailable {
    /// The executable is not installed (or not where it was configured).
    NotInstalled,
    /// The tool is installed but the probe failed (timeout, unusable output).
    ProbeFailed(String),
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled => f.write_str("not installed"),
            Self::ProbeFailed(reason) => write!(f, "probe failed: {reason}"),
        }
    }
}

impl std::error::Error for Unavailable {}

/// What one probe learned about one tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// Which tool this describes.
    pub id: ToolId,
    /// Version the tool reported, when it reported one.
    pub version: Option<String>,
    /// Capabilities the probe proved.
    pub capabilities: Vec<Capability>,
    /// Devices the tool can use (driver strings; never secrets).
    pub devices: Vec<String>,
    /// Why the tool is unavailable (`None` when it is available).
    pub unavailable: Option<Unavailable>,
}

impl Status {
    /// Whether the tool can serve work.
    pub fn is_available(&self) -> bool {
        self.unavailable.is_none()
    }
}

/// Where the tools live and how long a probe may take.
#[derive(Debug, Clone)]
pub struct Config {
    /// Blender executable (`None` → resolved on `PATH`).
    pub blender_path: Option<PathBuf>,
    /// `ComfyUI` base URL (`None` → [`DEFAULT_COMFYUI_URL`]).
    pub comfyui_url: Option<String>,
    /// Per-probe timeout.
    pub timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            blender_path: None,
            comfyui_url: None,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Registry of the external engines honkbox can drive.
#[derive(Debug)]
pub struct Registry {
    config: Config,
    client: reqwest::Client,
    statuses: Vec<Status>,
}

impl Registry {
    /// Probe every known tool once and cache the result.
    pub async fn discover(config: Config) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let mut registry = Self {
            config,
            client,
            statuses: Vec::new(),
        };
        registry.reload().await;
        registry
    }

    /// Re-probe every tool, replacing the cached statuses.
    pub async fn reload(&mut self) {
        let mut statuses = Vec::with_capacity(ToolId::ALL.len());
        for id in ToolId::ALL {
            let status = match id {
                ToolId::Blender => probe_blender(&self.config),
                ToolId::ComfyUi => probe_comfyui(&self.config, &self.client).await,
            };
            statuses.push(status);
        }
        self.statuses = statuses;
    }

    /// Cached status of every tool.
    pub fn statuses(&self) -> &[Status] {
        &self.statuses
    }

    /// Cached status of one tool (`None` before the first probe).
    pub fn status(&self, id: ToolId) -> Option<&Status> {
        self.statuses.iter().find(|status| status.id == id)
    }

    /// Whether one tool can serve work.
    pub fn is_available(&self, id: ToolId) -> bool {
        self.status(id).is_some_and(Status::is_available)
    }
}

/// Probe the Blender CLI: version first, then the Cycles device list.
fn probe_blender(config: &Config) -> Status {
    let bin = config
        .blender_path
        .as_deref()
        .unwrap_or_else(|| Path::new("blender"));
    let version_out = match run_tool(bin, &["--version"], config.timeout) {
        Ok(out) => out,
        Err(unavailable) => return unavailable_status(ToolId::Blender, unavailable),
    };
    let Some(version) = parse_blender_version(&version_out) else {
        return unavailable_status(
            ToolId::Blender,
            Unavailable::ProbeFailed("unexpected --version output".to_string()),
        );
    };

    let mut capabilities = vec![Capability::Render];
    let mut devices = Vec::new();
    let cycles_out = run_tool(
        bin,
        &[
            "-b",
            "--factory-startup",
            "-noaudio",
            "--python-expr",
            CYCLES_PROBE_EXPR,
        ],
        config.timeout,
    );
    if let Ok(out) = cycles_out
        && let Some(found) = parse_cycles_devices(&out)
    {
        capabilities.push(Capability::Cycles);
        devices = found;
    }

    Status {
        id: ToolId::Blender,
        version: Some(version),
        capabilities,
        devices,
        unavailable: None,
    }
}

/// Probe a `ComfyUI` server through its `/system_stats` endpoint.
async fn probe_comfyui(config: &Config, client: &reqwest::Client) -> Status {
    let base = config
        .comfyui_url
        .as_deref()
        .unwrap_or(DEFAULT_COMFYUI_URL)
        .trim_end_matches('/');
    let url = format!("{base}/system_stats");

    let Ok(response) = client.get(&url).send().await else {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{url}: unreachable")),
        );
    };
    if !response.status().is_success() {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{url}: HTTP {}", response.status())),
        );
    }
    let Ok(body) = response.json::<serde_json::Value>().await else {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{url}: invalid JSON body")),
        );
    };
    let Some(version) = body
        .pointer("/system/comfyui_version")
        .and_then(serde_json::Value::as_str)
    else {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{url}: no comfyui_version in the payload")),
        );
    };

    let devices = body
        .get("devices")
        .and_then(serde_json::Value::as_array)
        .map(|devices| {
            devices
                .iter()
                .filter_map(|device| {
                    device
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default();

    Status {
        id: ToolId::ComfyUi,
        version: Some(version.to_string()),
        capabilities: vec![Capability::Workflow],
        devices,
        unavailable: None,
    }
}

/// A status for a tool that could not be probed.
fn unavailable_status(id: ToolId, unavailable: Unavailable) -> Status {
    Status {
        id,
        version: None,
        capabilities: Vec::new(),
        devices: Vec::new(),
        unavailable: Some(unavailable),
    }
}

/// Run a tool with a timeout and return its stdout.
fn run_tool(bin: &Path, args: &[&str], timeout: Duration) -> Result<String, Unavailable> {
    let mut child = Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => Unavailable::NotInstalled,
            _ => Unavailable::ProbeFailed(format!("spawn failed: {err}")),
        })?;

    // Drain both pipes on their own threads: a full pipe would block the
    // child, and the timeout below would then kill a healthy tool.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = std::thread::spawn(move || stdout.map_or_else(String::new, drain_stream));
    let err_handle = std::thread::spawn(move || stderr.map_or_else(String::new, drain_stream));

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return Err(Unavailable::ProbeFailed(format!(
                        "timed out after {timeout:?}"
                    )));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => {
                let _ = out_handle.join();
                let _ = err_handle.join();
                return Err(Unavailable::ProbeFailed(format!("wait failed: {err}")));
            }
        }
    };

    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    if !status.success() {
        let detail = stderr.lines().next().unwrap_or_default().trim();
        let reason = if detail.is_empty() {
            format!("{status}")
        } else {
            format!("{status}: {detail}")
        };
        return Err(Unavailable::ProbeFailed(reason));
    }
    Ok(stdout)
}

/// Read a pipe to the end (tool output is bytes; invalid UTF-8 is replaced).
fn drain_stream(mut stream: impl Read) -> String {
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// Parse the `Blender <version>` token out of `blender --version` output.
fn parse_blender_version(stdout: &str) -> Option<String> {
    stdout.lines().find_map(|line| {
        let rest = line.trim().strip_prefix("Blender ")?;
        let token = rest.split_whitespace().next()?;
        (!token.is_empty()).then(|| token.to_string())
    })
}

/// One device reported by the Blender Cycles probe.
#[derive(Debug, serde::Deserialize)]
struct CyclesDevice {
    /// Device name as the driver reports it.
    name: String,
    /// Device backend (`CUDA`, `OPTIX`, `HIP`, `ONEAPI`, `CPU`, ...).
    #[serde(rename = "type")]
    kind: String,
}

/// Parse the Cycles device list out of the Blender probe output.
fn parse_cycles_devices(stdout: &str) -> Option<Vec<String>> {
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with(CYCLES_MARKER))?;
    let (_, payload) = line.trim_start().split_once(CYCLES_MARKER)?;
    let devices: Vec<CyclesDevice> = serde_json::from_str(payload.trim()).ok()?;
    Some(
        devices
            .into_iter()
            .map(|device| format!("{}: {}", device.kind, device.name))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// A fake Blender that reports a version and one Cycles device.
    #[cfg(unix)]
    const FAKE_BLENDER: &str = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Blender 4.2.1 LTS"
  exit 0
fi
echo 'HONKBOX_CYCLES=[{"name":"NVIDIA RTX 4090","type":"CUDA"}]'
"#;

    /// A fake Blender whose version output is unusable.
    #[cfg(unix)]
    const GARBAGE_BLENDER: &str = "#!/bin/sh\necho 'not blender at all'\n";

    /// Per-test scratch directory (unique per test tag and process).
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("honkbox-ext-tools-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Write an executable fake Blender and return its path.
    #[cfg(unix)]
    fn write_fake_blender(tag: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = scratch_dir(tag).join("blender");
        std::fs::write(&path, script).expect("fake blender");
        let mut perms = std::fs::metadata(&path).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod");
        path
    }

    /// Serve one canned JSON body on loopback; returns the base URL.
    fn spawn_comfy_mock(body: &'static str) -> (String, mpsc::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stop_rx.try_recv() {
                    Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                let Ok(mut stream) = stream else { break };
                let mut buf = [0_u8; 1024];
                let _ = stream.read(&mut buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://{addr}"), stop_tx)
    }

    /// A config that never finds a tool (no Blender, no `ComfyUI` server).
    fn empty_config() -> Config {
        Config {
            blender_path: Some(PathBuf::from("/nonexistent/honkbox-blender")),
            comfyui_url: Some("http://127.0.0.1:1".to_string()),
            timeout: Duration::from_secs(2),
        }
    }

    #[tokio::test]
    async fn every_known_tool_is_probed_once() {
        let registry = Registry::discover(empty_config()).await;
        assert_eq!(registry.statuses().len(), ToolId::ALL.len());
        for id in ToolId::ALL {
            assert_eq!(registry.status(id).map(|status| status.id), Some(id));
        }
    }

    #[tokio::test]
    async fn missing_blender_is_reported_as_not_installed() {
        let registry = Registry::discover(empty_config()).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(!status.is_available());
        assert_eq!(status.unavailable, Some(Unavailable::NotInstalled));
        assert!(status.version.is_none());
        assert_eq!(status.capabilities, Vec::new());
        assert!(!registry.is_available(ToolId::Blender));
    }

    #[tokio::test]
    async fn unreachable_comfyui_is_reported_as_probe_failed() {
        let registry = Registry::discover(empty_config()).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(!status.is_available());
        assert!(matches!(
            status.unavailable,
            Some(Unavailable::ProbeFailed(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn blender_probe_reports_version_and_cycles_devices() {
        let bin = write_fake_blender("blender-ok", FAKE_BLENDER);
        let config = Config {
            blender_path: Some(bin),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(status.is_available());
        assert_eq!(status.version.as_deref(), Some("4.2.1"));
        assert_eq!(
            status.capabilities,
            vec![Capability::Render, Capability::Cycles]
        );
        assert_eq!(status.devices, vec!["CUDA: NVIDIA RTX 4090".to_string()]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn blender_probe_rejects_unusable_version_output() {
        let bin = write_fake_blender("blender-garbage", GARBAGE_BLENDER);
        let config = Config {
            blender_path: Some(bin),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(!status.is_available());
        assert!(matches!(
            status.unavailable,
            Some(Unavailable::ProbeFailed(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reload_picks_up_a_newly_installed_tool() {
        let path = scratch_dir("reload").join("blender");
        let config = Config {
            blender_path: Some(path.clone()),
            ..empty_config()
        };
        let mut registry = Registry::discover(config).await;
        assert_eq!(
            registry
                .status(ToolId::Blender)
                .and_then(|status| status.unavailable.clone()),
            Some(Unavailable::NotInstalled)
        );

        write_fake_blender("reload", FAKE_BLENDER);
        registry.reload().await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(status.is_available());
        assert_eq!(status.version.as_deref(), Some("4.2.1"));
    }

    #[tokio::test]
    async fn comfyui_probe_reports_version_and_devices() {
        let (url, _stop) = spawn_comfy_mock(
            r#"{"system":{"comfyui_version":"0.3.40"},"devices":[{"name":"cuda:0 (NVIDIA RTX 4090)"}]}"#,
        );
        let config = Config {
            comfyui_url: Some(url),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(status.is_available());
        assert_eq!(status.version.as_deref(), Some("0.3.40"));
        assert_eq!(status.capabilities, vec![Capability::Workflow]);
        assert_eq!(status.devices, vec!["cuda:0 (NVIDIA RTX 4090)".to_string()]);
    }
}
