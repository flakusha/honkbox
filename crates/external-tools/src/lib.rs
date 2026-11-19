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
//!
//! Limitation, stated plainly: a timed-out probe kills the direct child only,
//! not a process group — `killpg` lives in `libc` and the workspace forbids
//! `unsafe` everywhere. Process-group lifecycle is the job of
//! `TASK-process-supervisor`; the Blender probe itself spawns no helpers.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

mod blender_addons;

pub use blender_addons::{AddonInfo, AddonTier, BlenderAddons, OperatorInfo};

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Headless rendering.
    Render,
    /// The Cycles add-on with a probed device list.
    Cycles,
    /// Workflow execution.
    Workflow,
    /// A probed add-on (`"addon:<id>"`).
    Addon(String),
    /// A probed operator that runs headless (`"op:<id>"`).
    Operator(String),
}

impl Capability {
    /// Stable lowercase identifier (never user data).
    pub fn as_str(&self) -> &str {
        match self {
            Self::Render => "render",
            Self::Cycles => "cycles",
            Self::Workflow => "workflow",
            Self::Addon(id) | Self::Operator(id) => id.as_str(),
        }
    }

    /// A probed add-on capability (`"addon:<id>"`).
    pub fn addon(id: &str) -> Self {
        Self::Addon(format!("addon:{id}"))
    }

    /// A probed headless operator capability (`"op:<id>"`).
    pub fn operator(id: &str) -> Self {
        Self::Operator(format!("op:{id}"))
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
    /// Add-on and operator records of the last Blender probe.
    blender_addons: BlenderAddons,
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
            blender_addons: BlenderAddons::default(),
        };
        registry.reload().await;
        registry
    }

    /// Re-probe every tool, replacing the cached statuses.
    ///
    /// The tools are independent - the Blender chain is a local subprocess
    /// chain, `ComfyUI` is an HTTP call - so they are probed CONCURRENTLY: a
    /// tool that hangs until the probe timeout must not serialize the other.
    /// The blocking chain runs on `spawn_blocking`, so it never occupies a
    /// runtime worker.
    pub async fn reload(&mut self) {
        let blender = tokio::task::spawn_blocking({
            let config = self.config.clone();
            move || probe_blender(&config)
        });
        let comfyui = probe_comfyui(&self.config, &self.client);
        let (blender, comfyui) = tokio::join!(blender, comfyui);
        let (blender_status, blender_addons) = match blender {
            Ok(probed) => probed,
            Err(join_error) => (
                unavailable_status(
                    ToolId::Blender,
                    Unavailable::ProbeFailed(join_error.to_string()),
                ),
                BlenderAddons::default(),
            ),
        };
        let mut statuses = Vec::with_capacity(ToolId::ALL.len());
        for id in ToolId::ALL {
            statuses.push(match id {
                ToolId::Blender => blender_status.clone(),
                ToolId::ComfyUi => comfyui.clone(),
            });
        }
        self.statuses = statuses;
        self.blender_addons = blender_addons;
    }

    /// Add-ons the last Blender probe found (empty when it did not run).
    pub fn blender_addons(&self) -> &[AddonInfo] {
        &self.blender_addons.addons
    }

    /// Operators the last Blender probe checked (empty when it did not run).
    pub fn blender_operators(&self) -> &[OperatorInfo] {
        &self.blender_addons.operators
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

/// Probe the Blender CLI: version first, then the Cycles device list, then
/// the installed add-ons and the baseline operators.
fn probe_blender(config: &Config) -> (Status, BlenderAddons) {
    let bin = config
        .blender_path
        .as_deref()
        .unwrap_or_else(|| Path::new("blender"));
    let version_out = match run_tool(bin, &["--version"], config.timeout) {
        Ok(out) => out,
        Err(unavailable) => {
            return (
                unavailable_status(ToolId::Blender, unavailable),
                BlenderAddons::default(),
            );
        }
    };
    let Some(version) = parse_blender_version(&version_out) else {
        return (
            unavailable_status(
                ToolId::Blender,
                Unavailable::ProbeFailed("unexpected --version output".to_string()),
            ),
            BlenderAddons::default(),
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

    // The add-on probe is best effort: a probe that cannot run must not take
    // the renderer down with it. Nothing is claimed it did not prove.
    let mut addons = BlenderAddons::default();
    if let Ok(found) = blender_addons::probe(bin, config.timeout) {
        for addon in &found.addons {
            // A detect-only add-on (unreadable license) is recorded, never
            // offered: honkbox must not run code whose license it cannot read.
            if addon.enabled && addon.tier.is_callable() {
                capabilities.push(Capability::addon(&addon.id));
            }
        }
        for operator in &found.operators {
            if operator.headless {
                capabilities.push(Capability::operator(&operator.id));
            }
        }
        addons = found;
    }

    (
        Status {
            id: ToolId::Blender,
            version: Some(version),
            capabilities,
            devices,
            unavailable: None,
        },
        addons,
    )
}

/// Probe a `ComfyUI` server through `/system_stats` (version + devices) and
/// `/object_info` (the node catalog that proves workflows can run).
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

    // A version alone does not prove workflows: `/object_info` must expose
    // the node catalog before `Capability::Workflow` can be claimed.
    let object_url = format!("{base}/object_info");
    let Ok(object_response) = client.get(&object_url).send().await else {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{object_url}: unreachable")),
        );
    };
    if !object_response.status().is_success() {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{object_url}: HTTP {}", object_response.status())),
        );
    }
    let Ok(nodes) = object_response.json::<serde_json::Value>().await else {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{object_url}: invalid JSON body")),
        );
    };
    if nodes.as_object().is_none_or(serde_json::Map::is_empty) {
        return unavailable_status(
            ToolId::ComfyUi,
            Unavailable::ProbeFailed(format!("{object_url}: no workflow nodes")),
        );
    }

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
pub(crate) fn run_tool(
    bin: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<String, Unavailable> {
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

    /// A fake Blender that also reports three installed add-ons: a built-in,
    /// a free-OSS one and a paid one with an unreadable license.
    #[cfg(unix)]
    const FAKE_BLENDER_WITH_ADDONS: &str = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Blender 4.2.1 LTS"
  exit 0
fi
echo 'HONKBOX_CYCLES=[{"name":"NVIDIA RTX 4090","type":"CUDA"}]'
echo 'HONKBOX_ADDONS={"scripts":"/opt/blender/4.5/scripts","addons":[{"id":"cycles","file":"/opt/blender/4.5/scripts/addons_core/cycles/__init__.py","version":"4.5.0","license":null,"enabled":true},{"id":"honkbox_free","file":"/home/u/.config/blender/4.5/scripts/addons/honkbox_free/__init__.py","version":"1.2.3","license":"GPL-3.0-or-later","enabled":true},{"id":"paid_tool","file":"/home/u/.config/blender/4.5/scripts/addons/paid_tool/__init__.py","version":"2.0","license":"Commercial","enabled":true}],"operators":[]}'
"#;

    /// A fake Blender whose version output is unusable.
    #[cfg(unix)]
    const GARBAGE_BLENDER: &str = "#!/bin/sh\necho 'not blender at all'\n";

    /// A fake Blender that never exits (`exec` so the kill reaches the sleep).
    #[cfg(unix)]
    const HANGING_BLENDER: &str = "#!/bin/sh\nexec sleep 30\n";

    /// A fake Blender that fails with a message on stderr.
    #[cfg(unix)]
    const FAILING_BLENDER: &str = "#!/bin/sh\necho 'blender: fatal error' >&2\nexit 3\n";

    /// A fake Blender that answers the version probe but prints no device list.
    #[cfg(unix)]
    const VERSION_ONLY_BLENDER: &str = r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Blender 4.2.1 LTS"
  exit 0
fi
echo 'no cycles here'
"#;

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

    /// Serve one canned response on loopback; returns the base URL.
    fn spawn_comfy_mock(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, mpsc::Sender<()>) {
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
                    "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://{addr}"), stop_tx)
    }

    /// A `ComfyUI` mock that marks the request as SERVED before answering.
    ///
    /// The marker is the rendezvous point for the concurrency test: a peer
    /// that waits for it can only be released while this probe is in flight.
    #[cfg(unix)]
    fn spawn_comfy_mock_marking_served(marker: &Path) -> (String, mpsc::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let marker = marker.to_path_buf();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_rx.try_recv().is_ok() {
                    break;
                }
                let Ok(mut stream) = stream else { break };
                let mut buf = [0_u8; 1024];
                let _ = stream.read(&mut buf);
                std::fs::write(&marker, b"served").expect("served marker");
                let body = r#"{"system":{"comfyui_version":"0.3.40"}}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://{addr}"), stop_tx)
    }

    /// Healthy `/system_stats`, broken `/object_info`: the version is known
    /// but no workflow node catalog can be proven.
    fn spawn_comfy_without_object_info() -> (String, mpsc::Sender<()>) {
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
                let request = String::from_utf8_lossy(&buf);
                let (status_line, body) = if request.contains("GET /object_info") {
                    ("500 Internal Server Error", "object info unavailable")
                } else {
                    (
                        "200 OK",
                        r#"{"system":{"comfyui_version":"0.3.40"},"devices":[]}"#,
                    )
                };
                let response = format!(
                    "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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

    /// Owns only its tag-scoped scratch directory (the fake Blender script):
    /// no port, no shared global, no order dependence.
    /// The two tool probes must OVERLAP: the fake Blender refuses to finish
    /// its probes until the `ComfyUI` mock has served a request, and that can
    /// only happen while both probes are in flight. A sequential
    /// implementation would time the fake out and report Blender as
    /// unavailable.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_tool_probes_run_concurrently() {
        let dir = scratch_dir("concurrent");
        let gate = dir.join("served");
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo \"Blender 4.2.1 LTS\"\n  exit 0\nfi\ni=0\nwhile [ $i -lt 100 ]; do\n  [ -f \"{0}\" ] && break\n  i=$((i+1))\n  sleep 0.1\ndone\n[ -f \"{0}\" ] || exit 1\necho 'HONKBOX_CYCLES=[{{\"name\":\"NVIDIA RTX 4090\",\"type\":\"CUDA\"}}]'\n",
            gate.display()
        );
        let bin = write_fake_blender("gated", &script);
        let (url, _stop) = spawn_comfy_mock_marking_served(&gate);
        let config = Config {
            blender_path: Some(bin),
            comfyui_url: Some(url),
            timeout: Duration::from_secs(30),
        };
        let registry = Registry::discover(config).await;
        let blender = registry.status(ToolId::Blender).expect("blender status");
        assert!(blender.is_available(), "{blender:?}");
        // The payload only exists once the gate opened, so a sequential probe
        // pair (which never overlaps) leaves the device list empty.
        assert_eq!(blender.devices, vec!["CUDA: NVIDIA RTX 4090".to_string()]);
        let comfy = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(comfy.is_available(), "{comfy:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn detect_only_addons_are_recorded_but_never_offered() {
        let bin = write_fake_blender("blender-addons", FAKE_BLENDER_WITH_ADDONS);
        let config = Config {
            blender_path: Some(bin),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(status.is_available());
        assert_eq!(
            status.capabilities,
            vec![
                Capability::Render,
                Capability::Cycles,
                Capability::addon("cycles"),
                Capability::addon("honkbox_free"),
            ]
        );
        let paid = registry
            .blender_addons()
            .iter()
            .find(|addon| addon.id == "paid_tool")
            .expect("paid add-on record");
        assert_eq!(paid.tier, AddonTier::PaidDetectOnly);
        assert!(paid.enabled, "the add-on is installed and enabled");
        assert!(!paid.tier.is_callable());
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
        // A recycled pid can hand us a scratch dir that still holds the tool
        // from an earlier run; the test asserts the "missing" state first.
        let _ = std::fs::remove_file(&path);
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
            "200 OK",
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

    #[cfg(unix)]
    #[tokio::test]
    async fn blender_probe_times_out_on_a_hanging_tool() {
        let bin = write_fake_blender("blender-hang", HANGING_BLENDER);
        let config = Config {
            blender_path: Some(bin),
            timeout: Duration::from_millis(200),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(!status.is_available());
        let Some(Unavailable::ProbeFailed(reason)) = status.unavailable.clone() else {
            panic!("expected a probe failure, got {:?}", status.unavailable);
        };
        assert!(reason.contains("timed out"), "reason was {reason:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn blender_probe_reports_a_failing_exit_status() {
        let bin = write_fake_blender("blender-fail", FAILING_BLENDER);
        let config = Config {
            blender_path: Some(bin),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(!status.is_available());
        let Some(Unavailable::ProbeFailed(reason)) = status.unavailable.clone() else {
            panic!("expected a probe failure, got {:?}", status.unavailable);
        };
        assert!(
            reason.contains("blender: fatal error"),
            "the stderr detail must survive, reason was {reason:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn blender_without_a_device_probe_stays_available_without_cycles() {
        let bin = write_fake_blender("blender-no-cycles", VERSION_ONLY_BLENDER);
        let config = Config {
            blender_path: Some(bin),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::Blender).expect("blender status");
        assert!(status.is_available());
        assert_eq!(status.version.as_deref(), Some("4.2.1"));
        assert_eq!(status.capabilities, vec![Capability::Render]);
        assert_eq!(status.devices, Vec::<String>::new());
    }

    #[tokio::test]
    async fn comfyui_server_error_is_reported_as_probe_failed() {
        let (url, _stop) = spawn_comfy_mock("500 Internal Server Error", "{}");
        let config = Config {
            comfyui_url: Some(url),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(!status.is_available());
        assert!(matches!(
            status.unavailable,
            Some(Unavailable::ProbeFailed(_))
        ));
    }

    #[tokio::test]
    async fn comfyui_invalid_body_is_reported_as_probe_failed() {
        let (url, _stop) = spawn_comfy_mock("200 OK", "not json at all");
        let config = Config {
            comfyui_url: Some(url),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(!status.is_available());
        assert!(matches!(
            status.unavailable,
            Some(Unavailable::ProbeFailed(_))
        ));
    }

    #[tokio::test]
    async fn comfyui_payload_without_a_version_is_reported_as_probe_failed() {
        let (url, _stop) = spawn_comfy_mock("200 OK", r#"{"system":{}}"#);
        let config = Config {
            comfyui_url: Some(url),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(!status.is_available());
        assert!(matches!(
            status.unavailable,
            Some(Unavailable::ProbeFailed(_))
        ));
    }

    #[tokio::test]
    async fn comfyui_base_url_with_a_trailing_slash_is_probed_once() {
        let (url, _stop) = spawn_comfy_mock("200 OK", r#"{"system":{"comfyui_version":"0.3.40"}}"#);
        let config = Config {
            comfyui_url: Some(format!("{url}/")),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(status.is_available());
        assert_eq!(status.version.as_deref(), Some("0.3.40"));
    }

    #[tokio::test]
    async fn comfyui_misconfigured_url_is_reported_as_probe_failed() {
        let config = Config {
            comfyui_url: Some("not-a-url".to_string()),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(!status.is_available());
        assert!(matches!(
            status.unavailable,
            Some(Unavailable::ProbeFailed(_))
        ));
    }

    #[tokio::test]
    async fn comfyui_without_object_info_never_claims_workflow() {
        let (url, _stop) = spawn_comfy_without_object_info();
        let config = Config {
            comfyui_url: Some(url),
            ..empty_config()
        };
        let registry = Registry::discover(config).await;
        let status = registry.status(ToolId::ComfyUi).expect("comfyui status");
        assert!(!status.is_available());
        assert_eq!(status.capabilities, [] as [Capability; 0]);
        assert!(
            matches!(&status.unavailable, Some(Unavailable::ProbeFailed(reason)) if reason.contains("object_info")),
            "{status:?}"
        );
    }
}
