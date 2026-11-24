// Managed lifecycle for operator-declared CLI apps (`FEAT-cli-app-manager`).
//
// honkbox starts/stops declared apps, owns readiness, capped logs, process-
// group teardown and the ledger lease. Nothing runs without an explicit
// `POST /v1/apps/{id}/start`: no auto-start, no auto-restart, no shell, no
// path or argv from an HTTP request.
//
// Trust boundary: a PATH binary is arbitrary code. Locations come only from
// config/env, are pinned by canonicalization at probe, and are rechecked at
// start. A TOCTOU window remains between the recheck and `exec`; that is
// accepted for an operator-owned host and never offered to a remote caller.
//
// ponytail: process groups are unix-only (Windows job objects are a separate
// feature); readiness restarts its countdown on a 503 but is bounded at 3x
// the configured timeout so a permanently "loading" app cannot hang a stop.
// size-allow: 440

use std::collections::BTreeMap;
use std::io::Read;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use honkbox_external_tools::llama_server::{DEFAULT_LLAMA_SERVER, LLAMA_SERVER_ENV};
use reqwest::{Client, Url};
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
use rustix::process::{Pid, Signal};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
#[cfg(unix)]
use std::os::unix::process::CommandExt as _;
#[cfg(unix)]
use std::process::Child;

use crate::admission::{self, GpuLease};
use crate::config::{ManagedAppConfig, ManagedAppLaunch, ManagedAppResource, ManagedAppsConfig};

/// Health budget floor (llama-swap semantics: 120 s default, 15 s minimum).
#[cfg(test)]
const MIN_HEALTH_SECS: u64 = 15;
/// Capped stderr tail carried into a `failed` status.
const STDERR_TAIL_BYTES: usize = 4096;
/// Readiness poll cadence.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Observable lifecycle state of one declared app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Stopped,
    Starting,
    Ready,
    Failed,
    Stopping,
}

impl AppState {
    /// Stable lowercase wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Failed => "failed",
            Self::Stopping => "stopping",
        }
    }
}

/// A failure with an HTTP status the admin routes can return verbatim.
#[derive(Debug)]
pub struct AppError {
    pub status: StatusCode,
    pub message: String,
}

impl AppError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self { status, message: message.into() }
    }
}

/// Snapshot of one app, mapped to the additive API DTO by the route layer.
#[derive(Debug, Clone)]
pub struct AppStatus {
    pub id: String,
    pub state: AppState,
    pub launch: ManagedAppLaunch,
    pub resource_kind: ManagedAppResource,
    pub endpoint: Option<String>,
    pub version: Option<String>,
    pub error: Option<String>,
}

/// Where readiness is probed. Each app has its own route: llama-server serves
/// `/health`, `ComfyUI` does not (it registers `/queue`). A unix socket needs a
/// hand-rolled GET: the workspace's reqwest has no `unix-socket` feature and
/// no new dep is allowed.
#[derive(Debug, Clone)]
enum Health {
    Tcp(String, &'static str),
    Unix(PathBuf),
}

/// Everything resolved from config at start time. Rebuilt on every explicit
/// start; never mutated by a request.
#[derive(Debug, Clone)]
struct Resolved {
    program: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
    health: Health,
    endpoint: String,
    /// `ComfyUI`'s pinned version file, when the launch mode uses an operator
    /// checkout. The venv python's `--version` would report Python, not SD.
    version_file: Option<PathBuf>,
}

/// Mutable per-app record. The `Child` itself lives in the supervisor task so
/// only one waiter owns it; the route layer only observes `state`.
#[derive(Debug)]
struct Entry {
    id: String,
    config: ManagedAppConfig,
    state: AppState,
    version: Option<String>,
    error: Option<String>,
    endpoint: Option<String>,
    pid: Option<u32>,
    token: Option<CancellationToken>,
}

fn entry_lock(entry: &Arc<Mutex<Entry>>) -> MutexGuard<'_, Entry> {
    entry.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Owns every declared app. `entries` is immutable after construction; each
/// entry has its own mutex, so status reads never block another app's start.
pub struct Manager {
    entries: BTreeMap<String, Arc<Mutex<Entry>>>,
    shutdown: CancellationToken,
    client: Client,
    data_dir: Option<PathBuf>,
    comfyui_url: String,
}

impl Manager {
    /// Build the manager. No process is started here.
    pub fn new(
        apps: ManagedAppsConfig,
        data_dir: Option<PathBuf>,
        comfyui_url: &str,
    ) -> Result<Self, String> {
        let comfyui_url = validated_loopback(comfyui_url)?;
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|err| format!("managed app HTTP client: {err}"))?;
        let mut entries = BTreeMap::new();
        for (id, config) in [
            ("llama_server", apps.llama_server),
            ("comfyui", apps.comfyui),
        ] {
            entries.insert(
                id.to_owned(),
                Arc::new(Mutex::new(Entry {
                    id: id.to_owned(),
                    config,
                    state: AppState::Stopped,
                    version: None,
                    error: None,
                    endpoint: None,
                    pid: None,
                    token: None,
                })),
            );
        }
        Ok(Self { entries, shutdown: CancellationToken::new(), client, data_dir, comfyui_url })
    }

    /// Whether an app was declared and enabled by the operator.
    pub fn configured(&self, id: &str) -> bool {
        self.entries.get(id).is_some_and(|entry| entry_lock(entry).config.enabled)
    }

    /// Shared bounded HTTP client for the typed `ComfyUI` read surface.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The configured loopback base URL used by the typed `/comfyui` routes.
    pub fn comfyui_url(&self) -> &str {
        &self.comfyui_url
    }

    /// Only a `ready` app contributes an endpoint to the capability surface.
    pub fn ready(&self, id: &str) -> bool {
        self.entries.get(id).is_some_and(|entry| entry_lock(entry).state == AppState::Ready)
    }

    /// Snapshot of one declared app, or `None` for an unknown id.
    pub fn status(&self, id: &str) -> Option<AppStatus> {
        let entry = self.entries.get(id)?;
        let entry = entry_lock(entry);
        Some(AppStatus {
            id: entry.id.clone(),
            state: entry.state,
            launch: entry.config.launch,
            resource_kind: entry.config.resource_kind,
            endpoint: entry.endpoint.clone(),
            version: entry.version.clone(),
            error: entry.error.clone(),
        })
    }

    /// All declared apps, in stable id order (disabled ones included).
    pub fn list(&self) -> Vec<AppStatus> {
        self.entries.keys().filter_map(|id| self.status(id)).collect()
    }

    /// Start one app. The caller must have admin scope; the path is config-only.
    pub async fn start(self: &Arc<Self>, id: &str) -> Result<(), AppError> {
        let entry = self.entries.get(id).ok_or_else(|| {
            AppError::new(StatusCode::NOT_FOUND, format!("unknown app {id}"))
        })?;
        {
            let state = entry_lock(entry);
            if !state.config.enabled {
                return Err(AppError::new(
                    StatusCode::NOT_IMPLEMENTED,
                    format!("app {id} is not enabled in config"),
                ));
            }
            if !matches!(state.state, AppState::Stopped | AppState::Failed) {
                return Err(AppError::new(
                    StatusCode::CONFLICT,
                    format!("app {id} is {}", state.state.as_str()),
                ));
            }
        }
        #[cfg(not(unix))]
        return Err(AppError::new(
            StatusCode::NOT_IMPLEMENTED,
            "managed apps require unix process groups",
        ));
        #[cfg(unix)]
        {
            let resolved = resolve(entry, self.data_dir.as_deref())
                .map_err(|err| AppError::new(StatusCode::BAD_REQUEST, err))?;
            let version = probe_version(&resolved.program, resolved.version_file.as_deref()).await;
            recheck(&resolved.program)
                .map_err(|err| AppError::new(StatusCode::BAD_REQUEST, err))?;
            let kind = entry_lock(entry).config.resource_kind;
            let lease = admission::acquire(match kind {
                ManagedAppResource::Llm => honkbox_residency::ResourceKind::Llm,
                ManagedAppResource::Comfy => honkbox_residency::ResourceKind::Comfy,
            })
            .await
            .map_err(|(status, message)| AppError::new(status, message))?;
            let mut command = Command::new(&resolved.program);
            command
                .args(&resolved.args)
                .current_dir(&resolved.cwd)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .process_group(0);
            let mut child = match command.spawn() {
                Ok(child) => child,
                Err(err) => {
                    drop(lease);
                    return Err(AppError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("spawn {id}: {err}"),
                    ));
                }
            };
            let group = Pid::from_child(&child);
            let log = Arc::new(Mutex::new(String::new()));
            if let Some(pipe) = child.stderr.take() {
                let sink = Arc::clone(&log);
                std::thread::spawn(move || drain_stderr(pipe, &sink));
            }
            let token = self.shutdown.child_token();
            let timeout = Duration::from_secs(entry_lock(entry).config.health_timeout_secs);
            {
                let mut state = entry_lock(entry);
                state.state = AppState::Starting;
                state.version = version;
                state.error = None;
                state.endpoint = Some(resolved.endpoint.clone());
                state.pid = Some(child.id());
                state.token = Some(token.clone());
            }
            let health = resolved.health.clone();
            let supervisor_entry = Arc::clone(entry);
            tokio::spawn(supervise(
                Arc::clone(self), supervisor_entry, child, group, lease, token,
                health, timeout, log,
            ));
            Ok(())
        }
    }

    /// Stop one app: cancel its token, kill the whole group, release the lease.
    /// Idempotent on an already stopped/failed app.
    pub async fn stop(&self, id: &str) -> Result<(), AppError> {
        let entry = self.entries.get(id).ok_or_else(|| {
            AppError::new(StatusCode::NOT_FOUND, format!("unknown app {id}"))
        })?;
        {
            let mut state = entry_lock(entry);
            match state.state {
                AppState::Stopped | AppState::Failed => return Ok(()),
                AppState::Stopping => {}
                _ => {
                    state.state = AppState::Stopping;
                    if let Some(token) = &state.token {
                        token.cancel();
                    }
                }
            }
        }
        if let Some(token) = entry_lock(entry).token.clone() {
            token.cancel();
        }
        for _ in 0..200 {
            if matches!(entry_lock(entry).state, AppState::Stopped | AppState::Failed) {
                return Ok(());
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        Err(AppError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("app {id} teardown timed out"),
        ))
    }

    /// Server shutdown: cancel every child and wait for group teardown.
    #[allow(dead_code)] // `#[path]` test harnesses do not call this
    pub async fn shutdown(&self) {
        self.shutdown.cancel();
        for entry in self.entries.values() {
            if let Some(token) = entry_lock(entry).token.clone() {
                token.cancel();
            }
        }
        for _ in 0..200 {
            if self.list().iter().all(|app| matches!(app.state, AppState::Stopped | AppState::Failed)) {
                return;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }
}

/// Reject a non-loopback, credentialed or non-HTTP `ComfyUI` base URL.
fn validated_loopback(input: &str) -> Result<String, String> {
    let url = Url::parse(input).map_err(|err| format!("apps.comfyui url: {err}"))?;
    let host = url.host_str().unwrap_or_default();
    if url.scheme() != "http"
        || !url.username().is_empty()
        || url.password().is_some()
        || !matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
    {
        return Err("apps.comfyui url must be credential-free loopback HTTP".to_owned());
    }
    Ok(input.trim_end_matches('/').to_owned())
}

/// Test-only PATH override: integration fixtures carry their own fake binaries
/// without mutating the process-global env under parallel tests.
#[cfg(test)]
static PATH_OVERRIDE: Mutex<Option<std::ffi::OsString>> = Mutex::new(None);

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn set_path_for_tests(path: Option<std::ffi::OsString>) {
    *PATH_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = path;
}

fn path_env() -> Option<std::ffi::OsString> {
    #[cfg(test)]
    if let Some(path) = PATH_OVERRIDE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() {
        return Some(path);
    }
    std::env::var_os("PATH")
}

/// Resolve an operator-supplied program: absolute paths and bare PATH names.
fn resolve_program(name: &Path) -> Result<PathBuf, String> {
    if name.components().count() > 1 {
        return std::fs::canonicalize(name).map_err(|err| format!("binary {}: {err}", name.display()));
    }
    let path = path_env().unwrap_or_default();
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return std::fs::canonicalize(&candidate)
                .map_err(|err| format!("binary {}: {err}", candidate.display()));
        }
    }
    Err(format!("binary {} not found on PATH", name.display()))
}

/// Pin at probe, recheck at start. The remaining TOCTOU window is documented
/// in the module header and accepted for operator-owned hosts.
fn recheck(program: &Path) -> Result<(), String> {
    let now = std::fs::canonicalize(program)
        .map_err(|err| format!("binary changed before start: {err}"))?;
    if now != program {
        return Err(format!("binary {} changed after probe", program.display()));
    }
    Ok(())
}

/// Generate an app-only API key from the OS RNG (no auth bearer is ever
/// passed to a child). `ulid` is the fallback for a host without urandom.
fn generated_key() -> String {
    if let Ok(mut file) = std::fs::File::open("/dev/urandom") {
        let mut bytes = [0_u8; 32];
        if file.read_exact(&mut bytes).is_ok() {
            use std::fmt::Write as _;
            let mut key = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                let _ = write!(key, "{byte:02x}");
            }
            return key;
        }
    }
    format!("{}-{}", ulid::Ulid::new(), std::process::id())
}

/// Resolve one app into an absolute program, allowlisted argv and endpoint.
#[allow(clippy::too_many_lines)] // one flat argv builder for all three launch modes
fn resolve(entry: &Arc<Mutex<Entry>>, data_dir: Option<&Path>) -> Result<Resolved, String> {
    let entry = entry_lock(entry);
    let config = &entry.config;
    let is_llama = entry.id == "llama_server";
    let port = if let Some(port) = config.port {
        port
    } else {
        let listener = TcpListener::bind((config.host.as_str(), 0))
            .map_err(|err| format!("ephemeral port for {}: {err}", entry.id))?;
        listener.local_addr().map_err(|err| format!("ephemeral port: {err}"))?.port()
    };
    let model = config.model.as_ref().map(|path| path.to_string_lossy().into_owned());
    let mut version_file = None;
    if is_llama && config.launch == ManagedAppLaunch::Path && model.is_none() {
        return Err("apps.llama_server.model is required".to_owned());
    }
    let (program, cwd, script_args) = match config.launch {
        ManagedAppLaunch::Path => {
            let name = config.binary.clone().unwrap_or_else(|| {
                if is_llama {
                    std::env::var_os(LLAMA_SERVER_ENV)
                        .map_or_else(|| PathBuf::from(DEFAULT_LLAMA_SERVER), PathBuf::from)
                } else {
                    PathBuf::from("comfyui")
                }
            });
            (
                resolve_program(&name)?,
                data_dir.map_or_else(|| std::env::current_dir().unwrap_or_default(), Path::to_path_buf),
                Vec::new(),
            )
        }
        ManagedAppLaunch::Installation | ManagedAppLaunch::Script => {
            if is_llama {
                return Err("apps.llama_server supports only launch = \"path\"".to_owned());
            }
            let root = std::fs::canonicalize(
                config.installation.as_ref().ok_or("apps.comfyui.installation is required")?,
            )
            .map_err(|err| format!("installation: {err}"))?;
            version_file = Some(root.join("comfyui_version.py"));
            let python = std::fs::canonicalize(root.join(".venv/bin/python"))
                .map_err(|err| format!("installation venv python: {err}"))?;
            let args = if config.launch == ManagedAppLaunch::Installation {
                vec![
                    std::fs::canonicalize(root.join("main.py"))
                        .map_err(|err| format!("main.py: {err}"))?
                        .to_string_lossy()
                        .into_owned(),
                ]
            } else {
                let script = config.script.as_ref().ok_or("apps.comfyui.script is required")?;
                let path = std::fs::canonicalize(script)
                    .or_else(|_| std::fs::canonicalize(root.join(script)))
                    .map_err(|err| format!("script: {err}"))?;
                vec![path.to_string_lossy().into_owned()]
            };
            (python, root, args)
        }
    };
    let mut args = script_args;
    let (health, endpoint) = if is_llama {
        if config.host == "unix" {
            #[cfg(unix)]
            {
                let dir = data_dir
                    .ok_or("data dir is required for unix socket apps")?
                    .join("apps")
                    .join(&entry.id);
                std::fs::create_dir_all(&dir).map_err(|err| format!("socket dir: {err}"))?;
                std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
                    .map_err(|err| format!("socket dir mode: {err}"))?;
                let socket = dir.join("llama.sock");
                let _ = std::fs::remove_file(&socket);
                let path = socket.to_string_lossy().into_owned();
                args.extend([
                    "--model".into(), model.unwrap_or_default(), "--host".into(), path.clone(),
                    "--api-key".into(), generated_key(), "--cors-origins".into(),
                    "localhost".into(), "--no-webui".into(),
                ]);
                (Health::Unix(socket), format!("unix:{path}"))
            }
            #[cfg(not(unix))]
            {
                let _ = (&mut args, model);
                return Err("unix sockets require unix".to_owned());
            }
        } else {
            args.extend([
                "--model".into(), model.unwrap_or_default(), "--host".into(), config.host.clone(),
                "--port".into(), port.to_string(), "--api-key".into(), generated_key(),
                "--cors-origins".into(), "localhost".into(), "--no-webui".into(),
            ]);
            let url = format!("http://{}:{}", config.host, port);
            (Health::Tcp(url.clone(), "/health"), url)
        }
    } else {
        // ComfyUI has no unix-socket mode: force loopback TCP. No tools/agent
        // flags exist here, and no node graph comes from an HTTP request.
        args.extend([
            "--listen".into(), config.host.clone(), "--port".into(), port.to_string(),
            "--disable-auto-launch".into(),
        ]);
        let url = format!("http://{}:{}", config.host, port);
        (Health::Tcp(url.clone(), "/queue"), url)
    };
    Ok(Resolved { program, args, cwd, health, endpoint, version_file })
}

/// Version probe at start. Failure is non-fatal: status simply has no version.
async fn probe_version(program: &Path, version_file: Option<&Path>) -> Option<String> {
    if let Some(path) = version_file {
        let text = std::fs::read_to_string(path).ok()?;
        return text.lines().find_map(|line| {
            let (key, value) = line.trim().split_once('=')?;
            (key.trim() == "__version__").then(|| {
                value.trim().trim_matches(|c| c == '\'' || c == '"').to_owned()
            })
        });
    }
    let binary = program.to_path_buf();
    let output = tokio::task::spawn_blocking(move || {
        Command::new(&binary).arg("--version").stdin(Stdio::null()).output()
    })
    .await
    .ok()?
    .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.split_whitespace()
        .find(|word| word.chars().any(|ch| ch.is_ascii_digit()))
        .map(str::to_owned)
}

/// Drain stderr into a capped tail so the child never blocks on a full pipe.
fn drain_stderr(mut pipe: impl Read, sink: &Arc<Mutex<String>>) {
    let mut buf = [0_u8; 8192];
    loop {
        match pipe.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                let mut tail = sink.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                tail.push_str(&String::from_utf8_lossy(&buf[..read]));
                if tail.len() > STDERR_TAIL_BYTES {
                    let from = tail.len() - STDERR_TAIL_BYTES;
                    let boundary = tail
                        .char_indices()
                        .find(|(index, _)| *index >= from)
                        .map_or(0, |(index, _)| index);
                    tail.drain(..boundary);
                }
            }
        }
    }
}

/// Why the supervisor task ended.
enum Outcome {
    Stopped,
    Failed(String),
}

/// One task owns the child, lease and readiness polling for an app's lifetime.
#[cfg(unix)]
#[allow(clippy::too_many_arguments)] // one task owns the child, lease and readiness
async fn supervise(
    manager: Arc<Manager>,
    entry: Arc<Mutex<Entry>>,
    mut child: Child,
    group: Pid,
    lease: GpuLease,
    token: CancellationToken,
    health: Health,
    timeout: Duration,
    log: Arc<Mutex<String>>,
) {
    let started = Instant::now();
    let mut deadline = started + timeout;
    let hard_deadline = started + timeout.saturating_mul(3);
    let mut ready = false;
    let outcome = loop {
        if token.is_cancelled() {
            break Outcome::Stopped;
        }
        match child.try_wait() {
            Ok(Some(_)) => break Outcome::Failed("app process exited before stop".to_owned()),
            Ok(None) => {}
            Err(err) => break Outcome::Failed(format!("app wait failed: {err}")),
        }
        if !ready {
            match health_ok(&health, &manager.client).await {
                Ok(true) => {
                    ready = true;
                    let mut state = entry_lock(&entry);
                    state.state = AppState::Ready;
                    state.error = None;
                }
                Ok(false) if Instant::now() < hard_deadline => {
                    // 503 "loading" restarts the countdown, with a hard ceiling.
                    deadline = Instant::now() + timeout;
                }
                _ => {} // 503 after the hard ceiling is handled by the deadline check below
            }
            if !ready && Instant::now() >= deadline {
                break Outcome::Failed(format!("readiness timed out after {}s", timeout.as_secs()));
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    };
    // Deterministic teardown: kill the whole group, reap, then drop the lease.
    let _ = rustix::process::kill_process_group(group, Signal::KILL);
    let _ = tokio::task::spawn_blocking(move || child.wait()).await;
    drop(lease);
    let tail = log.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    let mut state = entry_lock(&entry);
    // No endpoint, pid or token outlives a dead app.
    state.endpoint = None;
    state.pid = None;
    state.token = None;
    match outcome {
        Outcome::Stopped => {
            state.state = AppState::Stopped;
            state.error = None;
        }
        Outcome::Failed(reason) => {
            state.state = AppState::Failed;
            let message = if tail.is_empty() {
                reason
            } else {
                format!("{reason}; stderr: {tail}")
            };
            state.error = Some(message);
        }
    }
}

/// Poll the app's readiness route: 200 means ready, 503 means still loading.
async fn health_ok(target: &Health, client: &Client) -> Result<bool, String> {
    match target {
        Health::Tcp(url, path) => {
            let response = client
                .get(format!("{url}{path}"))
                .send()
                .await
                .map_err(|err| format!("health: {err}"))?;
            Ok(response.status() == StatusCode::OK)
        }
        #[cfg(unix)]
        Health::Unix(path) => {
            let path = path.clone();
            tokio::task::spawn_blocking(move || unix_health(&path))
                .await
                .map_err(|err| format!("health task: {err}"))?
        }
        #[cfg(not(unix))]
        Health::Unix(_) => Ok(false),
    }
}

/// Minimal HTTP/1.1 GET over a unix socket (no reqwest unix feature needed).
#[cfg(unix)]
fn unix_health(path: &Path) -> Result<bool, String> {
    use std::io::Write as _;
    let mut stream = UnixStream::connect(path).map_err(|err| format!("socket: {err}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|err| format!("socket timeout: {err}"))?;
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .map_err(|err| format!("socket write: {err}"))?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply).map_err(|err| format!("socket read: {err}"))?;
    Ok(reply.starts_with("HTTP/1.1 200") || reply.starts_with("HTTP/1.0 200"))
}

#[cfg(test)]
mod manager_tests {
    use super::*;

    #[test]
    fn all_apps_start_off_and_are_listed() {
        let manager = Manager::new(ManagedAppsConfig::default(), None, "http://127.0.0.1:8188").unwrap();
        let list = manager.list();
        assert_eq!(list.len(), 2);
        assert!(list.iter().all(|app| app.state == AppState::Stopped));
        assert!(!manager.ready("comfyui"));
    }

    #[test]
    fn non_loopback_upstream_is_rejected() {
        assert!(Manager::new(ManagedAppsConfig::default(), None, "http://0.0.0.0:8188").is_err());
        assert!(Manager::new(ManagedAppsConfig::default(), None, "http://user:pw@127.0.0.1:8188").is_err());
    }

    #[test]
    fn disabled_app_start_is_named_501() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let manager = Arc::new(
            Manager::new(ManagedAppsConfig::default(), None, "http://127.0.0.1:8188").unwrap(),
        );
        let err = rt.block_on(manager.start("comfyui")).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_IMPLEMENTED);
        assert!(err.message.contains("not enabled"));
    }

    #[test]
    fn health_timeout_floor_is_applied() {
        let file = crate::config::FileApps {
            comfyui: Some(crate::config::FileApp {
                enabled: Some(true),
                launch: Some("installation".into()),
                health_timeout_secs: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        };
        let apps = crate::config::managed_apps_config(file).unwrap();
        assert_eq!(apps.comfyui.launch, ManagedAppLaunch::Installation);
        assert_eq!(apps.comfyui.health_timeout_secs, MIN_HEALTH_SECS);
        assert!(!apps.llama_server.enabled);
    }
}