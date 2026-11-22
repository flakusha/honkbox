//! Model roots config: `[models]` + optional `[discovery]` in
//! `<data_dir>/honkbox.toml`, env wins over file. Also `[jobs]` GC settings
//! (`gc_ttl` / `gc_max_bytes`, seconds / bytes, env `HONKBOX_GC_TTL` /
//! `HONKBOX_GC_MAX_BYTES`), `[swap]` LLM swap settings (`enabled`,
//! `llm_ttl_secs`, `llm_max_loaded`, `max_ttl_secs`, `pin`, env
//! `HONKBOX_SWAP_LLM_TTL` / `HONKBOX_SWAP_LLM_MAX_LOADED` /
//! `HONKBOX_SWAP_MAX_TTL` / `HONKBOX_SWAP_PIN`), `[core] data_dir`, the
//! `[server] queue_wait_secs` admission-queue bound (REQ-6 Option A, env
//! `HONKBOX_QUEUE_WAIT_SECS`), the `[artifacts] max_artifact_bytes`
//! binary-artifact total cap (TASK-binary-artifact-api, env
//! `HONKBOX_ARTIFACT_MAX_BYTES`), and the `[ledger]` resource-ledger knobs
//! (TASK-resource-ledger-bounded-queue: `gpu_slots`, `vram_budget_bytes`,
//! `ram_budget_bytes`, `cpu_threads`, `oom_headroom_bytes`, `queue_cap`
//! with `HONKBOX_LEDGER_*` env stages, plus the file-only per-kind maps
//! `[ledger.queue_caps]` / `[ledger.vram_demand]`), and `[licenses]` serving
//! gates (`territory` default `unknown`, `commercial` default `true`, env
//! `HONKBOX_LICENSES_TERRITORY` / `HONKBOX_LICENSES_COMMERCIAL`).
//!
//! The ONE data-dir knob resolves `--data-dir` CLI flag > `HONKBOX_DATA_DIR`
//! env > `[core] data_dir` file key > `$XDG_STATE_HOME/honkbox` (empty or
//! relative env/XDG values are treated as unset; fallback
//! `$HOME/.local/state/honkbox`). Everything derives from it:
//! `<data_dir>/honkbox.toml` (config), `jobs/`, `logs/honkbox.log`,
//! `registry.sqlite3`, `models/{llm,sd,stt,tts,ocr}` (the audio starters plus
//! the managed download roots appended by `append_managed_roots`). The
//! file-*locating* default is
//! pure-env (no file read, no cwd), so the config file lives inside the data
//! dir without circularity; a file `[core] data_dir` refines the derived
//! roots but cannot move the file itself (one-pass).
//!
//! Precedence for model roots: built-in defaults → config file → env
//! (`HONKBOX_LLM_DIRS`, `HONKBOX_COMFYUI_DIR`, `HONKBOX_STT_DIRS`,
//! `HONKBOX_TTS_DIRS`, `HONKBOX_OCR_DIRS`, `HONKBOX_DISCOVERY_LLM_DIRS`,
//! `HONKBOX_DISCOVERY_COMFYUI_DIRS`, `HONKBOX_DISCOVERY_STT_DIRS`,
//! `HONKBOX_DISCOVERY_TTS_DIRS`, `HONKBOX_DISCOVERY_OCR_DIRS`); the curated
//! catalog's VRAM budget rides
//! `[models] vram_budget_bytes` with `HONKBOX_VRAM_BUDGET` as its env stage.
//!
//! Operator model bindings (REQ-2) ride `[[models.entry]]` — an array of
//! tables *under* the existing `[models]` table (a top-level `[[models]]`
//! array would collide with the roots table). Each entry targets a scan hit
//! by exact `slug` **or** by `match` on the weight file's name (the stable
//! choice for HF snapshots whose slug embeds a content sha); declaring both
//! or neither is a config error. It may declare `aliases` (operator-declared
//! only — never filename-derived) and `capabilities` overrides (`chat`,
//! `embeddings`, `tool_calling`, `vision`; each `Option<bool>`, absent keys
//! keep the kind-inferred default; a `chat = false` + `embeddings = true`
//! pair also reclassifies the emitted `kind` so embedding GGUFs under an
//! LLM root stay discoverable). Alias conflicts (two entries claiming one
//! alias, or one `match` hitting several files) are a **hard startup
//! error**, validated against the live scan in `routes::bind_entries`.
//! Structured tables have no env stage in this config's house pattern (all
//! `HONKBOX_*` env vars are scalars/path lists), so entries are file-only.
//!
//! Per-slug LLM cache controls ride `[[models.entry]] cache`: `n_ctx` (the
//! KV cache's context size in tokens), `kv_quant` (`"f16"` | `"q8_0"` —
//! `q8_0` halves the cache bytes) and `offload` (`false` keeps the cache in
//! RAM instead of the VRAM). The knobs bound the attention cache of the
//! loaded pipeline and are off until an entry declares them (absent keys
//! keep the backend default). The cache never exceeds the global
//! `[models] vram_budget_bytes` ceiling: the loader clamps `n_ctx` to it.
//!
//! Auth (REQ-7): `[server] auth` is `"none"` (default) or `"bearer"`;
//! `[auth] api_key_env` names the env var holding the bearer key (default
//! `HONKBOX_API_KEY`). Only the NAME is configured; the key VALUE is read
//! at request time via `std::env` in the route middleware (so rotation
//! works without a restart) and is never logged, never in metrics, and
//! never in any config dump — these types store the env-var name only.
//!
//! Signed URLs and scoped keys (file-only): `[auth] signed_url_secret_env`
//! names the env var holding the signed-URL HMAC secret (default
//! `HONKBOX_SIGNED_URL_SECRET`) and `signed_url_ttl_secs` bounds signed-URL
//! lifetime (default 300). `[[auth.scopes]]` entries carry `api_key_env` (the
//! NAME of a key env var) plus a `scopes` list, de-duplicated by
//! `api_key_env`. Only NAMES are stored — never secret VALUES.
//!
//! `[safety.nsfw]` holds the NSFW gate: `enabled` (default false), `policy`
//! (`flag|block|off`, default `flag`), `model_version` (default
//! `stub-nsfw-v1`), `threshold` (0..=1, default 0.8) and `classifier_bin`
//! (optional path, default unset).
//!
//! `[python]` holds the optional sidecar: `enabled` (default false), `venv`
//! (default `<data_dir>/sidecar/venv`, or `sidecar/venv` without a data dir),
//! `timeout_secs` (default 300) and `model_allowlist` (default empty).

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use honkbox_residency::ResourceKind;

use crate::jobs::{DEFAULT_GC_MAX_BYTES, DEFAULT_GC_TTL};

// Mesh proxy client + its `[mesh]` config (TASK-mesh-client-spike).
// Declared HERE with the jobs.rs/determinism.rs `#[path]` precedent: main.rs
// keeps only `mod config;`, and every test harness that includes this module
// transitively compiles the mesh config types too.
#[path = "mesh.rs"]
pub mod mesh;

/// Model roots handed to `honkbox_registry::scan_roots`: the `[models]`
/// base roots plus any `[discovery]` extras, merged in that order.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// LLM roots (HuggingFace-cache layout); every entry is scanned.
    pub llm_dirs: Vec<PathBuf>,
    /// `ComfyUI` base folders; every entry is scanned.
    pub comfyui_dirs: Vec<PathBuf>,
    /// Speech-to-text weight directories; every entry is scanned.
    pub stt_dirs: Vec<PathBuf>,
    /// Text-to-speech weight directories; every entry is scanned.
    pub tts_dirs: Vec<PathBuf>,
    /// OCR weight directories (ocrs `.rten`/`.onnx`); every entry is scanned.
    pub ocr_dirs: Vec<PathBuf>,
    /// `[jobs]` GC settings (startup garbage collection of job bundles).
    pub jobs: JobsConfig,
    /// `[swap]` LLM load-on-demand + idle-eviction settings.
    pub swap: SwapConfig,
    /// `[models] vram_budget_bytes` (env `HONKBOX_VRAM_BUDGET`): the
    /// user-declared VRAM ceiling driving the curated catalog's advisory
    /// fit pills. `None` → pills omitted (advice only, never enforced).
    pub vram_budget_bytes: Option<u64>,
    /// Operator model bindings (`[[models.entry]]`, REQ-2): aliases +
    /// capability overrides bound to scan hits by `routes::bind_entries`.
    pub entries: Vec<ModelEntryConfig>,
    /// The per-slug LLM cache controls (`[[models.entry]] cache`), keyed by
    /// the entry's registry slug. Empty = the backend defaults.
    pub cache: BTreeMap<String, CacheConfig>,
}

/// One operator model binding (`[[models.entry]]`): targets a scan hit by
/// exact `slug` or by `match` on the weight file's name, optionally naming
/// `aliases` and overriding `capabilities`. See the module docs for the
/// full contract (operator-declared only; conflicts are startup errors).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntryConfig {
    /// Exact registry slug this entry binds to (`slug = "llm:…"`). Mutually
    /// exclusive with `match`; exactly one must be present.
    pub slug: Option<String>,
    /// Weight file **name** to bind (`match = "Model-Q8_0.gguf"`): stable
    /// across HF snapshot re-downloads whose slug embeds a content sha.
    /// Matching is by exact name (the path's final component).
    pub file_match: Option<String>,
    /// Operator-declared request aliases; the first is the `id` served by
    /// `GET /v1/models`. All resolve on chat/transcribe/synthesize.
    pub aliases: Vec<String>,
    /// Capability overrides (`Option<bool>` per flag; absent keys keep the
    /// kind-inferred default).
    pub capabilities: ModelCapabilitiesOverride,
}

/// Per-slug LLM cache controls (`[[models.entry]] cache`).
///
/// The knobs bound the attention (KV) cache of the loaded pipeline: the
/// context size IS the cache size, the element type halves it, and
/// `offload = false` keeps it in RAM instead of the VRAM. Absent keys keep
/// the backend default, so an entry that declares no cache block behaves
/// exactly as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheConfig {
    /// The KV cache's context size in tokens (`n_ctx`).
    pub n_ctx: Option<u32>,
    /// The KV cache's element type (`kv_quant`).
    pub kv_quant: Option<KvQuantConfig>,
    /// Whether the KV cache is offloaded to the GPU (`offload`); `false`
    /// keeps the cache in RAM.
    pub offload: Option<bool>,
}

/// The element type an LLM's KV cache is stored in (config spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvQuantConfig {
    /// 16-bit floats (the backend default).
    F16,
    /// 8-bit quantized: half the cache bytes.
    Q8_0,
}

/// Per-flag capability overrides (`[[models.entry]] capabilities`).
/// `None` keeps the kind-inferred default for that flag; declaring
/// `chat = false` together with `embeddings = true` additionally
/// reclassifies the emitted `kind` as `embedding` (discoverability for
/// embedding GGUFs under LLM roots).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilitiesOverride {
    /// Chat-template conversation turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<bool>,
    /// Embedding vectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<bool>,
    /// OpenAI-style tool calling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calling: Option<bool>,
    /// Vision content parts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
}

impl ModelCapabilitiesOverride {
    /// Whether the declaration reclassifies the entry as an embedding model
    /// (explicit `chat = false` + `embeddings = true`).
    #[must_use]
    pub fn reclassifies_embedding(self) -> bool {
        self.chat == Some(false) && self.embeddings == Some(true)
    }

    /// JSON encoding for the registry-cache override column (declared keys
    /// only); `{}` when nothing is declared.
    #[must_use]
    pub fn cache_json(self) -> String {
        serde_json::to_string(&self).unwrap_or_else(|_| "{}".to_owned())
    }

    /// Decode [`Self::cache_json`] output; invalid JSON decodes empty
    /// (the cache is advisory and the next scan rewrites it).
    #[must_use]
    pub fn from_cache_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap_or_default()
    }
}

/// Default `[server] queue_wait_secs` (REQ-6 Option A): how long THE ONE
/// admission-queue waiter may wait for the GPU slot before the 429. `0`
/// disables queueing (immediate 429, the pre-REQ-6 behavior).
pub const DEFAULT_QUEUE_WAIT_SECS: u64 = 10;

/// Upper bound for `[server] queue_wait_secs`: the admission waiter adds the
/// bound to an `Instant` (which panics on overflow for `u64::MAX` seconds),
/// and a day already exceeds any real queue wait, so both config stages clamp
/// to this value.
pub const MAX_QUEUE_WAIT: Duration = Duration::from_hours(24);
/// Default `[swap]` idle TTL: 10 minutes (llama-swap defaults to never
/// unload; a bounded default is the safer choice for a single-GPU studio).
pub const DEFAULT_SWAP_TTL_SECS: u64 = 600;
/// Default `[swap]` warm capacity: one resident LLM (matches the
/// single-GPU residency slot; loading another slug swaps, not coexists).
pub const DEFAULT_SWAP_MAX_LOADED: usize = 1;
/// Default `[artifacts] max_artifact_bytes` (TASK-binary-artifact-api): the
/// bounded total size of ONE stored binary artifact. Generous for real
/// media (a 1 GiB `LoRA` bundle, an hour of WAV), still a hard wall against
/// unbounded disk growth; operators tighten or raise it via config/env.
pub const DEFAULT_ARTIFACT_MAX_BYTES: u64 = 1024 * 1024 * 1024;

/// Default declared jurisdiction: an undeclared territory fails the license
/// gate closed (TASK-model-license-registry).
pub const DEFAULT_LICENSE_TERRITORY: &str = "unknown";

/// Resolved `[licenses]` knobs (TASK-model-license-registry). `commercial`
/// defaults to true, so NC weights refuse until a research pipeline opts out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub territory: String,
    pub commercial: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            territory: DEFAULT_LICENSE_TERRITORY.to_owned(),
            commercial: true,
        }
    }
}

static LICENSE_CONFIG: OnceLock<Config> = OnceLock::new();

/// Effective knobs for request-time license gates. Defaults until
/// [`install_licenses`] publishes the resolved config at startup.
pub(crate) fn licenses() -> &'static Config {
    LICENSE_CONFIG.get_or_init(Config::default)
}

/// Publish the resolved `[licenses]` knobs (first call wins, like
/// `runtime_config::configure`).
pub(crate) fn install_licenses(config: Config) {
    let _ = LICENSE_CONFIG.set(config);
}

/// Resolve `[licenses]` file keys + env overrides. Pure (env as args) so
/// tests stay thread-safe; invalid boolean env values warn and keep the
/// previous stage, matching the GC/swap knobs.
pub(crate) fn resolve_licenses(
    file_territory: Option<&str>,
    file_commercial: Option<bool>,
    territory_env: Option<&str>,
    commercial_env: Option<&str>,
    warnings: &mut Vec<String>,
) -> Config {
    let mut territory = file_territory
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_LICENSE_TERRITORY)
        .to_owned();
    if let Some(value) = territory_env.map(str::trim).filter(|v| !v.is_empty()) {
        value.clone_into(&mut territory);
    }
    let mut commercial = file_commercial.unwrap_or(true);
    if let Some(value) = commercial_env {
        match value.trim().parse::<bool>() {
            Ok(parsed) => commercial = parsed,
            Err(_) => warnings.push(format!(
                "invalid HONKBOX_LICENSES_COMMERCIAL {value:?}; keeping {commercial}"
            )),
        }
    }
    Config {
        territory,
        commercial,
    }
}
/// Per-request `ttl_seconds` cap (`max_ttl_secs`): 24 h. Advice above the cap
/// is clamped at the route with a `warn!` (never a 400) so a third-party
/// client cannot out-stay the residency budget a server operator chose.
pub const DEFAULT_SWAP_MAX_TTL_SECS: u64 = 86_400;

/// `[swap]` section: LLM load-on-demand + idle eviction (P1 of the
/// llama-swap parity plan). Effective only when the chat backend is a real
/// file backend (`--llm-backend llama-cpp`); stub servers ignore it entirely
/// and keep the single static stub pipeline.
#[derive(Debug, Clone)]
pub struct SwapConfig {
    /// Master switch (file/env). Effective swap = this AND a real LLM
    /// backend, so a llama-cpp server gets JIT loading by default and a
    /// stub server keeps its zero-cost stub.
    pub enabled: bool,
    /// Idle time before an LLM pipeline is evictable (`llm_ttl_secs`); a
    /// request's `ttl_seconds` advice raises the entry's window per touch
    /// (`max(remaining, own)`), never past `max_ttl`.
    pub llm_ttl: Duration,
    /// Warm-pipeline LRU capacity (`llm_max_loaded`, clamped to >= 1);
    /// `1` = the incoming model replaces the resident.
    pub llm_max_loaded: usize,
    /// Ceiling on any per-entry residency window (`max_ttl_secs`, at least
    /// 1s): per-request `ttl_seconds` over the cap is clamped at the route,
    /// and the swap manager re-caps defensively.
    pub max_ttl: Duration,
    /// Slugs that never auto-evict and ignore per-request `ttl_seconds`
    /// entirely (`pin`; llama-swap `persistent`) — pin outranks client
    /// advice.
    pub pin: Vec<String>,
}

/// `[server] auth` mode (REQ-7): `none` (default, loopback-only
/// operation) or `bearer` (every route except the probes requires
/// `Authorization: Bearer <key>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AuthMode {
    /// No auth; the bind-address fail-closed check in `main::serve` is the
    /// safety net for non-loopback binds.
    #[default]
    None,
    /// Bearer-key auth against the `[auth] api_key_env` value.
    Bearer,
}

impl AuthMode {
    /// Parse the `[server] auth` file key; whitespace-insensitive,
    /// case-insensitive; anything but `none`/`bearer` is a config error.
    fn parse(raw: &str, where_: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "bearer" => Ok(Self::Bearer),
            other => Err(format!(
                "{where_}: invalid [server] auth value {other:?} (expected none or bearer)"
            )),
        }
    }
}

/// Resolved auth settings (REQ-7): the `[server] auth` mode plus the NAME
/// of the env var holding the bearer key (`[auth] api_key_env`). The key
/// VALUE is never stored here — the route middleware reads it per request
/// via `std::env`, so it cannot leak into logs, metrics, or dumps.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// `[server] auth`.
    pub mode: AuthMode,
    /// `[auth] api_key_env`: env var NAME read at request time.
    pub api_key_env: String,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            mode: AuthMode::None,
            api_key_env: "HONKBOX_API_KEY".to_owned(),
        }
    }
}

/// Resolved `[auth]` signed-URL and scoped-key facts. Kept separate from
/// [`AuthConfig`] so the existing single-key mode (and its struct literals)
/// stays unchanged. Only env-var NAMES are stored — never secret VALUES.
#[derive(Debug, Clone)]
pub struct AuthFactsConfig {
    /// `[auth] signed_url_secret_env`: NAME of the secret's env var.
    pub signed_url_secret_env: String,
    /// `[auth] signed_url_ttl_secs`: signed-URL lifetime in seconds.
    pub signed_url_ttl_secs: u64,
    /// `[[auth.scopes]]` entries, de-duplicated by `api_key_env`.
    pub scopes: Vec<FileScope>,
}

impl Default for AuthFactsConfig {
    fn default() -> Self {
        Self {
            signed_url_secret_env: DEFAULT_SIGNED_URL_SECRET_ENV.to_owned(),
            signed_url_ttl_secs: DEFAULT_SIGNED_URL_TTL_SECS,
            scopes: Vec::new(),
        }
    }
}

/// Default `[auth] signed_url_secret_env` (a NAME, never a value).
pub const DEFAULT_SIGNED_URL_SECRET_ENV: &str = "HONKBOX_SIGNED_URL_SECRET";
/// Default `[auth] signed_url_ttl_secs`.
pub const DEFAULT_SIGNED_URL_TTL_SECS: u64 = 300;
/// Default `[python] timeout_secs`.
pub const DEFAULT_PYTHON_TIMEOUT_SECS: u64 = 300;

/// One `[[auth.scopes]]` entry: a key env-var NAME plus the scopes granted to
/// keys read from it. Both fields are validated/cleaned in the file stage;
/// the secret VALUE behind `api_key_env` is never stored here.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct FileScope {
    /// NAME of the env var holding the key (trimmed, non-empty).
    #[serde(default)]
    pub api_key_env: String,
    /// Scopes granted to that key (trimmed, empty entries dropped).
    #[serde(default)]
    pub scopes: Vec<String>,
}

/// Resolved `[safety.nsfw]` gate. Facts only — no model is loaded here.
#[derive(Debug, Clone, PartialEq)]
pub struct NsfwConfig {
    pub enabled: bool,
    pub policy: String,
    pub model_version: String,
    pub threshold: f64,
    /// Optional explicit classifier binary path; `None` uses discovery.
    pub classifier_bin: Option<PathBuf>,
}

impl Default for NsfwConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy: "flag".to_owned(),
            model_version: "stub-nsfw-v1".to_owned(),
            threshold: 0.8,
            classifier_bin: None,
        }
    }
}

/// Resolved `[python]` sidecar settings. `venv` is derived from the final
/// data dir in `load` (default `<data_dir>/sidecar/venv`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonConfig {
    pub enabled: bool,
    pub venv: PathBuf,
    pub timeout_secs: u64,
    pub model_allowlist: Vec<String>,
}

impl Default for PythonConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            venv: PathBuf::from("sidecar/venv"),
            timeout_secs: DEFAULT_PYTHON_TIMEOUT_SECS,
            model_allowlist: Vec::new(),
        }
    }
}

/// Resolved `[server]` serving knobs. Currently one: the REQ-6 Option A
/// admission-queue bound (`queue_wait_secs`); `Duration::ZERO` keeps the
/// pre-REQ-6 immediate-429 behavior (queueing disabled).
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// `[server] queue_wait_secs` (env `HONKBOX_QUEUE_WAIT_SECS`), installed
    /// on the admission queue at serve startup.
    pub queue_wait: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            queue_wait: Duration::from_secs(DEFAULT_QUEUE_WAIT_SECS),
        }
    }
}

/// Validate a configured env-var NAME (trimmed, non-empty). The VALUE behind
/// it is read at request time and never stored. Pure.
fn parse_env_name(raw: &str, key: &str, where_: &str) -> Result<String, String> {
    let name = raw.trim();
    if name.is_empty() {
        Err(format!("{where_}: {key} must name an env var"))
    } else {
        Ok(name.to_owned())
    }
}

/// Validate `[auth] api_key_env` (an unnameable key would fail every request).
fn parse_api_key_env(raw: &str, where_: &str) -> Result<String, String> {
    parse_env_name(raw, "[auth] api_key_env", where_)
}

/// Validate `[safety.nsfw] policy`: `flag`, `block` or `off`. Pure; a typo
/// must be a loud config error, never a silently disabled gate.
fn parse_nsfw_policy(raw: &str, where_: &str) -> Result<String, String> {
    let policy = raw.trim();
    match policy {
        "flag" | "block" | "off" => Ok(policy.to_owned()),
        _ => Err(format!(
            "{where_}: [safety.nsfw] policy must be flag|block|off, got {policy:?}"
        )),
    }
}

/// Validate `[safety.nsfw] threshold`: a finite probability in `0..=1`. Pure.
fn parse_nsfw_threshold(raw: f64, where_: &str) -> Result<f64, String> {
    if raw.is_finite() && (0.0..=1.0).contains(&raw) {
        Ok(raw)
    } else {
        Err(format!(
            "{where_}: [safety.nsfw] threshold must be between 0 and 1, got {raw}"
        ))
    }
}

/// Resolve `[safety.nsfw]` from the file stage. Pure.
fn resolve_nsfw(file: &FileNsfw, where_: &str) -> Result<NsfwConfig, String> {
    let mut out = NsfwConfig::default();
    if let Some(enabled) = file.enabled {
        out.enabled = enabled;
    }
    if let Some(policy) = &file.policy {
        out.policy = parse_nsfw_policy(policy, where_)?;
    }
    if let Some(version) = &file.model_version {
        let version = version.trim();
        if version.is_empty() {
            return Err(format!(
                "{where_}: [safety.nsfw] model_version must not be empty"
            ));
        }
        version.clone_into(&mut out.model_version);
    }
    if let Some(threshold) = file.threshold {
        out.threshold = parse_nsfw_threshold(threshold, where_)?;
    }
    out.classifier_bin = match file.classifier_bin.as_deref() {
        Some(raw) if !raw.trim().is_empty() => Some(expand_home(raw.trim())),
        Some(_) => {
            return Err(format!(
                "{where_}: [safety.nsfw] classifier_bin must be a path"
            ));
        }
        None => None,
    };
    Ok(out)
}

/// Default `[python] venv`: `<data_dir>/sidecar/venv`, or the relative
/// `sidecar/venv` when no data dir is known. Pure.
fn python_venv_default(data_dir: Option<&Path>) -> PathBuf {
    data_dir.map_or_else(
        || PathBuf::from("sidecar/venv"),
        |dir| dir.join("sidecar/venv"),
    )
}

/// Validate + clean `[[auth.scopes]]`: each `api_key_env` is a trimmed,
/// non-empty NAME, scopes are trimmed with empties dropped, and entries are
/// de-duplicated by `api_key_env` (first declaration wins). Pure.
fn parse_scopes(raw: &[FileScope], where_: &str) -> Result<Vec<FileScope>, String> {
    let mut out: Vec<FileScope> = Vec::new();
    for (index, scope) in raw.iter().enumerate() {
        let name = scope.api_key_env.trim();
        if name.is_empty() {
            return Err(format!(
                "{where_}: [[auth.scopes]] #{index} must name an env var"
            ));
        }
        if out.iter().any(|kept| kept.api_key_env == name) {
            continue;
        }
        out.push(FileScope {
            api_key_env: name.to_owned(),
            scopes: clean_slugs(scope.scopes.iter().map(String::as_str)),
        });
    }
    Ok(out)
}

/// Fail-closed bind check (REQ-7 acceptance): `auth = none` on a
/// non-loopback bind refuses to start unless the operator passed the
/// explicit acknowledgement flag. `host` is loopback iff `localhost`,
/// `127.0.0.0/8`, or `::1`. Pure (args) so tests stay thread-safe.
pub fn check_bind_auth(
    mode: AuthMode,
    host: &str,
    allow_unauthenticated_bind: bool,
) -> Result<(), String> {
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    // Refuse when unauthenticated AND exposed; any of bearer auth, a
    // loopback bind, or the explicit ack makes the start safe.
    if mode == AuthMode::Bearer || loopback || allow_unauthenticated_bind {
        return Ok(());
    }
    Err(format!(
        "refusing to bind {host}: [server] auth = none on a non-loopback \
         address exposes every route without a key; set [server] auth = \
         bearer with [auth] api_key_env, or pass \
         --allow-unauthenticated-bind to acknowledge"
    ))
}

/// Parse the `--host` serve flag into a bind `IpAddr`: `localhost` binds
/// the IPv4 loopback (matching the historical default); anything else must
/// be a literal IP (`::1` included).
pub fn parse_bind_host(host: &str) -> Result<std::net::IpAddr, String> {
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(std::net::IpAddr::from([127, 0, 0, 1]));
    }
    host.parse::<std::net::IpAddr>()
        .map_err(|_| format!("invalid --host {host:?}: expected an IP or localhost"))
}

/// `[jobs]` section: startup GC of `<data_dir>/jobs` + stale temp PNGs.
#[derive(Debug, Clone)]
pub struct JobsConfig {
    /// Age past which job bundles and stale temp files are removed.
    pub gc_ttl: Duration,
    /// Total size cap across job bundles; oldest bundles evicted first.
    pub gc_max_bytes: u64,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            gc_ttl: DEFAULT_GC_TTL,
            gc_max_bytes: DEFAULT_GC_MAX_BYTES,
        }
    }
}

/// Resolve `[swap]` settings: defaults → file keys → env overrides.
/// `llm_ttl_secs` / `HONKBOX_SWAP_LLM_TTL` and `max_ttl_secs` /
/// `HONKBOX_SWAP_MAX_TTL` are seconds (`max_ttl` clamped to at least 1: a
/// `0` cap would use-then-unload every entry, silently overriding
/// `llm_ttl_secs`); `llm_max_loaded` / `HONKBOX_SWAP_LLM_MAX_LOADED` is a
/// count clamped to at least 1 (a `0` capacity would evict every entry
/// immediately, including the one a request just asked for). `pin` /
/// `HONKBOX_SWAP_PIN` is a slug list; the env form is comma-separated and
/// replaces the file key. Invalid env values warn-and-keep, matching the GC
/// knobs. Pure (env as args) so tests stay thread-safe.
#[allow(clippy::too_many_arguments)]
fn swap_settings(
    file: &FileSwap,
    ttl_env: Option<&str>,
    max_env: Option<&str>,
    max_ttl_env: Option<&str>,
    pin_env: Option<&str>,
) -> SwapConfig {
    let mut cfg = SwapConfig {
        enabled: true,
        llm_ttl: Duration::from_secs(DEFAULT_SWAP_TTL_SECS),
        llm_max_loaded: DEFAULT_SWAP_MAX_LOADED,
        max_ttl: Duration::from_secs(DEFAULT_SWAP_MAX_TTL_SECS),
        pin: Vec::new(),
    };
    if let Some(enabled) = file.enabled {
        cfg.enabled = enabled;
    }
    if let Some(secs) = file.llm_ttl_secs {
        cfg.llm_ttl = Duration::from_secs(secs);
    }
    if let Some(count) = file.llm_max_loaded {
        cfg.llm_max_loaded = count.max(1);
    }
    if let Some(secs) = file.max_ttl_secs {
        cfg.max_ttl = Duration::from_secs(secs.max(1));
    }
    if let Some(slugs) = &file.pin {
        cfg.pin = clean_slugs(slugs);
    }
    if let Some(v) = ttl_env
        && !v.is_empty()
        && let Ok(secs) = v.parse::<u64>()
    {
        cfg.llm_ttl = Duration::from_secs(secs);
    }
    if let Some(v) = max_env
        && !v.is_empty()
        && let Ok(count) = v.parse::<usize>()
    {
        cfg.llm_max_loaded = count.max(1);
    }
    if let Some(v) = max_ttl_env
        && !v.is_empty()
        && let Ok(secs) = v.parse::<u64>()
    {
        cfg.max_ttl = Duration::from_secs(secs.max(1));
    }
    if let Some(v) = pin_env
        && !v.is_empty()
    {
        cfg.pin = clean_slugs(v.split(','));
    }
    cfg
}

/// Trim whitespace and drop empty entries from a configured slug list.
fn clean_slugs<I: IntoIterator<Item = S>, S: AsRef<str>>(slugs: I) -> Vec<String> {
    slugs
        .into_iter()
        .map(|slug| slug.as_ref().trim().to_owned())
        .filter(|slug| !slug.is_empty())
        .collect()
}

/// Resolve GC settings: defaults → `[jobs]` file keys → env overrides.
/// `gc_ttl` / `HONKBOX_GC_TTL` are seconds; `gc_max_bytes` /
/// `HONKBOX_GC_MAX_BYTES` are bytes. Invalid values warn and keep the
/// previous stage. Pure (env as args) so tests stay thread-safe.
fn gc_settings(file: &FileJobs, ttl_env: Option<&str>, max_env: Option<&str>) -> JobsConfig {
    let mut cfg = JobsConfig::default();
    if let Some(secs) = file.gc_ttl {
        cfg.gc_ttl = Duration::from_secs(secs);
    }
    if let Some(bytes) = file.gc_max_bytes {
        cfg.gc_max_bytes = bytes;
    }
    if let Some(v) = ttl_env
        && !v.is_empty()
        && let Ok(secs) = v.parse::<u64>()
    {
        cfg.gc_ttl = Duration::from_secs(secs);
    }
    if let Some(v) = max_env
        && !v.is_empty()
        && let Ok(bytes) = v.parse::<u64>()
    {
        cfg.gc_max_bytes = bytes;
    }
    cfg
}

/// Resolve the VRAM budget (NEW-5): file key → env override. Pure (env as
/// args); empty or invalid env values keep the file value, matching the
/// GC/swap knobs. `None` → pills omitted (advice only, never enforced).
fn vram_budget_settings(file: Option<u64>, env: Option<&str>) -> Option<u64> {
    let mut budget = file;
    if let Some(raw) = env
        && !raw.is_empty()
        && let Ok(bytes) = raw.parse::<u64>()
    {
        budget = Some(bytes);
    }
    budget
}

/// Env var overriding the data dir (`--data-dir` flag wins over it).
pub const DATA_DIR_ENV: &str = "HONKBOX_DATA_DIR";

/// An absolute, non-empty path value; empty or relative values (a common
/// malformed `$XDG_STATE_HOME`/env) count as unset per the XDG spec.
fn clean(value: Option<OsString>) -> Option<PathBuf> {
    value
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
}

/// XDG-state default: `$XDG_STATE_HOME/honkbox`, else
/// `$HOME/.local/state/honkbox`. Pure (args, not process env).
fn default_data_dir(xdg: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    if let Some(dir) = clean(xdg) {
        return Some(dir.join("honkbox"));
    }
    clean(home).map(|h| h.join(".local/state/honkbox"))
}

/// Resolve the data dir WITHOUT the config-file stage (pure, so tests stay
/// thread-safe): `--data-dir` flag > `HONKBOX_DATA_DIR` env > XDG default.
/// This is the file-*locating* chain: it reads no file and no cwd, so
/// `<data_dir>/honkbox.toml` works without circularity.
fn resolve_data_dir(
    cli: Option<PathBuf>,
    env: Option<OsString>,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Option<PathBuf> {
    if let Some(dir) = cli {
        return Some(dir);
    }
    if let Some(dir) = clean(env) {
        return Some(dir);
    }
    default_data_dir(xdg, home)
}

/// `[tools]` external-engine discovery settings (TASK-external-tool-registry).
/// `None` paths/URLs let the probe crate use `PATH` / its loopback default.
#[derive(Debug, Clone)]
pub struct ExternalToolsConfig {
    /// Blender executable; `None` resolves `blender` on `PATH`.
    pub blender_path: Option<PathBuf>,
    /// `ComfyUI` base URL; `None` uses the crate's loopback default.
    pub comfyui_url: Option<String>,
    /// Optional `llama-server` base URL. `None` probes the binary only;
    /// `Some` also probes `/health` + `/v1/models` before claiming a capability.
    pub llama_server_url: Option<String>,
    /// Probe timeout; bounded so a hung binary never blocks startup.
    pub timeout: Duration,
}

impl ExternalToolsConfig {
    /// Hand the resolved settings to the discovery crate.
    pub fn into_probe_config(self) -> honkbox_external_tools::Config {
        honkbox_external_tools::Config {
            blender_path: self.blender_path,
            comfyui_url: self.comfyui_url,
            llama_server_url: self.llama_server_url,
            timeout: self.timeout,
        }
    }
}

/// Load result: model/job/swap/server config plus the final data dir (after
/// the `[core] data_dir` refinement).
#[derive(Debug)]
pub struct LoadedConfig {
    /// Model roots + `[jobs]` + `[swap]` settings.
    pub models: ModelConfig,
    /// Auth settings (REQ-7): `[server] auth` + `[auth] api_key_env` name.
    pub auth: AuthConfig,
    /// `[auth]` signed-URL + scoped-key facts (names only, no secret values).
    pub auth_facts: AuthFactsConfig,
    /// `[server]` serving knobs (REQ-6 admission-queue bound).
    pub server: ServerConfig,
    /// `[ledger]` resource-ledger settings (bounded per-kind admission
    /// queues + capacity budgets, TASK-resource-ledger-bounded-queue).
    pub ledger: LedgerSettings,
    /// `[artifacts] max_artifact_bytes`: the total size cap for ONE stored
    /// binary artifact (TASK-binary-artifact-api).
    pub artifact_max_bytes: u64,
    /// `[mesh]` upstream proxy config (TASK-mesh-client-spike): off by
    /// default; bearer tokens are redacted in `Debug` (which this struct
    /// derives).
    pub mesh: mesh::MeshConfig,
    /// `[tools]` external-engine discovery settings.
    pub tools: ExternalToolsConfig,
    /// `[safety.nsfw]` resolved gate facts.
    pub nsfw: NsfwConfig,
    /// `[python]` resolved sidecar settings.
    pub python: PythonConfig,
    /// `[licenses]` resolved serving-gate knobs.
    pub licenses: Config,
    /// Final data dir; `None` only when no absolute base can be derived.
    pub data_dir: Option<PathBuf>,
    /// Non-fatal config notices (invalid values that kept a previous stage):
    /// collected because `load` runs before `logging::init`, so the caller
    /// emits them (via `tracing::warn!`) right after installing the
    /// subscriber — that way they land in `honkbox.log`, not the void.
    pub warnings: Vec<String>,
}

/// Resolve the `[server] queue_wait_secs` knob (REQ-6 Option A): defaults
/// → file key → env override (`HONKBOX_QUEUE_WAIT_SECS`, seconds). Invalid
/// env values warn-and-keep the previous stage, matching the GC/swap
/// knobs; `0` disables queueing (immediate 429). Huge values are clamped to
/// [`MAX_QUEUE_WAIT`] so the admission deadline cannot overflow. Pure (env
/// as args) so tests stay thread-safe.
fn server_settings(file: Option<u64>, env: Option<&str>) -> ServerConfig {
    let mut cfg = ServerConfig::default();
    if let Some(secs) = file {
        cfg.queue_wait = Duration::from_secs(secs).min(MAX_QUEUE_WAIT);
    }
    if let Some(v) = env
        && !v.is_empty()
        && let Ok(secs) = v.parse::<u64>()
    {
        cfg.queue_wait = Duration::from_secs(secs).min(MAX_QUEUE_WAIT);
    }
    cfg
}

/// Resolve the `[artifacts] max_artifact_bytes` knob
/// (TASK-binary-artifact-api): defaults → file key → env override
/// (`HONKBOX_ARTIFACT_MAX_BYTES`). A `0` would reject every artifact, so it
/// warns and keeps the previous stage (a misconfigured cap should be loud,
/// not a silent full stop). Pure (env as args) so tests stay thread-safe;
/// warnings are returned for the caller to emit after `logging::init` (a
/// `tracing::warn!` here would fire before the subscriber exists).
fn artifact_settings(file: Option<u64>, env: Option<&str>, warnings: &mut Vec<String>) -> u64 {
    let mut cap = DEFAULT_ARTIFACT_MAX_BYTES;
    if let Some(bytes) = file {
        if bytes > 0 {
            cap = bytes;
        } else {
            warnings.push(
                "[artifacts] max_artifact_bytes = 0 would reject every artifact; keeping the default"
                    .to_owned(),
            );
        }
    }
    if let Some(raw) = env.filter(|raw| !raw.is_empty()) {
        match raw.parse::<u64>() {
            Ok(0) => warnings.push(format!(
                "HONKBOX_ARTIFACT_MAX_BYTES = 0 would reject every artifact; keeping {cap}"
            )),
            Ok(bytes) => cap = bytes,
            Err(_) => {
                warnings.push(format!(
                    "HONKBOX_ARTIFACT_MAX_BYTES {raw:?} is not a number; keeping {cap}"
                ));
            }
        }
    }
    cap
}

/// Default `[ledger] queue_cap` (TASK-resource-ledger-bounded-queue): ONE
/// waiter per kind — the REQ-6 depth-1 admission policy, per kind. `0` is
/// clamped to `1` (a zero queue would reject every busy request as
/// `queue_full` before the TTL ever runs).
pub const DEFAULT_LEDGER_QUEUE_CAP: usize = 1;

/// Resolved `[ledger]` knobs: the resource ledger's capacity budgets and
/// bounded admission-queue settings. Defaults keep the single-tenant dev
/// flow unchanged (one GPU slot, unmanaged byte/thread budgets, depth-1
/// queues); the per-kind maps refine specific kinds only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerSettings {
    /// GPU slot count (`gpu_slots`, env `HONKBOX_LEDGER_GPU_SLOTS`).
    pub gpu_slots: u32,
    /// VRAM budget in bytes (`vram_budget_bytes`, env
    /// `HONKBOX_LEDGER_VRAM_BUDGET`); `0` = unmanaged.
    pub vram_budget_bytes: u64,
    /// RAM budget in bytes (`ram_budget_bytes`, env
    /// `HONKBOX_LEDGER_RAM_BUDGET`); `0` = unmanaged.
    pub ram_budget_bytes: u64,
    /// CPU thread budget (`cpu_threads`, env `HONKBOX_LEDGER_CPU_THREADS`);
    /// `0` = unmanaged.
    pub cpu_threads: u64,
    /// OOM near-miss pressure margin in bytes (`oom_headroom_bytes`, env
    /// `HONKBOX_LEDGER_OOM_HEADROOM`); `0` = metric fires only at exact
    /// budget fill.
    pub oom_headroom_bytes: u64,
    /// Queue length per kind (`queue_cap`, env `HONKBOX_LEDGER_QUEUE_CAP`,
    /// clamped to >= 1); overridden per kind by `queue_caps`.
    pub queue_cap: usize,
    /// Per-kind queue-cap overrides (`[ledger.queue_caps]`, file-only).
    pub queue_caps: Vec<(ResourceKind, usize)>,
    /// Per-kind VRAM demand estimates in bytes (`[ledger.vram_demand]`,
    /// file-only); `0` (the default) = the kind claims no VRAM.
    pub vram_demand: Vec<(ResourceKind, u64)>,
}

impl Default for LedgerSettings {
    fn default() -> Self {
        Self {
            gpu_slots: 1,
            vram_budget_bytes: 0,
            ram_budget_bytes: 0,
            cpu_threads: 0,
            oom_headroom_bytes: 0,
            queue_cap: DEFAULT_LEDGER_QUEUE_CAP,
            queue_caps: Vec::new(),
            vram_demand: Vec::new(),
        }
    }
}

/// Resolve the `[ledger]` knobs: defaults → file keys → env overrides (the
/// scalar knobs only; the per-kind maps are file-only per the house
/// pattern). Invalid env values keep the previous stage, matching the
/// GC/swap knobs; unknown kind labels in the maps are config errors (a
/// silently ignored typo would admit far more than the operator wrote).
/// Pure (env as args) so tests stay thread-safe.
#[allow(clippy::too_many_arguments)]
fn ledger_settings(
    file: &FileLedger,
    gpu_slots_env: Option<&str>,
    vram_env: Option<&str>,
    ram_env: Option<&str>,
    cpu_env: Option<&str>,
    headroom_env: Option<&str>,
    queue_cap_env: Option<&str>,
) -> Result<LedgerSettings, String> {
    let mut cfg = LedgerSettings::default();
    if let Some(v) = file.gpu_slots {
        // A zero-slot ledger rejects every GPU demand forever; clamp like
        // queue_cap instead of blaming the request.
        cfg.gpu_slots = v.max(1);
    }
    if let Some(v) = file.vram_budget_bytes {
        cfg.vram_budget_bytes = v;
    }
    if let Some(v) = file.ram_budget_bytes {
        cfg.ram_budget_bytes = v;
    }
    if let Some(v) = file.cpu_threads {
        cfg.cpu_threads = v;
    }
    if let Some(v) = file.oom_headroom_bytes {
        cfg.oom_headroom_bytes = v;
    }
    if let Some(v) = file.queue_cap {
        cfg.queue_cap = v.max(1);
    }
    if let Some(v) = gpu_slots_env
        && !v.is_empty()
        && let Ok(parsed) = v.parse::<u32>()
    {
        cfg.gpu_slots = parsed.max(1);
    }
    if let Some(v) = vram_env
        && !v.is_empty()
        && let Ok(parsed) = v.parse::<u64>()
    {
        cfg.vram_budget_bytes = parsed;
    }
    if let Some(v) = ram_env
        && !v.is_empty()
        && let Ok(parsed) = v.parse::<u64>()
    {
        cfg.ram_budget_bytes = parsed;
    }
    if let Some(v) = cpu_env
        && !v.is_empty()
        && let Ok(parsed) = v.parse::<u64>()
    {
        cfg.cpu_threads = parsed;
    }
    if let Some(v) = headroom_env
        && !v.is_empty()
        && let Ok(parsed) = v.parse::<u64>()
    {
        cfg.oom_headroom_bytes = parsed;
    }
    if let Some(v) = queue_cap_env
        && !v.is_empty()
        && let Ok(parsed) = v.parse::<usize>()
    {
        cfg.queue_cap = parsed.max(1);
    }
    cfg.queue_caps = kind_overrides(file.queue_caps.as_ref(), "queue_caps")?;
    cfg.vram_demand = kind_overrides(file.vram_demand.as_ref(), "vram_demand")?;
    Ok(cfg)
}

/// Parse one file-only `[ledger.<map>]` kind table into validated
/// `(ResourceKind, value)` pairs; an unknown label is a config error.
fn kind_overrides<T: Copy>(
    map: Option<&BTreeMap<String, T>>,
    section: &str,
) -> Result<Vec<(ResourceKind, T)>, String> {
    let Some(map) = map else {
        return Ok(Vec::new());
    };
    map.iter()
        .map(|(label, value)| {
            let kind = ResourceKind::from_label(label).ok_or_else(|| {
                format!(
                    "[ledger] {section}: unknown kind {label:?} (expected one of \
                     sd, llm, ocr, embed, rag, comfy, blender, 3d)"
                )
            })?;
            Ok((kind, *value))
        })
        .collect()
}

/// Build the residency ledger configuration from the resolved settings plus
/// the queue TTL (`[server] queue_wait_secs`): the ledger's eviction clock
/// and the admission 429's "gave up waiting after Xs" share one knob, so
/// they can never disagree.
pub fn residency_config(
    settings: &LedgerSettings,
    queue_ttl: Duration,
) -> honkbox_residency::LedgerConfig {
    let mut config = honkbox_residency::LedgerConfig::new();
    config.capacities.gpu_slots = settings.gpu_slots;
    config.capacities.vram_bytes = settings.vram_budget_bytes;
    config.capacities.ram_bytes = settings.ram_budget_bytes;
    config.capacities.cpu_threads = settings.cpu_threads;
    config.capacities.oom_headroom_bytes = settings.oom_headroom_bytes;
    for kind in ResourceKind::ALL {
        let kind_settings = config.kind_mut(kind);
        kind_settings.ttl = queue_ttl;
        kind_settings.queue_cap = settings
            .queue_caps
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or(settings.queue_cap, |(_, cap)| *cap)
            .max(1);
        if let Some((_, vram)) = settings.vram_demand.iter().find(|(k, _)| *k == kind) {
            kind_settings.demand.vram_bytes = *vram;
        }
    }
    config
}

/// Resolve the REQ-7 auth knobs plus the signed-URL and scoped-key facts:
/// defaults → `[server] auth` + `[auth]` file keys. Missing keys keep the
/// defaults; invalid values are config errors. There is deliberately NO env
/// stage for these: the configured env vars hold VALUES at request time, not
/// config overrides. Pure (args) so tests stay thread-safe.
fn auth_settings(
    auth: Option<&str>,
    api_key_env: Option<&str>,
    signed_url_secret_env: Option<&str>,
    signed_url_ttl_secs: Option<u64>,
    scopes: &[FileScope],
    where_: &str,
) -> Result<(AuthConfig, AuthFactsConfig), String> {
    let mode = auth.map_or(Ok(AuthMode::None), |raw| AuthMode::parse(raw, where_))?;
    let env = api_key_env.map_or(Ok(AuthConfig::default().api_key_env), |raw| {
        parse_api_key_env(raw, where_)
    })?;
    let secret_env = signed_url_secret_env.map_or_else(
        || Ok(AuthFactsConfig::default().signed_url_secret_env),
        |raw| parse_env_name(raw, "[auth] signed_url_secret_env", where_),
    )?;
    let facts = AuthFactsConfig {
        signed_url_secret_env: secret_env,
        signed_url_ttl_secs: signed_url_ttl_secs.unwrap_or(DEFAULT_SIGNED_URL_TTL_SECS),
        scopes: parse_scopes(scopes, where_)?,
    };
    Ok((
        AuthConfig {
            mode,
            api_key_env: env,
        },
        facts,
    ))
}

/// Final data dir: flag > env > `[core] data_dir` file key > located default
/// (pure-env chain). The file key is skipped whenever the flag or env already
/// decided (they outrank it); an empty or relative file key is ignored with a
/// returned warning (absolute paths only). Pure (args, not process env); the
/// warning is returned because `load` runs before `logging::init`.
fn finalize_data_dir(
    cli_dir: Option<PathBuf>,
    env_dir: Option<OsString>,
    file: &FileCore,
    located: Option<PathBuf>,
    warnings: &mut Vec<String>,
) -> Option<PathBuf> {
    let explicit = cli_dir.or_else(|| clean(env_dir));
    explicit.or_else(|| {
        file.data_dir
            .as_deref()
            .and_then(|raw| {
                let expanded = expand_home(raw);
                if expanded.is_absolute() {
                    Some(expanded)
                } else {
                    warnings.push(format!(
                        "ignoring relative [core] data_dir {raw:?}; use an absolute path"
                    ));
                    None
                }
            })
            .or(located)
    })
}

/// Load model roots: defaults, then `config_path` (or
/// `<data_dir>/honkbox.toml`), then env overrides. `[discovery]` extras are
/// appended to the `[models]` base roots. A missing file falls back to
/// defaults; invalid TOML is an `Err`. `cli_dir` is the `--data-dir` flag;
/// the `HONKBOX_DATA_DIR` env and the `[core] data_dir` file key slot in
/// beneath it per the resolution chain in the module docs (one-pass: the
/// file key cannot relocate the config file mid-read).
/// The config file's side knobs, staged out of `load` (missing file ⇒
/// defaults; the file stage never re-reads the data-dir chain).
#[derive(Debug, Default)]
struct FileStages {
    /// `[jobs]` keys.
    jobs: FileJobs,
    /// `[swap]` keys.
    swap: FileSwap,
    /// `[artifacts]` keys (TASK-binary-artifact-api), resolved in `load`.
    artifacts: Option<u64>,
    /// `[core]` keys.
    core: FileCore,
    /// `[models] vram_budget_bytes` (NEW-5).
    vram_budget: Option<u64>,
    /// `[[models.entry]]` bindings (REQ-2), parse-validated.
    entries: Vec<ModelEntryConfig>,
    /// The per-slug LLM cache controls (`[[models.entry]] cache`), keyed by
    /// the entry's registry slug.
    cache: BTreeMap<String, CacheConfig>,
    /// `[server] auth` raw value (REQ-7), validated in `load`.
    auth: Option<String>,
    /// `[auth] api_key_env` raw value (REQ-7), validated in `load`.
    api_key_env: Option<String>,
    /// `[server] queue_wait_secs` raw value (REQ-6), resolved in `load`.
    queue_wait: Option<u64>,
    /// `[ledger]` keys (TASK-resource-ledger-bounded-queue), resolved in
    /// `load`.
    ledger: FileLedger,
    /// `[mesh]` keys (TASK-mesh-client-spike), resolved in `load`.
    mesh: mesh::FileMesh,
    /// `[tools]` keys (TASK-external-tool-registry), resolved in `load`.
    tools: FileTools,
    /// `[safety.nsfw]` keys, resolved in `load`.
    nsfw: FileNsfw,
    /// `[python]` keys, resolved in `load`.
    python: FilePython,
    /// `[licenses]` keys, resolved in `load`.
    licenses: FileLicenses,
    /// `[auth] signed_url_secret_env` raw value (a NAME only).
    signed_url_secret_env: Option<String>,
    /// `[auth] signed_url_ttl_secs` raw value.
    signed_url_ttl_secs: Option<u64>,
    /// `[[auth.scopes]]` raw entries, validated in `load`.
    scopes: Vec<FileScope>,
}

/// Read + apply the config-file stage: parse `path`'s TOML into the staged
/// roots (mutated in place) and return the file's side knobs. A missing
/// file keeps the defaults; other IO errors and bad TOML are fatal with a
/// message naming the path.
fn stage_config_file(
    path: Option<&Path>,
    staged: &mut RootStages,
    stt_from_file: &mut bool,
    tts_from_file: &mut bool,
) -> Result<FileStages, String> {
    let mut out = FileStages::default();
    let text = match path.map(std::fs::read_to_string).transpose() {
        Ok(Some(text)) => text,
        Ok(None) => return Ok(out),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => {
            let display = path.unwrap_or(Path::new("<none>"));
            return Err(format!("cannot read config {}: {e}", display.display()));
        }
    };
    let display = path.unwrap_or(Path::new("<none>"));
    let file: FileConfig =
        toml::from_str(&text).map_err(|e| format!("bad config {}: {e}", display.display()))?;
    if let Some(dirs) = file.models.llm_dirs {
        staged.llm = dirs.iter().map(|d| expand_home(d)).collect();
    }
    if let Some(dir) = file.models.comfyui_dir {
        staged.comfyui = vec![expand_home(&dir)];
    }
    if let Some(dirs) = file.discovery.llm_dirs {
        staged.extra_llm = dirs.iter().map(|d| expand_home(d)).collect();
    }
    if let Some(dirs) = file.discovery.comfyui_dirs {
        staged.extra_comfyui = dirs.iter().map(|d| expand_home(d)).collect();
    }
    if let Some(dirs) = file.models.stt_dirs {
        staged.stt = dirs.iter().map(|d| expand_home(d)).collect();
        *stt_from_file = true;
    }
    if let Some(dirs) = file.models.tts_dirs {
        staged.tts = dirs.iter().map(|d| expand_home(d)).collect();
        *tts_from_file = true;
    }
    if let Some(dirs) = file.discovery.stt_dirs {
        staged.extra_stt = dirs.iter().map(|d| expand_home(d)).collect();
    }
    if let Some(dirs) = file.discovery.tts_dirs {
        staged.extra_tts = dirs.iter().map(|d| expand_home(d)).collect();
    }
    if let Some(dirs) = file.models.ocr_dirs {
        staged.ocr = dirs.iter().map(|d| expand_home(d)).collect();
    }
    if let Some(dirs) = file.discovery.ocr_dirs {
        staged.extra_ocr = dirs.iter().map(|d| expand_home(d)).collect();
    }
    out.jobs = file.jobs;
    out.swap = file.swap;
    out.artifacts = file.artifacts.max_artifact_bytes;
    out.core = file.core;
    out.auth = file.server.auth;
    out.api_key_env = file.auth.api_key_env;
    out.queue_wait = file.server.queue_wait_secs;
    out.ledger = file.ledger;
    out.mesh = file.mesh;
    out.tools = file.tools;
    out.vram_budget = file.models.vram_budget_bytes;
    out.nsfw = file.safety.nsfw;
    out.python = file.python;
    out.licenses = file.licenses;
    out.signed_url_secret_env = file.auth.signed_url_secret_env;
    out.signed_url_ttl_secs = file.auth.signed_url_ttl_secs;
    out.scopes = file.auth.scopes;
    // The bindings + the per-slug cache knobs.
    let (entries, cache) = parse_entries(&file.models.entry)?;
    out.entries = entries;
    out.cache = cache;
    Ok(out)
}

/// The parsed `[[models.entry]]` bindings plus the per-slug cache knobs
/// (keyed by the entry's slug). A cache block rides a slug-targeting entry:
/// the slug is the key the policy looks up.
fn parse_entries(
    files: &[FileModelEntry],
) -> Result<(Vec<ModelEntryConfig>, BTreeMap<String, CacheConfig>), String> {
    let mut entries = Vec::new();
    let mut cache = BTreeMap::new();
    for file_entry in files {
        let entry = file_entry.parse()?;
        if let Some(knobs) = file_entry.cache.config()? {
            let Some(slug) = entry.slug.clone() else {
                return Err(
                    "[[models.entry]] cache needs a slug-targeting entry (the slug is the key)"
                        .to_owned(),
                );
            };
            cache.insert(slug, knobs);
        }
        entries.push(entry);
    }
    Ok((entries, cache))
}

/// One `[[models.entry]]` table as written (`match` is a Rust keyword, so
/// the field is renamed); validated + cleaned by [`FileModelEntry::parse`].
#[derive(Debug, Default, serde::Deserialize)]
struct FileModelEntry {
    /// Exact registry slug (`llm:…` / `comfy:…`).
    #[serde(default)]
    slug: Option<String>,
    /// Weight file name to match.
    #[serde(rename = "match", default)]
    file_match: Option<String>,
    /// Request aliases (operator-declared; never filename-derived).
    #[serde(default)]
    aliases: Option<Vec<String>>,
    /// Capability overrides (the `[models.entry.capabilities]` sub-table
    /// or an inline `capabilities = { … }` value).
    #[serde(default)]
    capabilities: ModelCapabilitiesOverride,
    /// The LLM cache controls (the `[models.entry.cache]` sub-table).
    #[serde(default)]
    cache: FileCacheConfig,
}

/// The `[models.entry.cache]` sub-table as written.
#[derive(Debug, Default, serde::Deserialize)]
struct FileCacheConfig {
    /// The KV cache's context size in tokens.
    #[serde(default)]
    n_ctx: Option<u32>,
    /// The KV cache's element type (`f16` | `q8_0`).
    #[serde(default)]
    kv_quant: Option<KvQuantConfig>,
    /// Whether the KV cache is offloaded to the GPU.
    #[serde(default)]
    offload: Option<bool>,
}

impl FileCacheConfig {
    /// The knobs, or `None` when the entry declares no cache block (the
    /// backend defaults then apply).
    fn config(&self) -> Result<Option<CacheConfig>, String> {
        if self.n_ctx.is_none() && self.kv_quant.is_none() && self.offload.is_none() {
            return Ok(None);
        }
        if self.n_ctx == Some(0) {
            return Err("[[models.entry]] cache.n_ctx must be greater than 0".to_owned());
        }
        Ok(Some(CacheConfig {
            n_ctx: self.n_ctx,
            kv_quant: self.kv_quant,
            offload: self.offload,
        }))
    }
}

impl FileModelEntry {
    /// Clean + validate: exactly one of `slug` / `match` must be present,
    /// aliases are trimmed/de-duplicated (empty and repeated aliases are
    /// config errors — a silently dropped alias would resolve differently
    /// than the operator wrote), and the capabilities flatten verbatim.
    fn parse(&self) -> Result<ModelEntryConfig, String> {
        let target = match (&self.slug, &self.file_match) {
            (Some(slug), None) => Some(clean_slug_key(slug, "slug")?),
            (None, Some(_)) => None,
            (Some(_), Some(_)) => {
                return Err(
                    "[[models.entry]] declares both slug and match; declare exactly one".to_owned(),
                );
            }
            (None, None) => {
                return Err(
                    "[[models.entry]] declares neither slug nor match; declare exactly one"
                        .to_owned(),
                );
            }
        };
        let file_match = match &self.file_match {
            Some(m) => Some(clean_slug_key(m, "match")?),
            None => None,
        };
        let mut aliases = Vec::new();
        for alias in clean_slugs(self.aliases.iter().flatten()) {
            if aliases.contains(&alias) {
                return Err(format!("[[models.entry]] declares alias {alias:?} twice"));
            }
            aliases.push(alias);
        }
        Ok(ModelEntryConfig {
            slug: target,
            file_match,
            aliases,
            capabilities: self.capabilities,
        })
    }
}

/// Trim one `slug`/`match` key; empty after trimming is a config error.
fn clean_slug_key(value: &str, key: &str) -> Result<String, String> {
    let trimmed = value.trim().to_owned();
    if trimmed.is_empty() {
        return Err(format!("[[models.entry]] {key} is empty"));
    }
    Ok(trimmed)
}

#[allow(clippy::too_many_lines)] // one stage per knob, by design
pub fn load(
    config_path: Option<PathBuf>,
    cli_dir: Option<PathBuf>,
) -> Result<LoadedConfig, String> {
    let env_dir = std::env::var_os(DATA_DIR_ENV);
    let located = resolve_data_dir(
        cli_dir.clone(),
        env_dir.clone(),
        std::env::var_os("XDG_STATE_HOME"),
        std::env::var_os("HOME"),
    );
    let path = config_path.or_else(|| located.as_ref().map(|d| d.join("honkbox.toml")));
    let mut staged = RootStages::defaults();
    let mut stt_from_file = false;
    let mut tts_from_file = false;
    let FileStages {
        jobs: file_jobs,
        swap: file_swap,
        core: file_core,
        vram_budget,
        entries,
        cache,
        auth,
        api_key_env,
        signed_url_secret_env,
        signed_url_ttl_secs,
        scopes,
        queue_wait: queue_wait_file,
        ledger: file_ledger,
        artifacts: artifact_cap_file,
        mesh: file_mesh,
        tools: file_tools,
        nsfw: file_nsfw,
        python: file_python,
        licenses: file_licenses,
    } = stage_config_file(
        path.as_deref(),
        &mut staged,
        &mut stt_from_file,
        &mut tts_from_file,
    )?;
    // Non-fatal notices for the caller to emit after `logging::init`.
    let mut warnings: Vec<String> = Vec::new();
    let data_dir = finalize_data_dir(cli_dir, env_dir, &file_core, located, &mut warnings);
    // Starter model dirs derive from the final data dir (rebasing here keeps
    // a file-refined `[core] data_dir` authoritative for them too); explicit
    // `[models] stt_dirs`/`tts_dirs` file keys win over the derived starters.
    if let Some(base) = &data_dir {
        if !stt_from_file {
            staged.stt = vec![base.join("models/stt")];
        }
        if !tts_from_file {
            staged.tts = vec![base.join("models/tts")];
        }
    }
    let jobs = gc_settings(
        &file_jobs,
        std::env::var("HONKBOX_GC_TTL").ok().as_deref(),
        std::env::var("HONKBOX_GC_MAX_BYTES").ok().as_deref(),
    );
    let swap = swap_settings(
        &file_swap,
        std::env::var("HONKBOX_SWAP_LLM_TTL").ok().as_deref(),
        std::env::var("HONKBOX_SWAP_LLM_MAX_LOADED").ok().as_deref(),
        std::env::var("HONKBOX_SWAP_MAX_TTL").ok().as_deref(),
        std::env::var("HONKBOX_SWAP_PIN").ok().as_deref(),
    );
    // `[models] vram_budget_bytes` (NEW-5): env wins over the file key.
    let vram_budget = vram_budget_settings(
        vram_budget,
        std::env::var("HONKBOX_VRAM_BUDGET").ok().as_deref(),
    );
    let mut models = apply_env(
        staged,
        std::env::var("HONKBOX_LLM_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_COMFYUI_DIR").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_LLM_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_COMFYUI_DIRS")
            .ok()
            .as_deref(),
        std::env::var("HONKBOX_STT_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_TTS_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_STT_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_TTS_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_OCR_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_OCR_DIRS").ok().as_deref(),
    )
    .merged(jobs, swap);
    models.vram_budget_bytes = vram_budget;
    models.entries = entries;
    models.cache = cache;
    // Managed download roots (NEW-4): `<data_dir>/models/{llm,sd,stt,tts}`
    // stay part of the scan no matter what file/env configured, so a
    // completed download shows up in `GET /v1/models` after the rescan
    // (exact-path dedupe keeps the starter dirs from doubling up).
    if let Some(base) = &data_dir {
        append_managed_roots(&mut models, base);
    }
    let display = path
        .as_deref()
        .map_or_else(|| "<none>".to_owned(), |p| p.display().to_string());
    let nsfw = resolve_nsfw(&file_nsfw, &display)?;
    let python = PythonConfig {
        enabled: file_python.enabled.unwrap_or(false),
        venv: file_python
            .venv
            .as_deref()
            .map_or_else(|| python_venv_default(data_dir.as_deref()), expand_home),
        timeout_secs: file_python
            .timeout_secs
            .unwrap_or(DEFAULT_PYTHON_TIMEOUT_SECS),
        model_allowlist: clean_slugs(
            file_python
                .model_allowlist
                .iter()
                .flatten()
                .map(String::as_str),
        ),
    };
    let licenses = resolve_licenses(
        file_licenses.territory.as_deref(),
        file_licenses.commercial,
        std::env::var("HONKBOX_LICENSES_TERRITORY").ok().as_deref(),
        std::env::var("HONKBOX_LICENSES_COMMERCIAL").ok().as_deref(),
        &mut warnings,
    );
    let (auth, auth_facts) = auth_settings(
        auth.as_deref(),
        api_key_env.as_deref(),
        signed_url_secret_env.as_deref(),
        signed_url_ttl_secs,
        &scopes,
        &display,
    )?;
    let server = server_settings(
        queue_wait_file,
        std::env::var("HONKBOX_QUEUE_WAIT_SECS").ok().as_deref(),
    );
    let ledger = ledger_settings(
        &file_ledger,
        std::env::var("HONKBOX_LEDGER_GPU_SLOTS").ok().as_deref(),
        std::env::var("HONKBOX_LEDGER_VRAM_BUDGET").ok().as_deref(),
        std::env::var("HONKBOX_LEDGER_RAM_BUDGET").ok().as_deref(),
        std::env::var("HONKBOX_LEDGER_CPU_THREADS").ok().as_deref(),
        std::env::var("HONKBOX_LEDGER_OOM_HEADROOM").ok().as_deref(),
        std::env::var("HONKBOX_LEDGER_QUEUE_CAP").ok().as_deref(),
    )?;
    let artifact_max_bytes = artifact_settings(
        artifact_cap_file,
        std::env::var("HONKBOX_ARTIFACT_MAX_BYTES").ok().as_deref(),
        &mut warnings,
    );
    let mesh = mesh::MeshConfig::resolve(&file_mesh)?;
    let tools = ExternalToolsConfig {
        blender_path: file_tools.blender_path.map(|p| expand_home(&p)),
        comfyui_url: file_tools.comfyui_url,
        llama_server_url: file_tools.llama_server_url,
        timeout: Duration::from_secs(file_tools.timeout_secs.unwrap_or(5).clamp(1, 3600)),
    };
    Ok(LoadedConfig {
        models,
        auth,
        auth_facts,
        server,
        ledger,
        artifact_max_bytes,
        mesh,
        tools,
        nsfw,
        python,
        licenses,
        data_dir,
        warnings,
    })
}

/// Append the managed download roots under `base/models` to a resolved
/// `ModelConfig`: `llm` under `models/llm`, `sd` as a `ComfyUI` base under
/// `models/sd` (its download subdir `models/sd/checkpoints` is inside it),
/// `stt` under `models/stt`, `tts` under `models/tts`, `ocr` under
/// `models/ocr`. Exact-path dedupe: the starter `stt`/`tts` dirs already
/// equal these paths.
fn append_managed_roots(models: &mut ModelConfig, base: &Path) {
    push_unique(&mut models.llm_dirs, base.join("models/llm"));
    push_unique(&mut models.comfyui_dirs, base.join("models/sd"));
    push_unique(&mut models.stt_dirs, base.join("models/stt"));
    push_unique(&mut models.tts_dirs, base.join("models/tts"));
    push_unique(&mut models.ocr_dirs, base.join("models/ocr"));
}

/// Push `path` unless an identical entry exists (the registry scan dedupes
/// canonical paths too; this keeps config output honest).
fn push_unique(dirs: &mut Vec<PathBuf>, path: PathBuf) {
    if !dirs.contains(&path) {
        dirs.push(path);
    }
}

/// Staged roots before the final merge: the `[models]` base plus the
/// `[discovery]` extras, each stage replaceable by its env var.
#[derive(Debug, Clone, Default)]
struct RootStages {
    llm: Vec<PathBuf>,
    comfyui: Vec<PathBuf>,
    stt: Vec<PathBuf>,
    tts: Vec<PathBuf>,
    ocr: Vec<PathBuf>,
    extra_llm: Vec<PathBuf>,
    extra_comfyui: Vec<PathBuf>,
    extra_stt: Vec<PathBuf>,
    extra_tts: Vec<PathBuf>,
    extra_ocr: Vec<PathBuf>,
}

impl RootStages {
    /// Built-in defaults used when no config file is present. The starter
    /// `stt`/`tts` roots derive from the data dir instead (set in `load`).
    fn defaults() -> Self {
        Self {
            llm: vec![expand_home("~/.cache/huggingface")],
            comfyui: vec![expand_home("~/git-ai/ai-models/Data/Models")],
            stt: Vec::new(),
            tts: Vec::new(),
            ocr: Vec::new(),
            extra_llm: Vec::new(),
            extra_comfyui: Vec::new(),
            extra_stt: Vec::new(),
            extra_tts: Vec::new(),
            extra_ocr: Vec::new(),
        }
    }

    /// Merge: base roots first, `[discovery]` extras appended (the scan
    /// dedupes equal canonical paths across roots).
    fn merged(self, jobs: JobsConfig, swap: SwapConfig) -> ModelConfig {
        let mut llm_dirs = self.llm;
        llm_dirs.extend(self.extra_llm);
        let mut comfyui_dirs = self.comfyui;
        comfyui_dirs.extend(self.extra_comfyui);
        let mut stt_dirs = self.stt;
        stt_dirs.extend(self.extra_stt);
        let mut tts_dirs = self.tts;
        tts_dirs.extend(self.extra_tts);
        let mut ocr_dirs = self.ocr;
        ocr_dirs.extend(self.extra_ocr);
        ModelConfig {
            llm_dirs,
            comfyui_dirs,
            stt_dirs,
            tts_dirs,
            ocr_dirs,
            jobs,
            swap,
            // Resolved separately in `load` (file key + env stage).
            vram_budget_bytes: None,
            // Operator bindings staged out of the file pass (`load` fills
            // them; no env stage — structured tables have none).
            entries: Vec::new(),
            // The per-slug cache knobs, staged with the bindings.
            cache: BTreeMap::new(),
        }
    }
}

/// Env wins over file, per stage: `HONKBOX_LLM_DIRS` (path-separated via
/// `split_paths`) replaces the `[models]` `llm_dirs`, `HONKBOX_COMFYUI_DIR`
/// replaces the `[models]` `comfyui_dir`, `HONKBOX_STT_DIRS` /
/// `HONKBOX_TTS_DIRS` replace the `[models]` `stt_dirs` / `tts_dirs`, and
/// `HONKBOX_DISCOVERY_LLM_DIRS` / `HONKBOX_DISCOVERY_COMFYUI_DIRS` /
/// `HONKBOX_DISCOVERY_STT_DIRS` / `HONKBOX_DISCOVERY_TTS_DIRS` replace the
/// `[discovery]` extras; `HONKBOX_OCR_DIRS` / `HONKBOX_DISCOVERY_OCR_DIRS`
/// do the same for OCR weights. Pure (args instead of process env) so tests
/// stay thread-safe.
#[allow(clippy::too_many_arguments)]
fn apply_env(
    mut staged: RootStages,
    llm_dirs: Option<&str>,
    comfyui_dir: Option<&str>,
    extra_llm_dirs: Option<&str>,
    extra_comfyui_dirs: Option<&str>,
    stt_dirs: Option<&str>,
    tts_dirs: Option<&str>,
    extra_stt_dirs: Option<&str>,
    extra_tts_dirs: Option<&str>,
    ocr_dirs: Option<&str>,
    extra_ocr_dirs: Option<&str>,
) -> RootStages {
    if let Some(v) = llm_dirs
        && !v.is_empty()
    {
        staged.llm = std::env::split_paths(v).collect();
    }
    if let Some(v) = comfyui_dir
        && !v.is_empty()
    {
        staged.comfyui = vec![PathBuf::from(v)];
    }
    if let Some(v) = extra_llm_dirs
        && !v.is_empty()
    {
        staged.extra_llm = std::env::split_paths(v).collect();
    }
    if let Some(v) = extra_comfyui_dirs
        && !v.is_empty()
    {
        staged.extra_comfyui = std::env::split_paths(v).collect();
    }
    if let Some(v) = stt_dirs
        && !v.is_empty()
    {
        staged.stt = std::env::split_paths(v).collect();
    }
    if let Some(v) = tts_dirs
        && !v.is_empty()
    {
        staged.tts = std::env::split_paths(v).collect();
    }
    if let Some(v) = extra_stt_dirs
        && !v.is_empty()
    {
        staged.extra_stt = std::env::split_paths(v).collect();
    }
    if let Some(v) = extra_tts_dirs
        && !v.is_empty()
    {
        staged.extra_tts = std::env::split_paths(v).collect();
    }
    if let Some(v) = ocr_dirs
        && !v.is_empty()
    {
        staged.ocr = std::env::split_paths(v).collect();
    }
    if let Some(v) = extra_ocr_dirs
        && !v.is_empty()
    {
        staged.extra_ocr = std::env::split_paths(v).collect();
    }
    staged
}

/// A leading `~` expands to `$HOME`; anything else passes through.
fn expand_home(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        return match home::home_dir() {
            Some(h) => h.join(rest),
            None => PathBuf::from(value),
        };
    }
    if value == "~" {
        return home::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    PathBuf::from(value)
}

/// Whole-file shape: `[core]` + `[models]` (base roots) + optional
/// `[discovery]` + optional `[jobs]` + optional `[swap]` + optional
/// `[server]`/`[auth]` (REQ-7).
#[derive(Debug, Default, serde::Deserialize)]
struct FileConfig {
    /// `[server]` keys (REQ-7 auth mode); absent table keeps the default.
    #[serde(default)]
    server: FileServer,
    /// `[ledger]` keys (TASK-resource-ledger-bounded-queue); absent table
    /// keeps the default (one GPU slot, unmanaged budgets, depth-1 queues).
    #[serde(default)]
    ledger: FileLedger,
    /// `[auth]` keys (REQ-7 key env-var NAME); absent table keeps the default.
    #[serde(default)]
    auth: FileAuth,
    /// `[safety]` keys (NSFW gate); absent table keeps defaults.
    #[serde(default)]
    safety: FileSafety,
    /// `[python]` keys (sidecar); absent table keeps defaults.
    #[serde(default)]
    python: FilePython,
    #[serde(default)]
    core: FileCore,
    #[serde(default)]
    models: FileModels,
    #[serde(default)]
    discovery: FileDiscovery,
    #[serde(default)]
    jobs: FileJobs,
    #[serde(default)]
    swap: FileSwap,
    /// `[artifacts]` keys (TASK-binary-artifact-api); absent table keeps
    /// the default (1 GiB total artifact cap).
    #[serde(default)]
    artifacts: FileArtifacts,
    /// `[mesh]` keys (TASK-mesh-client-spike); absent table keeps the
    /// default (mesh off, no upstreams).
    #[serde(default)]
    mesh: mesh::FileMesh,
    /// `[tools]` keys (TASK-external-tool-registry); absent table uses PATH
    /// / the loopback `ComfyUI` default, no attached `llama-server` endpoint
    /// and a 5 s probe timeout.
    #[serde(default)]
    tools: FileTools,
    /// `[licenses]` keys (TASK-model-license-registry); absent table keeps
    /// the defaults (`unknown` / `true`).
    #[serde(default)]
    licenses: FileLicenses,
}

/// `[licenses]` keys: `territory` (declared jurisdiction) and `commercial`
/// (false opts a research pipeline out of the NC refusal).
#[derive(Debug, Default, serde::Deserialize)]
struct FileLicenses {
    territory: Option<String>,
    commercial: Option<bool>,
}

/// `[core]` keys: `data_dir` (the one data-dir knob's file stage).
#[derive(Debug, Default, serde::Deserialize)]
struct FileCore {
    data_dir: Option<String>,
}

/// `[server]` keys: `auth` (`none` or `bearer`, REQ-7; parsed by
/// `AuthMode::parse` in the file stage - invalid values are config errors)
/// and `queue_wait_secs` (REQ-6 Option A admission-queue bound, seconds;
/// resolved by `server_settings`).
#[derive(Debug, Default, serde::Deserialize)]
struct FileServer {
    auth: Option<String>,
    queue_wait_secs: Option<u64>,
}

/// `[ledger]` keys (TASK-resource-ledger-bounded-queue): scalar capacities
/// and queue knobs plus the file-only per-kind maps. Each missing key keeps
/// its default (see [`LedgerSettings`]).
#[derive(Debug, Default, Clone, serde::Deserialize)]
struct FileLedger {
    /// Shared GPU slots (default 1).
    #[serde(default)]
    gpu_slots: Option<u32>,
    /// Total VRAM budget in bytes; `0`/absent = unmanaged.
    #[serde(default)]
    vram_budget_bytes: Option<u64>,
    /// Total RAM budget in bytes; `0`/absent = unmanaged.
    #[serde(default)]
    ram_budget_bytes: Option<u64>,
    /// CPU thread budget; `0`/absent = unmanaged.
    #[serde(default)]
    cpu_threads: Option<u64>,
    /// OOM near-miss pressure margin in bytes.
    #[serde(default)]
    oom_headroom_bytes: Option<u64>,
    /// Queue length per kind (clamped to >= 1; default 1).
    #[serde(default)]
    queue_cap: Option<usize>,
    /// Per-kind queue-cap overrides (`[ledger.queue_caps]`, file-only;
    /// labels are the fixed kind vocabulary, unknown labels are errors).
    #[serde(default)]
    queue_caps: Option<BTreeMap<String, usize>>,
    /// Per-kind VRAM demand estimates in bytes (`[ledger.vram_demand]`,
    /// file-only; absent = the kind claims no VRAM).
    #[serde(default)]
    vram_demand: Option<BTreeMap<String, u64>>,
}

/// `[auth]` keys: the bearer key env-var NAME (REQ-7), the signed-URL secret
/// env-var NAME + TTL, and `[[auth.scopes]]` entries. Only NAMES live here;
/// VALUES are read per request and never logged or dumped.
#[derive(Debug, Default, serde::Deserialize)]
struct FileAuth {
    api_key_env: Option<String>,
    #[serde(default)]
    signed_url_secret_env: Option<String>,
    #[serde(default)]
    signed_url_ttl_secs: Option<u64>,
    #[serde(default)]
    scopes: Vec<FileScope>,
}

/// `[safety]` table as written.
#[derive(Debug, Default, serde::Deserialize)]
struct FileSafety {
    #[serde(default)]
    nsfw: FileNsfw,
}

/// `[safety.nsfw]` keys; each missing key keeps its default.
#[derive(Debug, Default, serde::Deserialize)]
struct FileNsfw {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    policy: Option<String>,
    #[serde(default)]
    model_version: Option<String>,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    classifier_bin: Option<String>,
}

/// `[python]` keys; each missing key keeps its default (`venv` derives from
/// the final data dir in `load`).
#[derive(Debug, Default, serde::Deserialize)]
struct FilePython {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    venv: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    model_allowlist: Option<Vec<String>>,
}

/// `[models]` keys; each missing key keeps its default.
#[derive(Debug, Default, serde::Deserialize)]
struct FileModels {
    #[serde(default)]
    llm_dirs: Option<Vec<String>>,
    #[serde(default)]
    comfyui_dir: Option<String>,
    #[serde(default)]
    stt_dirs: Option<Vec<String>>,
    #[serde(default)]
    tts_dirs: Option<Vec<String>>,
    #[serde(default)]
    ocr_dirs: Option<Vec<String>>,
    /// Optional VRAM budget (bytes) for the curated catalog's fit pills.
    #[serde(default)]
    vram_budget_bytes: Option<u64>,
    /// Operator model bindings (`[[models.entry]]`, REQ-2).
    #[serde(default)]
    entry: Vec<FileModelEntry>,
}

/// Optional `[discovery]` extras APPENDED to the `[models]` roots (P0 of the
/// llama-swap parity plan); mirrors `[models]` with a plural `comfyui_dirs`
/// so extra `ComfyUI` trees can be added without replacing the base one.
/// Audio parity: `stt_dirs` / `tts_dirs` work exactly like the `[models]`
/// keys; `ocr_dirs` likewise for OCR weights. The research doc's `roots = [{kind, path}]` table plus `cache`/
/// `rescan_seconds` still land with the P1 adapter work; unknown
/// `[discovery]` keys are ignored.
#[derive(Debug, Default, serde::Deserialize)]
#[allow(clippy::struct_field_names)]
struct FileDiscovery {
    #[serde(default)]
    llm_dirs: Option<Vec<String>>,
    #[serde(default)]
    ocr_dirs: Option<Vec<String>>,
    #[serde(default)]
    comfyui_dirs: Option<Vec<String>>,
    #[serde(default)]
    stt_dirs: Option<Vec<String>>,
    #[serde(default)]
    tts_dirs: Option<Vec<String>>,
}

/// `[jobs]` GC keys: `gc_ttl` (seconds) and `gc_max_bytes` (bytes); each
/// missing key keeps its default.
#[derive(Debug, Default, Clone, Copy, serde::Deserialize)]
struct FileJobs {
    #[serde(default)]
    gc_ttl: Option<u64>,
    #[serde(default)]
    gc_max_bytes: Option<u64>,
}

/// `[swap]` keys: `enabled`, `llm_ttl_secs` (seconds), `llm_max_loaded`
/// (count), `max_ttl_secs` (seconds), `pin` (slug list); each missing key
/// keeps its default.
#[derive(Debug, Default, Clone, serde::Deserialize)]
struct FileSwap {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    llm_ttl_secs: Option<u64>,
    #[serde(default)]
    llm_max_loaded: Option<usize>,
    #[serde(default)]
    max_ttl_secs: Option<u64>,
    #[serde(default)]
    pin: Option<Vec<String>>,
}

/// `[artifacts]` keys (TASK-binary-artifact-api): `max_artifact_bytes` —
/// the total size cap for ONE stored binary artifact; a missing or invalid
/// key keeps [`DEFAULT_ARTIFACT_MAX_BYTES`].
#[derive(Debug, Default, Clone, Copy, serde::Deserialize)]
struct FileArtifacts {
    #[serde(default)]
    max_artifact_bytes: Option<u64>,
}

/// `[tools]` keys: explicit engine locations + probe timeout.
#[derive(Debug, Default, serde::Deserialize)]
struct FileTools {
    blender_path: Option<String>,
    comfyui_url: Option<String>,
    llama_server_url: Option<String>,
    timeout_secs: Option<u64>,
}

impl Default for SwapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            llm_ttl: Duration::from_secs(DEFAULT_SWAP_TTL_SECS),
            llm_max_loaded: DEFAULT_SWAP_MAX_LOADED,
            max_ttl: Duration::from_secs(DEFAULT_SWAP_MAX_TTL_SECS),
            pin: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_env_precedence() {
        let home = home::home_dir().expect("test needs $HOME");
        let cfg = RootStages::defaults().merged(JobsConfig::default(), SwapConfig::default());
        assert_eq!(cfg.llm_dirs, vec![home.join(".cache/huggingface")]);
        assert_eq!(
            cfg.comfyui_dirs,
            vec![home.join("git-ai/ai-models/Data/Models")]
        );
        // Starter audio roots derive from the data dir, not from `$HOME`.
        assert_eq!(cfg.stt_dirs, Vec::<PathBuf>::new());
        assert_eq!(cfg.tts_dirs, Vec::<PathBuf>::new());
        assert_eq!(cfg.ocr_dirs, Vec::<PathBuf>::new());

        let staged = apply_env(
            RootStages::defaults(),
            Some("/tmp/honkbox-a:/tmp/honkbox-b"),
            Some("/tmp/honkbox-comfy"),
            Some(""),
            Some(""),
            Some("/tmp/honkbox-stt:/tmp/honkbox-stt2"),
            Some("/tmp/honkbox-tts"),
            Some(""),
            Some(""),
            Some("/tmp/honkbox-ocr"),
            Some(""),
        );
        let over = staged
            .clone()
            .merged(JobsConfig::default(), SwapConfig::default());
        assert_eq!(
            over.llm_dirs,
            vec![
                PathBuf::from("/tmp/honkbox-a"),
                PathBuf::from("/tmp/honkbox-b")
            ]
        );
        assert_eq!(over.comfyui_dirs, vec![PathBuf::from("/tmp/honkbox-comfy")]);
        assert_eq!(
            over.stt_dirs,
            vec![
                PathBuf::from("/tmp/honkbox-stt"),
                PathBuf::from("/tmp/honkbox-stt2")
            ]
        );
        assert_eq!(over.tts_dirs, vec![PathBuf::from("/tmp/honkbox-tts")]);
        assert_eq!(over.ocr_dirs, vec![PathBuf::from("/tmp/honkbox-ocr")]);

        // Empty values keep the file/default values.
        let kept = apply_env(
            staged,
            Some(""),
            Some(""),
            Some(""),
            Some(""),
            Some(""),
            Some(""),
            Some(""),
            Some(""),
            Some(""),
            Some(""),
        )
        .merged(JobsConfig::default(), SwapConfig::default());
        assert_eq!(kept.llm_dirs, over.llm_dirs);
        assert_eq!(kept.comfyui_dirs, over.comfyui_dirs);
        assert_eq!(kept.stt_dirs, over.stt_dirs);
        assert_eq!(kept.tts_dirs, over.tts_dirs);
        assert_eq!(kept.ocr_dirs, over.ocr_dirs);

        // A missing file loads (defaults + ambient env), never errors.
        load(
            Some(PathBuf::from("/nonexistent-honkbox-config.toml")),
            None,
        )
        .expect("missing config file loads");
    }

    /// Resolution chain: flag > env > XDG default (empty/relative treated as
    /// unset) > `$HOME/.local/state/honkbox`; missing everything → `None`.
    #[test]
    fn resolve_data_dir_precedence() {
        let flag = || Some(PathBuf::from("/flag-dir"));
        let env = || Some("/env-dir".into());
        let xdg = || Some("/xdg-state".into());
        let home = || Some("/home/u".into());
        // Flag wins over everything, even malformed lower stages.
        assert_eq!(
            resolve_data_dir(flag(), env(), xdg(), home()),
            Some(PathBuf::from("/flag-dir"))
        );
        assert_eq!(
            resolve_data_dir(None, env(), xdg(), home()),
            Some(PathBuf::from("/env-dir"))
        );
        // XDG default with the env unset (relative XDG counts as unset).
        assert_eq!(
            resolve_data_dir(None, None, xdg(), home()),
            Some(PathBuf::from("/xdg-state/honkbox"))
        );
        assert_eq!(
            resolve_data_dir(None, None, Some("relative/xdg".into()), home()),
            Some(PathBuf::from("/home/u/.local/state/honkbox"))
        );
        assert_eq!(
            resolve_data_dir(None, None, Some(String::new().into()), home()),
            Some(PathBuf::from("/home/u/.local/state/honkbox"))
        );
        assert_eq!(
            resolve_data_dir(None, None, None, home()),
            Some(PathBuf::from("/home/u/.local/state/honkbox"))
        );
        // A relative HONKBOX_DATA_DIR counts as unset (falls through to XDG).
        assert_eq!(
            resolve_data_dir(None, Some("rel/dir".into()), xdg(), home()),
            Some(PathBuf::from("/xdg-state/honkbox"))
        );
        // Nothing resolvable → no data dir.
        assert_eq!(resolve_data_dir(None, None, None, None), None);
        assert_eq!(
            resolve_data_dir(None, None, None, Some(String::new().into())),
            None
        );
    }

    #[test]
    fn discovery_extras_append_per_stage() {
        // Discovery extras are a separate stage: base env does not touch them.
        let merged = apply_env(
            RootStages {
                extra_llm: vec![PathBuf::from("/tmp/extra-llm")],
                extra_comfyui: vec![PathBuf::from("/tmp/extra-comfy")],
                extra_ocr: vec![PathBuf::from("/tmp/extra-ocr")],
                ..RootStages::default()
            },
            Some("/tmp/base-llm"),
            Some("/tmp/base-comfy"),
            Some(""),
            Some("/tmp/replacement-comfy:/tmp/second-comfy"),
            Some(""),
            Some(""),
            Some("/tmp/replacement-stt"),
            Some("/tmp/replacement-tts"),
            Some(""),
            Some("/tmp/replacement-ocr"),
        )
        .merged(JobsConfig::default(), SwapConfig::default());
        assert_eq!(
            merged.llm_dirs,
            vec![
                PathBuf::from("/tmp/base-llm"),
                PathBuf::from("/tmp/extra-llm")
            ]
        );
        assert_eq!(
            merged.comfyui_dirs,
            vec![
                PathBuf::from("/tmp/base-comfy"),
                PathBuf::from("/tmp/replacement-comfy"),
                PathBuf::from("/tmp/second-comfy"),
            ]
        );
        // Audio stages behave the same: extras replace, base env does not touch them.
        assert_eq!(merged.stt_dirs, vec![PathBuf::from("/tmp/replacement-stt")]);
        assert_eq!(merged.tts_dirs, vec![PathBuf::from("/tmp/replacement-tts")]);
        assert_eq!(merged.ocr_dirs, vec![PathBuf::from("/tmp/replacement-ocr")]);
    }

    #[test]
    fn discovery_table_appends_to_models() {
        let base = std::env::temp_dir().join(format!("honkbox-discovery-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let path = base.join("honkbox.toml");
        std::fs::write(
            &path,
            "[models]\ncomfyui_dir = \"/tmp/base-comfy\"\nstt_dirs = [\"/tmp/base-stt\"]\nocr_dirs = [\"/tmp/base-ocr\"]\n\n[discovery]\nllm_dirs = [\"/tmp/extra-llm\"]\ncomfyui_dirs = [\"/tmp/extra-comfy\"]\nstt_dirs = [\"/tmp/extra-stt\"]\ntts_dirs = [\"/tmp/extra-tts\"]\nocr_dirs = [\"/tmp/extra-ocr\"]\n",
        )
        .expect("write config");
        let cfg = load(Some(path), None).expect("load config").models;
        std::fs::remove_dir_all(&base).ok();
        // `contains` keeps the test robust against ambient HONKBOX_* env.
        assert!(cfg.llm_dirs.contains(&PathBuf::from("/tmp/extra-llm")));
        assert!(
            cfg.comfyui_dirs
                .contains(&PathBuf::from("/tmp/extra-comfy"))
        );
        assert!(cfg.stt_dirs.contains(&PathBuf::from("/tmp/base-stt")));
        assert!(cfg.stt_dirs.contains(&PathBuf::from("/tmp/extra-stt")));
        assert!(cfg.tts_dirs.contains(&PathBuf::from("/tmp/extra-tts")));
        assert!(cfg.ocr_dirs.contains(&PathBuf::from("/tmp/base-ocr")));
        assert!(cfg.ocr_dirs.contains(&PathBuf::from("/tmp/extra-ocr")));
    }

    /// The `[core] data_dir` file key refines the default and relocates the
    /// starter model dirs; a relative file key is ignored (treated unset).
    #[test]
    fn file_core_data_dir_refines() {
        let base = std::env::temp_dir().join(format!("honkbox-core-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("my-state");
        std::fs::create_dir_all(&dir).expect("test dir");
        let path = dir.join("honkbox.toml");
        std::fs::write(
            &path,
            format!("[core]\ndata_dir = {:?}\n", dir.join("other").display()),
        )
        .expect("write config");
        let loaded = load(Some(path), None).expect("load config");
        assert_eq!(loaded.data_dir, Some(dir.join("other")));
        assert_eq!(loaded.models.stt_dirs, vec![dir.join("other/models/stt")]);
        assert_eq!(loaded.models.tts_dirs, vec![dir.join("other/models/tts")]);

        // Flag beats the file key.
        let path = dir.join("honkbox.toml");
        let loaded = load(Some(path.clone()), Some(dir.join("flag"))).expect("load config");
        assert_eq!(loaded.data_dir, Some(dir.join("flag")));
        assert_eq!(loaded.models.stt_dirs, vec![dir.join("flag/models/stt")]);

        // Relative file key is ignored → falls back to the located default
        // (temp dir = absolute XDG-ish default for this test).
        std::fs::write(&path, "[core]\ndata_dir = \"relative/dir\"\n").expect("write config");
        let loaded = load(Some(path), None).expect("load config");
        assert_ne!(loaded.data_dir, Some(PathBuf::from("relative/dir")));

        // No explicit data dir: the located default (XDG/HOME) still holds
        // even with an explicit config path; the file key cannot make it None.
        let path = base.join("plain.toml");
        std::fs::write(&path, "[models]\n").expect("write config");
        let loaded = load(Some(path), None).expect("load config");
        assert_eq!(
            loaded.data_dir,
            resolve_data_dir(
                None,
                std::env::var_os(DATA_DIR_ENV),
                std::env::var_os("XDG_STATE_HOME"),
                std::env::var_os("HOME")
            )
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// VRAM budget: defaults (unset) → `[models] vram_budget_bytes` →
    /// `HONKBOX_VRAM_BUDGET` (bytes); empty/invalid env keeps the file
    /// value (pure helper — same contract as the GC/swap knobs).
    #[test]
    fn vram_budget_precedence() {
        // Defaults: unset.
        assert_eq!(vram_budget_settings(None, None), None);
        // File key alone.
        assert_eq!(
            vram_budget_settings(Some(8_000_000_000), None),
            Some(8_000_000_000)
        );
        // Env wins over the file.
        assert_eq!(
            vram_budget_settings(Some(8_000_000_000), Some("4294967296")),
            Some(4_294_967_296)
        );
        // Env alone (no file key).
        assert_eq!(vram_budget_settings(None, Some("1024")), Some(1024));
        // Empty env keeps the file value; invalid env keeps it too.
        assert_eq!(
            vram_budget_settings(Some(8_000_000_000), Some("")),
            Some(8_000_000_000)
        );
        assert_eq!(
            vram_budget_settings(Some(8_000_000_000), Some("not-a-number")),
            Some(8_000_000_000)
        );
        // Empty/invalid env with no file key stays unset.
        assert_eq!(vram_budget_settings(None, Some("")), None);
        assert_eq!(vram_budget_settings(None, Some("nope")), None);
        // `load` actually wires the file key through (no env in this
        // process's test env unless a sibling test leaked it).
        let base = std::env::temp_dir().join(format!("honkbox-vram-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let path = base.join("honkbox.toml");
        std::fs::write(&path, "[models]\nvram_budget_bytes = 8_000_000_000\n").expect("write");
        let loaded = load(Some(path), Some(base.clone())).expect("load config");
        if std::env::var("HONKBOX_VRAM_BUDGET").is_err() {
            assert_eq!(loaded.models.vram_budget_bytes, Some(8_000_000_000));
        }
        std::fs::remove_dir_all(&base).ok();
    }

    /// Managed download roots (NEW-4): `<data_dir>/models/{llm,sd,stt,tts}`
    /// are appended to whatever file/env configured (exact-path dedupe keeps
    /// the starter audio dirs from doubling up); no data dir → no managed
    /// roots.
    #[test]
    fn managed_download_roots_appended() {
        let base = std::env::temp_dir().join(format!("honkbox-managed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let path = base.join("honkbox.toml");
        std::fs::write(
            &path,
            "[models]\nllm_dirs = [\"/custom/llm\"]\n\n[discovery]\ncomfyui_dirs = [\"/extra/comfy\"]\n",
        )
        .expect("write config");
        let cfg = load(Some(path), Some(base.clone()))
            .expect("load config")
            .models;
        // Custom roots survive; the managed roots ride along after them.
        assert!(cfg.llm_dirs.contains(&PathBuf::from("/custom/llm")));
        assert!(cfg.llm_dirs.contains(&base.join("models/llm")));
        assert!(cfg.comfyui_dirs.contains(&PathBuf::from("/extra/comfy")));
        assert!(cfg.comfyui_dirs.contains(&base.join("models/sd")));
        // The starter audio roots ARE the managed roots (deduped).
        assert_eq!(cfg.stt_dirs, vec![base.join("models/stt")]);
        assert_eq!(cfg.tts_dirs, vec![base.join("models/tts")]);

        // No explicit data dir: the located XDG default is still the data
        // dir, so exactly its managed root is the one appended.
        let plain = base.join("plain.toml");
        std::fs::write(&plain, "[models]\n").expect("write config");
        let cfg = load(Some(plain), None).expect("load config").models;
        let resolved = resolve_data_dir(
            None,
            std::env::var_os(DATA_DIR_ENV),
            std::env::var_os("XDG_STATE_HOME"),
            std::env::var_os("HOME"),
        );
        let managed = cfg
            .llm_dirs
            .iter()
            .filter(|d| d.ends_with("models/llm"))
            .count();
        if let Some(base) = resolved {
            assert_eq!(managed, 1);
            assert!(cfg.llm_dirs.contains(&base.join("models/llm")));
        } else {
            assert_eq!(managed, 0);
        }
        std::fs::remove_dir_all(&base).ok();
    }

    /// GC settings: defaults, then `[jobs]` keys, then env overrides win.
    #[test]
    fn gc_settings_precedence() {
        // Defaults.
        let d = gc_settings(&FileJobs::default(), None, None);
        assert_eq!(d.gc_ttl, DEFAULT_GC_TTL);
        assert_eq!(d.gc_max_bytes, DEFAULT_GC_MAX_BYTES);
        // File keys.
        let f = gc_settings(
            &FileJobs {
                gc_ttl: Some(3600),
                gc_max_bytes: Some(1000),
            },
            None,
            None,
        );
        assert_eq!(f.gc_ttl, Duration::from_secs(3600));
        assert_eq!(f.gc_max_bytes, 1000);
        // Env wins over file; empty env keeps the file value.
        let e = gc_settings(&e_file(), Some("60"), Some("7"));
        assert_eq!(e.gc_ttl, Duration::from_secs(60));
        assert_eq!(e.gc_max_bytes, 7);
        let kept = gc_settings(&e_file(), Some(""), Some(""));
        assert_eq!(kept.gc_ttl, Duration::from_secs(3600));
        // Invalid env keeps the file value.
        let bad = gc_settings(&e_file(), Some("abc"), Some("-3"));
        assert_eq!(bad.gc_ttl, Duration::from_secs(3600));
    }

    /// Swap settings: defaults, then `[swap]` keys, then env overrides win;
    /// `llm_max_loaded` clamps to >= 1, `max_ttl_secs` to >= 1s, and invalid
    /// env keeps the file value.
    #[test]
    fn swap_settings_precedence() {
        // Defaults: enabled, 10-minute idle TTL, single-resident capacity,
        // 24 h per-request TTL cap, nothing pinned.
        let d = swap_settings(&FileSwap::default(), None, None, None, None);
        assert!(d.enabled);
        assert_eq!(d.llm_ttl, Duration::from_secs(DEFAULT_SWAP_TTL_SECS));
        assert_eq!(d.llm_max_loaded, DEFAULT_SWAP_MAX_LOADED);
        assert_eq!(d.max_ttl, Duration::from_secs(DEFAULT_SWAP_MAX_TTL_SECS));
        assert_eq!(d.max_ttl, Duration::from_hours(24));
        assert_eq!(d.pin, Vec::<String>::new());
        // File keys.
        let f = swap_settings(
            &FileSwap {
                enabled: Some(false),
                llm_ttl_secs: Some(30),
                llm_max_loaded: Some(3),
                max_ttl_secs: Some(0),
                pin: Some(vec![
                    " llm:a.gguf ".to_owned(),
                    String::new(),
                    "llm:b".to_owned(),
                ]),
            },
            None,
            None,
            None,
            None,
        );
        assert!(!f.enabled);
        assert_eq!(f.llm_ttl, Duration::from_secs(30));
        assert_eq!(f.llm_max_loaded, 3);
        assert_eq!(f.max_ttl, Duration::from_secs(1), "a 0 cap clamps to 1s");
        assert_eq!(f.pin, vec!["llm:a.gguf", "llm:b"]);
        // Env wins over file; empty env keeps the file value.
        let e = swap_settings(
            &f_file(),
            Some("5"),
            Some("4"),
            Some("120"),
            Some("llm:c,llm:d"),
        );
        assert!(e.enabled);
        assert_eq!(e.llm_ttl, Duration::from_secs(5));
        assert_eq!(e.llm_max_loaded, 4);
        assert_eq!(e.max_ttl, Duration::from_secs(120));
        assert_eq!(e.pin, vec!["llm:c", "llm:d"]);
        let kept = swap_settings(&f_file(), Some(""), Some(""), Some(""), Some(""));
        assert_eq!(kept.llm_ttl, Duration::from_secs(30));
        assert_eq!(kept.llm_max_loaded, 3);
        assert_eq!(kept.max_ttl, Duration::from_secs(600));
        assert_eq!(kept.pin, vec!["llm:a"]);
        // Invalid env keeps the file value; a `0` capacity clamps to 1.
        let bad = swap_settings(&f_file(), Some("abc"), Some("0"), Some("abc"), None);
        assert_eq!(bad.llm_ttl, Duration::from_secs(30));
        assert_eq!(bad.llm_max_loaded, 1);
        assert_eq!(bad.max_ttl, Duration::from_secs(600));
    }

    fn f_file() -> FileSwap {
        FileSwap {
            enabled: None,
            llm_ttl_secs: Some(30),
            llm_max_loaded: Some(3),
            max_ttl_secs: Some(600),
            pin: Some(vec!["llm:a".to_owned()]),
        }
    }

    fn e_file() -> FileJobs {
        FileJobs {
            gc_ttl: Some(3600),
            gc_max_bytes: None,
        }
    }

    /// Write a config TOML and load it (isolated temp file per call).
    fn load_toml(tag: &str, text: &str) -> Result<LoadedConfig, String> {
        let base = std::env::temp_dir().join(format!("honkbox-entry-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let path = base.join("honkbox.toml");
        std::fs::write(&path, text).expect("write config");
        let loaded = load(Some(path), None);
        std::fs::remove_dir_all(&base).ok();
        loaded
    }

    /// `[[models.entry]]`: slug- and match-targeted entries parse with
    /// cleaned aliases and verbatim capability overrides.
    #[test]
    fn models_entry_parses_both_targets() {
        let loaded = load_toml(
            "ok",
            "[models]\n\
             llm_dirs = [\"/tmp/llm\"]\n\
             [[models.entry]]\n\
             slug = \"llm:models--org--Repo/snapshots/sha/Model-Q8_0.gguf\"\n\
             aliases = [\"qwen3\", \" embed-model \"]\n\
             [models.entry.capabilities]\n\
             chat = false\n\
             embeddings = true\n\
             [[models.entry]]\n\
             match = \"Whisper-Bin.bin\"\n\
             aliases = [\"whisper\"]\n\
             [models.entry.capabilities]\n\
             tool_calling = true\n",
        )
        .expect("load config");
        let entries = &loaded.models.entries;
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].slug.as_deref(),
            Some("llm:models--org--Repo/snapshots/sha/Model-Q8_0.gguf")
        );
        assert_eq!(entries[0].file_match, None);
        assert_eq!(entries[0].aliases, vec!["qwen3", "embed-model"]);
        assert_eq!(entries[0].capabilities.chat, Some(false));
        assert_eq!(entries[0].capabilities.embeddings, Some(true));
        assert_eq!(entries[0].capabilities.tool_calling, None);
        assert!(entries[0].capabilities.reclassifies_embedding());
        assert_eq!(entries[1].slug, None);
        assert_eq!(entries[1].file_match.as_deref(), Some("Whisper-Bin.bin"));
        assert_eq!(entries[1].aliases, vec!["whisper"]);
        assert_eq!(entries[1].capabilities.tool_calling, Some(true));
        // Cache JSON round-trips through the registry-cache column shape.
        let json = entries[0].capabilities.cache_json();
        assert_eq!(
            ModelCapabilitiesOverride::from_cache_json(&json),
            entries[0].capabilities
        );
    }

    /// `[[models.entry]] cache`: the per-slug KV cache knobs parse into the
    /// slug-keyed map, an entry without a cache block contributes nothing,
    /// and a zero context size (a zero-sized cache) is a config error.
    #[test]
    fn models_entry_cache_knobs_parse() {
        let loaded = load_toml(
            "cache",
            "[models]\n\
             [[models.entry]]\n\
             slug = \"llm:a.gguf\"\n\
             [models.entry.cache]\n\
             n_ctx = 4096\n\
             kv_quant = \"q8_0\"\n\
             offload = false\n",
        )
        .expect("load config");
        let knobs = loaded
            .models
            .cache
            .get("llm:a.gguf")
            .expect("cache knobs for the slug");
        assert_eq!(knobs.n_ctx, Some(4096));
        assert_eq!(knobs.kv_quant, Some(KvQuantConfig::Q8_0));
        assert_eq!(knobs.offload, Some(false));

        let loaded = load_toml(
            "cache-absent",
            "[models]\n\
             [[models.entry]]\n\
             slug = \"llm:a.gguf\"\n",
        )
        .expect("load config");
        assert!(loaded.models.cache.is_empty());

        let err = load_toml(
            "cache-zero",
            "[models]\n\
             [[models.entry]]\n\
             slug = \"llm:a.gguf\"\n\
             [models.entry.cache]\n\
             n_ctx = 0\n",
        )
        .expect_err("zero n_ctx");
        assert!(err.contains("n_ctx"), "{err}");

        // A cache block on a match-targeted entry has no slug to key on.
        let err = load_toml(
            "cache-match",
            "[models]\n\
             [[models.entry]]\n\
             match = \"a.gguf\"\n\
             [models.entry.cache]\n\
             n_ctx = 4096\n",
        )
        .expect_err("cache without a slug");
        assert!(err.contains("slug"), "{err}");
    }

    /// Malformed `[[models.entry]]` declarations are config errors (the
    /// whole load fails loudly — a half-applied binding is worse than a
    /// failed boot).
    #[test]
    fn models_entry_declarations_are_validated() {
        let both = load_toml(
            "both",
            "[[models.entry]]\nslug = \"llm:a.gguf\"\nmatch = \"a.gguf\"\n",
        )
        .expect_err("both targets");
        assert!(both.contains("both slug and match"), "{both}");

        let neither =
            load_toml("neither", "[[models.entry]]\naliases = [\"x\"]\n").expect_err("no target");
        assert!(neither.contains("neither slug nor match"), "{neither}");

        let empty =
            load_toml("empty", "[[models.entry]]\nslug = \"   \"\n").expect_err("empty slug");
        assert!(empty.contains("slug is empty"), "{empty}");

        let dup = load_toml(
            "dup",
            "[[models.entry]]\nmatch = \"a.gguf\"\naliases = [\"x\", \" x \"]\n",
        )
        .expect_err("duplicate alias");
        assert!(dup.contains("twice"), "{dup}");
    }

    /// REQ-7: `[server] auth` + `[auth] api_key_env` parse from the file;
    /// missing keys keep the defaults; invalid values are config errors.
    #[test]
    fn auth_settings_parse_and_validate() {
        // Defaults: no auth keys at all.
        let loaded = load_toml("auth-default", "[core]\n").expect("load");
        assert_eq!(loaded.auth.mode, AuthMode::None);
        assert_eq!(loaded.auth.api_key_env, "HONKBOX_API_KEY");

        // Both keys set (bearer + a custom env var NAME).
        let loaded = load_toml(
            "auth-bearer",
            concat!(
                "[server]\n",
                "auth = \"bearer\"\n",
                "\n",
                "[auth]\n",
                "api_key_env = \"MY_KEY_VAR\"\n"
            ),
        )
        .expect("load");
        assert_eq!(loaded.auth.mode, AuthMode::Bearer);
        assert_eq!(loaded.auth.api_key_env, "MY_KEY_VAR");

        // Explicit `none` is accepted (case/whitespace-insensitive).
        let loaded = load_toml("auth-none", "[server]\nauth = \" None \"\n").expect("load");
        assert_eq!(loaded.auth.mode, AuthMode::None);

        // An unknown mode is a config error naming the file.
        let err = load_toml("auth-bad", "[server]\nauth = \"magic\"\n").expect_err("bad auth mode");
        assert!(
            err.contains("honkbox.toml") && err.contains("magic"),
            "{err}"
        );

        // An empty env-var NAME is a config error.
        let err = load_toml("auth-empty-env", "[auth]\napi_key_env = \"   \"\n")
            .expect_err("empty env name");
        assert!(err.contains("api_key_env"), "{err}");
    }

    /// REQ-7: the fail-closed bind check — `none` + non-loopback refuses,
    /// the ack flag escapes, loopback and bearer pass.
    #[test]
    fn check_bind_auth_is_fail_closed() {
        // auth=none + loopback hosts: fine (127.x, ::1, localhost).
        for host in ["127.0.0.1", "localhost", "::1", "127.200.3.4"] {
            check_bind_auth(AuthMode::None, host, false).expect("loopback none");
        }
        // auth=none + non-loopback: refused unless acked.
        let err = check_bind_auth(AuthMode::None, "0.0.0.0", false).expect_err("no ack");
        assert!(err.contains("refusing to bind"), "{err}");
        check_bind_auth(AuthMode::None, "0.0.0.0", true).expect("acked");
        check_bind_auth(AuthMode::None, "192.168.1.10", false).expect_err("lan no ack");
        // bearer + non-loopback: allowed (the key is the protection).
        check_bind_auth(AuthMode::Bearer, "0.0.0.0", false).expect("bearer any bind");
    }

    /// REQ-6 Option A: `[server] queue_wait_secs` parses from the file with
    /// env override; invalid env warns-and-keeps; `0` disables queueing.
    #[test]
    fn server_queue_wait_resolves() {
        // Default (no keys).
        let loaded = load_toml("queue-default", "[core]\n").expect("load");
        assert_eq!(
            loaded.server.queue_wait,
            Duration::from_secs(DEFAULT_QUEUE_WAIT_SECS)
        );

        // File key wins over the default.
        let loaded = load_toml("queue-file", "[server]\nqueue_wait_secs = 3\n").expect("load");
        assert_eq!(loaded.server.queue_wait, Duration::from_secs(3));

        // `0` is meaningful: queueing disabled.
        let loaded = load_toml("queue-zero", "[server]\nqueue_wait_secs = 0\n").expect("load");
        assert_eq!(loaded.server.queue_wait, Duration::ZERO);

        // Env wins over the file; invalid env keeps the file value.
        let loaded = server_settings(Some(3), Some("7"));
        assert_eq!(loaded.queue_wait, Duration::from_secs(7));
        let kept = server_settings(Some(3), Some("not-a-number"));
        assert_eq!(kept.queue_wait, Duration::from_secs(3));
        let empty = server_settings(Some(3), Some(""));
        assert_eq!(empty.queue_wait, Duration::from_secs(3));
    }

    /// A `u64::MAX` bound must clamp before it reaches an `Instant` in the
    /// admission waiter (both the file and env stages).
    #[test]
    fn server_queue_wait_clamps_to_an_instant_safe_bound() {
        let from_file = server_settings(Some(u64::MAX), None);
        assert_eq!(from_file.queue_wait, MAX_QUEUE_WAIT);
        let from_env = server_settings(None, Some("18446744073709551615"));
        assert_eq!(from_env.queue_wait, MAX_QUEUE_WAIT);
        assert!(
            std::time::Instant::now()
                .checked_add(from_file.queue_wait)
                .is_some()
        );
    }

    /// `[ledger]` defaults keep the single-tenant dev flow: one GPU slot,
    /// unmanaged budgets, depth-1 queues, and the queue TTL wired to the
    /// `[server] queue_wait_secs` knob.
    #[test]
    fn ledger_defaults_keep_the_dev_flow() {
        let settings = ledger_settings(&FileLedger::default(), None, None, None, None, None, None)
            .expect("defaults");
        assert_eq!(settings.gpu_slots, 1);
        assert_eq!(settings.vram_budget_bytes, 0);
        assert_eq!(settings.ram_budget_bytes, 0);
        assert_eq!(settings.cpu_threads, 0);
        assert_eq!(settings.queue_cap, DEFAULT_LEDGER_QUEUE_CAP);
        assert_eq!(settings.queue_caps, Vec::<(ResourceKind, usize)>::new());
        assert_eq!(settings.vram_demand, Vec::<(ResourceKind, u64)>::new());

        let config = residency_config(&settings, Duration::from_secs(10));
        assert_eq!(config.capacities.gpu_slots, 1);
        assert_eq!(config.capacities.vram_bytes, 0);
        assert_eq!(config.kind(ResourceKind::Sd).queue_cap, 1);
        assert_eq!(config.kind(ResourceKind::Sd).ttl, Duration::from_secs(10));
        assert_eq!(config.kind(ResourceKind::Sd).demand.vram_bytes, 0);
        assert_eq!(config.kind(ResourceKind::Rag).demand.gpu_slots, 0);
    }

    /// `[ledger]` scalar knobs: env wins over the file, invalid env keeps
    /// the previous stage, `queue_cap = 0` clamps to `1`.
    #[test]
    fn ledger_scalars_resolve_file_then_env() {
        let file = FileLedger {
            gpu_slots: Some(2),
            vram_budget_bytes: Some(100),
            queue_cap: Some(0),
            ..Default::default()
        };
        let settings = ledger_settings(&file, Some("4"), Some("200"), None, None, None, Some("5"))
            .expect("env overrides");
        assert_eq!(settings.gpu_slots, 4);
        assert_eq!(settings.vram_budget_bytes, 200);
        assert_eq!(settings.queue_cap, 5);
        // Untouched scalars keep the file value.
        assert_eq!(settings.ram_budget_bytes, 0);

        // Invalid env keeps the file stage.
        let kept =
            ledger_settings(&file, Some("bogus"), None, None, None, None, None).expect("kept");
        assert_eq!(kept.gpu_slots, 2);
        // A zero queue cap clamps to 1 (a zero queue would reject every
        // busy request before its TTL ever ran).
        assert_eq!(kept.queue_cap, 1);
        // Empty env counts as unset.
        let empty = ledger_settings(&file, Some(""), None, None, None, None, None).expect("empty");
        assert_eq!(empty.gpu_slots, 2);
        // A zero slot count clamps to one from either source, like queue_cap.
        let zero_file = ledger_settings(
            &FileLedger {
                gpu_slots: Some(0),
                ..Default::default()
            },
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("zero file");
        assert_eq!(zero_file.gpu_slots, 1);
        let zero_env =
            ledger_settings(&file, Some("0"), None, None, None, None, None).expect("zero env");
        assert_eq!(zero_env.gpu_slots, 1);
    }

    /// `[ledger.queue_caps]` / `[ledger.vram_demand]` override per kind;
    /// unknown kind labels are config errors (a silently ignored typo
    /// would admit far more than the operator wrote).
    #[test]
    fn ledger_kind_maps_validate_and_override() {
        let file = FileLedger {
            queue_caps: Some(BTreeMap::from([("sd".to_owned(), 2), ("3d".to_owned(), 4)])),
            vram_demand: Some(BTreeMap::from([("llm".to_owned(), 4_000_000_000)])),
            ..Default::default()
        };
        let settings = ledger_settings(&file, None, None, None, None, None, None).expect("maps");
        assert!(settings.queue_caps.contains(&(ResourceKind::Sd, 2)));
        assert!(settings.queue_caps.contains(&(ResourceKind::ThreeD, 4)));
        let config = residency_config(&settings, Duration::from_secs(1));
        assert_eq!(config.kind(ResourceKind::Sd).queue_cap, 2);
        assert_eq!(config.kind(ResourceKind::ThreeD).queue_cap, 4);
        assert_eq!(config.kind(ResourceKind::Ocr).queue_cap, 1);
        assert_eq!(
            config.kind(ResourceKind::Llm).demand.vram_bytes,
            4_000_000_000
        );

        let bad = FileLedger {
            queue_caps: Some(BTreeMap::from([("audio".to_owned(), 2)])),
            ..Default::default()
        };
        let err = ledger_settings(&bad, None, None, None, None, None, None).unwrap_err();
        assert!(err.contains("unknown kind"), "{err}");
        assert!(err.contains("audio"), "{err}");
    }

    /// The `[ledger]` file table parses end-to-end through `load` (and a
    /// missing table keeps the defaults).
    #[test]
    fn ledger_file_table_parses() {
        let loaded = load_toml("ledger-absent", "[core]\n").expect("load");
        assert_eq!(loaded.ledger, LedgerSettings::default());

        let loaded = load_toml(
            "ledger-file",
            "[ledger]\n\
             gpu_slots = 2\n\
             vram_budget_bytes = 8_000_000_000\n\
             queue_cap = 3\n\
             [ledger.queue_caps]\n\
             sd = 2\n",
        )
        .expect("load");
        assert_eq!(loaded.ledger.gpu_slots, 2);
        assert_eq!(loaded.ledger.vram_budget_bytes, 8_000_000_000);
        assert_eq!(loaded.ledger.queue_cap, 3);
        assert_eq!(loaded.ledger.queue_caps, vec![(ResourceKind::Sd, 2)]);
    }

    /// `--host` parsing: `localhost` maps to the IPv4 loopback; real IPs
    /// (v4 and v6) pass through; garbage is an error.
    #[test]
    fn parse_bind_host_accepts_ips_and_localhost() {
        assert_eq!(
            parse_bind_host("localhost").expect("localhost"),
            std::net::IpAddr::from([127, 0, 0, 1])
        );
        assert_eq!(
            parse_bind_host("::1").expect("v6 loopback"),
            std::net::IpAddr::from([0, 0, 0, 0, 0, 0, 0, 1])
        );
        assert!(parse_bind_host("not-an-ip").is_err());
    }

    #[test]
    fn nsfw_file_stage_defaults_and_validation() {
        let where_ = "test.toml";
        let default = resolve_nsfw(&FileNsfw::default(), where_).expect("defaults");
        assert_eq!(default, NsfwConfig::default());
        assert_eq!(default.classifier_bin, None);

        let good = FileNsfw {
            enabled: Some(true),
            policy: Some(" block ".to_owned()),
            model_version: Some(" v2 ".to_owned()),
            threshold: Some(0.25),
            classifier_bin: Some(" ~/models/nsfw.bin ".to_owned()),
        };
        let got = resolve_nsfw(&good, where_).expect("valid nsfw");
        assert!(got.enabled);
        assert_eq!(got.policy, "block");
        assert_eq!(got.model_version, "v2");
        assert_eq!(got.threshold, 0.25);
        assert_eq!(got.classifier_bin, Some(expand_home("~/models/nsfw.bin")));

        let bad_policy = FileNsfw {
            policy: Some("quarantine".to_owned()),
            ..FileNsfw::default()
        };
        assert!(
            resolve_nsfw(&bad_policy, where_)
                .unwrap_err()
                .contains("[safety.nsfw] policy")
        );
        let bad_path = FileNsfw {
            classifier_bin: Some("  ".to_owned()),
            ..FileNsfw::default()
        };
        assert!(
            resolve_nsfw(&bad_path, where_)
                .unwrap_err()
                .contains("[safety.nsfw] classifier_bin")
        );
        for threshold in [-0.1, 1.1, f64::NAN] {
            let bad = FileNsfw {
                threshold: Some(threshold),
                ..FileNsfw::default()
            };
            assert!(
                resolve_nsfw(&bad, where_)
                    .unwrap_err()
                    .contains("[safety.nsfw] threshold")
            );
        }
    }

    #[test]
    fn scope_names_are_validated_and_deduped() {
        let where_ = "test.toml";
        let raw = vec![
            FileScope {
                api_key_env: " KEY_A ".to_owned(),
                scopes: vec![" read ".to_owned(), String::new(), "write".to_owned()],
            },
            FileScope {
                api_key_env: "KEY_A".to_owned(),
                scopes: vec!["admin".to_owned()],
            },
            FileScope {
                api_key_env: "KEY_B".to_owned(),
                scopes: vec![],
            },
        ];
        let got = parse_scopes(&raw, where_).expect("valid scopes");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].api_key_env, "KEY_A");
        assert_eq!(got[0].scopes, vec!["read", "write"]);
        assert_eq!(got[1].api_key_env, "KEY_B");

        let bad = vec![FileScope {
            api_key_env: "  ".to_owned(),
            scopes: vec![],
        }];
        assert!(parse_scopes(&bad, where_).unwrap_err().contains("#0"));
    }

    #[test]
    fn python_venv_default_follows_data_dir() {
        assert_eq!(python_venv_default(None), PathBuf::from("sidecar/venv"));
        assert_eq!(
            python_venv_default(Some(Path::new("/tmp/hb"))),
            PathBuf::from("/tmp/hb/sidecar/venv")
        );
    }

    #[test]
    fn env_names_are_trimmed_and_empty_rejected() {
        assert_eq!(
            parse_env_name("  HONKBOX_X  ", "[auth] signed_url_secret_env", "t.toml")
                .expect("name"),
            "HONKBOX_X"
        );
        assert!(parse_env_name("  ", "[auth] signed_url_secret_env", "t.toml").is_err());
    }
}
