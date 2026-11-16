//! Model roots config: `[models]` + optional `[discovery]` in
//! `<data_dir>/honkbox.toml`, env wins over file. Also `[jobs]` GC settings
//! (`gc_ttl` / `gc_max_bytes`, seconds / bytes, env `HONKBOX_GC_TTL` /
//! `HONKBOX_GC_MAX_BYTES`), `[swap]` LLM swap settings (`enabled`,
//! `llm_ttl_secs`, `llm_max_loaded`, `max_ttl_secs`, `pin`, env
//! `HONKBOX_SWAP_LLM_TTL` / `HONKBOX_SWAP_LLM_MAX_LOADED` /
//! `HONKBOX_SWAP_MAX_TTL` / `HONKBOX_SWAP_PIN`), and `[core] data_dir`.
//!
//! The ONE data-dir knob resolves `--data-dir` CLI flag > `HONKBOX_DATA_DIR`
//! env > `[core] data_dir` file key > `$XDG_STATE_HOME/honkbox` (empty or
//! relative env/XDG values are treated as unset; fallback
//! `$HOME/.local/state/honkbox`). Everything derives from it:
//! `<data_dir>/honkbox.toml` (config), `jobs/`, `logs/honkbox.log`,
//! `registry.sqlite3`, `models/{stt,tts}`. The file-*locating* default is
//! pure-env (no file read, no cwd), so the config file lives inside the data
//! dir without circularity; a file `[core] data_dir` refines the derived
//! roots but cannot move the file itself (one-pass).
//!
//! Precedence for model roots: built-in defaults → config file → env
//! (`HONKBOX_LLM_DIRS`, `HONKBOX_COMFYUI_DIR`, `HONKBOX_STT_DIRS`,
//! `HONKBOX_TTS_DIRS`, `HONKBOX_DISCOVERY_LLM_DIRS`,
//! `HONKBOX_DISCOVERY_COMFYUI_DIRS`, `HONKBOX_DISCOVERY_STT_DIRS`,
//! `HONKBOX_DISCOVERY_TTS_DIRS`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::warn;

use crate::jobs::{DEFAULT_GC_MAX_BYTES, DEFAULT_GC_TTL};

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
    /// `[jobs]` GC settings (startup garbage collection of job bundles).
    pub jobs: JobsConfig,
    /// `[swap]` LLM load-on-demand + idle-eviction settings.
    pub swap: SwapConfig,
}

/// Default `[swap]` idle TTL: 10 minutes (llama-swap defaults to never
/// unload; a bounded default is the safer choice for a single-GPU studio).
pub const DEFAULT_SWAP_TTL_SECS: u64 = 600;
/// Default `[swap]` warm capacity: one resident LLM (matches the
/// single-GPU residency slot; loading another slug swaps, not coexists).
pub const DEFAULT_SWAP_MAX_LOADED: usize = 1;
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

/// Load result: model/job/swap config plus the final data dir (after the
/// `[core] data_dir` refinement).
pub struct LoadedConfig {
    /// Model roots + `[jobs]` + `[swap]` settings.
    pub models: ModelConfig,
    /// Final data dir; `None` only when no absolute base can be derived.
    pub data_dir: Option<PathBuf>,
}

/// Final data dir: flag > env > `[core] data_dir` file key > located default
/// (pure-env chain). The file key is skipped whenever the flag or env already
/// decided (they outrank it); an empty or relative file key is ignored with a
/// `warn!` (absolute paths only). Pure (args, not process env).
fn finalize_data_dir(
    cli_dir: Option<PathBuf>,
    env_dir: Option<OsString>,
    file: &FileCore,
    located: Option<PathBuf>,
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
                    warn!("ignoring relative [core] data_dir {raw:?}; use an absolute path");
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
    let mut file_jobs = FileJobs::default();
    let mut file_swap = FileSwap::default();
    let mut file_core = FileCore::default();
    let mut stt_from_file = false;
    let mut tts_from_file = false;
    match path.as_deref().map(std::fs::read_to_string).transpose() {
        Ok(Some(text)) => {
            let display = path.as_deref().unwrap_or(Path::new("<none>"));
            let file: FileConfig = toml::from_str(&text)
                .map_err(|e| format!("bad config {}: {e}", display.display()))?;
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
                stt_from_file = true;
            }
            if let Some(dirs) = file.models.tts_dirs {
                staged.tts = dirs.iter().map(|d| expand_home(d)).collect();
                tts_from_file = true;
            }
            if let Some(dirs) = file.discovery.stt_dirs {
                staged.extra_stt = dirs.iter().map(|d| expand_home(d)).collect();
            }
            if let Some(dirs) = file.discovery.tts_dirs {
                staged.extra_tts = dirs.iter().map(|d| expand_home(d)).collect();
            }
            file_jobs = file.jobs;
            file_swap = file.swap;
            file_core = file.core;
        }
        Ok(None) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            let display = path.as_deref().unwrap_or(Path::new("<none>"));
            return Err(format!("cannot read config {}: {e}", display.display()));
        }
    }
    let data_dir = finalize_data_dir(cli_dir, env_dir, &file_core, located);
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
    let models = apply_env(
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
    )
    .merged(jobs, swap);
    Ok(LoadedConfig { models, data_dir })
}

/// Staged roots before the final merge: the `[models]` base plus the
/// `[discovery]` extras, each stage replaceable by its env var.
#[derive(Debug, Clone, Default)]
struct RootStages {
    llm: Vec<PathBuf>,
    comfyui: Vec<PathBuf>,
    stt: Vec<PathBuf>,
    tts: Vec<PathBuf>,
    extra_llm: Vec<PathBuf>,
    extra_comfyui: Vec<PathBuf>,
    extra_stt: Vec<PathBuf>,
    extra_tts: Vec<PathBuf>,
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
            extra_llm: Vec::new(),
            extra_comfyui: Vec::new(),
            extra_stt: Vec::new(),
            extra_tts: Vec::new(),
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
        ModelConfig {
            llm_dirs,
            comfyui_dirs,
            stt_dirs,
            tts_dirs,
            jobs,
            swap,
        }
    }
}

/// Env wins over file, per stage: `HONKBOX_LLM_DIRS` (path-separated via
/// `split_paths`) replaces the `[models]` `llm_dirs`, `HONKBOX_COMFYUI_DIR`
/// replaces the `[models]` `comfyui_dir`, `HONKBOX_STT_DIRS` /
/// `HONKBOX_TTS_DIRS` replace the `[models]` `stt_dirs` / `tts_dirs`, and
/// `HONKBOX_DISCOVERY_LLM_DIRS` / `HONKBOX_DISCOVERY_COMFYUI_DIRS` /
/// `HONKBOX_DISCOVERY_STT_DIRS` / `HONKBOX_DISCOVERY_TTS_DIRS` replace the
/// `[discovery]` extras. Pure (args instead of process env) so tests stay
/// thread-safe.
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
/// `[discovery]` + optional `[jobs]` + optional `[swap]`.
#[derive(Debug, Default, serde::Deserialize)]
struct FileConfig {
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
}

/// `[core]` keys: `data_dir` (the one data-dir knob's file stage).
#[derive(Debug, Default, serde::Deserialize)]
struct FileCore {
    data_dir: Option<String>,
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
}

/// Optional `[discovery]` extras APPENDED to the `[models]` roots (P0 of the
/// llama-swap parity plan); mirrors `[models]` with a plural `comfyui_dirs`
/// so extra `ComfyUI` trees can be added without replacing the base one.
/// Audio parity: `stt_dirs` / `tts_dirs` work exactly like the `[models]`
/// keys. The research doc's `roots = [{kind, path}]` table plus `cache`/
/// `rescan_seconds` still land with the P1 adapter work; unknown
/// `[discovery]` keys are ignored.
#[derive(Debug, Default, serde::Deserialize)]
#[allow(clippy::struct_field_names)]
struct FileDiscovery {
    #[serde(default)]
    llm_dirs: Option<Vec<String>>,
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
        )
        .merged(JobsConfig::default(), SwapConfig::default());
        assert_eq!(kept.llm_dirs, over.llm_dirs);
        assert_eq!(kept.comfyui_dirs, over.comfyui_dirs);
        assert_eq!(kept.stt_dirs, over.stt_dirs);
        assert_eq!(kept.tts_dirs, over.tts_dirs);

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
    }

    #[test]
    fn discovery_table_appends_to_models() {
        let base = std::env::temp_dir().join(format!("honkbox-discovery-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let path = base.join("honkbox.toml");
        std::fs::write(
            &path,
            "[models]\ncomfyui_dir = \"/tmp/base-comfy\"\nstt_dirs = [\"/tmp/base-stt\"]\n\n[discovery]\nllm_dirs = [\"/tmp/extra-llm\"]\ncomfyui_dirs = [\"/tmp/extra-comfy\"]\nstt_dirs = [\"/tmp/extra-stt\"]\ntts_dirs = [\"/tmp/extra-tts\"]\n",
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
}
