//! Model roots config: `[models]` + optional `[discovery]` in
//! `~/.honkbox/honkbox.toml`, env wins over file. Also `[jobs]` GC settings
//! (`gc_ttl` / `gc_max_bytes`, seconds / bytes, env `HONKBOX_GC_TTL` /
//! `HONKBOX_GC_MAX_BYTES`).
//!
//! Precedence: built-in defaults → config file → env (`HONKBOX_LLM_DIRS`,
//! `HONKBOX_COMFYUI_DIR`, `HONKBOX_STT_DIRS`, `HONKBOX_TTS_DIRS`,
//! `HONKBOX_DISCOVERY_LLM_DIRS`, `HONKBOX_DISCOVERY_COMFYUI_DIRS`,
//! `HONKBOX_DISCOVERY_STT_DIRS`, `HONKBOX_DISCOVERY_TTS_DIRS`).
//!
//! ONE-RELEASE compat: when the new path / env vars are absent, the old
//! `~/.sd-rip/sd-rip.toml` / `SD_RIP_*` vars are honored once with a `warn!`
//! pointing at the move to `~/.honkbox`.

use std::path::PathBuf;
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
}

/// `[jobs]` section: startup GC of `~/.honkbox/jobs` + stale temp PNGs.
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

/// Load model roots: defaults, then `config_path` (or `~/.honkbox/honkbox.toml`),
/// then env overrides. `[discovery]` extras are appended to the `[models]`
/// base roots. A missing file falls back to defaults; invalid TOML is an
/// `Err`.
pub fn load(config_path: Option<PathBuf>) -> Result<ModelConfig, String> {
    let path = config_path.unwrap_or_else(default_path);
    let mut staged = RootStages::defaults();
    let mut file_jobs = FileJobs::default();
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let file: FileConfig =
                toml::from_str(&text).map_err(|e| format!("bad config {}: {e}", path.display()))?;
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
            }
            if let Some(dirs) = file.models.tts_dirs {
                staged.tts = dirs.iter().map(|d| expand_home(d)).collect();
            }
            if let Some(dirs) = file.discovery.stt_dirs {
                staged.extra_stt = dirs.iter().map(|d| expand_home(d)).collect();
            }
            if let Some(dirs) = file.discovery.tts_dirs {
                staged.extra_tts = dirs.iter().map(|d| expand_home(d)).collect();
            }
            file_jobs = file.jobs;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("cannot read config {}: {e}", path.display())),
    }
    let jobs = gc_settings(
        &file_jobs,
        std::env::var("HONKBOX_GC_TTL").ok().as_deref(),
        std::env::var("HONKBOX_GC_MAX_BYTES").ok().as_deref(),
    );
    Ok(apply_env(
        staged,
        env_or_compat("HONKBOX_LLM_DIRS", "SD_RIP_LLM_DIRS").as_deref(),
        env_or_compat("HONKBOX_COMFYUI_DIR", "SD_RIP_COMFYUI_DIR").as_deref(),
        std::env::var("HONKBOX_DISCOVERY_LLM_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_COMFYUI_DIRS")
            .ok()
            .as_deref(),
        env_or_compat("HONKBOX_STT_DIRS", "SD_RIP_STT_DIRS").as_deref(),
        env_or_compat("HONKBOX_TTS_DIRS", "SD_RIP_TTS_DIRS").as_deref(),
        std::env::var("HONKBOX_DISCOVERY_STT_DIRS").ok().as_deref(),
        std::env::var("HONKBOX_DISCOVERY_TTS_DIRS").ok().as_deref(),
    )
    .merged(jobs))
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
    /// Built-in defaults used when no config file is present.
    fn defaults() -> Self {
        Self {
            llm: vec![expand_home("~/.cache/huggingface")],
            comfyui: vec![expand_home("~/git-ai/ai-models/Data/Models")],
            stt: vec![expand_home("~/.honkbox/models/stt")],
            tts: vec![expand_home("~/.honkbox/models/tts")],
            extra_llm: Vec::new(),
            extra_comfyui: Vec::new(),
            extra_stt: Vec::new(),
            extra_tts: Vec::new(),
        }
    }

    /// Merge: base roots first, `[discovery]` extras appended (the scan
    /// dedupes equal canonical paths across roots).
    fn merged(self, jobs: JobsConfig) -> ModelConfig {
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
        }
    }
}

/// New `HONKBOX_*` env wins; the old `SD_RIP_*` name is honored once with a
/// `warn!` when the new one is absent (one-release compat).
fn env_or_compat(new_key: &str, old_key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(new_key) {
        return Some(v);
    }
    let old = std::env::var(old_key).ok()?;
    warn!(
        "{old_key} is deprecated; use {new_key} (one-release compat: the old name will stop working)"
    );
    Some(old)
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

/// `~/.honkbox/honkbox.toml` (unexpanded when `$HOME` is unknown, which then
/// just falls back to defaults as a missing file).
fn default_path() -> PathBuf {
    match home::home_dir() {
        Some(h) => select_path(
            h.join(".honkbox/honkbox.toml"),
            h.join(".sd-rip/sd-rip.toml"),
        ),
        None => PathBuf::from("~/.honkbox/honkbox.toml"),
    }
}

/// New path wins; the old `~/.sd-rip` path is honored once with a `warn!`
/// when the new one is absent (one-release compat).
fn select_path(new_path: PathBuf, old_path: PathBuf) -> PathBuf {
    if new_path.exists() || !old_path.exists() {
        return new_path;
    }
    warn!(
        "{} is deprecated; move it to {} (one-release compat: the old path will stop working)",
        old_path.display(),
        new_path.display()
    );
    old_path
}

/// Whole-file shape: `[models]` (base roots) + optional `[discovery]` +
/// optional `[jobs]`.
#[derive(Debug, Default, serde::Deserialize)]
struct FileConfig {
    #[serde(default)]
    models: FileModels,
    #[serde(default)]
    discovery: FileDiscovery,
    #[serde(default)]
    jobs: FileJobs,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_env_precedence() {
        let home = home::home_dir().expect("test needs $HOME");
        let cfg = RootStages::defaults().merged(JobsConfig::default());
        assert_eq!(cfg.llm_dirs, vec![home.join(".cache/huggingface")]);
        assert_eq!(
            cfg.comfyui_dirs,
            vec![home.join("git-ai/ai-models/Data/Models")]
        );
        assert_eq!(cfg.stt_dirs, vec![home.join(".honkbox/models/stt")]);
        assert_eq!(cfg.tts_dirs, vec![home.join(".honkbox/models/tts")]);

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
        let over = staged.clone().merged(JobsConfig::default());
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
        .merged(JobsConfig::default());
        assert_eq!(kept.llm_dirs, over.llm_dirs);
        assert_eq!(kept.comfyui_dirs, over.comfyui_dirs);
        assert_eq!(kept.stt_dirs, over.stt_dirs);
        assert_eq!(kept.tts_dirs, over.tts_dirs);

        // A missing file loads (defaults + ambient env), never errors.
        assert!(load(Some(PathBuf::from("/nonexistent-honkbox-config.toml"))).is_ok());
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
        .merged(JobsConfig::default());
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
        let cfg = load(Some(path)).expect("load config");
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

    #[test]
    fn compat_path_prefers_new() {
        let base = std::env::temp_dir().join(format!("honkbox-compat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("test dir");
        let new = base.join("honkbox.toml");
        let old = base.join("sd-rip.toml");
        // Neither exists → new (a missing file falls back to defaults).
        assert_eq!(select_path(new.clone(), old.clone()), new);
        // Only old exists → old (one-release compat).
        std::fs::write(&old, "[models]\n").expect("old config");
        assert_eq!(select_path(new.clone(), old.clone()), old);
        // New exists → new even when old lingers.
        std::fs::write(&new, "[models]\n").expect("new config");
        assert_eq!(select_path(new.clone(), old.clone()), new);
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

    fn e_file() -> FileJobs {
        FileJobs {
            gc_ttl: Some(3600),
            gc_max_bytes: None,
        }
    }
}
