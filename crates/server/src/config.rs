//! Model roots config: `[models]` in `~/.sd-rip/sd-rip.toml`, env wins over file.
//!
//! Precedence: built-in defaults → config file → env (`SD_RIP_LLM_DIRS`,
//! `SD_RIP_COMFYUI_DIR`).

use std::path::PathBuf;

/// Model roots handed to `sd_rip_registry::scan_roots`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// LLM roots (HuggingFace-cache layout); every entry is scanned.
    pub llm_dirs: Vec<PathBuf>,
    /// `ComfyUI` base folder.
    pub comfyui_dir: PathBuf,
}

/// Load model roots: defaults, then `config_path` (or `~/.sd-rip/sd-rip.toml`),
/// then env overrides. A missing file falls back to defaults; invalid TOML
/// is an `Err`.
pub fn load(config_path: Option<PathBuf>) -> Result<ModelConfig, String> {
    let path = config_path.unwrap_or_else(default_path);
    let mut cfg = ModelConfig::defaults();
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let file: FileConfig =
                toml::from_str(&text).map_err(|e| format!("bad config {}: {e}", path.display()))?;
            if let Some(dirs) = file.models.llm_dirs {
                cfg.llm_dirs = dirs.iter().map(|d| expand_home(d)).collect();
            }
            if let Some(dir) = file.models.comfyui_dir {
                cfg.comfyui_dir = expand_home(&dir);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("cannot read config {}: {e}", path.display())),
    }
    Ok(apply_env(
        cfg,
        std::env::var("SD_RIP_LLM_DIRS").ok().as_deref(),
        std::env::var("SD_RIP_COMFYUI_DIR").ok().as_deref(),
    ))
}

impl ModelConfig {
    /// Built-in defaults used when no config file is present.
    fn defaults() -> Self {
        Self {
            llm_dirs: vec![expand_home("~/.cache/huggingface")],
            comfyui_dir: expand_home("~/git-ai/ai-models/Data/Models"),
        }
    }
}

/// Env wins over file: `SD_RIP_LLM_DIRS` (path-separated via `split_paths`)
/// replaces `llm_dirs`, `SD_RIP_COMFYUI_DIR` replaces `comfyui_dir`.
/// Pure (args instead of process env) so tests stay thread-safe.
fn apply_env(
    mut cfg: ModelConfig,
    llm_dirs: Option<&str>,
    comfyui_dir: Option<&str>,
) -> ModelConfig {
    if let Some(v) = llm_dirs
        && !v.is_empty()
    {
        cfg.llm_dirs = std::env::split_paths(v).collect();
    }
    if let Some(v) = comfyui_dir
        && !v.is_empty()
    {
        cfg.comfyui_dir = PathBuf::from(v);
    }
    cfg
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

/// `~/.sd-rip/sd-rip.toml` (unexpanded when `$HOME` is unknown, which then
/// just falls back to defaults as a missing file).
fn default_path() -> PathBuf {
    match home::home_dir() {
        Some(h) => h.join(".sd-rip/sd-rip.toml"),
        None => PathBuf::from("~/.sd-rip/sd-rip.toml"),
    }
}

/// Whole-file shape: only `[models]` is read.
#[derive(Debug, Default, serde::Deserialize)]
struct FileConfig {
    #[serde(default)]
    models: FileModels,
}

/// `[models]` keys; each missing key keeps its default.
#[derive(Debug, Default, serde::Deserialize)]
struct FileModels {
    #[serde(default)]
    llm_dirs: Option<Vec<String>>,
    #[serde(default)]
    comfyui_dir: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_env_precedence() {
        let home = home::home_dir().expect("test needs $HOME");
        let cfg = ModelConfig::defaults();
        assert_eq!(cfg.llm_dirs, vec![home.join(".cache/huggingface")]);
        assert_eq!(cfg.comfyui_dir, home.join("git-ai/ai-models/Data/Models"));

        let over = apply_env(
            cfg,
            Some("/tmp/sd-rip-a:/tmp/sd-rip-b"),
            Some("/tmp/sd-rip-comfy"),
        );
        assert_eq!(
            over.llm_dirs,
            vec![
                PathBuf::from("/tmp/sd-rip-a"),
                PathBuf::from("/tmp/sd-rip-b")
            ]
        );
        assert_eq!(over.comfyui_dir, PathBuf::from("/tmp/sd-rip-comfy"));

        // Empty values keep the file/default values.
        let kept = apply_env(over.clone(), Some(""), Some(""));
        assert_eq!(kept.llm_dirs, over.llm_dirs);
        assert_eq!(kept.comfyui_dir, over.comfyui_dir);

        // A missing file loads (defaults + ambient env), never errors.
        assert!(load(Some(PathBuf::from("/nonexistent-sd-rip-config.toml"))).is_ok());
    }
}
