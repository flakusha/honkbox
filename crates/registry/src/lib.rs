//! Model-dir scan: find LLM + `ComfyUI` weight files under configured roots.
//!
//! Sync, std-only. Slug contract: `llm:<posix-relative-path>` for files
//! under an `llm_dir`, `comfy:<posix-relative-path>` for files under the
//! `comfyui_dir` (e.g. `comfy:checkpoints/sdxl.safetensors`). Relative paths
//! are measured from the root; separators are always `/`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Which kind of root a model file was found under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootKind {
    /// A language-model directory (HF cache layout).
    Llm,
    /// The `ComfyUI` base folder (`checkpoints/`, `loras/`, … beneath it).
    Comfy,
}

/// Configured model roots to scan.
#[derive(Debug, Clone)]
pub struct ModelRoots {
    /// LLM roots (e.g. `~/.cache/huggingface/`); every entry is scanned.
    pub llm_dirs: Vec<PathBuf>,
    /// `ComfyUI` base folder (e.g. `~/git-ai/ai-models/Data/Models/`).
    pub comfyui_dir: PathBuf,
}

/// One discovered model file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FoundModel {
    /// Which kind of root this file was found under.
    pub root: RootKind,
    /// `ComfyUI`: first path component (`checkpoints`, `loras`, …) or `"root"`;
    /// LLM: parent dir file name or `"root"` if directly under the root.
    pub category: String,
    /// Absolute path to the file.
    pub path: PathBuf,
    /// `llm:` / `comfy:` + posix-style path relative to the root.
    pub slug: String,
    /// File size in bytes.
    pub bytes: u64,
}

impl serde::Serialize for RootKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match self {
            RootKind::Llm => "llm",
            RootKind::Comfy => "comfy",
        })
    }
}

/// Base extensions accepted everywhere under a `ComfyUI` root (lowercase, no dot).
const COMFY_BASE_EXTS: &[&str] = &["safetensors", "ckpt", "pt", "pth"];

/// Extensions accepted anywhere below an LLM root (lowercase, no dot).
const LLM_EXTS: &[&str] = &["gguf", "safetensors"];

/// Maximum scan depth, relative to the root (root entries are depth 1).
const MAX_DEPTH: usize = 8;

fn ext_of(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
}

fn comfy_allowed(category: &str, ext: &str) -> bool {
    if COMFY_BASE_EXTS.contains(&ext) {
        return true;
    }
    match category {
        // ponytail: match arms instead of a table; table if more per-category exts appear
        "checkpoints" | "diffusion_models" => ext == "gguf",
        "embeddings" => ext == "bin",
        _ => false,
    }
}

fn is_dot(name: &std::ffi::OsStr) -> bool {
    name.as_encoded_bytes().first() == Some(&b'.')
}

fn llm_category(path: &Path, rel: &str) -> String {
    if !rel.contains('/') {
        return "root".to_owned();
    }
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map_or_else(|| "root".to_owned(), ToOwned::to_owned)
}

fn comfy_category(rel: &str) -> String {
    rel.split('/')
        .next()
        .filter(|_| rel.contains('/'))
        .map_or_else(|| "root".to_owned(), ToOwned::to_owned)
}

/// Classify one file; returns (slug, category) if its extension is allowed.
fn classify(kind: RootKind, path: &Path, rel: &str, ext: &str) -> Option<(String, String)> {
    match kind {
        RootKind::Llm => LLM_EXTS
            .contains(&ext)
            .then(|| (format!("llm:{rel}"), llm_category(path, rel))),
        RootKind::Comfy => {
            let cat = comfy_category(rel);
            comfy_allowed(&cat, ext).then(|| (format!("comfy:{rel}"), cat))
        }
    }
}

fn rel_posix(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root).ok().map(|rel| {
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    })
}

/// Scan all configured roots, returning hits sorted by slug for determinism.
///
/// A missing root dir yields an empty result for that root, never an error.
#[must_use]
pub fn scan_roots(roots: &ModelRoots) -> Vec<FoundModel> {
    let mut out = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    for dir in &roots.llm_dirs {
        scan_one(dir, RootKind::Llm, &mut visited, &mut out);
    }
    scan_one(&roots.comfyui_dir, RootKind::Comfy, &mut visited, &mut out);
    out.sort_by(|a, b| a.slug.cmp(&b.slug));
    out
}

/// Handle one directory entry: push subdirs onto the stack, push model files
/// into `out`. Every per-file IO failure skips the entry (dangling symlinks,
/// unreadable files) — the walk never panics below a missing root.
fn visit(
    kind: RootKind,
    root: &Path,
    entry: &std::fs::DirEntry,
    depth: usize,
    stack: &mut Vec<(PathBuf, usize)>,
    out: &mut Vec<FoundModel>,
) {
    if is_dot(&entry.file_name()) {
        return;
    }
    let path = entry.path();
    let Ok(file_type) = entry.file_type() else {
        return;
    };
    // Symlinked dirs count as dirs; dangling links fail `is_dir` below and
    // fall through to the file path, where the extension filter drops them.
    if file_type.is_dir() || (file_type.is_symlink() && path.is_dir()) {
        if depth < MAX_DEPTH {
            stack.push((path, depth + 1));
        }
        return;
    }
    if depth + 1 > MAX_DEPTH {
        return;
    }
    let Some(rel) = rel_posix(root, &path) else {
        return;
    };
    let ext = ext_of(&path).unwrap_or_default();
    let Some((slug, category)) = classify(kind, &path, &rel, &ext) else {
        return;
    };
    // `bytes` is best-effort: a file vanishing mid-scan reports 0, not an error.
    let bytes = path.metadata().map_or(0, |m| m.len());
    out.push(FoundModel {
        root: kind,
        category,
        path,
        slug,
        bytes,
    });
}

fn scan_one(
    root: &Path,
    kind: RootKind,
    visited: &mut HashSet<PathBuf>,
    out: &mut Vec<FoundModel>,
) {
    if !root.is_dir() {
        return;
    }
    // Stack of (dir, depth); root itself is depth 0, its entries depth 1.
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        // Symlink-cycle guard: skip dirs already visited by canonical path.
        let canon = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !visited.insert(canon) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            visit(kind, root, &entry, depth, &mut stack, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comfy_ext_table() {
        assert!(comfy_allowed("checkpoints", "safetensors"));
        assert!(comfy_allowed("loras", "ckpt"));
        assert!(comfy_allowed("checkpoints", "gguf"));
        assert!(comfy_allowed("diffusion_models", "gguf"));
        assert!(!comfy_allowed("loras", "gguf"));
        assert!(comfy_allowed("embeddings", "bin"));
        assert!(!comfy_allowed("checkpoints", "bin"));
        assert!(!comfy_allowed("vae", "gguf"));
        assert!(!comfy_allowed("vae", "json"));
    }

    #[test]
    fn rel_posix_uses_slashes() {
        let rel = rel_posix(Path::new("/r"), Path::new("/r/a/b.safetensors"));
        assert_eq!(rel.as_deref(), Some("a/b.safetensors"));
    }

    #[test]
    fn scan_tmp_tree() {
        let base = std::env::temp_dir().join(format!("sd-rip-scan-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let llm = base.join("llm");
        let comfy = base.join("comfy");
        std::fs::create_dir_all(llm.join("org/model")).unwrap();
        std::fs::create_dir_all(comfy.join("checkpoints")).unwrap();
        std::fs::create_dir_all(comfy.join("loras")).unwrap();
        std::fs::write(llm.join("org/model/q.gguf"), b"x").unwrap();
        std::fs::write(llm.join("top.safetensors"), b"yy").unwrap();
        std::fs::create_dir_all(llm.join("blobs")).unwrap();
        std::fs::write(llm.join("blobs/abc123"), b"content-addressed, no ext").unwrap();
        std::fs::write(llm.join("skip.json"), b"{}").unwrap();
        std::fs::write(comfy.join("checkpoints/sdxl.safetensors"), b"zzz").unwrap();
        std::fs::write(comfy.join("loras/a.gguf"), b"no").unwrap();
        std::fs::write(comfy.join(".hidden.safetensors"), b"no").unwrap();
        let roots = ModelRoots {
            llm_dirs: vec![llm.clone(), base.join("missing-root")],
            comfyui_dir: comfy.clone(),
        };
        let found = scan_roots(&roots);
        // Cleanup before asserting: no debris even when an assert fails.
        std::fs::remove_dir_all(&base).unwrap();
        let slugs: Vec<&str> = found.iter().map(|m| m.slug.as_str()).collect();
        assert_eq!(
            slugs,
            [
                "comfy:checkpoints/sdxl.safetensors",
                "llm:org/model/q.gguf",
                "llm:top.safetensors",
            ]
        );
        assert_eq!(found[0].category, "checkpoints");
        assert_eq!(found[1].category, "model");
        assert_eq!(found[2].category, "root");
        assert_eq!(found[0].bytes, 3);
    }
}
