//! `generate` bundle writer: `~/.sd-rip/jobs/{ulid}/` + atomic writes.
//!
//! Layout per job dir: `request.json` (full params + slug), `final_00.png` +
//! sidecar `final_00.json`, `metrics.jsonl` (one line), `run.log`. Every file
//! lands via `<name>.part` + best-effort fsync + rename; `*.part` leftovers
//! are sweepable garbage.

use std::fs;
use std::path::{Path, PathBuf};

/// Bundle inputs: everything the sidecars record.
#[derive(Debug, Clone)]
pub struct BundleInputs {
    /// Registry slug (e.g. `comfy:checkpoints/sdxl.safetensors`).
    pub model_slug: String,
    /// Text prompt.
    pub prompt: String,
    /// RNG seed.
    pub seed: u64,
    /// Diffusion steps.
    pub steps: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `SdPipeline::backend_id` (currently `"stub"`).
    pub backend_id: &'static str,
}

/// Parse `WxH` (e.g. `1024x1024`) into `(width, height)`; zero dims rejected.
pub fn parse_size(raw: &str) -> Result<(u32, u32), String> {
    let err = || format!("bad --size {raw:?}: want WxH like 1024x1024");
    let (w, h) = raw.split_once(['x', 'X']).ok_or_else(err)?;
    let width: u32 = w.parse().map_err(|_| err())?;
    let height: u32 = h.parse().map_err(|_| err())?;
    if width == 0 || height == 0 {
        return Err(format!("bad --size {raw:?}: dimensions must be > 0"));
    }
    Ok((width, height))
}

/// Write a full job bundle under `~/.sd-rip/jobs/{ulid}/`; returns the job dir.
pub fn write_bundle(
    inputs: &BundleInputs,
    png_bytes: &[u8],
    elapsed_ms: u64,
) -> Result<PathBuf, String> {
    let jobs = home::home_dir()
        .map(|h| h.join(".sd-rip").join("jobs"))
        .ok_or_else(|| "cannot resolve home directory".to_string())?;
    let dir = jobs.join(ulid::Ulid::new().to_string());
    fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot create job dir {}: {e}", dir.display()))?;
    write_bundle_into(&dir, inputs, png_bytes, elapsed_ms)?;
    Ok(dir)
}

/// Write the bundle files into an existing `dir` (split out for tests).
pub fn write_bundle_into(
    dir: &Path,
    inputs: &BundleInputs,
    png_bytes: &[u8],
    elapsed_ms: u64,
) -> Result<(), String> {
    let size = format!("{}x{}", inputs.width, inputs.height);
    let request = serde_json::json!({
        "model": inputs.model_slug,
        "prompt": inputs.prompt,
        "seed": inputs.seed,
        "steps": inputs.steps,
        "size": size,
    });
    let sidecar = serde_json::json!({
        "prompt": inputs.prompt,
        "seed": inputs.seed,
        "steps": inputs.steps,
        "sampler": "euler",
        "scheduler": "normal",
        "size": size,
        "model_slug": inputs.model_slug,
        "backend_id": inputs.backend_id,
    });
    let metrics = serde_json::json!({
        "event": "done",
        "seed": inputs.seed,
        "steps": inputs.steps,
        "width": inputs.width,
        "height": inputs.height,
        "elapsed_ms": elapsed_ms,
    });
    let log = format!(
        "model={}\nprompt={}\nseed={} steps={} size={} backend={}\ndone elapsed_ms={elapsed_ms}\n",
        inputs.model_slug, inputs.prompt, inputs.seed, inputs.steps, size, inputs.backend_id,
    );
    write_atomic(
        dir,
        "request.json",
        serde_json::to_string_pretty(&request)
            .map_err(|e| format!("cannot encode request.json: {e}"))?
            .as_bytes(),
    )?;
    write_atomic(dir, "final_00.png", png_bytes)?;
    write_atomic(
        dir,
        "final_00.json",
        serde_json::to_string_pretty(&sidecar)
            .map_err(|e| format!("cannot encode final_00.json: {e}"))?
            .as_bytes(),
    )?;
    let mut metrics_line =
        serde_json::to_string(&metrics).map_err(|e| format!("cannot encode metrics: {e}"))?;
    metrics_line.push('\n');
    write_atomic(dir, "metrics.jsonl", metrics_line.as_bytes())?;
    write_atomic(dir, "run.log", log.as_bytes())?;
    Ok(())
}

/// Write `bytes` to `dir/name` via `<name>.part` + best-effort fsync + rename.
fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let tmp = dir.join(format!("{name}.part"));
    fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    if let Ok(f) = fs::File::open(&tmp) {
        let _ = f.sync_all();
    }
    fs::rename(&tmp, dir.join(name)).map_err(|e| format!("cannot commit {name}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> BundleInputs {
        BundleInputs {
            model_slug: "comfy:checkpoints/x.safetensors".to_string(),
            prompt: "a red fox".to_string(),
            seed: 42,
            steps: 20,
            width: 16,
            height: 16,
            backend_id: "stub",
        }
    }

    #[test]
    fn parse_size_ok() {
        assert_eq!(parse_size("1024x1024"), Ok((1024, 1024)));
        assert_eq!(parse_size("512x768"), Ok((512, 768)));
    }

    #[test]
    fn parse_size_rejects() {
        for bad in ["", "1024", "0x64", "1024x0", "axb", "1024x", "x1024"] {
            assert!(parse_size(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn bundle_layout_exact() {
        let dir = std::env::temp_dir().join(format!("sd-rip-jobs-test-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("test dir");
        write_bundle_into(&dir, &inputs(), &[0u8; 8], 7).expect("bundle");
        for name in [
            "request.json",
            "final_00.png",
            "final_00.json",
            "metrics.jsonl",
            "run.log",
        ] {
            let bytes = fs::read(dir.join(name)).expect(name);
            assert!(!bytes.is_empty(), "{name}");
        }
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert_eq!(side["sampler"], "euler");
        assert_eq!(side["scheduler"], "normal");
        assert_eq!(side["backend_id"], "stub");
        assert_eq!(side["model_slug"], "comfy:checkpoints/x.safetensors");
        let metrics = fs::read_to_string(dir.join("metrics.jsonl")).expect("metrics");
        assert_eq!(metrics.lines().count(), 1);
        assert!(serde_json::from_str::<serde_json::Value>(metrics.trim()).is_ok());
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .expect("ls")
            .map(|e| e.expect("entry").file_name())
            .filter(|n| {
                std::path::Path::new(n)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("part"))
            })
            .collect();
        assert_eq!(leftovers, [] as [std::ffi::OsString; 0]);
        fs::remove_dir_all(&dir).ok();
    }
}
