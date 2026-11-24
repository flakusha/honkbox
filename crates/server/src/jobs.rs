//! `generate` bundle writer: `<data_dir>/jobs/{ulid}/` + atomic writes.
//!
//! Layout per job dir: `request.json` (full params + slug), `final_00.{png,jpg,webp}` +
//! sidecar `final_00.json`, `metrics.jsonl` (one line), `run.log`. Every file
//! lands via `<name>.part` + best-effort fsync + rename; `*.part` leftovers
//! are swept by `gc_jobs` at server startup (plus stray
//! `$TMPDIR/honkbox-*.png` scratch files from sd-cpp).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tracing::{debug, warn};

// Determinism machinery (TASK-determinism-contract) is registered here —
// not in `main.rs` — because main.rs is owned by the in-flight
// binary-formats wave (see .tmp/intercom/WHO-DOES-WHAT.md). Promoting it
// to a top-level module later moves this one `mod` line and re-points the
// `determinism::` call site below (and determinism.rs's `super::` use)
// to `crate::`.
#[path = "determinism.rs"]
pub mod determinism;

/// How long a job bundle or stray temp file survives before `gc_jobs` removes
/// it (default 7 days, override via `[jobs] gc_ttl` / `HONKBOX_GC_TTL`).
pub const DEFAULT_GC_TTL: Duration = Duration::from_hours(7 * 24);
/// Total size cap across all job bundles (default 4 GiB, override via
/// `[jobs] gc_max_bytes` / `HONKBOX_GC_MAX_BYTES`).
pub const DEFAULT_GC_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;

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
    /// Output container: art file `final_00.{ext}` + sidecar `format` field.
    pub format: honkbox_sd_core::OutputFormat,
    /// `SdPipeline::backend_id` (currently `"stub"`).
    pub backend_id: &'static str,
    /// Verified sha256 of the weight file actually used, from the
    /// download provenance sidecar (`<file>.sha256.json`, written only
    /// when the source published a checksum and the streamed bytes
    /// matched). `None` = unknown — locally-placed files, unverified
    /// downloads, and the CLI path carry no digest; the bundle records
    /// null, never a fabricated hash (TASK-determinism-contract).
    pub model_sha256: Option<String>,
    /// Vendored engine identity (`sd-core::engine_version_for`); `None`
    /// when the backend has no separate engine (the stub is in-process
    /// honkbox code, covered by the record's `code_version`).
    pub engine_version: Option<&'static str>,
    /// Sampler wire name actually used (e.g. `"euler"`).
    pub sampler: String,
    /// Scheduler wire name actually used (`"normal"` = vendor discrete).
    pub scheduler: String,
    /// CFG guidance scale actually used (default 7.0).
    pub cfg_scale: f32,
    /// Negative prompt actually used; `None`/empty = none.
    pub negative_prompt: Option<String>,
    /// `LoRA` adapters actually applied (HTTP generations only; empty
    /// otherwise).
    pub loras: Vec<LoraRecord>,
    /// img2img/inpaint denoise strength actually used; `None` = txt2img.
    pub strength: Option<f32>,
    /// Mask mode actually used: `Some("inpaint")` when a mask PNG was
    /// supplied, `Some("replace")` for full-image img2img without one,
    /// `None` for txt2img. The vendor chooses scope by mask presence; the
    /// route records the choice explicitly (TASK-sd-inpaint-mask-semantics).
    pub mask_mode: Option<&'static str>,
    /// Lowercase sha256 of the staged mask PNG bytes; `None` without a mask.
    pub mask_sha256: Option<String>,
    /// Scan-time base-model tag of the checkpoint actually used.
    pub checkpoint_base_model: Option<String>,
    /// Named generation recipe actually applied, verbatim from the request
    /// (explicit per-field overrides may have replaced individual recipe
    /// values — the sampler/scheduler/cfg/steps fields above are what ran).
    /// `None` = no recipe named.
    pub recipe: Option<String>,
    /// Hires-fix second pass actually applied (HTTP generations only;
    /// `None` = single-pass).
    pub hires: Option<HiresRecord>,
    /// Named annotations attached to this bundle (sidecar record).
    pub annotations: Vec<BundleAnnotation>,
}

/// A single named annotation attached to a generation bundle (sidecar
/// record). `name` is a stable key; `value` is any JSON value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BundleAnnotation {
    pub name: String,
    pub value: serde_json::Value,
}

/// One `LoRA` adapter actually applied to a generation (sidecar record).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LoraRecord {
    /// Catalog slug as resolved for the request.
    pub slug: String,
    /// Multiplier actually used.
    pub strength: f32,
    /// Scan-time base-model tag, when the `LoRA` header carries one.
    pub base_model: Option<String>,
}

/// Hires-fix second pass actually applied (sidecar record).
#[derive(Debug, Clone, serde::Serialize)]
pub struct HiresRecord {
    /// Upscaler used: always `"latent"` today (the vendor interpolation
    /// pass; the model-backed ESRGAN variant is not wired).
    pub upscaler: &'static str,
    /// Final output size `WxH` (base * scale, rounded).
    pub target: String,
    /// Upscale factor over the base size.
    pub scale: f32,
    /// Second-pass denoising strength actually used.
    pub denoising_strength: f32,
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

/// Write a full job bundle under `<data_dir>/jobs/{ulid}/`; returns the job dir.
pub fn write_bundle(
    data_dir: &Path,
    inputs: &BundleInputs,
    image_bytes: &[u8],
    elapsed_ms: u64,
) -> Result<PathBuf, String> {
    let dir = data_dir.join("jobs").join(ulid::Ulid::new().to_string());
    fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot create job dir {}: {e}", dir.display()))?;
    write_bundle_into(&dir, inputs, image_bytes, elapsed_ms)?;
    Ok(dir)
}

/// `record_for_slug` maps a slug to a curated row through a catalog's
/// slug/file-name pairs. Bundle writers have no `AppConfig`, so build that
/// map from the checked-in curated manifest — the authoritative list of
/// license rows — rather than from a scan snapshot.
fn license_catalog() -> honkbox_api_types::ModelCatalog {
    use honkbox_api_types::{ModelCapabilities, ModelCatalogEntry};
    honkbox_api_types::ModelCatalog {
        object: "list".to_owned(),
        data: crate::catalog::manifest()
            .iter()
            .map(|row| ModelCatalogEntry {
                modality: honkbox_api_types::ModelModalities::empty(),
                id: row.id.clone(),
                slug: row.id.clone(),
                kind: row.kind,
                owned_by: "curated".to_owned(),
                context_length: None,
                max_output: None,
                thinking: false,
                tool_calling: false,
                modalities: Vec::new(),
                capabilities: ModelCapabilities::for_kind(row.kind),
                category: "curated".to_owned(),
                path: row.file.clone(),
                bytes: 0,
                metadata: None,
            })
            .collect(),
    }
}

/// Sidecar license provenance from the same curated rows capabilities
/// publish. Null when the slug has no curated record — never fabricated.
fn model_license_record(slug: &str) -> Option<serde_json::Value> {
    crate::catalog::licenses::record_for_slug(slug, &license_catalog()).map(|record| {
        let non_commercial = record.commercial_terms == "non-commercial";
        serde_json::json!({
            "license": record.license,
            "license_url": record.license_url,
            "territory": record.territory,
            "commercial_terms": record.commercial_terms,
            "gated": record.gated,
            "source_url": record.source_url,
            "sha256": record.sha256,
            "attribution_text": record.attribution_text,
            "non_commercial": non_commercial,
        })
    })
}

/// CC0 pack license rows recorded by the asset-pack fetch job. Read from
/// `<data_dir>/asset-packs` rather than a global registry: every bundle
/// manifest carries the same rows the fetch wrote, joined on sha256. QAL
/// rows can never be present — the fetch guard refuses them before write.
fn pack_license_rows(bundle_dir: &Path) -> Vec<serde_json::Value> {
    let Some(data_dir) = bundle_dir.parent().and_then(Path::parent) else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(data_dir.join("asset-packs")) else {
        return Vec::new();
    };
    let mut rows: Vec<serde_json::Value> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".license.json"))
            {
                return None;
            }
            let raw = fs::read_to_string(path).ok()?;
            let row: serde_json::Value = serde_json::from_str(&raw).ok()?;
            // Defense in depth: a stray row never bundles a QAL pack.
            if row["license"].as_str() == Some("QAL-1.0") {
                return None;
            }
            Some(row)
        })
        .collect();
    rows.sort_by(|a, b| {
        a["pack_id"]
            .as_str()
            .unwrap_or_default()
            .cmp(b["pack_id"].as_str().unwrap_or_default())
    });
    rows
}

/// Write the bundle files into an existing `dir` (split out for tests).
pub fn write_bundle_into(
    dir: &Path,
    inputs: &BundleInputs,
    image_bytes: &[u8],
    elapsed_ms: u64,
) -> Result<(), String> {
    let size = format!("{}x{}", inputs.width, inputs.height);
    let art = format!("final_00.{}", inputs.format.extension());
    // Canonical input key (TASK-determinism-contract): the L0
    // reproducibility handle over everything recorded below. Two runs of
    // the same request carry the same value; volatile per-run state (ULID
    // dir name, elapsed_ms) is excluded by the canonical form.
    let inputs_sha256 = determinism::bundle_record_hash(inputs);
    let model_license = model_license_record(&inputs.model_slug);
    let pack_licenses = pack_license_rows(dir);
    let request = serde_json::json!({
        "model": inputs.model_slug,
        "model_sha256": inputs.model_sha256,
        "engine_version": inputs.engine_version,
        "pack_licenses": pack_licenses.clone(),
        "prompt": inputs.prompt,
        "seed": inputs.seed,
        "steps": inputs.steps,
        "size": size,
        "format": inputs.format.name(),
        "sampler": inputs.sampler,
        "scheduler": inputs.scheduler,
        "cfg_scale": inputs.cfg_scale,
        "negative_prompt": inputs.negative_prompt.clone().unwrap_or_default(),
        "loras": inputs.loras,
        "strength": inputs.strength,
        "mask_mode": inputs.mask_mode,
        "mask_sha256": inputs.mask_sha256,
        "checkpoint_base_model": inputs.checkpoint_base_model,
        "recipe": inputs.recipe,
        "hires": inputs.hires,
        "inputs_sha256": inputs_sha256,
    });
    let sidecar = serde_json::json!({
        "prompt": inputs.prompt,
        "seed": inputs.seed,
        "steps": inputs.steps,
        "sampler": inputs.sampler,
        "scheduler": inputs.scheduler,
        "cfg_scale": inputs.cfg_scale,
        "negative_prompt": inputs.negative_prompt.clone().unwrap_or_default(),
        "size": size,
        "model_slug": inputs.model_slug,
        "format": inputs.format.name(),
        "backend_id": inputs.backend_id,
        "model_sha256": inputs.model_sha256,
        "engine_version": inputs.engine_version,
        "model_license": model_license,
        "pack_licenses": pack_licenses,
        "loras": inputs.loras,
        "strength": inputs.strength,
        "mask_mode": inputs.mask_mode,
        "mask_sha256": inputs.mask_sha256,
        "checkpoint_base_model": inputs.checkpoint_base_model,
        "recipe": inputs.recipe,
        "hires": inputs.hires,
        "annotations": inputs.annotations,
        "inputs_sha256": inputs_sha256,
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
    // F5: commit the artifact BEFORE any sidecar. A crash mid-bundle then
    // leaves at most an orphan artifact (swept by GC), never a sidecar set
    // that makes a half-written bundle look complete.
    write_atomic(dir, &art, image_bytes)?;
    write_atomic(
        dir,
        "request.json",
        serde_json::to_string_pretty(&request)
            .map_err(|e| format!("cannot encode request.json: {e}"))?
            .as_bytes(),
    )?;
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

/// One recorded input of an image-ops job bundle (sidecar record): where
/// the bytes came from and what they hashed to. Data-URI payloads are
/// recorded by DIGEST only — no payload bytes ever land in a sidecar.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OpsInputRecord {
    /// Provenance label: `"data_uri"` or `"artifact:<bundle>/<index>"`.
    pub source: String,
    /// SHA-256 hex digest of the decoded input bytes.
    pub sha256: String,
}

/// One recorded output of an image-ops job bundle (sidecar record).
#[derive(Debug, Clone, serde::Serialize)]
pub struct OpsOutputRecord {
    /// File name within the bundle dir (e.g. `op_00.png`).
    pub file: String,
    /// SHA-256 hex digest of the file bytes.
    pub sha256: String,
}

/// Everything the image-ops bundle records about the command
/// (TASK-image-ops-job-route): command hash, per-input/output hashes,
/// tool version, attempts, outcome. No data-URI payload, no token, and no
/// full URL ever lands in a bundle.
#[derive(Debug, Clone)]
pub struct OpsBundleInputs {
    /// Op wire name (`resize`, `crop`, …).
    pub op: String,
    /// SHA-256 hex over the canonical command JSON (the retry-dedup key's
    /// hash: command + params + output format + compression).
    pub command_sha256: String,
    /// The request's idempotency key, verbatim.
    pub request_id: String,
    /// Caller identity, verbatim (`None` = not supplied).
    pub client: Option<String>,
    /// Per-input provenance + digests, in order.
    pub inputs: Vec<OpsInputRecord>,
    /// Output container.
    pub format: honkbox_sd_core::OutputFormat,
}

/// Write an image-ops job bundle under `<data_dir>/jobs/{ulid}/`; returns
/// the bundle dir. Same atomic-write discipline as [`write_bundle_into`]
/// (`.part` + fsync + rename) and the same GC lifecycle (`gc_jobs`).
///
/// Layout: `command.json` (the bounded command — no payloads), one
/// `op_XX.{ext}` file per output, `ops.json` (the sidecar: command hash,
/// input/output hashes, `seed: null` — pixel ops are seedless, the null
/// keeps the record shape uniform with generation bundles — tool version,
/// attempts, outcome), and `run.log`.
pub fn write_ops_bundle(
    data_dir: &Path,
    inputs: &OpsBundleInputs,
    outputs: &[OpsOutputRecord],
    output_bytes: &[&[u8]],
    elapsed_ms: u64,
) -> Result<PathBuf, String> {
    // F4: never let ops.json claim outputs that were not written. Check
    // before creating any dir so a mismatch leaves no partial bundle.
    if outputs.len() != output_bytes.len() {
        return Err(format!(
            "ops bundle output count mismatch: {} records but {} byte buffers",
            outputs.len(),
            output_bytes.len()
        ));
    }
    let dir = data_dir.join("jobs").join(ulid::Ulid::new().to_string());
    fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot create job dir {}: {e}", dir.display()))?;
    let command = serde_json::json!({
        "op": inputs.op,
        "output_format": inputs.format.name(),
        "request_id": inputs.request_id,
        "client": inputs.client,
    });
    let tool = serde_json::json!({
        "name": "honkbox-image-ops",
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "image-rs",
    });
    let sidecar = serde_json::json!({
        "command_sha256": inputs.command_sha256,
        "request_id": inputs.request_id,
        "op": inputs.op,
        "output_format": inputs.format.name(),
        "client": inputs.client,
        "inputs": inputs.inputs,
        "outputs": outputs,
        "seed": serde_json::Value::Null,
        "tool": tool,
        "attempts": 1,
        "outcome": "done",
        "elapsed_ms": elapsed_ms,
    });
    let log = format!(
        "op={} outputs={} format={} client={:?}\ndone elapsed_ms={elapsed_ms}\n",
        inputs.op,
        outputs.len(),
        inputs.format.name(),
        inputs.client,
    );
    write_atomic(
        &dir,
        "command.json",
        serde_json::to_string_pretty(&command)
            .map_err(|e| format!("cannot encode command.json: {e}"))?
            .as_bytes(),
    )?;
    write_atomic(
        &dir,
        "ops.json",
        serde_json::to_string_pretty(&sidecar)
            .map_err(|e| format!("cannot encode ops.json: {e}"))?
            .as_bytes(),
    )?;
    for (record, bytes) in outputs.iter().zip(output_bytes) {
        write_atomic(&dir, &record.file, bytes)?;
    }
    write_atomic(&dir, "run.log", log.as_bytes())?;
    Ok(dir)
}

/// Write `bytes` to `dir/name` via `<name>.part` + best-effort fsync + rename.
/// Public: every bundle writer commits the same way (the artifact first,
/// the sidecars after), so a crash never leaves a half-written file that
/// looks complete.
pub fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let tmp = dir.join(format!("{name}.part"));
    fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    if let Ok(f) = fs::File::open(&tmp) {
        let _ = f.sync_all();
    }
    fs::rename(&tmp, dir.join(name)).map_err(|e| format!("cannot commit {name}: {e}"))?;
    Ok(())
}

/// Copy `src` into `dir/name` via `<name>.part` + fsync + rename; returns the
/// copied byte count. Streamed, so a large artifact never lands in memory.
pub fn copy_file_atomic(dir: &Path, name: &str, src: &Path) -> Result<u64, String> {
    let tmp = dir.join(format!("{name}.part"));
    let bytes = fs::copy(src, &tmp).map_err(|e| format!("cannot copy {}: {e}", src.display()))?;
    if let Ok(f) = fs::File::open(&tmp) {
        let _ = f.sync_all();
    }
    fs::rename(&tmp, dir.join(name)).map_err(|e| format!("cannot commit {name}: {e}"))?;
    Ok(bytes)
}

/// Startup garbage collection over `<data_dir>/jobs` and the temp dir.
///
/// Removes: job dirs whose mtime is older than `max_age`; then, oldest
/// first, job dirs until the total bundle size is under `max_bytes`; stray
/// `*.part` files left by a crash mid-write; and stale
/// `$TMPDIR/honkbox-*.png` files (the sd-cpp decode scratch template
/// `honkbox-{pid}-{seed}-{nanos}.png`) older than `max_age`. Never fails:
/// problems log at `warn!` and GC continues; removals log at `debug!`.
/// `jobs_root: None` (unresolvable data dir) still sweeps the temp dir.
pub fn gc_jobs(jobs_root: Option<&Path>, max_age: Duration, max_bytes: u64) {
    if let Some(root) = jobs_root {
        gc_dir(root, max_age, max_bytes);
        // Sidecar work dirs (`fizgig/<id>`) hold caches and resume state.
        // Sweep them with the same retention so a long training run's leftovers
        // cannot grow forever; each root has its own size cap.
        if let Some(base) = root.parent() {
            gc_dir(&base.join("fizgig"), max_age, max_bytes);
        }
    }
    sweep_stale_temp_pngs(max_age, SystemTime::now());
}

/// GC one jobs root (split out for tests): sweep stray root-level `*.part`
/// files, remove expired job dirs, evict oldest-first under the size cap,
/// and sweep `*.part` leftovers inside the surviving job dirs.
fn gc_dir(root: &Path, max_age: Duration, max_bytes: u64) {
    let now = SystemTime::now();
    sweep_parts_in(root);
    let mut total = 0u64;
    let mut dirs: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
    match fs::read_dir(root) {
        Ok(entries) => {
            for entry in entries.flatten() {
                if !entry.file_type().is_ok_and(|t| t.is_dir()) {
                    continue;
                }
                let path = entry.path();
                let (mtime, size) = dir_stats(&path);
                if now.duration_since(mtime).unwrap_or(Duration::ZERO) > max_age {
                    if let Err(e) = fs::remove_dir_all(&path) {
                        warn!("gc: cannot remove {}: {e}", path.display());
                    } else {
                        debug!("gc: removed expired job dir {}", path.display());
                    }
                    continue;
                }
                total += size;
                dirs.push((path, mtime, size));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!("gc: cannot list {}: {e}", root.display()),
    }
    // Size cap: evict oldest-first until under cap.
    dirs.sort_by_key(|(_, mtime, _)| *mtime);
    for (path, _, size) in dirs {
        if total <= max_bytes {
            break;
        }
        if let Err(e) = fs::remove_dir_all(&path) {
            warn!("gc: cannot remove {}: {e}", path.display());
        } else {
            debug!("gc: evicted {} ({}) for size cap", path.display(), size);
        }
        total -= size;
    }
    // Crash leftovers inside surviving bundles.
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                sweep_parts_in(&entry.path());
            }
        }
    }
}

/// Remove `*.part` files directly inside `dir` (crash leftovers that never
/// got renamed into place).
fn sweep_parts_in(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("part"))
        {
            if let Err(e) = fs::remove_file(&path) {
                warn!("gc: cannot remove {}: {e}", path.display());
            } else {
                debug!("gc: swept stray part file {}", path.display());
            }
        }
    }
}

/// Remove stale `$TMPDIR/honkbox-*.png` scratch files (sd-cpp decode output
/// template `honkbox-{pid}-{seed}-{nanos}.png`) older than `max_age`.
fn sweep_stale_temp_pngs(max_age: Duration, now: SystemTime) {
    let Ok(entries) = fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_scratch =
            name.starts_with("honkbox-") && name.ends_with(".png") && entry.path().is_file();
        if !is_scratch {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if now.duration_since(mtime).unwrap_or(Duration::ZERO) <= max_age {
            continue;
        }
        if let Err(e) = fs::remove_file(entry.path()) {
            warn!("gc: cannot remove {}: {e}", entry.path().display());
        } else {
            debug!("gc: swept stale temp png {}", entry.path().display());
        }
    }
}

/// `(mtime, recursive byte size)` of a directory; unreadable entries count as
/// epoch mtime / zero bytes so GC can still proceed.
fn dir_stats(dir: &Path) -> (SystemTime, u64) {
    let mut size = 0u64;
    let mut mtime = SystemTime::UNIX_EPOCH;
    let Ok(entries) = fs::read_dir(dir) else {
        return (mtime, 0);
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            let (sub_mtime, sub_size) = dir_stats(&entry.path());
            size += sub_size;
            mtime = mtime.max(sub_mtime);
        } else {
            size += meta.len();
            if let Ok(t) = meta.modified() {
                mtime = mtime.max(t);
            }
        }
    }
    (mtime, size)
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
            format: honkbox_sd_core::OutputFormat::Png,
            backend_id: "stub",
            model_sha256: None,
            engine_version: None,
            sampler: "euler".to_string(),
            scheduler: "normal".to_string(),
            cfg_scale: 7.0,
            negative_prompt: None,
            loras: Vec::new(),
            strength: None,
            mask_mode: None,
            mask_sha256: None,
            checkpoint_base_model: None,
            recipe: None,
            hires: None,
            annotations: Vec::new(),
        }
    }

    /// Backdate `path`'s mtime to `age_secs` before now.
    fn backdate(path: &Path, age_secs: u64) {
        let t = SystemTime::now()
            .checked_sub(Duration::from_secs(age_secs))
            .expect("age below now");
        let times = fs::FileTimes::new().set_modified(t).set_accessed(t);
        fs::File::options()
            .write(true)
            .open(path)
            .expect("open for backdate")
            .set_times(times)
            .expect("backdate");
    }

    fn fresh_root(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("honkbox-jobs-gc-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("test root");
        root
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
    fn format_changes_art_filename_and_sidecar() {
        for (format, ext, name) in [
            (honkbox_sd_core::OutputFormat::Png, "final_00.png", "png"),
            (honkbox_sd_core::OutputFormat::Jpeg, "final_00.jpg", "jpeg"),
            (honkbox_sd_core::OutputFormat::Webp, "final_00.webp", "webp"),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "honkbox-jobs-fmt-{}-{}",
                std::process::id(),
                name
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("test dir");
            let mut bundle = inputs();
            bundle.format = format;
            write_bundle_into(&dir, &bundle, &[0u8; 8], 7).expect("bundle");
            assert!(fs::read(dir.join(ext)).is_ok(), "{ext}");
            let side: serde_json::Value =
                serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                    .expect("sidecar json");
            assert_eq!(side["format"], name, "{ext}");
            fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn sidecar_records_requested_recipe() {
        let dir = std::env::temp_dir().join(format!("honkbox-jobs-recipe-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("test dir");
        let mut bundle = inputs();
        bundle.sampler = "dpm++2m".to_string();
        bundle.scheduler = "karras".to_string();
        bundle.cfg_scale = 4.5;
        bundle.negative_prompt = Some("blurry".to_string());
        write_bundle_into(&dir, &bundle, &[0u8; 8], 7).expect("bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert_eq!(side["sampler"], "dpm++2m");
        assert_eq!(side["scheduler"], "karras");
        assert_eq!(side["cfg_scale"], 4.5);
        assert_eq!(side["negative_prompt"], "blurry");
        fs::remove_dir_all(&dir).ok();
    }

    /// Provenance (TASK-determinism-contract): the verified model digest
    /// and vendored engine identity land in the sidecar when known and
    /// stay explicit nulls (never absent, never fabricated) otherwise.
    #[test]
    fn sidecar_records_provenance() {
        let dir = std::env::temp_dir().join(format!("honkbox-jobs-prov-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("test dir");

        let mut bundle = inputs();
        bundle.model_sha256 = Some("a".repeat(64));
        bundle.engine_version = Some("stable-diffusion.cpp test");
        write_bundle_into(&dir, &bundle, &[0u8; 8], 7).expect("bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert_eq!(side["model_sha256"], "a".repeat(64));
        assert_eq!(side["engine_version"], "stable-diffusion.cpp test");
        let req: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("request.json")).expect("request"))
                .expect("request json");
        assert_eq!(req["model_sha256"], "a".repeat(64));

        write_bundle_into(&dir, &inputs(), &[0u8; 8], 7).expect("plain bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert!(side["model_sha256"].is_null());
        assert!(side["engine_version"].is_null());
        fs::remove_dir_all(&dir).ok();
    }

    /// `LoRA` adapters and the img2img strength land in the sidecar when set,
    /// and stay `[]`/`null` (not missing) for plain txt2img records.
    #[test]
    fn sidecar_records_loras_and_strength() {
        let dir = std::env::temp_dir().join(format!("honkbox-jobs-lora-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("test dir");

        let mut bundle = inputs();
        bundle.loras = vec![LoraRecord {
            slug: "comfy:loras/detail.safetensors".to_string(),
            strength: 0.8,
            base_model: Some("sdxl".to_owned()),
        }];
        bundle.strength = Some(0.6);
        write_bundle_into(&dir, &bundle, &[0u8; 8], 7).expect("bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        // f32 strengths widen to f64 in the sidecar JSON: compare exactly
        // representable values with a tolerance, not equality.
        assert!((side["loras"][0]["strength"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(side["loras"][0]["slug"], "comfy:loras/detail.safetensors");
        assert!((side["strength"].as_f64().unwrap() - 0.6).abs() < 1e-6);

        write_bundle_into(&dir, &inputs(), &[0u8; 8], 7).expect("plain bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert_eq!(side["loras"], serde_json::json!([]));
        assert!(side["strength"].is_null());
        assert!(side["recipe"].is_null());
        assert!(side["hires"].is_null());
        fs::remove_dir_all(&dir).ok();
    }

    /// The named recipe and the hires-fix pass land verbatim in the sidecar
    /// when set (the recipe name + the pass actually applied).
    #[test]
    fn sidecar_records_recipe_and_hires() {
        let dir =
            std::env::temp_dir().join(format!("honkbox-jobs-recipe-hires-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("test dir");

        let mut bundle = inputs();
        bundle.recipe = Some("sdxl".to_string());
        bundle.hires = Some(HiresRecord {
            upscaler: "latent",
            target: "2048x1024".to_string(),
            scale: 2.0,
            denoising_strength: 0.55,
        });
        write_bundle_into(&dir, &bundle, &[0u8; 8], 7).expect("bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert_eq!(side["recipe"], "sdxl");
        assert_eq!(side["hires"]["upscaler"], "latent");
        assert_eq!(side["hires"]["target"], "2048x1024");
        assert!((side["hires"]["scale"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((side["hires"]["denoising_strength"].as_f64().unwrap() - 0.55).abs() < 1e-6);

        write_bundle_into(&dir, &inputs(), &[0u8; 8], 7).expect("plain bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("final_00.json")).expect("sidecar"))
                .expect("sidecar json");
        assert!(side["recipe"].is_null());
        assert!(side["hires"].is_null());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bundle_layout_exact() {
        let dir = std::env::temp_dir().join(format!("honkbox-jobs-test-{}", std::process::id()));
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
        assert_eq!(side["cfg_scale"], 7.0);
        assert_eq!(side["negative_prompt"], "");
        assert_eq!(side["backend_id"], "stub");
        assert_eq!(side["format"], "png");
        assert_eq!(side["model_slug"], "comfy:checkpoints/x.safetensors");
        let req: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("request.json")).expect("request"))
                .expect("request json");
        assert_eq!(req["format"], "png");
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

    /// Old job dir removed, fresh job dir kept.
    #[test]
    fn gc_removes_expired_keeps_fresh() {
        let root = fresh_root("old");
        let old = root.join("01OLD");
        let new = root.join("01NEW");
        for dir in [&old, &new] {
            fs::create_dir_all(dir).expect("dir");
            fs::write(dir.join("final_00.png"), [0u8; 16]).expect("file");
        }
        backdate(&old.join("final_00.png"), 7200);
        // `dir_stats` takes the max child mtime, so backdating the child is
        // enough; assert the helper agrees before sweeping.
        gc_dir(&root, Duration::from_secs(1), u64::MAX);
        assert!(!old.exists(), "expired dir removed");
        assert!(new.exists(), "fresh dir kept");
        fs::remove_dir_all(&root).ok();
    }

    /// Size cap evicts the oldest dir first while over budget.
    #[test]
    fn gc_size_cap_evicts_oldest() {
        let root = fresh_root("cap");
        let a = root.join("01A");
        let b = root.join("02B");
        let c = root.join("03C");
        for dir in [&a, &b, &c] {
            fs::create_dir_all(dir).expect("dir");
            fs::write(dir.join("blob"), [0u8; 100]).expect("blob");
        }
        for (dir, age) in [(&a, 600), (&b, 500), (&c, 400)] {
            backdate(&dir.join("blob"), age);
        }
        // Cap fits only two 100-byte bundles.
        gc_dir(&root, Duration::from_secs(3600), 200);
        assert!(!a.exists(), "oldest evicted");
        assert!(b.exists() && c.exists(), "newer kept");
        fs::remove_dir_all(&root).ok();
    }

    /// Stray `*.part` files in the root and inside job dirs are swept.
    #[test]
    fn gc_sweeps_part_files() {
        let root = fresh_root("part");
        let job = root.join("01J");
        fs::create_dir_all(&job).expect("dir");
        fs::write(root.join("final_00.png.part"), b"x").expect("root part");
        fs::write(job.join("run.log.part"), b"x").expect("job part");
        gc_dir(&root, Duration::from_secs(3600), u64::MAX);
        assert!(!root.join("final_00.png.part").exists(), "root part swept");
        assert!(!job.join("run.log.part").exists(), "job part swept");
        assert!(job.exists(), "fresh job kept");
        fs::remove_dir_all(&root).ok();
    }

    /// Fizgig work dirs share the job retention policy.
    #[test]
    fn gc_jobs_sweeps_fizgig_work_dirs() {
        let base = fresh_root("fizgig-gc");
        let jobs = base.join("jobs");
        let work = base.join("fizgig").join("job_OLD");
        fs::create_dir_all(&jobs).expect("jobs root");
        fs::create_dir_all(&work).expect("work dir");
        fs::write(work.join("cache.bin"), [0_u8; 16]).expect("cache");
        backdate(&work.join("cache.bin"), 7200);
        gc_jobs(Some(&jobs), Duration::from_secs(1), u64::MAX);
        assert!(!work.exists(), "expired fizgig work dir removed");
        fs::remove_dir_all(&base).ok();
    }

    /// F5: the artifact is committed before any sidecar. Forcing the
    /// artifact commit to fail (its target path is a directory) must leave
    /// NO sidecar behind — under the old request.json-first order a
    /// complete-looking bundle survived exactly this kind of crash.
    #[test]
    fn artifact_is_written_before_sidecars() {
        let dir = std::env::temp_dir().join(format!("honkbox-jobs-order-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("test dir");
        fs::create_dir(dir.join("final_00.png")).expect("artifact blocker dir");
        assert!(write_bundle_into(&dir, &inputs(), &[0u8; 8], 7).is_err());
        for sidecar in ["request.json", "final_00.json", "metrics.jsonl", "run.log"] {
            assert!(
                !dir.join(sidecar).exists(),
                "{sidecar} must not be written before the artifact"
            );
        }
        fs::remove_dir_all(&dir).ok();
    }

    /// F4: a short byte vector is rejected before ANY file is written, so
    /// ops.json can never list a file that has no bytes. Equal lengths still
    /// succeed.
    #[test]
    fn ops_bundle_rejects_output_byte_count_mismatch() {
        let root = fresh_root("ops-mismatch");
        let bundle = OpsBundleInputs {
            op: "resize".to_string(),
            command_sha256: "a".repeat(64),
            request_id: "req-1".to_string(),
            client: None,
            inputs: Vec::new(),
            format: honkbox_sd_core::OutputFormat::Png,
        };
        let outputs = [
            OpsOutputRecord {
                file: "op_00.png".to_string(),
                sha256: "b".repeat(64),
            },
            OpsOutputRecord {
                file: "op_01.png".to_string(),
                sha256: "c".repeat(64),
            },
        ];
        let one: [&[u8]; 1] = [&[1u8, 2, 3]];
        let err = write_ops_bundle(&root, &bundle, &outputs, &one, 1)
            .expect_err("mismatched byte count rejected");
        assert!(err.contains("mismatch"), "{err}");
        assert!(
            !root.join("jobs").exists(),
            "no partial job dir on mismatch"
        );

        let both: [&[u8]; 2] = [&[1u8], &[2u8]];
        let dir = write_ops_bundle(&root, &bundle, &outputs, &both, 1).expect("matched bundle");
        let side: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("ops.json")).expect("sidecar"))
                .expect("sidecar json");
        assert_eq!(side["outputs"].as_array().map_or(0, Vec::len), 2);
        fs::remove_dir_all(&root).ok();
    }
}
