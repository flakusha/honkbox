//! Runner for one admitted Fizgig job. Kept apart from the HTTP surface so
//! the argv plan, lease lifetime and commit path are readable in one place.
//! One lease spans all three stages; the supervisor still owns process-group
//! teardown, log caps and timeouts.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;

use honkbox_residency::ResourceKind;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::fizgig::{LiveJob, Plan};
use super::supervisor::{self, Spec, Termination};
use crate::admission;

type Jobs = Mutex<HashMap<String, LiveJob>>;

/// Keep supervisor stderr bounded in the status DTO (logs stay on disk).
fn bounded(err: impl fmt::Display) -> String {
    err.to_string().chars().take(512).collect()
}

fn set(jobs: &Jobs, id: &str, status: &str, stage_name: &str) {
    let mut map = jobs
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(job) = map.get_mut(id) {
        status.clone_into(&mut job.state);
        stage_name.clone_into(&mut job.stage);
    }
}

fn fail(jobs: &Jobs, id: &str, err: impl fmt::Display) {
    let mut map = jobs
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(job) = map.get_mut(id) {
        "failed".clone_into(&mut job.state);
        job.error = Some(bounded(err));
    }
}

/// Quote a path as a TOML basic string (Windows paths carry backslashes).
fn toml_string(path: &Path) -> String {
    path.display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn dataset_toml(plan: &Plan) -> String {
    format!(
        "[general]\nresolution = [1024, 1024]\ncaption_extension = \".txt\"\nbatch_size = 1\nnum_repeats = 1\nenable_bucket = true\nbucket_no_upscale = true\n\n[[datasets]]\nimage_directory = \"{}\"\ncache_directory = \"{}\"\n",
        toml_string(&plan.dataset_dir),
        toml_string(&plan.work_dir.join("cache"))
    )
}

/// The three documented Fizgig stages, in execution order.
#[derive(Debug, Clone, Copy)]
enum Stage {
    Latents,
    Text,
    Train,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Latents => "latents",
            Self::Text => "text",
            Self::Train => "train",
        }
    }

    fn script(self, plan: &Plan) -> &Path {
        match self {
            Self::Train => &plan.train_py,
            Self::Latents | Self::Text => &plan.cache_py,
        }
    }

    fn args(self, plan: &Plan) -> Vec<String> {
        let dataset = plan.work_dir.join("dataset.toml").display().to_string();
        let mut args = vec![
            self.script(plan).display().to_string(),
            "--family".to_owned(),
            plan.family.clone(),
            "--dataset_config".to_owned(),
            dataset,
        ];
        match self {
            Self::Latents => args.extend([
                "--stage".to_owned(),
                "latents".to_owned(),
                "--model".to_owned(),
                plan.vae.as_ref().unwrap_or(&plan.dit).display().to_string(),
            ]),
            Self::Text => args.extend([
                "--stage".to_owned(),
                "text".to_owned(),
                "--model".to_owned(),
                plan.text_encoder
                    .as_ref()
                    .unwrap_or(&plan.dit)
                    .display()
                    .to_string(),
            ]),
            Self::Train => {
                args.extend([
                    "--dit".to_owned(),
                    plan.dit.display().to_string(),
                    "--output_dir".to_owned(),
                    plan.output_dir.display().to_string(),
                    "--output_name".to_owned(),
                    plan.output_name.clone(),
                    "--network_dim".to_owned(),
                    plan.rank.to_string(),
                    "--network_alpha".to_owned(),
                    plan.rank.to_string(),
                    "--max_train_epochs".to_owned(),
                    plan.epochs.to_string(),
                    "--seed".to_owned(),
                    plan.seed.to_string(),
                    "--save_every_n_epochs".to_owned(),
                    "1".to_owned(),
                    "--save_state".to_owned(),
                    "--keep_last_n_states".to_owned(),
                    "1".to_owned(),
                ]);
                for (flag, path) in [("--vae", &plan.vae), ("--text_encoder", &plan.text_encoder)] {
                    if let Some(path) = path {
                        args.push(flag.to_owned());
                        args.push(path.display().to_string());
                    }
                }
            }
        }
        args
    }
}

/// Run all stages, then commit the produced `LoRA`s to the job bundle.
// ponytail: the lease, cancel watchdog and stage loop belong in one scope;
// split only if the runner grows another responsibility.
#[allow(clippy::too_many_lines)]
pub(super) async fn run_job(plan: Plan, jobs: &'static Jobs, id: String) {
    let soft = {
        let map = jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match map.get(&id) {
            Some(job) => job.cancel.clone(),
            None => return,
        }
    };
    let marker = plan.output_dir.join(".pause_requested");
    let is_cancelled = || soft.is_cancelled() || marker.exists();
    if let Err(err) = fs::create_dir_all(plan.work_dir.join("cache")) {
        fail(jobs, &id, format!("cannot create job work dir: {err}"));
        return;
    }
    let logs = plan.work_dir.join("logs");
    if let Err(err) = fs::create_dir_all(&logs) {
        fail(jobs, &id, format!("cannot create log dir: {err}"));
        return;
    }
    let dataset = dataset_toml(&plan);
    if let Err(err) = fs::write(plan.work_dir.join("dataset.toml"), dataset) {
        fail(jobs, &id, format!("cannot write dataset config: {err}"));
        return;
    }
    // Reserve the external-GPU class for the whole run. Until the dedicated
    // training class lands, `Comfy` is the reserved external-process slot.
    let lease = match admission::acquire(ResourceKind::Comfy).await {
        Ok(lease) => lease,
        Err((_, message)) => {
            fail(jobs, &id, message);
            return;
        }
    };
    let _lease = lease;
    for stage in [Stage::Latents, Stage::Text, Stage::Train] {
        if is_cancelled() {
            break;
        }
        set(jobs, &id, "running", stage.name());
        let hard = CancellationToken::new();
        let mut spec = Spec::new(
            plan.python.clone(),
            stage.args(&plan),
            plan.work_dir.clone(),
            plan.timeout,
        );
        spec.cancel = Some(hard.clone());
        // ponytail: fixed grace from config; marker first, group kill only as
        // a fallback for CLIs that never check it.
        let grace = plan.grace;
        let watchdog = {
            let soft = soft.clone();
            tokio::spawn(async move {
                soft.cancelled().await;
                tokio::time::sleep(grace).await;
                hard.cancel();
            })
        };
        let outcome = match tokio::task::spawn_blocking(move || supervisor::run(spec)).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(err)) => {
                watchdog.abort();
                fail(
                    jobs,
                    &id,
                    format!("{} stage could not start: {err}", stage.name()),
                );
                return;
            }
            Err(err) => {
                watchdog.abort();
                fail(
                    jobs,
                    &id,
                    format!("{} stage task failed: {err}", stage.name()),
                );
                return;
            }
        };
        watchdog.abort();
        for (stream, text) in [("stdout", &outcome.stdout), ("stderr", &outcome.stderr)] {
            let name = format!("{}.{stream}.log", stage.name());
            if let Err(err) = crate::jobs::write_atomic(&logs, &name, text.as_bytes()) {
                fail(jobs, &id, format!("cannot write {name}: {err}"));
                return;
            }
        }
        if outcome.termination == Termination::Cancelled && is_cancelled() {
            break;
        }
        if outcome.termination != Termination::Exited(0) {
            fail(
                jobs,
                &id,
                format!(
                    "{} stage {:?}: {}",
                    stage.name(),
                    outcome.termination,
                    bounded(outcome.stderr)
                ),
            );
            return;
        }
        if marker.exists() {
            // A pause requested during caching need not run the remaining stages.
            break;
        }
    }
    let cancelled = is_cancelled();
    match commit(&plan, &id, cancelled) {
        Ok(bundle) => {
            let state = if cancelled { "cancelled" } else { "completed" };
            set(jobs, &id, state, "done");
            let mut map = jobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(job) = map.get_mut(&id) {
                job.bundle = Some(bundle);
            }
        }
        Err(err) => fail(jobs, &id, err),
    }
}

/// Stream a file digest; `LoRA` checkpoints are far too large to read at once.
fn sha256(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("cannot hash artifact: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot hash artifact: {e}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").map_err(|e| format!("cannot encode digest: {e}"))?;
    }
    Ok(encoded)
}

/// Copy the `LoRA` files, their digests and the dataset config into
/// `<data_dir>/jobs/{id}/` using the shared atomic bundle writers.
fn commit(plan: &Plan, id: &str, cancelled: bool) -> Result<String, String> {
    let data_dir = plan
        .work_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "job work dir has no data dir".to_owned())?;
    let bundle_dir = data_dir.join("jobs").join(id);
    fs::create_dir_all(&bundle_dir).map_err(|e| format!("cannot create bundle dir: {e}"))?;
    crate::jobs::copy_file_atomic(
        &bundle_dir,
        "dataset.toml",
        &plan.work_dir.join("dataset.toml"),
    )?;
    let mut artifacts = Vec::new();
    let entries =
        fs::read_dir(&plan.output_dir).map_err(|e| format!("cannot read outputs: {e}"))?;
    for entry in entries {
        let path = entry
            .map_err(|e| format!("cannot read output entry: {e}"))?
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "non-UTF-8 LoRA filename".to_owned())?
            .to_owned();
        let size_bytes = crate::jobs::copy_file_atomic(&bundle_dir, &name, &path)?;
        artifacts.push(json!({
            "name": name,
            "sha256": sha256(&path)?,
            "size_bytes": size_bytes,
        }));
    }
    for stage in [Stage::Latents, Stage::Text, Stage::Train] {
        for stream in ["stdout", "stderr"] {
            let name = format!("{}.{stream}.log", stage.name());
            let src = plan.work_dir.join("logs").join(&name);
            if src.is_file() {
                crate::jobs::copy_file_atomic(&bundle_dir, &name, &src)?;
            }
        }
    }
    if artifacts.is_empty() && !cancelled {
        return Err("training produced no LoRA safetensors".to_owned());
    }
    let manifest = json!({
        "family": plan.family,
        "rank": plan.rank,
        "epochs": plan.epochs,
        "seed": plan.seed,
        "artifacts": artifacts,
    });
    let bytes =
        serde_json::to_vec(&manifest).map_err(|e| format!("cannot encode manifest: {e}"))?;
    crate::jobs::write_atomic(&bundle_dir, "manifest.json", &bytes)?;
    Ok(format!("jobs/{id}"))
}
