//! The supervised Blender run behind `POST /v1/jobs/blender`.
//!
//! The route validated the request; this module owns the PROCESS: the
//! recipe files are written into the job's staging directory, Blender runs
//! under the process supervisor (its own process group, a wall-clock bound,
//! a group-wide teardown) and the artifacts are committed into the job
//! bundle the way every other bundle is written (`.part` + fsync +
//! rename).
//!
//! Cancel is a STAGE-BOUNDARY cancel: the recipe checks a marker file
//! between stages, so a cancel never interrupts a bake mid-flight. The
//! supervisor's own cancel token is deliberately left unset — the job's
//! deadline is the only thing that kills the process.

use std::path::PathBuf;

use tokio_util::sync::CancellationToken;

use crate::admission::GpuLease;
use crate::jobs_registry::JobGuard;

use super::blender::JobPlan;

/// Start the job in the background. The guard moves into the task: the
/// registry entry stays live (and the lease held) until the work ends.
pub(super) fn start(
    plan: JobPlan,
    bundle: PathBuf,
    binary: PathBuf,
    token: CancellationToken,
    lease: GpuLease,
    guard: JobGuard,
) {
    #[cfg(unix)]
    {
        tokio::task::spawn_blocking(move || {
            let _guard = guard;
            unix::run(plan, bundle, binary, token, lease);
        });
    }
    #[cfg(not(unix))]
    {
        // Unreachable: the route's platform gate answers 501 first. The
        // values are dropped so nothing leaks (the guard unregisters the
        // job, the lease frees the slot).
        drop((plan, bundle, binary, token, lease, guard));
    }
}

/// The unix implementation: the process supervisor owns the process group
/// (and with it the teardown that leaves no orphan process behind).
#[cfg(unix)]
mod unix {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use honkbox_external_tools::{
        AssetError, AssetJob, AssetManifest, AssetSource, build_manifest, prepare,
    };
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::admission::GpuLease;

    use super::super::blender::JobPlan;
    use super::super::supervisor::{self, Termination};

    /// How often the cancel watcher checks the job's token.
    const CANCEL_POLL: Duration = Duration::from_millis(100);

    /// What one run produced.
    enum Outcome {
        /// The recipe finished; the manifest describes what it wrote.
        Done(AssetManifest),
        /// The job was cancelled at a stage boundary.
        Cancelled,
        /// The job failed; the message says why.
        Failed(String),
    }

    /// Run one job: the recipe files, the supervised process, and the
    /// bundle the artifacts land in.
    pub(super) fn run(
        plan: JobPlan,
        bundle: PathBuf,
        binary: PathBuf,
        token: CancellationToken,
        lease: GpuLease,
    ) {
        let staging = bundle.join(".staging");
        let (status, error) = match execute(&plan, &staging, &binary, &token, lease) {
            Outcome::Done(manifest) => match commit(&plan, &bundle, &manifest) {
                Ok(()) => ("done", None),
                Err(err) => ("failed", Some(err)),
            },
            Outcome::Cancelled => ("cancelled", None),
            Outcome::Failed(err) => ("failed", Some(err)),
        };
        let _ = write_status(&bundle, status, error.as_deref());
        let _ = std::fs::remove_dir_all(&staging);
    }

    /// Hand the job to Blender and collect what it produced.
    fn execute(
        plan: &JobPlan,
        staging: &Path,
        binary: &Path,
        token: &CancellationToken,
        lease: GpuLease,
    ) -> Outcome {
        let job = AssetJob {
            source: AssetSource::File {
                path: plan.input.clone(),
                format: plan.format,
            },
            stages: plan.stages.clone(),
            output_dir: staging.to_path_buf(),
            prefix: plan.prefix.clone(),
            seed: plan.seed,
        };
        let files = match prepare(&job, staging) {
            Ok(files) => files,
            Err(err) => return Outcome::Failed(err.to_string()),
        };
        let mut spec = supervisor::Spec::new(
            binary,
            vec![
                "--background".to_string(),
                "--factory-startup".to_string(),
                "-noaudio".to_string(),
                "--python".to_string(),
                files.script.display().to_string(),
                "--python-exit-code".to_string(),
                "1".to_string(),
                "--".to_string(),
                files.job.display().to_string(),
                files.result.display().to_string(),
            ],
            staging,
            plan.timeout,
        );
        // The lease is held for the child's whole lifetime and released the
        // moment it is reaped (the supervisor owns the reservation).
        spec.lease = Some(lease);
        // The cancel is a STAGE-BOUNDARY cancel: the watcher drops the
        // recipe's marker file, and the supervisor's own cancel token stays
        // unset so a bake is never interrupted mid-flight.
        let watcher = CancelWatcher::spawn(token.clone(), files.cancel.clone());
        let outcome = match supervisor::run(spec) {
            Ok(outcome) => outcome,
            Err(err) => {
                watcher.stop();
                return Outcome::Failed(err.to_string());
            }
        };
        watcher.stop();
        match outcome.termination {
            Termination::Exited(0) => {
                // A truncated report cannot be parsed: the recipe's JSON is
                // cut in half. Say that instead of reporting a parse error.
                if outcome.stdout_truncated {
                    return Outcome::Failed(
                        "the recipe's report was truncated (the log cap was hit)".to_string(),
                    );
                }
                match build_manifest(&job, &files.result, &outcome.stdout) {
                    Ok(manifest) => Outcome::Done(manifest),
                    Err(AssetError::Cancelled) => Outcome::Cancelled,
                    Err(err) => Outcome::Failed(err.to_string()),
                }
            }
            Termination::Exited(code) => Outcome::Failed(format!(
                "blender exited with code {code}: {}{}",
                tail(&outcome.stderr, 20),
                if outcome.stderr_truncated {
                    " (the log was truncated)"
                } else {
                    ""
                }
            )),
            Termination::TimedOut => Outcome::Failed("the job exceeded its deadline".to_string()),
            Termination::Cancelled => Outcome::Cancelled,
            Termination::Signalled(signal) => {
                Outcome::Failed(format!("blender was killed by signal {signal}"))
            }
            Termination::RssExceeded => {
                Outcome::Failed("the job exceeded its memory cap".to_string())
            }
        }
    }

    /// Copy the artifacts into the bundle and write the manifest. The byte
    /// cap is checked BEFORE anything is copied: an oversized result never
    /// lands in the bundle.
    fn commit(plan: &JobPlan, bundle: &Path, manifest: &AssetManifest) -> Result<(), String> {
        let total: u64 = manifest
            .artifacts
            .iter()
            .map(|artifact| artifact.bytes as u64)
            .sum();
        if total > plan.max_output_bytes {
            return Err(format!(
                "the job produced {total} bytes; the cap is {}",
                plan.max_output_bytes
            ));
        }
        let mut artifacts = Vec::with_capacity(manifest.artifacts.len());
        for artifact in &manifest.artifacts {
            let name = artifact
                .path
                .file_name()
                .ok_or_else(|| "the recipe produced an artifact without a name".to_string())?
                .to_string_lossy()
                .to_string();
            // The preview is the one artifact the bundle keeps in another
            // container: Blender writes PNG, the bundle keeps webp (the
            // format every consumer reads).
            if artifact.kind == "preview" {
                let webp = to_webp(&artifact.path)?;
                let name = format!("{}.webp", name.trim_end_matches(".png"));
                crate::jobs::write_atomic(bundle, &name, &webp)?;
                artifacts.push(json!({
                    "name": name,
                    "kind": artifact.kind,
                    "bytes": webp.len(),
                    "sha256": sha256_hex(&webp),
                }));
                continue;
            }
            let bytes = crate::jobs::copy_file_atomic(bundle, &name, &artifact.path)?;
            artifacts.push(json!({
                "name": name,
                "kind": artifact.kind,
                "bytes": bytes,
                "sha256": artifact.sha256,
            }));
        }
        // The bundle manifest is path-free: the bundle's own file names,
        // never the staging paths the recipe wrote.
        let bytes = serde_json::to_vec_pretty(&manifest_record(plan, manifest, artifacts))
            .map_err(|err| format!("cannot encode the manifest: {err}"))?;
        crate::jobs::write_atomic(bundle, "manifest.json", &bytes)
    }

    /// The bundle manifest record — the shape `manifest.json` carries. This
    /// is ALSO the API contract: `honkbox_api_types::BlenderJobCompletion`
    /// mirrors it field for field, and the unit test below serializes the
    /// DTO over the same inputs and requires value equality, so the file on
    /// disk and the documented DTO cannot drift apart.
    fn manifest_record(
        plan: &JobPlan,
        manifest: &AssetManifest,
        artifacts: Vec<serde_json::Value>,
    ) -> serde_json::Value {
        json!({
            "manifest": manifest.manifest,
            "blender_version": manifest.blender_version,
            "recipe_version": manifest.recipe_version,
            "recipe_sha256": manifest.recipe_sha256,
            "job": {
                "input": {
                    "artifact_id": plan.input_id,
                    "format": plan.format,
                },
                "stages": plan.echo,
                "prefix": plan.prefix,
                "seed": plan.seed,
            },
            "artifacts": artifacts,
            "warnings": manifest.warnings,
            "progress": manifest.progress,
        })
    }

    /// The preview PNG as webp: Blender writes PNG, the bundle keeps webp.
    fn to_webp(path: &Path) -> Result<Vec<u8>, String> {
        let bytes =
            std::fs::read(path).map_err(|err| format!("cannot read {}: {err}", path.display()))?;
        let image = image::load_from_memory(&bytes)
            .map_err(|err| format!("cannot decode {}: {err}", path.display()))?;
        let rgba = image.to_rgba8();
        let mut out = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut out)
            .encode(
                rgba.as_raw(),
                rgba.width(),
                rgba.height(),
                image::ExtendedColorType::Rgba8,
            )
            .map_err(|err| format!("cannot encode {}: {err}", path.display()))?;
        Ok(out)
    }

    /// The sha256 of the bytes (hex): the digest the bundle manifest records
    /// for an artifact the server itself produced.
    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest as _, Sha256};
        use std::fmt::Write as _;

        let mut hex = String::with_capacity(64);
        for byte in Sha256::digest(bytes) {
            let _ = write!(hex, "{byte:02x}");
        }
        hex
    }

    /// The job's terminal state. The registry is a liveness view, so the
    /// outcome has to land in the bundle.
    fn write_status(bundle: &Path, status: &str, error: Option<&str>) -> Result<(), String> {
        let mut record = json!({ "status": status });
        if let Some(error) = error {
            record["error"] = json!(error);
        }
        let bytes = serde_json::to_vec_pretty(&record)
            .map_err(|err| format!("cannot encode the status: {err}"))?;
        crate::jobs::write_atomic(bundle, "status.json", &bytes)
    }

    /// The last lines of Blender's own message.
    fn tail(text: &str, lines: usize) -> String {
        let collected: Vec<&str> = text.lines().collect();
        let start = collected.len().saturating_sub(lines);
        collected[start..].join("\n")
    }

    /// Watches the job's cancel token and drops the recipe's cancel marker:
    /// the recipe checks it BETWEEN stages, so a cancel never interrupts a
    /// bake mid-flight.
    struct CancelWatcher {
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl CancelWatcher {
        fn spawn(token: CancellationToken, marker: PathBuf) -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    if token.is_cancelled() {
                        let _ = std::fs::write(&marker, b"cancel\n");
                        return;
                    }
                    std::thread::sleep(CANCEL_POLL);
                }
            });
            Self {
                stop,
                handle: Some(handle),
            }
        }

        fn stop(mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use honkbox_api_types::{
            BlenderJobCompletion, BlenderJobCompletionArtifact, BlenderJobCompletionInput,
            BlenderJobCompletionJob, BlenderJobCompletionProgress, BlenderStage,
        };
        use honkbox_external_tools::{AssetArtifact, MeshFormat, Primitive, ProgressEvent};

        use super::*;

        /// The typed completion DTO and the bundle manifest are ONE shape:
        /// the DTO serialized over the same inputs must equal, value for
        /// value, the record the writer emits. A field the writer gains or
        /// drops breaks this test instead of silently diverging from the
        /// documented API contract.
        #[test]
        fn the_completion_dto_mirrors_the_bundle_manifest() {
            let plan = JobPlan {
                stages: Vec::new(),
                echo: vec![BlenderStage::ExportGlb],
                input: PathBuf::from("unused-mesh.glb"),
                format: MeshFormat::Glb,
                input_id: "a".repeat(64),
                prefix: "output".to_owned(),
                seed: 7,
                timeout: Duration::from_secs(600),
                max_output_bytes: 1024,
            };
            let manifest = AssetManifest {
                manifest: 1,
                blender_version: "4.5.15".to_owned(),
                recipe_version: "0.3.0".to_owned(),
                recipe_sha256: "b".repeat(64),
                job: AssetJob {
                    source: AssetSource::Primitive {
                        primitive: Primitive::Cube,
                        size: 2.0,
                    },
                    stages: Vec::new(),
                    output_dir: PathBuf::from("unused-output"),
                    prefix: "output".to_owned(),
                    seed: 7,
                },
                artifacts: vec![AssetArtifact {
                    path: PathBuf::from("unused/output.glb"),
                    kind: "model".to_owned(),
                    bytes: 3,
                    sha256: "c".repeat(64),
                }],
                warnings: vec!["the recipe had a note".to_owned()],
                progress: vec![ProgressEvent {
                    stage: "export_glb".to_owned(),
                    event: "done".to_owned(),
                }],
            };
            // The artifact row exactly as commit() emits it for this
            // manifest: the bundle name, the byte count, the bundle digest.
            let artifacts = vec![json!({
                "name": "output.glb",
                "kind": "model",
                "bytes": 3,
                "sha256": "c".repeat(64),
            })];

            let dto = BlenderJobCompletion {
                manifest: manifest.manifest,
                blender_version: manifest.blender_version.clone(),
                recipe_version: manifest.recipe_version.clone(),
                recipe_sha256: manifest.recipe_sha256.clone(),
                job: BlenderJobCompletionJob {
                    input: BlenderJobCompletionInput {
                        artifact_id: plan.input_id.clone(),
                        // The writer serializes the mesh format through the
                        // same snake_case serde as the request enum.
                        format: "glb".to_owned(),
                    },
                    stages: plan.echo.clone(),
                    prefix: plan.prefix.clone(),
                    seed: plan.seed,
                },
                artifacts: vec![BlenderJobCompletionArtifact {
                    name: "output.glb".to_owned(),
                    kind: "model".to_owned(),
                    bytes: 3,
                    sha256: "c".repeat(64),
                }],
                warnings: manifest.warnings.clone(),
                progress: manifest
                    .progress
                    .iter()
                    .map(|event| BlenderJobCompletionProgress {
                        stage: event.stage.clone(),
                        event: event.event.clone(),
                    })
                    .collect(),
            };
            assert_eq!(
                serde_json::to_value(&dto).expect("the completion DTO serializes"),
                manifest_record(&plan, &manifest, artifacts)
            );
        }
    }
}
