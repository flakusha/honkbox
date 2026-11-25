//! The supervised Blender run behind `POST /v1/jobs/blender`.
//!
//! The route validated the request; this module owns the PROCESS: the
//! recipe files are written into the job's staging directory, Blender runs
//! under the process supervisor (its own process group, a wall-clock bound,
//! a group-wide teardown) and the whole output bundle is assembled in a
//! FRESH directory via `.part` + fsync + rename. One directory swap publishes
//! every artifact and `manifest.json` together; the terminal record is then
//! written LAST into the bundle with the same atomic helper, so `done` can
//! never appear without a complete manifest behind it.
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
    use super::super::job_content::content_type_for_name;
    use super::super::supervisor::{self, Termination};

    /// How often the cancel watcher checks the job's token.
    const CANCEL_POLL: Duration = Duration::from_millis(100);

    /// What one run produced.
    #[allow(clippy::large_enum_variant)] // the manifest is the normal outcome
    pub(crate) enum Outcome {
        /// The recipe finished; the manifest describes what it wrote.
        Done(AssetManifest),
        /// The job was cancelled at a stage boundary.
        Cancelled,
        /// The job failed; the message says why.
        Failed(String),
    }

    /// Run one job: the recipe files, the supervised process, and the
    /// bundle the artifacts land in.
    #[allow(clippy::needless_pass_by_value)] // owned inputs cross the spawned-run boundary
    pub(super) fn run(
        plan: JobPlan,
        bundle: PathBuf,
        binary: PathBuf,
        token: CancellationToken,
        lease: GpuLease,
    ) {
        let staging = bundle.join(".staging");
        // Reconcile at exec start: a reused bundle must not keep an old
        // manifest, terminal record or artifact beside the new outputs.
        if let Err(err) = clear_bundle(&bundle).and_then(|()| {
            std::fs::create_dir_all(&bundle)
                .map_err(|e| format!("cannot create bundle {}: {e}", bundle.display()))
        }) {
            publish(&plan, &bundle, Outcome::Failed(err), |_| Ok(()));
            return;
        }
        let outcome = execute(&plan, &staging, &binary, &token, lease);
        // Publication owns the terminal record: every outcome leaves one,
        // and the terminal record is the last file written.
        publish(&plan, &bundle, outcome, |_| Ok(()));
        // Publication only returns once the terminal record is durable.
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
                return Outcome::Failed(err.clone());
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

    /// One ordered promotion checkpoint. Production passes a no-op; the
    /// integration tests fail at each step to prove no success can leak.
    #[allow(dead_code)] // variants are constructed by fault-injection tests
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum PublishStep {
        /// Before promoting artifact `index`.
        Artifact(usize),
        /// Before writing the success manifest.
        Manifest,
        /// Before writing the terminal record into the staged bundle.
        Terminal,
        /// Before the directory swap that publishes the bundle.
        Publish,
    }

    /// One artifact's registry identity, retained until the whole bundle is
    /// durable: `job_result` must never precede the terminal commit.
    struct Promoted {
        id: String,
        sha256: String,
        size: u64,
        name: String,
        kind: String,
        index: u32,
        primary: bool,
    }

    /// Reconcile a bundle before reuse: no previous manifest, terminal
    /// record or artifacts survive into the new run.
    fn clear_bundle(bundle: &Path) -> Result<(), String> {
        match std::fs::symlink_metadata(bundle) {
            Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(bundle)
                .map_err(|e| format!("cannot clear bundle {}: {e}", bundle.display())),
            Ok(_) => std::fs::remove_file(bundle)
                .map_err(|e| format!("cannot clear bundle {}: {e}", bundle.display())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(format!("cannot inspect bundle {}: {err}", bundle.display())),
        }
    }

    /// Publish a fully staged bundle in one rename: artifacts and manifest
    /// become visible together. The caller writes the terminal record into
    /// the bundle LAST, so `done` never outruns a complete manifest.
    fn publish_bundle(staged: &Path, bundle: &Path) -> Result<(), String> {
        clear_bundle(bundle)?;
        if let Some(parent) = bundle.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create bundle parent {}: {e}", parent.display()))?;
        }
        sync_dir(staged);
        std::fs::rename(staged, bundle)
            .map_err(|e| format!("cannot publish bundle {}: {e}", bundle.display()))?;
        if let Some(parent) = bundle.parent() {
            sync_dir(parent);
        }
        Ok(())
    }

    /// Best-effort fsync of directory entries, matching the file fsync of
    /// `jobs::write_atomic`.
    fn sync_dir(dir: &Path) {
        if let Ok(file) = std::fs::File::open(dir) {
            let _ = file.sync_all();
        }
    }

    /// A fresh, never-reused staging directory beside the bundle (same
    /// filesystem, so the final publish is one rename).
    fn fresh_stage_dir(bundle: &Path) -> PathBuf {
        bundle.with_file_name(format!(".publish-{}", ulid::Ulid::new()))
    }

    /// Commit one terminal outcome. Success is staged in a FRESH directory
    /// and published by a single rename; every failure leaves only a failed
    /// (or cancelled) terminal record, never a success manifest.
    pub(crate) fn publish(
        plan: &JobPlan,
        bundle: &Path,
        outcome: Outcome,
        mut checkpoint: impl FnMut(PublishStep) -> Result<(), String>,
    ) {
        let staged = fresh_stage_dir(bundle);
        // A background task has no caller to return to: a failed terminal
        // write is reported, never silently dropped. If the failed record
        // cannot be written either, storage itself is unwritable and no
        // terminal record can exist.
        let terminal = match outcome {
            Outcome::Done(manifest) => {
                match stage_and_publish(plan, &staged, bundle, &manifest, &mut checkpoint) {
                    Ok(()) => Ok(()),
                    Err(err) => publish_terminal(bundle, "failed", Some(&err)),
                }
            }
            Outcome::Cancelled => publish_terminal(bundle, "cancelled", None),
            Outcome::Failed(err) => publish_terminal(bundle, "failed", Some(&err)),
        };
        if let Err(err) = terminal {
            tracing::warn!(error = %err, "cannot write the job terminal record");
        }
        // On success the rename consumed `staged`; on every other path it
        // holds partial outputs that must never be mistaken for a bundle.
        let _ = std::fs::remove_dir_all(&staged);
    }

    /// Promote every artifact and then the manifest into a staged bundle,
    /// swap the whole directory into place, then write the terminal record
    /// LAST. The byte cap is checked BEFORE anything is staged: an oversized
    /// result never lands in the bundle.
    fn stage_and_publish(
        plan: &JobPlan,
        staged: &Path,
        bundle: &Path,
        manifest: &AssetManifest,
        checkpoint: &mut impl FnMut(PublishStep) -> Result<(), String>,
    ) -> Result<(), String> {
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
        std::fs::create_dir_all(staged)
            .map_err(|e| format!("cannot create staging dir {}: {e}", staged.display()))?;
        // One winner for the whole job, chosen before any frame is emitted:
        // the first `model`, falling back to index 0 when a recipe has none.
        let primary_index = manifest
            .artifacts
            .iter()
            .position(|artifact| artifact.kind == "model")
            .unwrap_or(0);
        let mut promoted = Vec::with_capacity(manifest.artifacts.len());
        let mut rows = Vec::with_capacity(manifest.artifacts.len());
        for (index, artifact) in manifest.artifacts.iter().enumerate() {
            checkpoint(PublishStep::Artifact(index))?;
            let name = artifact
                .path
                .file_name()
                .ok_or_else(|| "the recipe produced an artifact without a name".to_string())?
                .to_string_lossy()
                .to_string();
            // The preview is the one artifact the bundle keeps in another
            // container: Blender writes PNG, the bundle keeps webp (the
            // format every consumer reads). Every output is ALSO promoted
            // into the content-addressed registry; its digest is the id the
            // manifest and the job_result frame carry.
            let (name, bytes, content_type) = if artifact.kind == "preview" {
                let webp = to_webp(&artifact.path)?;
                let name = format!("{}.webp", name.trim_end_matches(".png"));
                crate::jobs::write_atomic(staged, &name, &webp)?;
                (name, webp, "image/webp")
            } else {
                crate::jobs::copy_file_atomic(staged, &name, &artifact.path)?;
                let bytes = std::fs::read(staged.join(&name))
                    .map_err(|err| format!("cannot read committed {name}: {err}"))?;
                let content_type = content_type_for_name(&name);
                (name, bytes, content_type)
            };
            let info = plan
                .artifacts
                .put_bytes(&bytes, content_type)
                .map_err(|err| format!("cannot promote {name}: {err}"))?;
            promoted.push(Promoted {
                id: info.id.clone(),
                sha256: info.sha256.clone(),
                size: info.size_bytes,
                name: name.clone(),
                kind: artifact.kind.clone(),
                index: u32::try_from(index).unwrap_or(u32::MAX),
                primary: index == primary_index,
            });
            rows.push(json!({
                "name": name,
                "kind": artifact.kind,
                "bytes": bytes.len(),
                "sha256": info.sha256,
                "artifact_id": info.id,
            }));
        }
        // The bundle manifest is path-free: the bundle's own file names,
        // never the staging paths the recipe wrote. It is written after
        // every artifact is durable, so a partial stage carries no manifest.
        checkpoint(PublishStep::Manifest)?;
        let bytes = serde_json::to_vec_pretty(&manifest_record(plan, manifest, &rows))
            .map_err(|err| format!("cannot encode the manifest: {err}"))?;
        crate::jobs::write_atomic(staged, "manifest.json", &bytes)?;
        checkpoint(PublishStep::Publish)?;
        // Artifacts and manifest become visible together in one rename.
        publish_bundle(staged, bundle)?;
        // The terminal record is the universal commit marker and the LAST
        // write, so `done` can never outrun a complete manifest. Its error
        // propagates: a terminal failure then gets its own failed record.
        checkpoint(PublishStep::Terminal)?;
        write_terminal(bundle, "done", None)?;
        // Only now, with the whole bundle durable, are artifacts announced.
        for row in &promoted {
            crate::jobs_registry::global().emit_result(
                &plan.job_id,
                &row.id,
                &row.sha256,
                row.size,
                crate::jobs_registry::ArtifactFrameMeta {
                    name: &row.name,
                    kind: &row.kind,
                    index: row.index,
                    primary: row.primary,
                },
            );
        }
        Ok(())
    }

    /// The bundle manifest record — the shape `manifest.json` carries. This
    /// is ALSO the API contract: `honkbox_api_types::BlenderJobCompletion`
    /// mirrors it field for field, and the unit test below serializes the
    /// DTO over the same inputs and requires value equality, so the file on
    /// disk and the documented DTO cannot drift apart.
    fn manifest_record(
        plan: &JobPlan,
        manifest: &AssetManifest,
        artifacts: &[serde_json::Value],
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

    /// Write the universal terminal record via the shared atomic helper.
    /// This is the LAST write on every path; its error propagates to the
    /// publication caller.
    fn write_terminal(bundle: &Path, status: &str, error: Option<&str>) -> Result<(), String> {
        let mut record = json!({ "status": status });
        if let Some(error) = error {
            record["error"] = json!(error);
        }
        let bytes = serde_json::to_vec_pretty(&record)
            .map_err(|err| format!("cannot encode the terminal record: {err}"))?;
        crate::jobs::write_atomic(bundle, "status.json", &bytes)
    }

    /// Reconcile the bundle and leave ONLY a durable terminal record. A
    /// previous manifest and its artifacts are removed first, so a failed
    /// rerun cannot expose stale success metadata.
    fn publish_terminal(bundle: &Path, status: &str, error: Option<&str>) -> Result<(), String> {
        clear_bundle(bundle)?;
        std::fs::create_dir_all(bundle)
            .map_err(|err| format!("cannot create bundle {}: {err}", bundle.display()))?;
        write_terminal(bundle, status, error)
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
            BlenderJobCompletionJob, BlenderJobCompletionProgress, BlenderStage, JobEventType,
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
                artifacts: Arc::new(crate::artifacts::Registry::new(None, 1024)),
                job_id: "01J00000000000000000000000".to_owned(),
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
            // The artifact row exactly as stage_and_publish() emits it:
            // the bundle name, the byte count, the bundle digest.
            let artifacts = vec![json!({
                "name": "output.glb",
                "kind": "model",
                "bytes": 3,
                "sha256": "c".repeat(64),
                "artifact_id": "d".repeat(64),
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
                    artifact_id: "d".repeat(64),
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
                manifest_record(&plan, &manifest, &artifacts)
            );
        }

        /// Reclaims a unique test root even when an assertion panics.
        struct TempRoot(PathBuf);

        impl Drop for TempRoot {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        fn test_manifest(artifacts: Vec<AssetArtifact>) -> AssetManifest {
            AssetManifest {
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
                artifacts,
                warnings: Vec::new(),
                progress: Vec::new(),
            }
        }

        /// A real multi-artifact bundle: each committed output emits one
        /// result frame AFTER the whole bundle is durable, the first
        /// `model` is the only primary, and the live snapshot agrees with
        /// the published manifest and terminal record.
        #[test]
        fn multi_artifact_commit_emits_one_primary_matching_manifest() {
            let root =
                std::env::temp_dir().join(format!("honkbox-blender-meta-{}", ulid::Ulid::new()));
            let _root_cleanup = TempRoot(root.clone());
            let staging = root.join("staging");
            let bundle = root.join("bundle");
            std::fs::create_dir_all(&staging).expect("staging dir");
            std::fs::create_dir_all(&bundle).expect("bundle dir");

            let rows = [
                ("map_diffuse.png", "map", &b"map-bytes"[..]),
                ("output.glb", "model", &b"glb-bytes"[..]),
                ("map_normal.png", "map", &b"normal-bytes"[..]),
            ];
            let artifacts: Vec<AssetArtifact> = rows
                .iter()
                .map(|(name, kind, bytes)| {
                    std::fs::write(staging.join(name), bytes).expect("staged artifact");
                    AssetArtifact {
                        path: staging.join(name),
                        kind: (*kind).to_owned(),
                        bytes: bytes.len(),
                        sha256: "c".repeat(64),
                    }
                })
                .collect();
            let manifest = test_manifest(artifacts);

            let registry = crate::jobs_registry::global();
            let mut receiver = registry.subscribe();
            let guard = registry.register(crate::jobs_registry::JobKind::Blender, None, false);
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
                artifacts: Arc::new(crate::artifacts::Registry::new(Some(root.clone()), 10_000)),
                job_id: guard.id().to_owned(),
            };
            publish(&plan, &bundle, Outcome::Done(manifest), |_| Ok(()));

            let terminal: serde_json::Value = serde_json::from_slice(
                &std::fs::read(bundle.join("status.json")).expect("terminal record"),
            )
            .expect("terminal json");
            assert_eq!(terminal["status"], "done");

            let mut frames: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok())
                .filter(|event| event.event == JobEventType::Result && event.id == guard.id())
                .collect();
            frames.sort_by_key(|frame| frame.artifact_index.unwrap_or(u32::MAX));
            assert_eq!(frames.len(), 3, "one frame per manifest artifact");
            let expected = [
                ("map_diffuse.png", "map", 0, false),
                ("output.glb", "model", 1, true),
                ("map_normal.png", "map", 2, false),
            ];
            for (frame, (name, kind, index, primary)) in frames.iter().zip(expected) {
                assert_eq!(frame.artifact_name.as_deref(), Some(name));
                assert_eq!(frame.artifact_kind.as_deref(), Some(kind));
                assert_eq!(frame.artifact_index, Some(index));
                assert_eq!(frame.artifact_primary, Some(primary));
            }
            assert_eq!(
                frames
                    .iter()
                    .filter(|f| f.artifact_primary == Some(true))
                    .count(),
                1,
                "exactly one primary"
            );

            let recorded: serde_json::Value = serde_json::from_slice(
                &std::fs::read(bundle.join("manifest.json")).expect("manifest.json"),
            )
            .expect("manifest json");
            let manifest_rows = recorded["artifacts"].as_array().expect("artifacts");
            for (frame, row) in frames.iter().zip(manifest_rows) {
                assert_eq!(frame.artifact_id.as_deref(), row["artifact_id"].as_str());
                assert_eq!(frame.artifact_name.as_deref(), row["name"].as_str());
                assert_eq!(frame.artifact_kind.as_deref(), row["kind"].as_str());
            }
            let refs = registry
                .list()
                .into_iter()
                .find(|row| row.id == guard.id())
                .and_then(|row| row.artifacts)
                .expect("live snapshot");
            assert_eq!(refs.len(), 3);
            assert_eq!(refs.iter().filter(|row| row.primary).count(), 1);
            assert_eq!(refs[1].name, "output.glb");
            assert_eq!(
                refs[1].artifact_id,
                frames[1].artifact_id.as_deref().unwrap()
            );

            drop(guard);
        }
    }
}

/// Test-only access to the publication seam. The integration test drives
/// the exact production path with fault injection at each promotion step.
#[cfg(all(test, unix))]
pub(crate) mod test_support {
    // Used by the integration-test binary; the unit-test build reaches
    // these items through `super::*` instead.
    #[allow(unused_imports)]
    pub(crate) use super::unix::{Outcome, PublishStep, publish};
}
