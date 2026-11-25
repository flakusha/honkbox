//! Blender bundle publication safety: the whole bundle is staged in a fresh
//! directory, the success manifest is written only after every artifact is
//! durable, one terminal record is published atomically, and every terminal
//! failure leaves a durable failed record. No lib target on this binary
//! crate, so the modules are included directly.
//!
//! Resource contract: every test owns a unique ULID temp root under the
//! system temp dir, a unique id in the process-global job registry, and a
//! private broadcast receiver. No fixed paths, ports, or shared mutable
//! state; tests pass alone, in any order, and in parallel. `Fixture::drop`
//! reclaims the root even on panic.
#![cfg(unix)]

#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "../src/artifacts.rs"]
mod artifacts;
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/container.rs"]
mod container;
#[allow(dead_code)]
#[path = "../src/downloads.rs"]
mod downloads;
#[allow(dead_code)]
#[path = "../src/formats.rs"]
mod formats;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/jobs_registry.rs"]
mod jobs_registry;
#[allow(dead_code)]
#[path = "../src/metrics.rs"]
mod metrics;
#[allow(dead_code)]
#[path = "../src/ora.rs"]
mod ora;
#[allow(dead_code)]
#[path = "../src/raster.rs"]
mod raster;
#[allow(dead_code)]
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;

/// The module under test at a known parent: its `super::` seams resolve to
/// these test-local stand-ins instead of the full router tree.
mod harness {
    /// Only the fields publication reads.
    pub(crate) mod blender {
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::time::Duration;

        use honkbox_api_types::BlenderStage;
        use honkbox_external_tools::{AssetStage, MeshFormat};

        #[derive(Debug)]
        pub(crate) struct JobPlan {
            pub(crate) stages: Vec<AssetStage>,
            pub(crate) echo: Vec<BlenderStage>,
            pub(crate) input: PathBuf,
            pub(crate) format: MeshFormat,
            pub(crate) input_id: String,
            pub(crate) prefix: String,
            pub(crate) seed: u64,
            pub(crate) timeout: Duration,
            pub(crate) max_output_bytes: u64,
            pub(crate) artifacts: Arc<crate::artifacts::Registry>,
            pub(crate) job_id: String,
        }
    }

    /// The real MIME mapping; publication records it in each artifact row.
    pub(crate) mod job_content {
        pub(crate) fn content_type_for_name(name: &str) -> &'static str {
            match name.rsplit_once('.').map_or("", |(_, ext)| ext) {
                "glb" => "model/gltf-binary",
                "webp" => "image/webp",
                "png" => "image/png",
                "json" => "application/json",
                _ => "application/octet-stream",
            }
        }
    }

    /// `execute` is never called here; this only compiles its imports.
    pub(crate) mod supervisor {
        use std::path::Path;
        use std::time::Duration;

        use crate::admission::GpuLease;

        pub(crate) struct Spec {
            pub(crate) lease: Option<GpuLease>,
        }

        impl Spec {
            pub(crate) fn new(
                _binary: &Path,
                _args: Vec<String>,
                _cwd: &Path,
                _timeout: Duration,
            ) -> Self {
                Self { lease: None }
            }
        }

        pub(crate) enum Termination {
            Exited(i32),
            TimedOut,
            Cancelled,
            Signalled(i32),
            RssExceeded,
        }

        pub(crate) struct RunOutcome {
            pub(crate) termination: Termination,
            pub(crate) stdout: String,
            pub(crate) stderr: String,
            pub(crate) stdout_truncated: bool,
            pub(crate) stderr_truncated: bool,
        }

        pub(crate) fn run(_spec: Spec) -> Result<RunOutcome, String> {
            Err("the process supervisor is not used in publication tests".to_owned())
        }
    }

    #[path = "../../src/routes/blender_exec.rs"]
    pub(crate) mod blender_exec;
}

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use honkbox_api_types::{BlenderStage, JobEvent, JobEventType};
use honkbox_external_tools::{
    AssetArtifact, AssetJob, AssetManifest, AssetSource, MeshFormat, Primitive,
};
use serde_json::Value;

use harness::blender::JobPlan;
use harness::blender_exec::test_support::{Outcome, PublishStep, publish};
use jobs_registry::JobKind;

/// One test-scoped root, unique per run (tests share no bundle path).
fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("honkbox-terminal-{name}-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&dir).expect("test temp dir");
    dir
}

/// The artifact files a finished recipe leaves in its staging directory.
fn stage_outputs(root: &Path, rows: &[(&str, &[u8])]) -> Vec<AssetArtifact> {
    let dir = root.join("recipe");
    std::fs::create_dir_all(&dir).expect("recipe dir");
    rows.iter()
        .map(|(name, bytes)| {
            std::fs::write(dir.join(name), bytes).expect("staged artifact");
            AssetArtifact {
                path: dir.join(name),
                kind: if name.ends_with(".glb") {
                    "model"
                } else {
                    "map"
                }
                .to_owned(),
                bytes: bytes.len(),
                sha256: "c".repeat(64),
            }
        })
        .collect()
}

fn manifest(artifacts: Vec<AssetArtifact>) -> AssetManifest {
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
            output_dir: PathBuf::from("unused-recipe-output"),
            prefix: "output".to_owned(),
            seed: 7,
        },
        artifacts,
        warnings: Vec::new(),
        progress: Vec::new(),
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("read json")).expect("parse json")
}

fn artifact_rows(bundle: &Path) -> Vec<Value> {
    let record = read_json(&bundle.join("manifest.json"));
    record["artifacts"]
        .as_array()
        .expect("manifest artifacts")
        .clone()
}

/// A bundle path plus a live registry id, so result frames can be attributed
/// to exactly this run even when tests execute in parallel.
struct Fixture {
    root: PathBuf,
    bundle: PathBuf,
    plan: JobPlan,
    guard: jobs_registry::JobGuard,
    receiver: tokio::sync::broadcast::Receiver<JobEvent>,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = tmp_dir(name);
        let bundle = root.join("jobs").join("bundle");
        std::fs::create_dir_all(&bundle).expect("bundle dir");
        let registry = jobs_registry::global();
        let receiver = registry.subscribe();
        let guard = registry.register(JobKind::Blender, None, true);
        let plan = JobPlan {
            stages: Vec::new(),
            echo: vec![BlenderStage::ExportGlb],
            input: root.join("unused-mesh.glb"),
            format: MeshFormat::Glb,
            input_id: "a".repeat(64),
            prefix: "output".to_owned(),
            seed: 7,
            timeout: Duration::from_secs(600),
            max_output_bytes: 1024 * 1024,
            artifacts: Arc::new(artifacts::Registry::new(Some(root.clone()), 100_000)),
            job_id: guard.id().to_owned(),
        };
        Self {
            root,
            bundle,
            plan,
            guard,
            receiver,
        }
    }

    fn result_frames(&mut self) -> Vec<JobEvent> {
        let id = self.guard.id().to_owned();
        std::iter::from_fn(|| self.receiver.try_recv().ok())
            .filter(|event| event.event == JobEventType::Result && event.id == id)
            .collect()
    }
}

// Each fixture owns a unique ULID temp root, a unique registry job id and a
// private event receiver; nothing is shared or order-dependent. Drop removes
// the root even when an assertion panics, so a failed test leaves no residue.
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A fault at ANY promotion step leaves no `done` record, no success
/// manifest, no artifacts (old or new), and no completion frames.
#[test]
fn faults_at_each_promotion_step_never_publish_success() {
    let steps = [
        PublishStep::Artifact(0),
        PublishStep::Artifact(1),
        PublishStep::Manifest,
        PublishStep::Terminal,
        PublishStep::Publish,
    ];
    for step in steps {
        let mut fx = Fixture::new("fault");
        // A prior run's success metadata must not survive reconciliation.
        std::fs::write(
            fx.bundle.join("manifest.json"),
            br#"{"artifacts":[{"name":"stale.glb"}]}"#,
        )
        .expect("stale manifest");
        std::fs::write(fx.bundle.join("stale.glb"), b"stale").expect("stale artifact");
        let staged = stage_outputs(&fx.root, &[("a.glb", &b"aaa"[..]), ("b.png", &b"bbb"[..])]);
        publish(
            &fx.plan,
            &fx.bundle,
            Outcome::Done(manifest(staged)),
            |hit| {
                if hit == step {
                    Err(format!("injected fault at {hit:?}"))
                } else {
                    Ok(())
                }
            },
        );

        let terminal = read_json(&fx.bundle.join("status.json"));
        assert_ne!(terminal["status"], "done", "fault at {step:?}");
        assert_eq!(terminal["status"], "failed", "fault at {step:?}");
        assert!(
            !fx.bundle.join("manifest.json").exists(),
            "fault at {step:?}"
        );
        assert!(!fx.bundle.join("stale.glb").exists(), "fault at {step:?}");
        assert!(!fx.bundle.join("a.glb").exists(), "fault at {step:?}");
        assert!(!fx.bundle.join("b.png").exists(), "fault at {step:?}");
        assert!(fx.result_frames().is_empty(), "fault at {step:?}");
        assert!(
            std::fs::read_dir(fx.root.join("jobs"))
                .expect("jobs root")
                .all(|entry| !entry
                    .expect("job entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".publish-")),
            "the fresh staging dir must be swept on every path"
        );
    }
}

/// A reader cannot see a success manifest without its artifacts: artifacts
/// and manifest appear in ONE swap, the terminal record is the last write,
/// and no result frame precedes either.
#[test]
fn success_manifest_never_precedes_its_artifacts() {
    let mut fx = Fixture::new("reader");
    let id = fx.guard.id().to_owned();
    let mut watcher = jobs_registry::global().subscribe();
    let staged = stage_outputs(
        &fx.root,
        &[("output.glb", &b"glb"[..]), ("map.png", &b"png"[..])],
    );
    let bundle = fx.bundle.clone();
    let mut published_together = false;
    let mut terminal_last = false;
    publish(
        &fx.plan,
        &fx.bundle,
        Outcome::Done(manifest(staged)),
        |step| {
            let no_frames = !std::iter::from_fn(|| watcher.try_recv().ok())
                .any(|event| event.event == JobEventType::Result && event.id == id);
            match step {
                PublishStep::Publish => {
                    // If the manifest is already visible, every artifact it
                    // names must be there too: both land in the same swap.
                    let complete = if bundle.join("manifest.json").exists() {
                        let record: Value = serde_json::from_slice(
                            &std::fs::read(bundle.join("manifest.json")).expect("manifest"),
                        )
                        .expect("manifest json");
                        record["artifacts"]
                            .as_array()
                            .expect("artifacts")
                            .iter()
                            .all(|row| bundle.join(row["name"].as_str().expect("name")).is_file())
                    } else {
                        true
                    };
                    published_together =
                        complete && !bundle.join("status.json").exists() && no_frames;
                }
                PublishStep::Terminal => {
                    terminal_last = !bundle.join("status.json").exists() && no_frames;
                }
                PublishStep::Artifact(_) | PublishStep::Manifest => {}
            }
            Ok(())
        },
    );
    assert!(
        published_together,
        "artifacts and manifest publish together"
    );
    assert!(terminal_last, "the terminal record is the last write");

    assert_eq!(read_json(&fx.bundle.join("status.json"))["status"], "done");
    let rows = artifact_rows(&fx.bundle);
    assert_eq!(rows.len(), 2, "manifest lists every artifact");
    for row in &rows {
        let name = row["name"].as_str().expect("artifact name");
        let path = fx.bundle.join(name);
        assert!(path.is_file(), "manifest names a missing artifact: {name}");
        assert_eq!(
            std::fs::metadata(&path).expect("artifact metadata").len(),
            row["bytes"].as_u64().expect("artifact bytes")
        );
        assert_eq!(row["sha256"].as_str().expect("sha256").len(), 64);
        assert_eq!(row["artifact_id"].as_str().expect("artifact id").len(), 64);
    }
    assert_eq!(fx.result_frames().len(), 2);
}

/// No result frame is observable at ANY promotion checkpoint, and a fault
/// at each one leaves only a failed terminal record — never success.
#[test]
fn no_result_frame_precedes_the_durable_terminal_record() {
    for step in [
        PublishStep::Artifact(0),
        PublishStep::Manifest,
        PublishStep::Publish,
        PublishStep::Terminal,
    ] {
        let mut fx = Fixture::new("no-early-result");
        let id = fx.guard.id().to_owned();
        let mut watcher = jobs_registry::global().subscribe();
        let staged = stage_outputs(&fx.root, &[("output.glb", &b"glb"[..])]);
        publish(
            &fx.plan,
            &fx.bundle,
            Outcome::Done(manifest(staged)),
            |hit| {
                assert!(
                    !std::iter::from_fn(|| watcher.try_recv().ok())
                        .any(|event| event.event == JobEventType::Result && event.id == id),
                    "result frame published at {hit:?}, before the terminal record"
                );
                if hit == step {
                    Err(format!("injected fault at {hit:?}"))
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(
            read_json(&fx.bundle.join("status.json"))["status"],
            "failed",
            "fault at {step:?}"
        );
        assert!(
            !fx.bundle.join("manifest.json").exists(),
            "fault at {step:?}"
        );
        assert!(fx.result_frames().is_empty(), "fault at {step:?}");
    }
}

/// Re-running on the same bundle path never exposes stale success metadata,
/// whether the rerun succeeds or fails at a promotion step.
#[test]
fn rerun_reconciles_stale_success_metadata() {
    let mut fx = Fixture::new("rerun");
    let first = stage_outputs(&fx.root, &[("output.glb", &b"old"[..])]);
    publish(&fx.plan, &fx.bundle, Outcome::Done(manifest(first)), |_| {
        Ok(())
    });
    assert!(fx.bundle.join("output.glb").is_file());
    let _ = fx.result_frames(); // drain the first run

    let second = stage_outputs(&fx.root, &[("final.glb", &b"new"[..])]);
    publish(
        &fx.plan,
        &fx.bundle,
        Outcome::Done(manifest(second)),
        |_| Ok(()),
    );
    assert!(!fx.bundle.join("output.glb").exists());
    let rows = artifact_rows(&fx.bundle);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "final.glb");
    let _ = fx.result_frames(); // drain the second run

    let third = stage_outputs(&fx.root, &[("partial.glb", &b"partial"[..])]);
    publish(
        &fx.plan,
        &fx.bundle,
        Outcome::Done(manifest(third)),
        |step| {
            if step == PublishStep::Artifact(0) {
                Err("injected fault".to_owned())
            } else {
                Ok(())
            }
        },
    );
    assert_eq!(
        read_json(&fx.bundle.join("status.json"))["status"],
        "failed"
    );
    assert!(!fx.bundle.join("manifest.json").exists());
    assert!(!fx.bundle.join("final.glb").exists());
    assert!(!fx.bundle.join("partial.glb").exists());
    assert!(fx.result_frames().is_empty());
}

/// Completion events agree with the terminal record: result frames exist
/// only for `done`, and their ids/hashes/sizes match the published manifest.
#[test]
fn completion_events_agree_with_terminal_record() {
    let mut fx = Fixture::new("events-done");
    let staged = stage_outputs(&fx.root, &[("a.glb", &b"aaa"[..]), ("b.png", &b"bbb"[..])]);
    publish(
        &fx.plan,
        &fx.bundle,
        Outcome::Done(manifest(staged)),
        |_| Ok(()),
    );
    assert_eq!(read_json(&fx.bundle.join("status.json"))["status"], "done");
    let rows = artifact_rows(&fx.bundle);
    let frames = fx.result_frames();
    assert_eq!(frames.len(), rows.len());
    for (frame, row) in frames.iter().zip(rows.iter()) {
        assert_eq!(frame.artifact_name.as_deref(), row["name"].as_str());
        assert_eq!(frame.artifact_id.as_deref(), row["artifact_id"].as_str());
        assert_eq!(frame.sha256.as_deref(), row["sha256"].as_str());
        assert_eq!(frame.size_bytes, row["bytes"].as_u64());
    }

    let mut fx = Fixture::new("events-failed");
    publish(
        &fx.plan,
        &fx.bundle,
        Outcome::Failed("boom".to_owned()),
        |_| Ok(()),
    );
    let terminal = read_json(&fx.bundle.join("status.json"));
    assert_eq!(terminal["status"], "failed");
    assert_eq!(terminal["error"], "boom");
    assert!(!fx.bundle.join("manifest.json").exists());
    assert!(fx.result_frames().is_empty());

    let mut fx = Fixture::new("events-cancelled");
    publish(&fx.plan, &fx.bundle, Outcome::Cancelled, |_| Ok(()));
    assert_eq!(
        read_json(&fx.bundle.join("status.json"))["status"],
        "cancelled"
    );
    assert!(!fx.bundle.join("manifest.json").exists());
    assert!(fx.result_frames().is_empty());
}
