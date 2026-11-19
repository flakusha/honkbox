//! Blender asset jobs (`FEAT-blender-asset-preflight`).
//!
//! A typed job description is handed to a headless Blender process:
//!
//! ```text
//! blender --background --factory-startup -noaudio \\
//!     --python <recipe.py> --python-exit-code 1 -- <job.json> <result.json>
//! ```
//!
//! The recipe (embedded in this binary, see [`RECIPE`]) applies the requested
//! stages, writes the artifacts into the job's output directory and reports
//! progress as `HONKBOX_PROGRESS={json}` lines on stdout.
//!
//! The manifest is honest by construction: it records the Blender version the
//! recipe reported, the recipe version, the sha256 of the recipe source and the
//! sha256 of every artifact. A failed job carries Blender's own message; a
//! missing Blender is [`AssetError::NotInstalled`].
//!
//! Cancel lands on a stage boundary, never mid-bake: the runner drops a marker
//! file, the recipe only looks for it between stages, so a bake in flight is
//! never interrupted half-written. The job's own deadline is the only thing
//! that kills the process.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The recipe script, embedded so a job needs no script file on disk.
const RECIPE: &str = include_str!("blender_asset.py");

/// The recipe version recorded in every manifest.
pub const RECIPE_VERSION: &str = "1";

/// The manifest schema version.
const MANIFEST_VERSION: u32 = 1;

/// The marker the recipe prints before every progress event.
const PROGRESS_PREFIX: &str = "HONKBOX_PROGRESS=";

/// Where the mesh of an asset job comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssetSource {
    /// A generated primitive: no input file, the fixture path.
    Primitive {
        /// Which primitive to generate.
        primitive: Primitive,
        /// Its size in Blender units.
        size: f64,
    },
    /// A mesh file on disk.
    File {
        /// The mesh to import.
        path: PathBuf,
        /// The format the file is in.
        format: MeshFormat,
    },
}

/// The primitives an asset job can start from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primitive {
    /// A cube.
    Cube,
    /// A UV sphere.
    Sphere,
    /// An icosphere.
    IcoSphere,
}

/// The mesh formats an asset job can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshFormat {
    /// Binary glTF.
    Glb,
    /// Wavefront OBJ.
    Obj,
    /// Stanford PLY.
    Ply,
}

/// One stage of an asset job, in execution order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AssetStage {
    /// Smart UV project.
    UvUnpack {
        /// The island margin in Blender units.
        margin: f64,
    },
    /// Pack the file-backed images into the blend.
    UvPack,
    /// Voxel remesh (deterministic: the recipe uses no random seed).
    VoxelRemesh {
        /// The voxel size in Blender units.
        voxel_size: f64,
    },
    /// Quadriflow retopology (only where the Blender build has the operator).
    Quadriflow {
        /// The target face count.
        target_faces: u64,
    },
    /// Attach a texture through a `ShaderNodeTexImage`.
    AttachTexture {
        /// The image to attach.
        image: PathBuf,
        /// The node name to create.
        node: String,
    },
    /// Bake maps to PNG files.
    Bake {
        /// The maps to bake.
        maps: Vec<BakeMap>,
        /// The square resolution of the baked maps.
        resolution: u32,
    },
    /// Export the scene as a binary glTF.
    ExportGlb,
}

/// The maps a bake can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BakeMap {
    /// The diffuse colour map.
    Color,
    /// The tangent-space normal map.
    Normal,
    /// The roughness map.
    Roughness,
}

/// One Blender asset job: where the mesh comes from and what to do to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetJob {
    /// Where the mesh comes from.
    pub source: AssetSource,
    /// The stages, in execution order.
    pub stages: Vec<AssetStage>,
    /// Where the artifacts land (created when missing).
    pub output_dir: PathBuf,
    /// The artifact name prefix.
    pub prefix: String,
    /// The seed the job was planned with (recorded in the manifest).
    pub seed: u64,
}

/// One progress event the recipe reported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// The stage the event belongs to.
    pub stage: String,
    /// What happened: `start`, `done` or `cancelled`.
    pub event: String,
}

/// One artifact the recipe produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetArtifact {
    /// Where the artifact landed.
    pub path: PathBuf,
    /// What it is (`map` or `model`).
    pub kind: String,
    /// Its size in bytes.
    pub bytes: usize,
    /// Its sha256.
    pub sha256: String,
}

/// What one asset job produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetManifest {
    /// The manifest schema version.
    pub manifest: u32,
    /// The Blender version that ran the job.
    pub blender_version: String,
    /// The recipe version that ran the job.
    pub recipe_version: String,
    /// The sha256 of the recipe source that ran the job.
    pub recipe_sha256: String,
    /// The job, echoed back.
    pub job: AssetJob,
    /// The produced artifacts.
    pub artifacts: Vec<AssetArtifact>,
    /// Non-fatal notes from the recipe.
    pub warnings: Vec<String>,
    /// The progress events the recipe reported.
    pub progress: Vec<ProgressEvent>,
}

/// Why an asset job did not finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetError {
    /// Blender is not installed (or not where it was pointed).
    NotInstalled,
    /// The input mesh does not exist.
    InputMissing(String),
    /// The job description cannot be handed to the recipe.
    InvalidJob(String),
    /// The job outlived its deadline and was killed.
    Timeout(Duration),
    /// The job was cancelled at a stage boundary.
    Cancelled,
    /// Blender failed; the reason is Blender's own message.
    Failed(String),
    /// Blender finished without writing a usable result.
    NoResult(String),
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled => f.write_str("blender is not installed"),
            Self::InputMissing(path) => write!(f, "input mesh not found: {path}"),
            Self::InvalidJob(reason) => write!(f, "invalid job: {reason}"),
            Self::Timeout(limit) => write!(f, "the job exceeded its {limit:?} deadline"),
            Self::Cancelled => f.write_str("the job was cancelled"),
            Self::Failed(reason) => write!(f, "blender failed: {reason}"),
            Self::NoResult(reason) => write!(f, "no result: {reason}"),
        }
    }
}

impl std::error::Error for AssetError {}

/// Run one asset job in a headless Blender.
///
/// The Blender process is killed when the job outlives `timeout`; a cancel
/// request is delivered as a marker file and takes effect at the next stage
/// boundary. The manifest is written to `<output_dir>/manifest.json`.
pub fn run(
    job: &AssetJob,
    blender_bin: &Path,
    timeout: Duration,
    cancel: &AtomicBool,
) -> Result<AssetManifest, AssetError> {
    if let AssetSource::File { path, .. } = &job.source
        && !path.exists()
    {
        return Err(AssetError::InputMissing(path.display().to_string()));
    }
    std::fs::create_dir_all(&job.output_dir).map_err(|e| {
        AssetError::Failed(format!("cannot create {}: {e}", job.output_dir.display()))
    })?;

    let scratch = scratch_dir()?;
    let outcome = run_recipe(job, blender_bin, timeout, cancel, &scratch);
    let _ = std::fs::remove_dir_all(&scratch);
    let manifest = outcome?;
    write_manifest(&job.output_dir, &manifest)?;
    Ok(manifest)
}

/// Hand the job to Blender and collect what it produced.
fn run_recipe(
    job: &AssetJob,
    blender_bin: &Path,
    timeout: Duration,
    cancel: &AtomicBool,
    scratch: &Path,
) -> Result<AssetManifest, AssetError> {
    let recipe_path = scratch.join("blender_asset.py");
    std::fs::write(&recipe_path, RECIPE)
        .map_err(|e| AssetError::Failed(format!("cannot write the recipe: {e}")))?;
    let result_path = scratch.join("result.json");
    let cancel_path = scratch.join("cancel");
    let job_path = scratch.join("job.json");

    let recipe_job = RecipeJob {
        source: &job.source,
        stages: &job.stages,
        output_dir: &job.output_dir,
        prefix: &job.prefix,
        result_path: &result_path,
        cancel_path: &cancel_path,
    };
    let job_bytes =
        serde_json::to_vec(&recipe_job).map_err(|e| AssetError::InvalidJob(e.to_string()))?;
    std::fs::write(&job_path, job_bytes)
        .map_err(|e| AssetError::Failed(format!("cannot write the job: {e}")))?;

    let mut child = Command::new(blender_bin)
        .args(["--background", "--factory-startup", "-noaudio"])
        .arg("--python")
        .arg(&recipe_path)
        .args(["--python-exit-code", "1"])
        .arg("--")
        .arg(&job_path)
        .arg(&result_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| spawn_error(blender_bin, &e))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = std::thread::spawn(move || stdout.map_or_else(String::new, drain));
    let err_handle = std::thread::spawn(move || stderr.map_or_else(String::new, drain));

    let status = wait_with_cancel(&mut child, &cancel_path, timeout, cancel)?;
    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    if !status.success() {
        return Err(AssetError::Failed(tail(&stderr, 40)));
    }
    build_manifest(job, &result_path, &stdout)
}

/// Wait for Blender, delivering a cancel request as a marker file.
fn wait_with_cancel(
    child: &mut std::process::Child,
    cancel_path: &Path,
    timeout: Duration,
    cancel: &AtomicBool,
) -> Result<std::process::ExitStatus, AssetError> {
    let deadline = Instant::now() + timeout;
    let mut cancel_sent = false;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(e) => {
                return Err(AssetError::Failed(format!("cannot wait for blender: {e}")));
            }
        }
        if !cancel_sent && cancel.load(Ordering::Relaxed) {
            cancel_sent = true;
            let _ = std::fs::write(cancel_path, b"cancel\n");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AssetError::Timeout(timeout));
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Read the recipe's result and turn it into a manifest.
fn build_manifest(
    job: &AssetJob,
    result_path: &Path,
    stdout: &str,
) -> Result<AssetManifest, AssetError> {
    let raw = std::fs::read_to_string(result_path)
        .map_err(|e| AssetError::NoResult(format!("the recipe wrote no result: {e}")))?;
    let result: RecipeResult = serde_json::from_str(&raw)
        .map_err(|e| AssetError::NoResult(format!("unreadable result: {e}")))?;
    if result.cancelled {
        return Err(AssetError::Cancelled);
    }
    let artifacts = result
        .artifacts
        .iter()
        .map(hash_artifact)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AssetManifest {
        manifest: MANIFEST_VERSION,
        blender_version: result.blender_version,
        recipe_version: RECIPE_VERSION.to_string(),
        recipe_sha256: sha256_hex(RECIPE.as_bytes()),
        job: job.clone(),
        artifacts,
        warnings: result.warnings,
        progress: parse_progress(stdout),
    })
}

/// Hash one artifact the recipe produced.
fn hash_artifact(artifact: &RecipeArtifact) -> Result<AssetArtifact, AssetError> {
    let bytes = std::fs::read(&artifact.path)
        .map_err(|e| AssetError::Failed(format!("cannot read {}: {e}", artifact.path.display())))?;
    Ok(AssetArtifact {
        path: artifact.path.clone(),
        kind: artifact.kind.clone(),
        bytes: bytes.len(),
        sha256: sha256_hex(&bytes),
    })
}

/// Write the manifest into the output directory.
fn write_manifest(output_dir: &Path, manifest: &AssetManifest) -> Result<(), AssetError> {
    let path = output_dir.join("manifest.json");
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| AssetError::Failed(format!("cannot encode the manifest: {e}")))?;
    write_atomic(&path, &bytes)
}

/// Write a file the way the artifacts are written: `.part`, fsync, rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), AssetError> {
    let mut part = path.as_os_str().to_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    std::fs::write(&part, bytes)
        .map_err(|e| AssetError::Failed(format!("cannot write {}: {e}", part.display())))?;
    let file = std::fs::File::open(&part)
        .map_err(|e| AssetError::Failed(format!("cannot open {}: {e}", part.display())))?;
    file.sync_all()
        .map_err(|e| AssetError::Failed(format!("cannot sync {}: {e}", part.display())))?;
    std::fs::rename(&part, path)
        .map_err(|e| AssetError::Failed(format!("cannot rename into {}: {e}", path.display())))
}

/// A unique scratch directory for one job's recipe files.
fn scratch_dir() -> Result<PathBuf, AssetError> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let dir = std::env::temp_dir().join(format!(
        "honkbox-blender-asset-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)
        .map_err(|e| AssetError::Failed(format!("cannot create {}: {e}", dir.display())))?;
    Ok(dir)
}

/// The progress events the recipe printed on stdout.
fn parse_progress(stdout: &str) -> Vec<ProgressEvent> {
    stdout
        .lines()
        .filter_map(|line| line.trim().strip_prefix(PROGRESS_PREFIX))
        .filter_map(|payload| serde_json::from_str(payload).ok())
        .collect()
}

/// The sha256 of a byte slice, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Read a pipe to the end (Blender output is bytes; invalid UTF-8 is replaced).
fn drain(mut stream: impl Read) -> String {
    let mut buffer = Vec::new();
    let _ = stream.read_to_end(&mut buffer);
    String::from_utf8_lossy(&buffer).into_owned()
}

/// The last lines of a tool message.
fn tail(text: &str, lines: usize) -> String {
    let collected: Vec<&str> = text.lines().collect();
    let start = collected.len().saturating_sub(lines);
    collected[start..].join("\n")
}

/// Why Blender could not be started.
fn spawn_error(bin: &Path, err: &std::io::Error) -> AssetError {
    if err.kind() == std::io::ErrorKind::NotFound {
        AssetError::NotInstalled
    } else {
        AssetError::Failed(format!("cannot start {}: {err}", bin.display()))
    }
}

/// The job description handed to the recipe.
#[derive(Debug, Serialize)]
struct RecipeJob<'a> {
    source: &'a AssetSource,
    stages: &'a [AssetStage],
    output_dir: &'a Path,
    prefix: &'a str,
    result_path: &'a Path,
    cancel_path: &'a Path,
}

/// What the recipe wrote to its result file.
#[derive(Debug, Deserialize)]
struct RecipeResult {
    #[serde(default)]
    artifacts: Vec<RecipeArtifact>,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(default)]
    cancelled: bool,
    #[serde(default)]
    blender_version: String,
}

/// One artifact the recipe produced.
#[derive(Debug, Deserialize)]
struct RecipeArtifact {
    path: PathBuf,
    #[serde(default)]
    kind: String,
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A scratch directory for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("honkbox-asset-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// A fake Blender that runs the given shell body.
    fn fake_blender(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("fake-blender.sh");
        std::fs::write(&path, body).expect("fake blender");
        // btrfs keeps a write reference until the writeback lands; exec'ing the
        // script right away would fail with ETXTBSY. Flush it first.
        std::fs::File::open(&path)
            .and_then(|file| file.sync_all())
            .expect("flush");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        path
    }

    /// A job that writes into the given directory.
    fn job_for(dir: &Path) -> AssetJob {
        AssetJob {
            source: AssetSource::Primitive {
                primitive: Primitive::Cube,
                size: 2.0,
            },
            stages: vec![AssetStage::Bake {
                maps: vec![BakeMap::Color],
                resolution: 32,
            }],
            output_dir: dir.join("out"),
            prefix: "honkbox_asset".to_string(),
            seed: 7,
        }
    }

    #[test]
    fn a_missing_blender_is_reported_as_not_installed() {
        let dir = scratch("missing");
        let job = job_for(&dir);
        let err = run(
            &job,
            Path::new("/nonexistent/blender"),
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .expect_err("must fail");
        assert_eq!(err, AssetError::NotInstalled);
    }

    #[test]
    fn a_failing_recipe_reports_blenders_own_message() {
        let dir = scratch("failing");
        let bin = fake_blender(
            &dir,
            "#!/bin/sh\necho 'boom: the mesh is broken' >&2\nexit 1\n",
        );
        let job = job_for(&dir);
        let err = run(&job, &bin, Duration::from_secs(5), &AtomicBool::new(false))
            .expect_err("must fail");
        match &err {
            AssetError::Failed(reason) => {
                assert!(reason.contains("boom: the mesh is broken"), "{reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn a_successful_job_writes_a_manifest_with_hashes() {
        let dir = scratch("ok");
        let artifact = dir.join("map.png");
        std::fs::write(&artifact, b"\x89PNG\r\n\x1a\n").expect("artifact");
        let bin = fake_blender(
            &dir,
            &format!(
                "#!/bin/sh\nresult=\"\"\nfor arg in \"$@\"; do result=\"$arg\"; done\n\
                 cat > \"$result\" <<'JSON'\n\
                 {{\"artifacts\": [{{\"path\": \"{}\", \"kind\": \"map\"}}], \
                 \"warnings\": [], \"cancelled\": false, \"blender_version\": \"fake 1.0\"}}\n\
                 JSON\n\
                 echo 'HONKBOX_PROGRESS={{\"stage\": \"bake\", \"event\": \"done\"}}'\n",
                artifact.display()
            ),
        );
        let job = job_for(&dir);
        let manifest =
            run(&job, &bin, Duration::from_secs(5), &AtomicBool::new(false)).expect("manifest");
        assert_eq!(manifest.blender_version, "fake 1.0");
        assert_eq!(manifest.recipe_version, RECIPE_VERSION);
        assert_eq!(manifest.recipe_sha256.len(), 64);
        assert_eq!(manifest.artifacts.len(), 1);
        assert_eq!(manifest.artifacts[0].bytes, 8);
        assert_eq!(
            manifest.artifacts[0].sha256,
            sha256_hex(b"\x89PNG\r\n\x1a\n")
        );
        assert_eq!(manifest.progress.len(), 1);
        assert_eq!(manifest.progress[0].stage, "bake");
        assert!(dir.join("out/manifest.json").exists(), "manifest on disk");
    }

    #[test]
    fn a_cancel_stops_the_job_at_a_stage_boundary() {
        let dir = scratch("cancel");
        let bin = fake_blender(
            &dir,
            "#!/bin/sh\nprev=\"\"\nlast=\"\"\nfor arg in \"$@\"; do prev=\"$last\"; last=\"$arg\"; done\n\
             job=\"$prev\"\nresult=\"$last\"\nscratch=$(dirname \"$job\")\n\
             while [ ! -f \"$scratch/cancel\" ]; do sleep 0.02; done\n\
             cat > \"$result\" <<'JSON'\n\
             {\"artifacts\": [], \"warnings\": [], \"cancelled\": true, \"blender_version\": \"fake 1.0\"}\n\
             JSON\n",
        );
        let job = job_for(&dir);
        let err = run(&job, &bin, Duration::from_secs(10), &AtomicBool::new(true))
            .expect_err("must fail");
        assert_eq!(err, AssetError::Cancelled);
    }

    #[test]
    fn a_job_that_outlives_its_deadline_is_killed() {
        let dir = scratch("timeout");
        let bin = fake_blender(&dir, "#!/bin/sh\nexec sleep 30\n");
        let job = job_for(&dir);
        let err = run(
            &job,
            &bin,
            Duration::from_millis(200),
            &AtomicBool::new(false),
        )
        .expect_err("must fail");
        assert_eq!(err, AssetError::Timeout(Duration::from_millis(200)));
    }

    #[test]
    fn a_missing_input_mesh_is_reported_before_blender_starts() {
        let dir = scratch("input");
        let mut job = job_for(&dir);
        job.source = AssetSource::File {
            path: dir.join("nope.glb"),
            format: MeshFormat::Glb,
        };
        let err = run(
            &job,
            Path::new("/nonexistent/blender"),
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .expect_err("must fail");
        assert!(matches!(err, AssetError::InputMissing(_)), "{err:?}");
    }

    #[test]
    #[ignore = "needs a local Blender install"]
    fn a_real_bake_produces_maps_and_a_manifest() {
        let dir = scratch("real");
        let bin = std::env::var("HONKBOX_BLENDER").unwrap_or_else(|_| "blender".to_string());
        let job = AssetJob {
            source: AssetSource::Primitive {
                primitive: Primitive::Cube,
                size: 2.0,
            },
            stages: vec![
                AssetStage::UvUnpack { margin: 0.02 },
                AssetStage::Bake {
                    maps: vec![BakeMap::Color, BakeMap::Normal],
                    resolution: 64,
                },
            ],
            output_dir: dir.join("out"),
            prefix: "honkbox_asset".to_string(),
            seed: 1,
        };
        let manifest = run(
            &job,
            Path::new(&bin),
            Duration::from_secs(600),
            &AtomicBool::new(false),
        )
        .expect("manifest");
        assert_eq!(manifest.artifacts.len(), 2);
        for artifact in &manifest.artifacts {
            assert!(
                artifact.path.exists(),
                "{} missing",
                artifact.path.display()
            );
            assert_eq!(artifact.sha256.len(), 64);
        }
        assert!(dir.join("out/manifest.json").exists(), "manifest on disk");
    }
}
