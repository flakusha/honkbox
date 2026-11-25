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
//!
//! A bake contract ([`BakeContract`]) is resolved from the job *before* any
//! process starts: it pins the source hash, the UV layer, every output path,
//! the bake type, the resolution, the exact Blender argv and the deterministic
//! Cycles settings. Jobs with no UV source or with two targets writing the
//! same path are rejected here, not halfway through a bake.
//!
//! The legacy [`AssetManifest`] keeps its original shape for existing
//! callers. The Godot-facing record (role, colorspace, dimensions, sha256 and
//! tool version for every emitted map and thumbnail) is written as a second
//! sidecar, `bake-manifest.json`, by [`build_bake_manifest`].

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ColorSpace, MapRole};

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
    /// Render a turntable preview: `frames` views around the object, tiled
    /// into one image.
    Turntable {
        /// How many views to render.
        frames: u32,
        /// The square resolution of each view.
        resolution: u32,
    },
    /// Emit an asset-browser thumbnail: one still plus `frames` turntable
    /// views. Additive on top of [`AssetStage::Turntable`], which still emits
    /// its single tiled strip exactly as before.
    Thumbnail {
        /// How many turntable views to keep.
        frames: u32,
        /// The square resolution of each view.
        resolution: u32,
    },
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
    /// The metalness map (linear).
    Metallic,
    /// The ambient-occlusion map (linear).
    Ao,
    /// The height/displacement map (linear).
    Height,
}

impl BakeMap {
    /// The filename token the recipe uses: `<prefix>_<name>.png`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::Normal => "normal",
            Self::Roughness => "roughness",
            Self::Metallic => "metallic",
            Self::Ao => "ao",
            Self::Height => "height",
        }
    }

    /// The Godot import role of this map.
    #[must_use]
    pub const fn role(self) -> MapRole {
        match self {
            Self::Color => MapRole::Albedo,
            Self::Normal => MapRole::Normal,
            Self::Roughness => MapRole::Rough,
            Self::Metallic => MapRole::Metal,
            Self::Ao => MapRole::Ao,
            Self::Height => MapRole::Height,
        }
    }
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

/// The bake pass a contract pins. CPU Cycles with one sample and no denoising
/// keeps the output free of GPU and driver drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BakeType {
    /// CPU Cycles (the recipe's deterministic default).
    CpuCycles,
}

/// Deterministic settings pinned by a bake contract. These mirror the
/// recipe's bake setup; changing either side is a recipe version bump.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BakeSettings {
    /// Render engine (`CYCLES`).
    pub engine: String,
    /// Bake device (`CPU`).
    pub device: String,
    /// Samples per map (`1`).
    pub samples: u32,
    /// Denoising (off; it is not deterministic across builds).
    pub denoise: bool,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            engine: "CYCLES".to_owned(),
            device: "CPU".to_owned(),
            samples: 1,
            denoise: false,
        }
    }
}

/// One resolved output of a bake contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BakeTarget {
    /// Godot-facing role (`albedo`, `normal`, `rough`, `metal`, `ao`,
    /// `height`, or `thumbnail`).
    pub role: String,
    /// The exact file the stage writes.
    pub path: PathBuf,
}

/// A validated headless bake job: every value the process needs, resolved
/// before the process starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BakeContract {
    /// The source mesh, exactly as handed to the recipe.
    pub source: AssetSource,
    /// sha256 of the mesh bytes, or of the primitive descriptor.
    pub source_hash: String,
    /// The UV layer the maps bake into (`UVMap`).
    pub uv_map: String,
    /// Every output path, in emission order; no two may be equal.
    pub targets: Vec<BakeTarget>,
    /// The pinned bake pass.
    pub bake_type: BakeType,
    /// Edge length of the square maps (0 when the job bakes no maps).
    pub resolution: u32,
    /// The exact Blender argv, binary included.
    pub argv: Vec<String>,
    /// The recipe version recorded in the manifest.
    pub recipe_version: String,
    /// The sha256 of the embedded recipe.
    pub recipe_sha256: String,
    /// Deterministic settings pinned by this contract.
    pub settings: BakeSettings,
}

/// One emitted map or thumbnail as recorded in `bake-manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BakeArtifact {
    /// Where the artifact landed.
    pub path: PathBuf,
    /// Godot-facing role of the file.
    pub role: String,
    /// `srgb` for albedo, `linear` for every data map.
    pub colorspace: String,
    /// PNG width in pixels (0 for non-image artifacts).
    pub width: u32,
    /// PNG height in pixels (0 for non-image artifacts).
    pub height: u32,
    /// sha256 of the file bytes.
    pub sha256: String,
    /// The Blender version that emitted the file.
    pub tool_version: String,
}

/// The Godot-facing bake sidecar written to `<output_dir>/bake-manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BakeManifest {
    /// The sidecar schema version.
    pub manifest: u32,
    /// The contract that produced these artifacts.
    pub contract: BakeContract,
    /// The Blender version the recipe reported.
    pub blender_version: String,
    /// Every emitted map and thumbnail, with role and dimensions.
    pub artifacts: Vec<BakeArtifact>,
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
/// boundary. The manifests are written to `<output_dir>/manifest.json` and
/// `<output_dir>/bake-manifest.json`.
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

    let scratch = scratch_dir("blender-asset")?;
    let outcome = run_recipe(job, blender_bin, timeout, cancel, &scratch);
    let _ = std::fs::remove_dir_all(&scratch);
    let (manifest, bake) = outcome?;
    write_manifest(&job.output_dir, &manifest)?;
    write_bake_manifest(&job.output_dir, &bake)?;
    Ok(manifest)
}

/// The files one recipe run needs on disk. A caller that owns its own
/// process lifecycle (the server route drives Blender through the process
/// supervisor) writes them with [`prepare`] and runs the script itself.
#[derive(Debug, Clone)]
pub struct RecipeFiles {
    /// The recipe script to hand to Blender (`--python`).
    pub script: PathBuf,
    /// The job JSON the recipe reads.
    pub job: PathBuf,
    /// Where the recipe writes its result.
    pub result: PathBuf,
    /// The cancel marker the recipe checks between stages.
    pub cancel: PathBuf,
}

impl RecipeFiles {
    /// The exact argv handed to Blender. The contract records this same
    /// vector, so an external driver cannot drift from the built-in runner.
    #[must_use]
    pub fn argv(&self, blender_bin: &Path) -> Vec<String> {
        vec![
            blender_bin.to_string_lossy().into_owned(),
            "--background".to_owned(),
            "--factory-startup".to_owned(),
            "-noaudio".to_owned(),
            "--python".to_owned(),
            self.script.to_string_lossy().into_owned(),
            "--python-exit-code".to_owned(),
            "1".to_owned(),
            "--".to_owned(),
            self.job.to_string_lossy().into_owned(),
            self.result.to_string_lossy().into_owned(),
        ]
    }
}

/// Write the recipe and the job description into `scratch` and report the
/// files Blender needs. The job JSON is the recipe's own protocol, so the
/// built-in runner and an external driver share one description.
pub fn prepare(job: &AssetJob, scratch: &Path) -> Result<RecipeFiles, AssetError> {
    validate_job(job)?;
    std::fs::create_dir_all(scratch)
        .map_err(|e| AssetError::Failed(format!("cannot create {}: {e}", scratch.display())))?;
    let script = scratch.join("blender_asset.py");
    std::fs::write(&script, RECIPE)
        .map_err(|e| AssetError::Failed(format!("cannot write the recipe: {e}")))?;
    let result = scratch.join("result.json");
    let cancel = scratch.join("cancel");
    let job_path = scratch.join("job.json");
    let recipe_job = RecipeJob {
        source: &job.source,
        stages: &job.stages,
        output_dir: &job.output_dir,
        prefix: &job.prefix,
        result_path: &result,
        cancel_path: &cancel,
    };
    let job_bytes =
        serde_json::to_vec(&recipe_job).map_err(|e| AssetError::InvalidJob(e.to_string()))?;
    std::fs::write(&job_path, job_bytes)
        .map_err(|e| AssetError::Failed(format!("cannot write the job: {e}")))?;
    Ok(RecipeFiles {
        script,
        job: job_path,
        result,
        cancel,
    })
}

/// The sha256 of the embedded recipe: the script version a caller pins in
/// its engine contract.
#[must_use]
pub fn recipe_sha256() -> String {
    sha256_hex(RECIPE.as_bytes())
}

/// Resolve a job into its contract: source hash, UV layer, every output path,
/// exact argv and the pinned deterministic settings. Nothing is written and
/// no process is started, so a caller can reject or log the job first.
pub fn plan_bake(
    job: &AssetJob,
    blender_bin: &Path,
    files: &RecipeFiles,
) -> Result<BakeContract, AssetError> {
    validate_job(job)?;
    let bytes = match &job.source {
        AssetSource::File { path, .. } => std::fs::read(path)
            .map_err(|e| AssetError::Failed(format!("cannot hash {}: {e}", path.display())))?,
        primitive @ AssetSource::Primitive { .. } => serde_json::to_vec(primitive)
            .map_err(|e| AssetError::InvalidJob(format!("cannot hash the source: {e}")))?,
    };
    let (targets, resolution) = planned_targets(job)?;
    Ok(BakeContract {
        source_hash: sha256_hex(&bytes),
        source: job.source.clone(),
        // The smart-project stage and Blender's primitives both use `UVMap`.
        uv_map: "UVMap".to_owned(),
        targets,
        bake_type: BakeType::CpuCycles,
        resolution,
        argv: files.argv(blender_bin),
        recipe_version: RECIPE_VERSION.to_owned(),
        recipe_sha256: recipe_sha256(),
        settings: BakeSettings::default(),
    })
}

/// Every output path a job claims, with the Godot role of each. Shared by
/// validation and the contract so the two cannot drift apart.
fn planned_targets(job: &AssetJob) -> Result<(Vec<BakeTarget>, u32), AssetError> {
    let mut uv_ready = matches!(&job.source, AssetSource::Primitive { .. });
    let mut resolution = 0;
    let mut targets: Vec<BakeTarget> = Vec::new();
    let mut claim = |path: PathBuf, role: &str| -> Result<(), AssetError> {
        if targets
            .iter()
            .any(|target: &BakeTarget| target.path == path)
        {
            return Err(AssetError::InvalidJob(format!(
                "overlapping bake targets: {}",
                path.display()
            )));
        }
        targets.push(BakeTarget {
            role: role.to_owned(),
            path,
        });
        Ok(())
    };
    for stage in &job.stages {
        match stage {
            AssetStage::UvUnpack { .. } => uv_ready = true,
            AssetStage::Bake {
                maps,
                resolution: edge,
            } => {
                if maps.is_empty() {
                    return Err(AssetError::InvalidJob("bake needs at least one map".into()));
                }
                if !uv_ready {
                    return Err(AssetError::InvalidJob(
                        "bake needs a UV map: add a uv_unpack stage before it".into(),
                    ));
                }
                resolution = *edge;
                if resolution == 0 {
                    return Err(AssetError::InvalidJob(
                        "bake resolution must be positive".into(),
                    ));
                }
                for map in maps {
                    claim(
                        job.output_dir
                            .join(format!("{}_{}.png", job.prefix, map.name())),
                        map.role().as_str(),
                    )?;
                }
            }
            AssetStage::Turntable { frames, .. } => {
                if *frames == 0 {
                    return Err(AssetError::InvalidJob(
                        "turntable needs at least one view".into(),
                    ));
                }
                claim(
                    job.output_dir.join(format!("{}_turntable.png", job.prefix)),
                    "thumbnail",
                )?;
            }
            AssetStage::Thumbnail { frames, .. } => {
                if *frames == 0 {
                    return Err(AssetError::InvalidJob(
                        "thumbnail needs at least one view".into(),
                    ));
                }
                claim(
                    job.output_dir.join(format!("{}_thumbnail.png", job.prefix)),
                    "thumbnail",
                )?;
                for frame in 0..*frames {
                    claim(
                        job.output_dir
                            .join(format!("{}_thumbnail_{frame:02}.png", job.prefix)),
                        "thumbnail",
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok((targets, resolution))
}

/// Reject a job that cannot produce an honest result before any process starts.
fn validate_job(job: &AssetJob) -> Result<(), AssetError> {
    if let AssetSource::File { path, .. } = &job.source
        && !path.exists()
    {
        return Err(AssetError::InputMissing(path.display().to_string()));
    }
    if job.prefix.is_empty() {
        return Err(AssetError::InvalidJob(
            "the artifact prefix is empty".into(),
        ));
    }
    planned_targets(job).map(|_| ())
}

/// Read the recipe's result and enrich each artifact with the fields the
/// Godot/loop-lore manifest contract requires.
pub fn build_bake_manifest(
    contract: &BakeContract,
    result_path: &Path,
) -> Result<BakeManifest, AssetError> {
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
        .map(|artifact| {
            let role = contract
                .targets
                .iter()
                .find(|target| target.path == artifact.path)
                .map_or_else(|| role_of(artifact), |target| target.role.clone());
            let bytes = std::fs::read(&artifact.path).map_err(|e| {
                AssetError::Failed(format!("cannot read {}: {e}", artifact.path.display()))
            })?;
            let (width, height) = if role == "model" {
                (0, 0)
            } else {
                png_dimensions(&bytes)?
            };
            let colorspace = if role == MapRole::Albedo.as_str() || role == "thumbnail" {
                ColorSpace::Srgb.as_str()
            } else {
                ColorSpace::Linear.as_str()
            };
            Ok(BakeArtifact {
                path: artifact.path.clone(),
                role,
                colorspace: colorspace.to_owned(),
                width,
                height,
                sha256: sha256_hex(&bytes),
                tool_version: result.blender_version.clone(),
            })
        })
        .collect::<Result<Vec<_>, AssetError>>()?;
    Ok(BakeManifest {
        manifest: MANIFEST_VERSION,
        contract: contract.clone(),
        blender_version: result.blender_version,
        artifacts,
    })
}

/// Role from the artifact itself, for results an external driver produced
/// without a contract target for the file.
fn role_of(artifact: &RecipeArtifact) -> String {
    let name = artifact
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    for map in [
        BakeMap::Color,
        BakeMap::Normal,
        BakeMap::Roughness,
        BakeMap::Metallic,
        BakeMap::Ao,
        BakeMap::Height,
    ] {
        if name.ends_with(&format!("_{}.png", map.name())) {
            return map.role().as_str().to_owned();
        }
    }
    match artifact.kind.as_str() {
        "preview" => "thumbnail",
        "model" => "model",
        _ => "map",
    }
    .to_owned()
}

/// Validate the PNG signature and read the IHDR dimensions.
fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32), AssetError> {
    if bytes.len() < 24 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") || &bytes[12..16] != b"IHDR" {
        return Err(AssetError::NoResult(
            "an emitted map is not a PNG with an IHDR".to_owned(),
        ));
    }
    Ok((
        u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
    ))
}

/// Write the Godot-facing bake sidecar into the output directory. External
/// drivers that run `prepare` themselves use this after `build_bake_manifest`.
pub fn write_bake_manifest(output_dir: &Path, manifest: &BakeManifest) -> Result<(), AssetError> {
    let path = output_dir.join("bake-manifest.json");
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| AssetError::Failed(format!("cannot encode the bake manifest: {e}")))?;
    write_atomic(&path, &bytes)
}

/// Hand the job to Blender and collect what it produced.
fn run_recipe(
    job: &AssetJob,
    blender_bin: &Path,
    timeout: Duration,
    cancel: &AtomicBool,
    scratch: &Path,
) -> Result<(AssetManifest, BakeManifest), AssetError> {
    let files = prepare(job, scratch)?;
    let argv = files.argv(blender_bin);

    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| spawn_error(blender_bin, &e))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = std::thread::spawn(move || stdout.map_or_else(String::new, drain));
    let err_handle = std::thread::spawn(move || stderr.map_or_else(String::new, drain));

    let status = wait_with_cancel(&mut child, &files.cancel, timeout, cancel)?;
    let stdout = out_handle.join().unwrap_or_default();
    let stderr = err_handle.join().unwrap_or_default();
    if !status.success() {
        return Err(AssetError::Failed(tail(&stderr, 40)));
    }
    let contract = plan_bake(job, blender_bin, &files)?;
    Ok((
        build_manifest(job, &files.result, &stdout)?,
        build_bake_manifest(&contract, &files.result)?,
    ))
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

/// Read the recipe's result and turn it into a manifest. Public so a caller
/// that drove Blender itself (the server route, through the process
/// supervisor) reads the result exactly like the built-in runner does.
pub fn build_manifest(
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

/// Machine-temp root for honkbox scratch: `$HONKBOX_TMP_ROOT` (or
/// `$LOOP_LORE_TMP_ROOT`) when set and non-blank, else the process temp
/// dir, with the `honkbox` app folder below it. Never a bare entry
/// directly under the temp dir.
fn tmp_root() -> PathBuf {
    let base = std::env::var("HONKBOX_TMP_ROOT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| {
            std::env::var("LOOP_LORE_TMP_ROOT")
                .ok()
                .filter(|v| !v.trim().is_empty())
        })
        .map_or_else(std::env::temp_dir, PathBuf::from);
    base.join("honkbox")
}

/// A unique scratch directory: `<root>/<subsystem>/<pid>-<nanos>`.
///
/// `subsystem` is one path segment (`blender-asset`, `bake`, `ext-addons`,
/// `jobs-gc`, `jobs-recipe`, `log-rotate`, `log-base`, ...). The parents are
/// created with mode `0700`; the leaf uses `create_dir` (fail-if-exists,
/// a few nanos retries) so two processes can never share it. The caller
/// owns cleanup.
pub fn scratch_dir(subsystem: &str) -> Result<PathBuf, AssetError> {
    let root = tmp_root();
    let parent = root.join(subsystem);
    std::fs::create_dir_all(&parent)
        .map_err(|e| AssetError::Failed(format!("cannot create {}: {e}", parent.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::Permissions::from_mode(0o700);
        let _ = std::fs::set_permissions(&root, mode.clone());
        let _ = std::fs::set_permissions(&parent, mode);
    }
    let pid = std::process::id();
    for attempt in 0..8u32 {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let leaf = if attempt == 0 {
            format!("{pid}-{stamp}")
        } else {
            format!("{pid}-{stamp}-{attempt}")
        };
        let dir = parent.join(leaf);
        match std::fs::create_dir(&dir) {
            Ok(()) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
                }
                return Ok(dir);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(AssetError::Failed(format!(
                    "cannot create {}: {e}",
                    dir.display()
                )));
            }
        }
    }
    Err(AssetError::Failed(format!(
        "cannot pick a unique name under {}",
        parent.display()
    )))
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
    use std::collections::HashSet;

    /// One test's temp dir: unique by pid and tag, removed even on panic.
    struct BakeTemp(PathBuf);

    impl BakeTemp {
        fn new(_tag: &str) -> Self {
            Self(scratch_dir("bake").expect("scratch dir"))
        }
    }

    impl std::ops::Deref for BakeTemp {
        type Target = Path;

        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for BakeTemp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A scratch directory for one test.
    fn scratch(name: &str) -> BakeTemp {
        BakeTemp::new(name)
    }

    /// A minimal PNG: valid magic and IHDR, with the requested dimensions.
    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes
    }

    /// Where a bake map lands for a job.
    fn map_path(job: &AssetJob, map: BakeMap) -> PathBuf {
        job.output_dir
            .join(format!("{}_{}.png", job.prefix, map.name()))
    }

    /// The result JSON a fake recipe writes for the given files.
    fn result(files: &[(&Path, &str)], cancelled: bool) -> String {
        let artifacts: Vec<_> = files
            .iter()
            .map(|(path, kind)| {
                serde_json::json!({
                    "path": path.display().to_string(),
                    "kind": kind,
                })
            })
            .collect();
        serde_json::json!({
            "artifacts": artifacts,
            "warnings": [],
            "cancelled": cancelled,
            "blender_version": "fake 4.5",
        })
        .to_string()
    }

    /// A fake Blender that writes the given result JSON to its last argument.
    fn writing_blender(dir: &Path, body: &str) -> PathBuf {
        fake_blender(
            dir,
            &format!(
                "#!/bin/sh\nresult=\"\"\nfor arg in \"$@\"; do result=\"$arg\"; done\n\
                 cat > \"$result\" <<'JSON'\n{body}\nJSON\n"
            ),
        )
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
        std::fs::write(&artifact, png(32, 32)).expect("artifact");
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
        assert_eq!(manifest.artifacts[0].bytes, 33);
        assert_eq!(manifest.artifacts[0].sha256, sha256_hex(&png(32, 32)));
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
    fn a_real_turntable_renders_one_preview() {
        let dir = scratch("turntable");
        let bin = std::env::var("HONKBOX_BLENDER").unwrap_or_else(|_| "blender".to_string());
        let job = AssetJob {
            source: AssetSource::Primitive {
                primitive: Primitive::Cube,
                size: 2.0,
            },
            stages: vec![AssetStage::Turntable {
                frames: 3,
                resolution: 64,
            }],
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
        assert_eq!(manifest.artifacts.len(), 1);
        let artifact = &manifest.artifacts[0];
        assert_eq!(artifact.kind, "preview");
        assert!(artifact.path.exists(), "the preview is on disk");
        assert!(artifact.bytes > 0);
        assert_eq!(artifact.sha256.len(), 64);
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

    #[test]
    fn emitted_pngs_have_valid_magic_and_exact_dimensions() {
        let dir = scratch("dims");
        let job = job_for(&dir);
        std::fs::create_dir_all(&job.output_dir).expect("out dir");
        let map = map_path(&job, BakeMap::Color);
        let bytes = png(8, 12);
        std::fs::write(&map, &bytes).expect("map");
        let result_path = dir.join("result.json");
        std::fs::write(&result_path, result(&[(&map, "map")], false)).expect("result");
        let files = RecipeFiles {
            script: dir.join("blender_asset.py"),
            job: dir.join("job.json"),
            result: result_path.clone(),
            cancel: dir.join("cancel"),
        };
        let contract = plan_bake(&job, Path::new("/usr/bin/blender"), &files).expect("contract");
        let bake = build_bake_manifest(&contract, &result_path).expect("bake manifest");
        assert_eq!(png_dimensions(&bytes).expect("png"), (8, 12));
        assert_eq!(bake.artifacts.len(), 1);
        assert_eq!(bake.artifacts[0].width, 8);
        assert_eq!(bake.artifacts[0].height, 12);

        let mut bad = bytes;
        bad[1] = 0;
        assert!(png_dimensions(&bad).is_err(), "bad magic must be rejected");
    }

    #[test]
    fn the_bake_manifest_chains_hashes_versions_and_argv() {
        let dir = scratch("chain");
        let job = job_for(&dir);
        std::fs::create_dir_all(&job.output_dir).expect("out dir");
        let map = map_path(&job, BakeMap::Color);
        let bytes = png(16, 16);
        std::fs::write(&map, &bytes).expect("map");
        let bin = writing_blender(&dir, &result(&[(&map, "map")], false));
        run(&job, &bin, Duration::from_secs(5), &AtomicBool::new(false)).expect("run");

        let raw = std::fs::read_to_string(job.output_dir.join("bake-manifest.json"))
            .expect("bake sidecar on disk");
        let bake: BakeManifest = serde_json::from_str(&raw).expect("decode sidecar");
        assert_eq!(bake.manifest, MANIFEST_VERSION);
        assert_eq!(bake.blender_version, "fake 4.5");
        assert_eq!(bake.artifacts.len(), 1);
        let artifact = &bake.artifacts[0];
        assert_eq!(artifact.sha256, sha256_hex(&bytes));
        assert_eq!(artifact.tool_version, "fake 4.5");
        assert_eq!(artifact.role, "albedo");
        assert_eq!(artifact.colorspace, "srgb");
        assert_eq!(
            bake.contract.source_hash,
            sha256_hex(&serde_json::to_vec(&job.source).expect("source json"))
        );
        assert_eq!(bake.contract.uv_map, "UVMap");
        assert_eq!(bake.contract.bake_type, BakeType::CpuCycles);
        assert_eq!(bake.contract.settings, BakeSettings::default());
        assert_eq!(bake.contract.recipe_version, RECIPE_VERSION);
        assert_eq!(bake.contract.recipe_sha256, recipe_sha256());
        assert_eq!(bake.contract.argv[0], bin.display().to_string());
        assert!(
            bake.contract
                .argv
                .iter()
                .any(|arg| arg == "--python-exit-code")
        );
    }

    #[test]
    fn every_extended_map_and_thumbnail_has_a_role() {
        let dir = scratch("roles");
        let mut job = job_for(&dir);
        let maps = [
            BakeMap::Color,
            BakeMap::Normal,
            BakeMap::Roughness,
            BakeMap::Metallic,
            BakeMap::Ao,
            BakeMap::Height,
        ];
        job.stages = vec![
            AssetStage::UvUnpack { margin: 0.01 },
            AssetStage::Bake {
                maps: maps.to_vec(),
                resolution: 8,
            },
            AssetStage::Thumbnail {
                frames: 2,
                resolution: 8,
            },
        ];
        std::fs::create_dir_all(&job.output_dir).expect("out dir");
        let mut files: Vec<(PathBuf, &str)> = Vec::new();
        for map in maps {
            let path = map_path(&job, map);
            std::fs::write(&path, png(8, 8)).expect("map");
            files.push((path, "map"));
        }
        let still = job.output_dir.join(format!("{}_thumbnail.png", job.prefix));
        std::fs::write(&still, png(8, 8)).expect("still");
        files.push((still, "preview"));
        for frame in 0..2 {
            let path = job
                .output_dir
                .join(format!("{}_thumbnail_{frame:02}.png", job.prefix));
            std::fs::write(&path, png(8, 8)).expect("frame");
            files.push((path, "preview"));
        }
        let refs: Vec<(&Path, &str)> = files
            .iter()
            .map(|(path, kind)| (path.as_path(), *kind))
            .collect();
        let bin = writing_blender(&dir, &result(&refs, false));
        run(&job, &bin, Duration::from_secs(5), &AtomicBool::new(false)).expect("run");

        let raw = std::fs::read_to_string(job.output_dir.join("bake-manifest.json"))
            .expect("bake sidecar on disk");
        let bake: BakeManifest = serde_json::from_str(&raw).expect("decode sidecar");
        let roles: HashSet<_> = bake
            .artifacts
            .iter()
            .map(|item| item.role.as_str())
            .collect();
        assert_eq!(
            roles,
            HashSet::from([
                "albedo",
                "normal",
                "rough",
                "metal",
                "ao",
                "height",
                "thumbnail"
            ])
        );
        assert_eq!(bake.artifacts.len(), 9);
        assert!(
            bake.artifacts
                .iter()
                .all(|item| item.width == 8 && item.height == 8)
        );
    }

    #[test]
    fn a_bake_without_a_uv_source_is_rejected_before_blender_starts() {
        let dir = scratch("no-uv");
        let mesh = dir.join("mesh.glb");
        std::fs::write(&mesh, b"glb").expect("mesh");
        let mut job = job_for(&dir);
        job.source = AssetSource::File {
            path: mesh,
            format: MeshFormat::Glb,
        };
        let err = run(
            &job,
            Path::new("/nonexistent/blender"),
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .expect_err("must fail");
        assert!(
            matches!(&err, AssetError::InvalidJob(reason) if reason.contains("UV")),
            "{err:?}"
        );
    }

    #[test]
    fn overlapping_bake_targets_are_rejected_before_blender_starts() {
        let dir = scratch("overlap");
        let mut job = job_for(&dir);
        job.stages = vec![
            AssetStage::Bake {
                maps: vec![BakeMap::Color],
                resolution: 32,
            },
            AssetStage::Bake {
                maps: vec![BakeMap::Color],
                resolution: 32,
            },
        ];
        let err = run(
            &job,
            Path::new("/nonexistent/blender"),
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .expect_err("must fail");
        assert!(
            matches!(&err, AssetError::InvalidJob(reason) if reason.contains("overlapping")),
            "{err:?}"
        );
    }

    #[test]
    fn the_cancel_marker_is_part_of_the_recipe_protocol() {
        let dir = scratch("marker");
        let job = job_for(&dir);
        let files = prepare(&job, &dir).expect("recipe files");
        assert_eq!(files.cancel, dir.join("cancel"));
        let encoded = std::fs::read_to_string(&files.job).expect("job json");
        let value: serde_json::Value = serde_json::from_str(&encoded).expect("decode job json");
        assert_eq!(value["cancel_path"], files.cancel.display().to_string());
    }
}
