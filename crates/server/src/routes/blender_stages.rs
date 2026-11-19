//! The typed stage vocabulary of `POST /v1/jobs/blender`
//! (FEAT-blender-job-route): one closed list of stages, each with its own
//! bounds, mapped onto the recipe's own vocabulary.
//!
//! The vocabulary is closed on purpose: the recipe knows exactly these
//! stages, and a stage the recipe does not know is a 400 at the route
//! instead of a failure inside Blender.

use axum::http::StatusCode;
use honkbox_api_types::{BlenderBakeMap, BlenderMeshFormat, BlenderStage};
use honkbox_external_tools::{AssetStage, BakeMap, MeshFormat};

use super::blender::{MAX_INPUT_BYTES, Rejection};

/// The retopology face-count cap.
pub(super) const MAX_TARGET_FACES: u64 = 200_000;

/// The bake resolution bounds (a square map).
pub(super) const MIN_BAKE_RESOLUTION: u32 = 4;
pub(super) const MAX_BAKE_RESOLUTION: u32 = 4096;

/// The turntable bounds (views per turntable, square resolution per view).
pub(super) const MAX_TURNTABLE_FRAMES: u32 = 8;
pub(super) const MIN_TURNTABLE_RESOLUTION: u32 = 16;
pub(super) const MAX_TURNTABLE_RESOLUTION: u32 = 1024;

/// The UV island margin bounds (Blender units).
const MAX_UV_MARGIN: f64 = 100.0;

/// The voxel size bounds (Blender units).
const MIN_VOXEL_SIZE: f64 = 0.001;
const MAX_VOXEL_SIZE: f64 = 1000.0;

/// No `job_progress` emission here: this is a pure, id-less validation pass
/// over the whole recipe (it runs before the job executes), so there is
/// neither a step index nor a job id to report. The emit belongs in
/// `routes/blender_exec.rs::run`/`execute`, where the live guard id is in
/// scope and the recipe's `HONKBOX_PROGRESS` events arrive via
/// `honkbox_external_tools::blender_assets::parse_progress`
/// (`AssetManifest::progress`) — translating those, not inventing steps.
///
/// Map one typed stage onto the recipe's own stage vocabulary, resolving
/// the artifacts it names.
#[allow(clippy::too_many_lines)] // one closed stage per branch, by design
pub(super) fn map_stage(
    stage: &BlenderStage,
    artifacts: &crate::artifacts::Registry,
) -> Result<AssetStage, Rejection> {
    match stage {
        BlenderStage::UvUnpack { margin } => {
            if !margin.is_finite() || *margin < 0.0 || *margin > MAX_UV_MARGIN {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("uv_unpack margin must be 0..={MAX_UV_MARGIN}"),
                    "blender_stage_margin",
                ));
            }
            Ok(AssetStage::UvUnpack { margin: *margin })
        }
        BlenderStage::UvPack => Ok(AssetStage::UvPack),
        BlenderStage::VoxelRemesh { voxel_size } => {
            if !voxel_size.is_finite()
                || *voxel_size < MIN_VOXEL_SIZE
                || *voxel_size > MAX_VOXEL_SIZE
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("voxel_remesh voxel_size must be {MIN_VOXEL_SIZE}..={MAX_VOXEL_SIZE}"),
                    "blender_stage_voxel_size",
                ));
            }
            Ok(AssetStage::VoxelRemesh {
                voxel_size: *voxel_size,
            })
        }
        BlenderStage::Quadriflow { target_faces } => {
            if *target_faces == 0 || *target_faces > MAX_TARGET_FACES {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("quadriflow target_faces must be 1..={MAX_TARGET_FACES}"),
                    "blender_stage_target_faces",
                ));
            }
            Ok(AssetStage::Quadriflow {
                target_faces: *target_faces,
            })
        }
        BlenderStage::AttachTexture { image_id, node } => {
            let (path, bytes) = artifacts.stored_blob(image_id).ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("texture artifact {image_id:?} is not stored"),
                    "blender_texture_not_found",
                )
            })?;
            if bytes > MAX_INPUT_BYTES {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("the texture is {bytes} bytes; the cap is {MAX_INPUT_BYTES}"),
                    "blender_texture_too_large",
                ));
            }
            if node.is_empty() || node.len() > 64 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "attach_texture node must be 1..=64 characters".to_string(),
                    "blender_stage_node",
                ));
            }
            Ok(AssetStage::AttachTexture {
                image: path,
                node: node.clone(),
            })
        }
        BlenderStage::Bake { maps, resolution } => {
            if maps.is_empty() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "bake needs at least one map".to_string(),
                    "blender_stage_maps",
                ));
            }
            if *resolution < MIN_BAKE_RESOLUTION || *resolution > MAX_BAKE_RESOLUTION {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "bake resolution must be {MIN_BAKE_RESOLUTION}..={MAX_BAKE_RESOLUTION}"
                    ),
                    "blender_stage_resolution",
                ));
            }
            let maps = maps
                .iter()
                .map(|map| match map {
                    BlenderBakeMap::Color => BakeMap::Color,
                    BlenderBakeMap::Normal => BakeMap::Normal,
                    BlenderBakeMap::Roughness => BakeMap::Roughness,
                })
                .collect();
            Ok(AssetStage::Bake {
                maps,
                resolution: *resolution,
            })
        }
        BlenderStage::ExportGlb => Ok(AssetStage::ExportGlb),
        BlenderStage::Turntable { frames, resolution } => {
            if *frames == 0 || *frames > MAX_TURNTABLE_FRAMES {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("turntable frames must be 1..={MAX_TURNTABLE_FRAMES}"),
                    "blender_stage_frames",
                ));
            }
            if *resolution < MIN_TURNTABLE_RESOLUTION || *resolution > MAX_TURNTABLE_RESOLUTION {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "turntable resolution must be {MIN_TURNTABLE_RESOLUTION}..={MAX_TURNTABLE_RESOLUTION}"
                    ),
                    "blender_stage_turntable_resolution",
                ));
            }
            Ok(AssetStage::Turntable {
                frames: *frames,
                resolution: *resolution,
            })
        }
    }
}

/// The recipe's mesh format for a typed one.
pub(super) fn mesh_format(format: BlenderMeshFormat) -> MeshFormat {
    match format {
        BlenderMeshFormat::Glb => MeshFormat::Glb,
        BlenderMeshFormat::Obj => MeshFormat::Obj,
        BlenderMeshFormat::Ply => MeshFormat::Ply,
    }
}
