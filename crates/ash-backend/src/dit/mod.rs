//! Shared single-stream flow-DiT substrate, independent of any model family.
//!
//! Shapes are explicit and row-major. The portable CPU path is the reference;
//! the Vulkan path reuses the existing matmul and attention kernels and keeps
//! elementwise gating on the host until a fused shader is actually needed.

pub mod block;
pub mod flow;
#[cfg(feature = "ash-compute")]
pub mod gpu;
pub mod positions;
pub mod predict;
pub mod spec;

pub use block::{DitBlockWeights, dit_block, patch_embed, patch_flatten};
pub use flow::{euler_step, shifted_schedule, sigma_embedding};
pub use positions::position_grid;
pub use predict::{FlowPredictor, FlowPredictorWeights};
pub use spec::{FlowDitSpec, PositionAxes};

pub(crate) fn invalid(field: &str, detail: &str) -> crate::VkError {
    crate::VkError::Device(format!("flow_dit: {field} {detail}"))
}
