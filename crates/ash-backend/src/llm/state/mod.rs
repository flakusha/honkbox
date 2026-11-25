//! Request-local autoregressive state: validated plans, masks and sequences.

mod mask;
mod plan;
mod sequence;

pub use mask::{AdditiveMask, MaskMode, TokenPosition, build_mask};
pub use plan::{ConvolutionPlan, KvPlan, LayerPlan, MatrixPlan, RecurrentPlan, StatePlan};
pub use sequence::{KvAttentionViews, LayerUpdate, SequenceState, StateBytes, TokenUpdate};
