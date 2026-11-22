//! Ops vocabulary and declarative model contracts.
//!
//! A model is an ordered list of compute passes. New passes arrive as new
//! [`Op`] variants plus a kernel module; dispatch, buffers and admission stay
//! shared. Checkpoint key mapping is deliberately absent until a real model
//! lands — nothing consumes it yet.

use crate::{
    VkError,
    device::{CapacityRequirements, PhysicalDeviceInfo},
};

/// Compute passes the backend can schedule. Add a variant when a kernel lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `out[i] = lhs[i] + rhs[i]`.
    ElementwiseAdd,
    /// Row-major `C = A * B`.
    MatMul,
    /// Stable row-wise normalization.
    Softmax,
    /// Scaled dot-product attention (composed from matmul + softmax).
    Attention,
    /// Affine row-wise layer norm.
    LayerNorm,
}

impl Op {
    /// Static floor for this pass; per-call shapes are checked again in `run`.
    pub fn requirements(self) -> CapacityRequirements {
        match self {
            Self::ElementwiseAdd => CapacityRequirements {
                workgroup_count: [1, 1, 1],
                storage_range: 4, // length uniform
                ..CapacityRequirements::default()
            },
            Self::MatMul | Self::Attention | Self::LayerNorm => CapacityRequirements {
                workgroup_count: [1, 1, 1],
                storage_range: 12, // matmul shape buffer
                ..CapacityRequirements::default()
            },
            Self::Softmax => CapacityRequirements {
                workgroup_count: [1, 1, 1],
                storage_range: 8, // two shape u32s
                ..CapacityRequirements::default()
            },
        }
    }
}

/// A model expressed as an ordered pass list.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// Human-readable model family name (diagnostics only).
    pub name: &'static str,
    /// Passes in execution order.
    pub ops: &'static [Op],
}

impl ModelSpec {
    /// Reject a device that cannot satisfy any pass's static floor.
    pub fn validate(self, info: &PhysicalDeviceInfo) -> Result<(), VkError> {
        for &op in self.ops {
            info.admit(&op.requirements())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceType, tests::fake_info};

    const ADD_ONLY: ModelSpec = ModelSpec {
        name: "add-only",
        ops: &[Op::ElementwiseAdd],
    };
    const FULL: ModelSpec = ModelSpec {
        name: "mixed",
        ops: &[Op::ElementwiseAdd, Op::MatMul],
    };

    #[test]
    fn ops_declare_their_static_floor() {
        assert_eq!(Op::ElementwiseAdd.requirements().storage_range, 4);
        assert_eq!(Op::MatMul.requirements().storage_range, 12);
        assert_eq!(Op::Softmax.requirements().storage_range, 8);
    }

    #[test]
    fn model_spec_validates_every_op_against_the_device() {
        let dev = fake_info(DeviceType::Cpu, 1 << 30, 1, true);
        assert_eq!(ADD_ONLY.validate(&dev), Ok(()));
        assert_eq!(FULL.validate(&dev), Ok(()));
    }

    #[test]
    fn model_spec_reports_the_missing_capability() {
        let mut dev = fake_info(DeviceType::Cpu, 1 << 30, 1, true);
        dev.limits.max_storage_buffer_range = 4;
        assert_eq!(ADD_ONLY.validate(&dev), Ok(()));
        assert!(matches!(
            FULL.validate(&dev),
            Err(VkError::InsufficientCapacity {
                capability: "maxStorageBufferRange",
                required: 12,
                available: 4,
            })
        ));
    }

    #[test]
    fn model_spec_rejects_a_device_without_a_compute_queue() {
        let dev = fake_info(DeviceType::Cpu, 1 << 30, 1, false);
        assert_eq!(ADD_ONLY.validate(&dev), Err(VkError::NoComputeQueue));
    }
}
