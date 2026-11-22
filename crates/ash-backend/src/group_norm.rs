//! NCHW group normalization with per-channel affine parameters.

use crate::{
    VkError,
    compute::{GROUP_NORM_SPIRV, GpuBuffer, Kernel, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const ELEMENTS_PER_GROUP: u32 = 64;

/// Shapes for one NCHW group norm. `channels` must divide by `groups`.
#[derive(Debug, Clone, Copy)]
pub struct GroupNormShape {
    /// Batch size (`N`).
    pub batch: usize,
    /// Channel count (`C`).
    pub channels: usize,
    /// Spatial height (`H`).
    pub height: usize,
    /// Spatial width (`W`).
    pub width: usize,
    /// Number of channel groups (`G`).
    pub groups: usize,
}

/// Normalize each `(batch, group)` slice over `C/G × H × W` elements.
pub struct GroupNorm {
    kernel: Kernel,
}

fn validate(
    values: &[f32],
    gamma: &[f32],
    beta: &[f32],
    shape: GroupNormShape,
) -> Result<usize, VkError> {
    let invalid = || VkError::Device("group_norm: invalid shape or buffer length".into());
    let len = shape
        .batch
        .checked_mul(shape.channels)
        .and_then(|n| n.checked_mul(shape.height))
        .and_then(|n| n.checked_mul(shape.width))
        .ok_or_else(invalid)?;
    if shape.batch == 0
        || shape.channels == 0
        || shape.height == 0
        || shape.width == 0
        || shape.groups == 0
        || !shape.channels.is_multiple_of(shape.groups)
        || values.len() != len
        || gamma.len() != shape.channels
        || beta.len() != shape.channels
    {
        return Err(invalid());
    }
    Ok(len)
}

impl GroupNorm {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(GROUP_NORM_SPIRV))?,
        })
    }

    /// Normalize each group, then apply per-channel `gamma` / `beta`.
    pub fn run(
        &self,
        values: &[f32],
        gamma: &[f32],
        beta: &[f32],
        shape: GroupNormShape,
        eps: f32,
    ) -> Result<Vec<f32>, VkError> {
        let len = validate(values, gamma, beta, shape)?;
        let dim = |value: usize| {
            u32::try_from(value)
                .map_err(|_| VkError::Device("group_norm: dimension exceeds u32".into()))
        };
        let dims = [
            dim(shape.batch)?,
            dim(shape.channels)?,
            dim(shape.height)?,
            dim(shape.width)?,
            dim(shape.groups)?,
            eps.to_bits(),
        ];
        let as_bytes = |count: usize| {
            count
                .checked_mul(size_of::<f32>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| VkError::Device("group_norm: buffer too large".into()))
        };
        let bytes = as_bytes(len)?;
        let affine_len = shape.channels * 2;
        let affine_bytes = as_bytes(affine_len)?;
        let workgroups = u32::try_from(len.div_ceil(ELEMENTS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("group_norm: too many workgroups".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: bytes.max(affine_bytes).max(24),
            ..CapacityRequirements::default()
        })?;

        let affine: Vec<f32> = gamma.iter().zip(beta).flat_map(|(g, b)| [*g, *b]).collect();
        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let affine_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(&affine))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; len * size_of::<f32>()])?;
        let shape_buffer = GpuBuffer::new(
            &self.kernel.device,
            &dims
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        let result = self.kernel.dispatch(
            &[input, affine_buffer, output, shape_buffer],
            [bytes, affine_bytes, bytes, 24],
            2,
            len * size_of::<f32>(),
            [workgroups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_group_norm(
        values: &[f32],
        gamma: &[f32],
        beta: &[f32],
        shape: GroupNormShape,
        eps: f32,
    ) -> Vec<f32> {
        let spatial = shape.height * shape.width;
        let cpg = shape.channels / shape.groups;
        let count = f32::from(u16::try_from(cpg * spatial).expect("test group size"));
        let mut out = vec![0.0; values.len()];
        for n in 0..shape.batch {
            for group in 0..shape.groups {
                let mut mean = 0.0;
                for c in group * cpg..(group + 1) * cpg {
                    for s in 0..spatial {
                        mean += values[(n * shape.channels + c) * spatial + s];
                    }
                }
                mean /= count;
                let mut variance = 0.0;
                for c in group * cpg..(group + 1) * cpg {
                    for s in 0..spatial {
                        let delta = values[(n * shape.channels + c) * spatial + s] - mean;
                        variance += delta * delta;
                    }
                }
                variance /= count;
                let inv_std = 1.0 / (variance + eps).sqrt();
                for c in group * cpg..(group + 1) * cpg {
                    for s in 0..spatial {
                        let index = (n * shape.channels + c) * spatial + s;
                        out[index] = (values[index] - mean) * inv_std * gamma[c] + beta[c];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn group_norm_rejects_bad_shapes() {
        let shape = GroupNormShape {
            batch: 1,
            channels: 4,
            height: 2,
            width: 2,
            groups: 2,
        };
        let values = vec![0.0; 16];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        assert!(validate(&values, &gamma, &beta, shape).is_ok());
        assert!(
            validate(
                &values,
                &gamma,
                &beta,
                GroupNormShape { groups: 3, ..shape }
            )
            .is_err()
        );
        assert!(validate(&values, &gamma, &beta[..3], shape).is_err());
    }

    #[test]
    fn group_norm_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let shape = GroupNormShape {
            batch: 1,
            channels: 4,
            height: 2,
            width: 2,
            groups: 2,
        };
        let value = |i: usize| f32::from(u8::try_from(i % 13).expect("0..13")) * 0.375 - 2.0;
        let values: Vec<f32> = (0..16).map(value).collect();
        let gamma = vec![0.5_f32, 1.0, 1.5, 2.0];
        let beta = vec![0.1_f32, -0.1, 0.2, -0.2];

        let kernel = GroupNorm::new(device).expect("pipeline creation");
        let actual = kernel
            .run(&values, &gamma, &beta, shape, 1e-5)
            .expect("gpu dispatch");
        let expected = cpu_group_norm(&values, &gamma, &beta, shape, 1e-5);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}
