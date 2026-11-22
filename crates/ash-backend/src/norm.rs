//! Row-wise layer norm with learnable scale/shift on the shared scaffold.

use crate::{
    VkError,
    compute::{GpuBuffer, Kernel, LAYER_NORM_SPIRV, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const ROWS_PER_GROUP: u32 = 64;

/// `out[row, i] = (x[row, i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]`.
pub struct LayerNorm {
    kernel: Kernel,
}

fn validate(
    values: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<usize, VkError> {
    let invalid = || {
        VkError::Device("layer_norm: non-empty shape and matching affine lengths required".into())
    };
    let len = rows.checked_mul(cols).ok_or_else(invalid)?;
    if rows == 0 || cols == 0 || values.len() != len || gamma.len() != cols || beta.len() != cols {
        return Err(invalid());
    }
    Ok(len)
}

impl LayerNorm {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(LAYER_NORM_SPIRV))?,
        })
    }

    /// Normalize each row, then apply per-column `gamma` / `beta`.
    pub fn run(
        &self,
        values: &[f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, VkError> {
        let len = validate(values, rows, cols, gamma, beta)?;
        let dims = [
            u32::try_from(rows).map_err(|_| VkError::Device("layer_norm: too many rows".into()))?,
            u32::try_from(cols)
                .map_err(|_| VkError::Device("layer_norm: too many columns".into()))?,
            eps.to_bits(),
        ];
        let bytes = len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("layer_norm: byte length overflow".into()))?;
        let affine_bytes = bytes
            .checked_mul(2)
            .ok_or_else(|| VkError::Device("layer_norm: affine byte length overflow".into()))?;
        let workgroups = u32::try_from(rows.div_ceil(ROWS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("layer_norm: too many workgroups".into()))?;
        let as_u64 = |len: usize| {
            u64::try_from(len).map_err(|_| VkError::Device("layer_norm: buffer too large".into()))
        };

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: as_u64(affine_bytes.max(bytes))?,
            ..CapacityRequirements::default()
        })?;

        let affine: Vec<f32> = gamma.iter().zip(beta).flat_map(|(g, b)| [*g, *b]).collect();
        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let affine_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(&affine))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let shape = GpuBuffer::new(
            &self.kernel.device,
            &dims
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        let result = self.kernel.dispatch(
            &[input, affine_buffer, output, shape],
            [as_u64(bytes)?, as_u64(affine_bytes)?, as_u64(bytes)?, 12],
            2,
            bytes,
            [workgroups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_layer_norm(
        values: &[f32],
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(values.len());
        let count = f32::from(u16::try_from(cols).expect("test cols"));
        for row in values.chunks(cols) {
            let mean: f32 = row.iter().sum::<f32>() / count;
            let variance: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / count;
            let inv_std = 1.0 / (variance + eps).sqrt();
            for (i, value) in row.iter().enumerate() {
                out.push((value - mean) * inv_std * gamma[i] + beta[i]);
            }
        }
        out
    }

    #[test]
    fn layer_norm_rejects_bad_shapes() {
        let values = vec![1.0; 6];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        assert!(validate(&values, 2, 3, &gamma, &beta).is_ok());
        assert!(validate(&values, 2, 3, &gamma, &beta[..2]).is_err());
        assert!(validate(&values, 2, 0, &gamma, &beta).is_err());
    }

    #[test]
    fn layer_norm_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let cols = 4;
        let values = vec![
            1.0_f32, 2.0, 4.0, 8.0, -3.0, 0.5, 7.0, 2.5, 9.0, -1.0, 0.0, 3.0,
        ];
        let gamma = vec![0.8_f32, 1.2, -0.5, 2.0];
        let beta = vec![0.1_f32, -0.2, 0.3, -0.4];
        let kernel = LayerNorm::new(device).expect("pipeline creation");
        let actual = kernel
            .run(&values, 3, cols, &gamma, &beta, 1e-5)
            .expect("gpu dispatch");
        let expected = cpu_layer_norm(&values, cols, &gamma, &beta, 1e-5);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}
