//! Second real compute op: row-major matrix multiplication.
//!
//! Uses the same [`Kernel`] scaffold as elementwise add — only the shader,
//! shape metadata and workgroup geometry differ. Correctness first: the shader
//! is one thread per output element, no tiling or subgroup reduction yet.

use crate::{
    VkError,
    compute::{GpuBuffer, Kernel, MATMUL_SPIRV, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

/// `local_size.x` and `local_size.y` from `shaders/matmul.comp`.
const TILE: u32 = 8;

fn invalid_shape() -> VkError {
    VkError::Device("matmul: shapes must be non-zero and match both input lengths".into())
}

/// Reusable Vulkan pipeline for `C = A * B` (row-major f32 matrices).
pub struct MatMul {
    kernel: Kernel,
}

impl MatMul {
    /// Build the pipeline from the compile-time-embedded SPIR-V.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(MATMUL_SPIRV))?,
        })
    }

    /// Multiply `lhs` (rows × inner) by `rhs` (inner × cols).
    ///
    /// Shape is explicit rather than inferred because a flattened buffer is
    /// ambiguous; mismatches are rejected before any GPU resource is made.
    pub fn run(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, VkError> {
        let lhs_len = rows.checked_mul(inner).ok_or_else(invalid_shape)?;
        let rhs_len = inner.checked_mul(cols).ok_or_else(invalid_shape)?;
        let output_len = rows.checked_mul(cols).ok_or_else(invalid_shape)?;
        if rows == 0 || inner == 0 || cols == 0 || lhs.len() != lhs_len || rhs.len() != rhs_len {
            return Err(invalid_shape());
        }
        let dims = [
            u32::try_from(rows).map_err(|_| invalid_shape())?,
            u32::try_from(inner).map_err(|_| invalid_shape())?,
            u32::try_from(cols).map_err(|_| invalid_shape())?,
        ];
        let dims_bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
        let lhs_bytes = lhs_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(invalid_shape)?;
        let rhs_bytes = rhs_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(invalid_shape)?;
        let output_bytes = output_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(invalid_shape)?;
        let tile = usize::try_from(TILE).expect("8 fits usize");
        let groups_x = u32::try_from(rows.div_ceil(tile)).map_err(|_| invalid_shape())?;
        let groups_y = u32::try_from(cols.div_ceil(tile)).map_err(|_| invalid_shape())?;
        let as_u64 = |len: usize| {
            u64::try_from(len).map_err(|_| VkError::Device("matmul: buffer too large".into()))
        };
        // One storage binding may be as large as the biggest input or output.
        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [groups_x, groups_y, 1],
            storage_range: as_u64(lhs_bytes.max(rhs_bytes).max(output_bytes))?,
            ..CapacityRequirements::default()
        })?;

        let left = GpuBuffer::new(&self.kernel.device, &f32_bytes(lhs))?;
        let right = GpuBuffer::new(&self.kernel.device, &f32_bytes(rhs))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; output_bytes])?;
        let shape = GpuBuffer::new(&self.kernel.device, &dims_bytes)?;
        let bytes = self.kernel.dispatch(
            &[left, right, output, shape],
            [
                as_u64(lhs_bytes)?,
                as_u64(rhs_bytes)?,
                as_u64(output_bytes)?,
                as_u64(dims_bytes.len())?,
            ],
            2,
            output_bytes,
            [groups_x, groups_y, 1],
        )?;
        Ok(f32_from_bytes(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let kernel = MatMul::new(device).expect("pipeline creation");
        // Non-multiples of the 8x8 tile exercise the shader's bounds guard.
        let (rows, inner, cols) = (5usize, 7usize, 3usize);
        let lhs: Vec<f32> = (0..rows * inner)
            .map(|i| f32::from(u8::try_from(i % 5).expect("0..5")) + 1.0)
            .collect();
        let rhs: Vec<f32> = (0..inner * cols)
            .map(|i| f32::from(u8::try_from(i % 3).expect("0..3")) + 0.5)
            .collect();
        let actual = kernel
            .run(&lhs, &rhs, rows, inner, cols)
            .expect("gpu dispatch");
        assert_eq!(actual.len(), rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let expected: f32 = (0..inner)
                    .map(|k| lhs[row * inner + k] * rhs[k * cols + col])
                    .sum();
                assert!(
                    (actual[row * cols + col] - expected).abs() < 1e-3,
                    "M[{row},{col}]: {} != {expected}",
                    actual[row * cols + col]
                );
            }
        }
        assert!(matches!(
            kernel.run(&lhs, &rhs, rows, inner, cols + 1),
            Err(VkError::Device(_))
        ));
    }
}
