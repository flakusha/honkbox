//! Row/column transpose on the shared Vulkan compute scaffold.

use crate::{
    VkError,
    compute::{GpuBuffer, Kernel, TRANSPOSE_SPIRV, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const ELEMENTS_PER_GROUP: usize = 64;

/// Reusable pipeline for `out[col * rows + row] = in[row * cols + col]`.
pub struct Transpose {
    kernel: Kernel,
}

impl Transpose {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(TRANSPOSE_SPIRV))?,
        })
    }

    /// Transpose a `rows × cols` row-major matrix into `cols × rows`.
    pub fn run(&self, values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, VkError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| VkError::Device("transpose: shape overflow".into()))?;
        if rows == 0 || cols == 0 || values.len() != expected {
            return Err(VkError::Device(
                "transpose: non-empty shape must match input length".into(),
            ));
        }
        let dims = [
            u32::try_from(rows).map_err(|_| VkError::Device("transpose: too many rows".into()))?,
            u32::try_from(cols).map_err(|_| VkError::Device("transpose: too many cols".into()))?,
        ];
        let dims_bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
        let bytes = expected
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("transpose: byte length overflow".into()))?;
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| VkError::Device("transpose: buffer too large".into()))?;
        let workgroups = u32::try_from(expected.div_ceil(ELEMENTS_PER_GROUP))
            .map_err(|_| VkError::Device("transpose: too many workgroups".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: bytes_u64,
            ..CapacityRequirements::default()
        })?;

        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let spare = GpuBuffer::new(&self.kernel.device, &[0u8; 4])?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let shape = GpuBuffer::new(&self.kernel.device, &dims_bytes)?;
        let result = self.kernel.dispatch(
            &[input, spare, output, shape],
            [bytes_u64, 4, bytes_u64, 8],
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

    #[test]
    fn transpose_rejects_bad_shapes() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => return,
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let transpose = Transpose::new(device).expect("pipeline");
        assert!(transpose.run(&[1.0, 2.0, 3.0], 2, 2).is_err());
        assert!(transpose.run(&[], 0, 2).is_err());
    }

    #[test]
    fn transpose_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let transpose = Transpose::new(device).expect("pipeline");
        let input: Vec<f32> = (0..15)
            .map(|i| f32::from(u8::try_from(i).expect("0..15")) * 0.5)
            .collect();
        let actual = transpose.run(&input, 3, 5).expect("gpu transpose");
        let expected: Vec<f32> = (0..15)
            .map(|index| input[(index % 3) * 5 + index / 3])
            .collect();
        assert_eq!(actual, expected);
    }
}
