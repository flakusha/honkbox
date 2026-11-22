//! `SiLU` (`x · sigmoid(x)`) activation on the shared Vulkan compute scaffold.

use crate::{
    VkError,
    compute::{GpuBuffer, Kernel, SILU_SPIRV, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const ELEMENTS_PER_GROUP: u32 = 64;

/// Reusable pipeline for `result[i] = values[i] / (1 + exp(-values[i]))`.
pub struct Silu {
    kernel: Kernel,
}

impl Silu {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(SILU_SPIRV))?,
        })
    }

    /// Apply `SiLU` elementwise; rejects an empty input.
    pub fn run(&self, values: &[f32]) -> Result<Vec<f32>, VkError> {
        if values.is_empty() {
            return Err(VkError::Device("silu: input must not be empty".into()));
        }
        let count = u32::try_from(values.len())
            .map_err(|_| VkError::Device("silu: too many elements".into()))?;
        let bytes = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("silu: byte length overflow".into()))?;
        let bytes_u64 =
            u64::try_from(bytes).map_err(|_| VkError::Device("silu: buffer too large".into()))?;
        let workgroups = u32::try_from(values.len().div_ceil(ELEMENTS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("silu: too many workgroups".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: bytes_u64,
            ..CapacityRequirements::default()
        })?;

        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let spare = GpuBuffer::new(&self.kernel.device, &[0u8; 4])?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let shape = GpuBuffer::new(&self.kernel.device, &count.to_le_bytes())?;
        let result = self.kernel.dispatch(
            &[input, spare, output, shape],
            [bytes_u64, 4, bytes_u64, 4],
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
    fn silu_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let values: Vec<f32> = (0..129)
            .map(|i| f32::from(u8::try_from(i % 11).expect("0..11")) * 0.5 - 2.5)
            .collect();
        let kernel = Silu::new(device).expect("pipeline creation");
        let actual = kernel.run(&values).expect("gpu dispatch");
        let expected: Vec<f32> = values.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }
}
