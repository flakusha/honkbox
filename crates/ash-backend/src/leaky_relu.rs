//! `LeakyReLU(0.2)` activation on the shared Vulkan compute scaffold.

use crate::{
    VkError,
    compute::{GpuBuffer, Kernel, LEAKY_RELU_SPIRV, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const ELEMENTS_PER_GROUP: u32 = 64;

/// Reusable pipeline for `result[i] = x >= 0 ? x : 0.2 * x`.
pub struct LeakyRelu {
    kernel: Kernel,
}

impl LeakyRelu {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(LEAKY_RELU_SPIRV))?,
        })
    }

    /// Apply `LeakyReLU(0.2)` elementwise; rejects an empty input.
    pub fn run(&self, values: &[f32]) -> Result<Vec<f32>, VkError> {
        if values.is_empty() {
            return Err(VkError::Device(
                "leaky_relu: input must not be empty".into(),
            ));
        }
        let count = u32::try_from(values.len())
            .map_err(|_| VkError::Device("leaky_relu: too many elements".into()))?;
        let bytes = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("leaky_relu: byte length overflow".into()))?;
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| VkError::Device("leaky_relu: buffer too large".into()))?;
        let workgroups = u32::try_from(values.len().div_ceil(ELEMENTS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("leaky_relu: too many workgroups".into()))?;

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
    fn leaky_relu_dispatches_on_vulkan_device() {
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
        let kernel = LeakyRelu::new(device).expect("pipeline creation");
        let actual = kernel.run(&values).expect("gpu dispatch");
        let expected: Vec<f32> = values
            .iter()
            .map(|x| if *x < 0.0 { 0.2 * x } else { *x })
            .collect();
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        assert!(matches!(kernel.run(&[]), Err(VkError::Device(_))));
    }
}
