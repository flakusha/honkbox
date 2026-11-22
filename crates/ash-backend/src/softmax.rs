//! Stable row-wise softmax on the shared Vulkan compute scaffold.

use crate::{
    VkError,
    compute::{GpuBuffer, Kernel, SOFTMAX_SPIRV, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

/// One invocation per row (`local_size_x` of `shaders/softmax.comp`).
const ROWS_PER_GROUP: u32 = 64;

/// Reusable pipeline for `out[row, :] = softmax(values[row, :])`.
pub struct Softmax {
    kernel: Kernel,
}

impl Softmax {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(SOFTMAX_SPIRV))?,
        })
    }

    /// Normalize each row of a `rows × cols` f32 matrix.
    ///
    /// The shape is explicit: a flattened buffer cannot tell rows apart.
    pub fn run(&self, values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, VkError> {
        let invalid = || {
            VkError::Device("softmax: non-empty shape matching the input length required".into())
        };
        let len = rows.checked_mul(cols).ok_or_else(invalid)?;
        if rows == 0 || cols == 0 || values.len() != len {
            return Err(invalid());
        }
        let dims = [
            u32::try_from(rows).map_err(|_| invalid())?,
            u32::try_from(cols).map_err(|_| invalid())?,
        ];
        let bytes = len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("softmax: byte length overflow".into()))?;
        let workgroups = u32::try_from(rows.div_ceil(ROWS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("softmax: too many rows".into()))?;
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| VkError::Device("softmax: buffer too large".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: bytes_u64,
            ..CapacityRequirements::default()
        })?;

        let input = GpuBuffer::new(&self.kernel.device, &f32_bytes(values))?;
        let spare = GpuBuffer::new(&self.kernel.device, &[0u8; 4])?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let shape = GpuBuffer::new(
            &self.kernel.device,
            &dims
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
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

    fn cpu_softmax(values: &[f32], cols: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(values.len());
        for row in values.chunks(cols) {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = row.iter().map(|v| (v - maximum).exp()).collect();
            let total: f32 = weights.iter().sum();
            out.extend(weights.iter().map(|w| w / total));
        }
        out
    }

    #[test]
    fn softmax_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let kernel = Softmax::new(device).expect("pipeline creation");
        // Three rows exercise the partial-workgroup guard; 1000.0 checks
        // numerical stability against overflow in exp().
        let cols = 5;
        let values = vec![
            1000.0, 1001.0, 999.0, -4.0, 0.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, //
            -20.0, -10.0, 0.0, 10.0, 20.0,
        ];
        let actual = kernel.run(&values, 3, cols).expect("gpu dispatch");
        let expected = cpu_softmax(&values, cols);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        for row in actual.chunks(cols) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row sum {sum}");
        }
    }
}
