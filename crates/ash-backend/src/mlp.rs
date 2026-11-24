//! Transformer MLP stage: `GELU` activation and a two-matrix feed-forward
//! block composed from the shared Vulkan kernels.
//!
//! Projections and biases stay with the caller — no model weights are bundled
//! and no token sequence is produced here.

use crate::{
    VkError,
    compute::{
        ElementwiseAdd, GELU_SPIRV, GpuBuffer, Kernel, f32_bytes, f32_from_bytes, spirv_words,
    },
    device::{CapacityRequirements, VulkanDevice},
    matmul::MatMul,
};
use std::sync::Arc;

const ELEMENTS_PER_GROUP: usize = 64;

/// Reusable pipeline for elementwise GELU (tanh approximation).
pub struct Gelu {
    kernel: Kernel,
}

impl Gelu {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(GELU_SPIRV))?,
        })
    }

    /// Apply GELU elementwise; rejects an empty input.
    pub fn run(&self, values: &[f32]) -> Result<Vec<f32>, VkError> {
        if values.is_empty() {
            return Err(VkError::Device("gelu: input must not be empty".into()));
        }
        let count = u32::try_from(values.len())
            .map_err(|_| VkError::Device("gelu: too many elements".into()))?;
        let bytes = values
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("gelu: byte length overflow".into()))?;
        let bytes_u64 =
            u64::try_from(bytes).map_err(|_| VkError::Device("gelu: buffer too large".into()))?;
        let workgroups = u32::try_from(values.len().div_ceil(ELEMENTS_PER_GROUP))
            .map_err(|_| VkError::Device("gelu: too many workgroups".into()))?;

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

/// `w2 · GELU(w1 · x)` for a `rows × dim` input batch.
pub struct FeedForward {
    expand: MatMul,
    activation: Gelu,
    project: MatMul,
    add: ElementwiseAdd,
}

impl FeedForward {
    /// Three pipelines: expand matmul, GELU, project matmul.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            expand: MatMul::new(Arc::clone(&device))?,
            activation: Gelu::new(Arc::clone(&device))?,
            project: MatMul::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// Run one MLP block with optional row-wise biases. Empty slices mean
    /// "no bias", preserving the original call shape for bias-free weights.
    #[allow(clippy::too_many_arguments)] // kernel signature: two matrices + two biases + shape
    pub fn run_with_bias(
        &self,
        input: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        rows: usize,
        dim: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, VkError> {
        let mut expanded = self.expand.run(input, w1, rows, dim, hidden)?;
        if !b1.is_empty() {
            expanded = self.add.run_broadcast(&expanded, b1)?;
        }
        let activated = self.activation.run(&expanded)?;
        let mut projected = self.project.run(&activated, w2, rows, hidden, dim)?;
        if !b2.is_empty() {
            projected = self.add.run_broadcast(&projected, b2)?;
        }
        Ok(projected)
    }

    /// Run one MLP block with caller-supplied `[dim, hidden]` and
    /// `[hidden, dim]` row-major weight matrices.
    pub fn run(
        &self,
        input: &[f32],
        w1: &[f32],
        w2: &[f32],
        rows: usize,
        dim: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, VkError> {
        self.run_with_bias(input, w1, &[], w2, &[], rows, dim, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<Arc<VulkanDevice>> {
        match VulkanDevice::headless() {
            Ok(device) => Some(Arc::new(device)),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                None
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        }
    }

    fn cpu_gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x)).tanh())
    }

    #[test]
    fn gelu_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let gelu = Gelu::new(device).expect("pipeline");
        let values: Vec<f32> = (0..130)
            .map(|i| f32::from(u8::try_from(i % 13).expect("0..13")) * 0.5 - 3.0)
            .collect();
        let actual = gelu.run(&values).expect("gpu gelu");
        let expected: Vec<f32> = values.iter().copied().map(cpu_gelu).collect();
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
        assert!(gelu.run(&[]).is_err());
    }

    #[test]
    fn feed_forward_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let block = FeedForward::new(device).expect("pipeline");
        let input = [0.2_f32, -0.4, 0.6, -0.8];
        let w1 = [0.1_f32, 0.2, -0.3, 0.4, 0.5, -0.6];
        let w2 = [0.7_f32, -0.2, 0.3, 0.1, -0.1, 0.4];
        let hidden = w1.len() / 2;
        let expanded: Vec<f32> = (0..2)
            .flat_map(|row| {
                (0..hidden).map(move |col| {
                    input[2 * row] * w1[col] + input[2 * row + 1] * w1[hidden + col]
                })
            })
            .collect();
        let expected: Vec<f32> = expanded
            .chunks(hidden)
            .flat_map(|row| {
                (0..2).map(move |col| {
                    (0..hidden)
                        .map(|h| cpu_gelu(row[h]) * w2[h * 2 + col])
                        .sum::<f32>()
                })
            })
            .collect();
        let actual = block.run(&input, &w1, &w2, 2, 2, hidden).expect("ffn");
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }
}
