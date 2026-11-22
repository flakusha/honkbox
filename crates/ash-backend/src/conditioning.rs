//! Classifier-free guidance and the text-conditioning container it consumes.
//!
//! Text encoding is deliberately not here: CLIP tokenization needs vocab and
//! merge assets plus an encoder checkpoint. Callers supply embeddings; nothing
//! is derived from a prompt string, so no embedding or image is faked.

use crate::{
    VkError,
    compute::{CFG_GUIDANCE_SPIRV, GpuBuffer, Kernel, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const LOCAL_SIZE: usize = 64;

/// Positive/negative text embeddings for one conditioning batch.
#[derive(Debug, Clone, PartialEq)]
pub struct TextConditioning {
    positive: Vec<f32>,
    negative: Vec<f32>,
    tokens: usize,
    dim: usize,
}

impl TextConditioning {
    /// Accept only finite, equally shaped positive and negative embeddings.
    pub fn new(
        positive: Vec<f32>,
        negative: Vec<f32>,
        tokens: usize,
        dim: usize,
    ) -> Result<Self, VkError> {
        let invalid = || {
            VkError::Device("conditioning: finite, equally shaped token embeddings required".into())
        };
        let len = tokens.checked_mul(dim).ok_or_else(invalid)?;
        if tokens == 0
            || dim == 0
            || positive.len() != len
            || negative.len() != len
            || !positive.iter().all(|v| v.is_finite())
            || !negative.iter().all(|v| v.is_finite())
        {
            return Err(invalid());
        }
        Ok(Self {
            positive,
            negative,
            tokens,
            dim,
        })
    }

    /// Number of tokens in each embedding.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Per-token embedding width.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply classifier-free guidance to this conditioning.
    pub fn guided(&self, guidance: &Cfg, scale: f32) -> Result<Vec<f32>, VkError> {
        guidance.run(&self.positive, &self.negative, scale)
    }
}

/// Reusable pipeline for `uncond + scale * (cond - uncond)`.
pub struct Cfg {
    kernel: Kernel,
}

impl Cfg {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(CFG_GUIDANCE_SPIRV))?,
        })
    }

    /// Blend conditional and unconditional vectors with guidance `scale`.
    ///
    /// `scale == 1.0` returns the conditional vector unchanged.
    pub fn run(
        &self,
        conditional: &[f32],
        unconditional: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, VkError> {
        let invalid = || {
            VkError::Device("cfg: non-empty vectors of equal length and scale >= 1 required".into())
        };
        if !scale.is_finite() || scale < 1.0 {
            return Err(invalid());
        }
        if conditional.is_empty() || conditional.len() != unconditional.len() {
            return Err(invalid());
        }
        let length = u32::try_from(conditional.len()).map_err(|_| invalid())?;
        let bytes = conditional
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("cfg: byte length overflow".into()))?;
        let bytes_u64 =
            u64::try_from(bytes).map_err(|_| VkError::Device("cfg: buffer too large".into()))?;
        let workgroups = u32::try_from(conditional.len().div_ceil(LOCAL_SIZE))
            .map_err(|_| VkError::Device("cfg: too many elements".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: bytes_u64,
            ..CapacityRequirements::default()
        })?;

        let params = [length, scale.to_bits()]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>();
        let cond_buf = GpuBuffer::new(&self.kernel.device, &f32_bytes(conditional))?;
        let uncond_buf = GpuBuffer::new(&self.kernel.device, &f32_bytes(unconditional))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let params_buf = GpuBuffer::new(&self.kernel.device, &params)?;
        let result = self.kernel.dispatch(
            &[cond_buf, uncond_buf, output, params_buf],
            [bytes_u64, bytes_u64, bytes_u64, 8],
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

    fn cpu_guidance(cond: &[f32], uncond: &[f32], scale: f32) -> Vec<f32> {
        cond.iter()
            .zip(uncond)
            .map(|(c, u)| u + scale * (c - u))
            .collect()
    }

    #[test]
    fn conditioning_rejects_bad_embeddings() {
        assert!(TextConditioning::new(vec![], vec![], 0, 4).is_err());
        assert!(TextConditioning::new(vec![0.1; 8], vec![0.2; 4], 2, 4).is_err());
        assert!(TextConditioning::new(vec![f32::NAN; 4], vec![0.1; 4], 1, 4).is_err());
        let ok = TextConditioning::new(vec![0.1; 8], vec![0.2; 8], 2, 4).expect("conditioning");
        assert_eq!((ok.tokens(), ok.dim()), (2, 4));
    }

    #[test]
    fn cfg_dispatches_on_vulkan_device() {
        let Some(device) = device() else { return };
        let guidance = Cfg::new(device).expect("pipeline");
        let cond = [0.8_f32, -0.4, 0.2, 1.1, -0.9, 0.3, 0.6, -0.1];
        let uncond = [0.1_f32, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
        let scale = 7.5;
        let expected = cpu_guidance(&cond, &uncond, scale);
        let actual = guidance.run(&cond, &uncond, scale).expect("cfg");
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn cfg_scale_one_keeps_conditional_values() {
        let Some(device) = device() else { return };
        let guidance = Cfg::new(device).expect("pipeline");
        let cond = [0.5_f32, -1.0, 0.25, 0.75];
        let uncond = [-0.2_f32, 0.3, 0.9, -0.4];
        let actual = guidance.run(&cond, &uncond, 1.0).expect("cfg");
        for (got, want) in actual.iter().zip(&cond) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn cfg_rejects_bad_scales_and_shapes() {
        let Some(device) = device() else { return };
        let guidance = Cfg::new(device).expect("pipeline");
        assert!(guidance.run(&[0.1], &[0.2], 0.5).is_err());
        assert!(guidance.run(&[0.1], &[0.2], f32::NAN).is_err());
        assert!(guidance.run(&[], &[], 7.0).is_err());
        assert!(guidance.run(&[0.1], &[], 7.0).is_err());
    }
}
