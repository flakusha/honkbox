//! Sampling primitives for the direct-Vulkan path: seeded latent noise and
//! the DDIM (eta=0) denoising step.
//!
//! Nothing here starts a generation: the scheduler needs a model prediction,
//! and `generate` still names the missing pieces instead of faking an image.

use crate::{
    VkError,
    compute::{DDIM_STEP_SPIRV, GpuBuffer, Kernel, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
#[cfg(feature = "ash-sd")]
use honkbox_sd_core::splitmix64;
use std::sync::Arc;

const LOCAL_SIZE: usize = 64;

/// Square-root coefficients of consecutive cumulative alphas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdimAlphas {
    /// `sqrt(alpha_t)`.
    pub sqrt_alpha: f32,
    /// `sqrt(1 - alpha_t)`.
    pub sqrt_one_minus_alpha: f32,
    /// `sqrt(alpha_{t-1})`.
    pub sqrt_alpha_prev: f32,
    /// `sqrt(1 - alpha_{t-1})`.
    pub sqrt_one_minus_alpha_prev: f32,
}

impl DdimAlphas {
    /// Derive the coefficients from two cumulative alphas (current, previous).
    pub fn from_cumprod(alpha: f32, alpha_prev: f32) -> Result<Self, VkError> {
        let valid = |v: f32| v.is_finite() && v > 0.0 && v <= 1.0;
        if !valid(alpha) || !valid(alpha_prev) {
            return Err(VkError::Device(
                "ddim: cumulative alphas must be finite and in (0, 1]".into(),
            ));
        }
        Ok(Self {
            sqrt_alpha: alpha.sqrt(),
            sqrt_one_minus_alpha: (1.0 - alpha).sqrt(),
            sqrt_alpha_prev: alpha_prev.sqrt(),
            sqrt_one_minus_alpha_prev: (1.0 - alpha_prev).sqrt(),
        })
    }
}

/// Reusable pipeline for one DDIM eta=0 update over a latent tensor.
pub struct DdimStep {
    kernel: Kernel,
}

impl DdimStep {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(DDIM_STEP_SPIRV))?,
        })
    }

    /// `x_{t-1} = sqrt(a_prev) * (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)
    ///           + sqrt(1-a_prev) * eps`
    pub fn run(
        &self,
        latent: &[f32],
        prediction: &[f32],
        alphas: DdimAlphas,
    ) -> Result<Vec<f32>, VkError> {
        let invalid = || {
            VkError::Device("ddim: non-empty latent and prediction of equal length required".into())
        };
        if latent.is_empty() || latent.len() != prediction.len() {
            return Err(invalid());
        }
        let length = u32::try_from(latent.len()).map_err(|_| invalid())?;
        let bytes = latent
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("ddim: byte length overflow".into()))?;
        let bytes_u64 =
            u64::try_from(bytes).map_err(|_| VkError::Device("ddim: buffer too large".into()))?;
        let workgroups = u32::try_from(latent.len().div_ceil(LOCAL_SIZE))
            .map_err(|_| VkError::Device("ddim: too many elements".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: bytes_u64,
            ..CapacityRequirements::default()
        })?;

        let params = [
            length,
            alphas.sqrt_alpha.to_bits(),
            alphas.sqrt_one_minus_alpha.to_bits(),
            alphas.sqrt_alpha_prev.to_bits(),
            alphas.sqrt_one_minus_alpha_prev.to_bits(),
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<u8>>();
        let latent_buf = GpuBuffer::new(&self.kernel.device, &f32_bytes(latent))?;
        let prediction_buf = GpuBuffer::new(&self.kernel.device, &f32_bytes(prediction))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; bytes])?;
        let params_buf = GpuBuffer::new(&self.kernel.device, &params)?;
        let result = self.kernel.dispatch(
            &[latent_buf, prediction_buf, output, params_buf],
            [bytes_u64, bytes_u64, bytes_u64, 20],
            2,
            bytes,
            [workgroups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

/// Deterministic initial latent, derived only from a seed and an index.
///
/// ponytail: uniform `[-1, 1]` rather than `N(0, 1)`; add Box-Muller when a
/// real sampler consumes it and requires Gaussian noise.
#[cfg(feature = "ash-sd")]
pub fn seeded_uniform_noise(len: usize, seed: u64) -> Result<Vec<f32>, VkError> {
    let count = u64::try_from(len)
        .map_err(|_| VkError::Device("latent noise: length exceeds u64".into()))?;
    (0..count)
        .map(|index| {
            let hash = splitmix64(seed ^ index);
            let bits = u32::try_from(hash >> 32).expect("shifted hash fits u32");
            Ok(f32::from_bits(0x3f80_0000 | (bits & 0x7f_ffff)) - 1.0)
        })
        .collect()
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

    fn cpu_step(latent: &[f32], prediction: &[f32], alphas: DdimAlphas) -> Vec<f32> {
        latent
            .iter()
            .zip(prediction)
            .map(|(x, eps)| {
                let x0 = (x - alphas.sqrt_one_minus_alpha * eps) / alphas.sqrt_alpha;
                alphas.sqrt_alpha_prev * x0 + alphas.sqrt_one_minus_alpha_prev * eps
            })
            .collect()
    }

    #[test]
    fn alphas_reject_out_of_range_cumprod() {
        assert!(DdimAlphas::from_cumprod(0.0, 0.9).is_err());
        assert!(DdimAlphas::from_cumprod(1.1, 0.9).is_err());
        assert!(DdimAlphas::from_cumprod(f32::NAN, 0.9).is_err());
        assert!(DdimAlphas::from_cumprod(0.8, -0.1).is_err());
        let alphas = DdimAlphas::from_cumprod(0.8, 0.9).expect("valid alphas");
        assert!((alphas.sqrt_alpha - 0.8_f32.sqrt()).abs() < 1e-6);
        assert!((alphas.sqrt_alpha_prev - 0.9_f32.sqrt()).abs() < 1e-6);
    }

    #[cfg(feature = "ash-sd")]
    #[test]
    fn seeded_noise_is_reproducible_and_seed_sensitive() {
        let first = seeded_uniform_noise(96, 42).expect("noise");
        let again = seeded_uniform_noise(96, 42).expect("noise");
        let other = seeded_uniform_noise(96, 43).expect("noise");
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert!(first.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn ddim_step_dispatches_on_vulkan_device() {
        let Some(device) = device() else { return };
        let sampler = DdimStep::new(device).expect("pipeline");
        let latent = [0.4_f32, -0.2, 0.9, -0.7, 0.1, 0.6, -0.3, 0.8];
        let prediction = [-0.1_f32, 0.5, 0.2, -0.6, 0.3, -0.4, 0.7, 0.15];
        let alphas = DdimAlphas::from_cumprod(0.8, 0.9).expect("alphas");
        let expected = cpu_step(&latent, &prediction, alphas);
        let actual = sampler
            .run(&latent, &prediction, alphas)
            .expect("ddim step");
        assert_eq!(actual.len(), latent.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn ddim_step_rejects_bad_shapes() {
        let Some(device) = device() else { return };
        let sampler = DdimStep::new(device).expect("pipeline");
        let alphas = DdimAlphas::from_cumprod(0.8, 0.9).expect("alphas");
        assert!(sampler.run(&[], &[], alphas).is_err());
        assert!(sampler.run(&[0.1], &[], alphas).is_err());
        assert!(sampler.run(&[0.1], &[0.2, 0.3], alphas).is_err());
    }
}
