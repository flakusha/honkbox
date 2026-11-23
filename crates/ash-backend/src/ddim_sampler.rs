//! Eta=0 DDIM loop over caller-supplied latents, schedule and predictions.
//!
//! The diffusion backbone is not implemented here: `predict` receives the
//! current latent and step index and returns its noise prediction. Callers can
//! capture a conditioned embedding in that closure once the `UNet` lands.

use crate::{
    VkError,
    device::VulkanDevice,
    sampler::{DdimAlphas, DdimStep},
};
use std::sync::Arc;

/// Reusable single-step kernel plus the schedule loop around it.
pub struct DdimSampler {
    step: DdimStep,
}

impl DdimSampler {
    /// One pipeline: the DDIM eta=0 update.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            step: DdimStep::new(device)?,
        })
    }

    /// Walk `cumprod` from `t` to `t-1`, calling `predict` once per step.
    ///
    /// `cumprod` holds `steps + 1` values in (0, 1], non-increasing from the
    /// noisiest timestep to the cleanest; the initial latent is supplied by
    /// the caller. The closure receives that latent and the zero-based step
    /// index.
    pub fn run<F>(
        &self,
        initial_latent: &[f32],
        cumprod: &[f32],
        mut predict: F,
    ) -> Result<Vec<f32>, VkError>
    where
        F: FnMut(&[f32], usize) -> Result<Vec<f32>, VkError>,
    {
        if cumprod.len() < 2 || initial_latent.is_empty() {
            return Err(VkError::Device(
                "ddim sampler: need at least two alphas and a non-empty latent".into(),
            ));
        }
        let mut latent = initial_latent.to_vec();
        for index in 0..cumprod.len() - 1 {
            let out_of_range =
                || VkError::Device("ddim sampler: schedule index out of range".into());
            let alpha = cumprod.get(index).copied().ok_or_else(out_of_range)?;
            let alpha_prev = cumprod.get(index + 1).copied().ok_or_else(out_of_range)?;
            if alpha_prev > alpha {
                return Err(VkError::Device(
                    "ddim sampler: cumulative alphas must be non-increasing".into(),
                ));
            }
            let alphas = DdimAlphas::from_cumprod(alpha, alpha_prev)?;
            let prediction = predict(&latent, index)?;
            latent = self.step.run(&latent, &prediction, alphas)?;
        }
        Ok(latent)
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

    fn cpu_step(latent: &[f32], prediction: &[f32], alphas: DdimAlphas) -> Vec<f32> {
        latent
            .iter()
            .zip(prediction)
            .map(|(x, eps)| {
                alphas.sqrt_alpha_prev
                    * ((x - alphas.sqrt_one_minus_alpha * eps) / alphas.sqrt_alpha)
                    + alphas.sqrt_one_minus_alpha_prev * eps
            })
            .collect()
    }

    #[test]
    fn loop_matches_sequential_cpu_reference() {
        let Some(device) = device() else { return };
        let sampler = DdimSampler::new(device).expect("sampler");
        let latent = [0.4_f32, -0.2, 0.7, 0.1];
        let schedule = [0.9_f32, 0.6, 0.3, 0.1];
        let mut expected = latent.to_vec();
        for index in 0..schedule.len() - 1 {
            let prediction: Vec<f32> = expected.iter().map(|x| x * 0.25).collect();
            let alphas = DdimAlphas::from_cumprod(schedule[index], schedule[index + 1])
                .expect("test alphas");
            expected = cpu_step(&expected, &prediction, alphas);
        }
        let mut seen = Vec::new();
        let actual = sampler
            .run(&latent, &schedule, |current, index| {
                seen.push(index);
                Ok(current.iter().map(|x| x * 0.25).collect())
            })
            .expect("sample");
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn loop_rejects_bad_schedules_and_predictions() {
        let Some(device) = device() else { return };
        let sampler = DdimSampler::new(device).expect("sampler");
        let latent = [0.5_f32, -0.5];
        let good = [0.9_f32, 0.4];
        let bad_scale = |_current: &[f32], _index: usize| Ok(vec![0.0; 3]);
        let failed =
            |_current: &[f32], _index: usize| Err(VkError::Device("predictor failed".into()));

        assert!(sampler.run(&latent, &[0.9], |c, _| Ok(c.to_vec())).is_err());
        assert!(sampler.run(&[], &good, |c, _| Ok(c.to_vec())).is_err());
        assert!(
            sampler
                .run(&latent, &[0.4, 0.9], |c, _| Ok(c.to_vec()))
                .is_err()
        );
        assert!(sampler.run(&latent, &good, bad_scale).is_err());
        assert!(matches!(
            sampler.run(&latent, &good, failed),
            Err(VkError::Device(msg)) if msg.contains("predictor failed")
        ));
    }
}
