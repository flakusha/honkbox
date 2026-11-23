//! Sinusoidal timestep embedding through two caller-weighted linear layers.
//!
//! ponytail: the frequency table is computed on the host (a few dozen f32
//! values) rather than by a shader; add a kernel only if profiling shows it
//! matters.

use crate::{VkError, device::VulkanDevice, matmul::MatMul, silu::Silu};
use std::sync::Arc;

/// Projection weights for the timestep MLP.
pub struct TimestepWeights<'a> {
    /// `[embedding_dim, hidden]` first linear layer.
    pub w1: &'a [f32],
    /// `[hidden, channels]` second linear layer.
    pub w2: &'a [f32],
}

/// Sinusoidal encoding → linear → `SiLU` → linear.
pub struct TimestepEmbed {
    first: MatMul,
    activation: Silu,
    second: MatMul,
}

fn word(value: usize) -> Result<f32, VkError> {
    let converted = u16::try_from(value)
        .map_err(|_| VkError::Device("timestep: value exceeds u16::MAX".into()))?;
    Ok(f32::from(converted))
}

/// CLIP-style sinusoidal frequencies: even slots are cos, odd slots sin.
pub fn sinusoidal(timestep: usize, dim: usize, max_period: usize) -> Result<Vec<f32>, VkError> {
    if dim == 0 || max_period == 0 {
        return Err(VkError::Device(
            "timestep: embedding width and max period must be positive".into(),
        ));
    }
    let width = word(dim)?;
    let period = word(max_period)?;
    let step = word(timestep)?;
    let log_period = period.ln();
    let mut out = vec![0.0_f32; dim];
    for index in 0..dim / 2 {
        let exponent = -2.0 * word(index)? * log_period / width;
        let angle = step * exponent.exp();
        out[2 * index] = angle.cos();
        out[2 * index + 1] = angle.sin();
    }
    Ok(out)
}

impl TimestepEmbed {
    /// Two matmuls plus the shared `SiLU` kernel.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            first: MatMul::new(Arc::clone(&device))?,
            activation: Silu::new(Arc::clone(&device))?,
            second: MatMul::new(device)?,
        })
    }

    /// Return the `channels`-wide embedding for one diffusion timestep.
    pub fn run(
        &self,
        timestep: usize,
        weights: &TimestepWeights<'_>,
        embedding_dim: usize,
        hidden: usize,
        channels: usize,
    ) -> Result<Vec<f32>, VkError> {
        let embedding = sinusoidal(timestep, embedding_dim, 10_000)?;
        let projected = self
            .first
            .run(&embedding, weights.w1, 1, embedding_dim, hidden)?;
        let activated = self.activation.run(&projected)?;
        self.second.run(&activated, weights.w2, 1, hidden, channels)
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

    #[test]
    fn sinusoidal_layout_and_rejections() {
        let values = sinusoidal(0, 6, 10_000).expect("table");
        assert_eq!(values.len(), 6);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        assert!(values[2].abs() <= 1.0 && values[3].abs() <= 1.0);
        assert!(sinusoidal(5, 0, 10_000).is_err());
        assert!(sinusoidal(5, 4, 0).is_err());
    }

    #[test]
    fn timestep_mlp_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let embed = TimestepEmbed::new(device).expect("pipeline");
        let (embedding_dim, hidden, channels, timestep) = (4usize, 3usize, 2usize, 5usize);
        let w1 = [
            0.1_f32, 0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 0.1, 0.2,
        ];
        let w2 = [0.3_f32, -0.2, 0.1, 0.4, -0.5, 0.6];
        let weights = TimestepWeights { w1: &w1, w2: &w2 };
        let input = sinusoidal(timestep, embedding_dim, 10_000).expect("table");

        let activated: Vec<f32> = (0..hidden)
            .map(|h| {
                (0..embedding_dim)
                    .map(|d| input[d] * w1[d * hidden + h])
                    .sum::<f32>()
            })
            .map(|x| x / (1.0 + (-x).exp()))
            .collect();
        let expected: Vec<f32> = (0..channels)
            .map(|c| {
                (0..hidden)
                    .map(|h| activated[h] * w2[h * channels + c])
                    .sum::<f32>()
            })
            .collect();

        let actual = embed
            .run(timestep, &weights, embedding_dim, hidden, channels)
            .expect("gpu embed");
        assert_eq!(actual.len(), channels);
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }

        let short = TimestepWeights {
            w1: &w1[..3],
            w2: &w2,
        };
        assert!(
            embed
                .run(timestep, &short, embedding_dim, hidden, channels)
                .is_err()
        );
    }
}
