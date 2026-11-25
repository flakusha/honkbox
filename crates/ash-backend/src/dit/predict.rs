//! Model-agnostic velocity predictor: `(latent, sigma, conditioning) -> velocity`.
#![allow(clippy::cast_precision_loss)]

use super::block::{DitBlockWeights, dit_block, matmul_rows, patch_embed, patch_flatten};
use super::flow::{euler_step, sigma_embedding};
use super::positions::position_grid;
use super::spec::{FlowDitSpec, PositionAxes};
use super::{FlowDitSpec as Spec, invalid};
use crate::VkError;
/// Weights for one synthetic predictor. All vectors are row-major f32.
#[derive(Debug)]
pub struct FlowPredictorWeights {
    /// `[patch_volume, width]` patch projection.
    pub embed: Vec<f32>,
    /// `[width]` patch bias.
    pub embed_bias: Vec<f32>,
    /// Sigma MLP first layer `[width, width]`.
    pub sigma_mlp_1: Vec<f32>,
    /// Sigma MLP second layer `[width, width]`.
    pub sigma_mlp_2: Vec<f32>,
    /// Shared single-stream block weights.
    pub block: DitBlockWeights,
    /// `[width, patch_volume]` velocity head.
    pub head: Vec<f32>,
    /// `[patch_volume]` velocity bias.
    pub head_bias: Vec<f32>,
}

/// A single-stream flow predictor over a declared patch grid.
#[derive(Debug)]
pub struct FlowPredictor {
    spec: Spec,
    patch: Vec<usize>,
    weights: FlowPredictorWeights,
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

impl FlowPredictor {
    /// Validate geometry and every weight length before any work is done.
    pub fn new(
        spec: FlowDitSpec,
        patch: &[usize],
        weights: FlowPredictorWeights,
    ) -> Result<Self, VkError> {
        spec.validate()?;
        let dims = spec.axes.dims();
        let axis_count = match spec.axes {
            PositionAxes::Image { .. } => 2,
            PositionAxes::Video { .. } => 3,
        };
        if patch.len() != axis_count {
            return Err(invalid("patch", "axis count must match position axes"));
        }
        let mut volume = spec.latent_channels;
        for (axis, &size) in patch.iter().enumerate() {
            if size == 0 {
                return Err(invalid("patch", &format!("axis {axis} must be non-zero")));
            }
            dims[axis]
                .checked_mul(size)
                .ok_or_else(|| invalid("patch", "latent span overflow"))?;
            volume = volume
                .checked_mul(size)
                .ok_or_else(|| invalid("patch_volume", "overflow"))?;
        }
        if volume != spec.patch_volume {
            return Err(invalid(
                "patch_volume",
                "must equal latent_channels * patch product",
            ));
        }
        let width = spec.width;
        let expected = [
            ("embed", weights.embed.len(), volume * width),
            ("embed_bias", weights.embed_bias.len(), width),
            ("sigma_mlp_1", weights.sigma_mlp_1.len(), width * width),
            ("sigma_mlp_2", weights.sigma_mlp_2.len(), width * width),
            ("head", weights.head.len(), width * volume),
            ("head_bias", weights.head_bias.len(), volume),
        ];
        for (field, actual, wanted) in expected {
            if actual != wanted {
                return Err(invalid(field, &format!("length {actual} must be {wanted}")));
            }
        }
        Ok(Self {
            spec,
            patch: patch.to_vec(),
            weights,
        })
    }

    /// Predict velocity in the same row-major patch order as the flattened input.
    pub fn velocity(&self, latent: &[f32], sigma: f32, cond: &[f32]) -> Result<Vec<f32>, VkError> {
        let spec = self.spec;
        let dims = spec.axes.dims();
        let axis_count = match spec.axes {
            PositionAxes::Image { .. } => 2,
            PositionAxes::Video { .. } => 3,
        };
        let spatial: Vec<usize> = dims[..axis_count]
            .iter()
            .zip(&self.patch)
            .map(|(grid, size)| {
                grid.checked_mul(*size)
                    .ok_or_else(|| invalid("patch", "latent span overflow"))
            })
            .collect::<Result<_, _>>()?;
        let tokens = spec.tokens()?;
        let grid = spatial.iter().product::<usize>();
        let wanted = spec
            .latent_channels
            .checked_mul(grid)
            .ok_or_else(|| invalid("latent", "element count overflow"))?;
        if latent.len() != wanted {
            return Err(invalid(
                "latent",
                "wrong length for declared axes and channels",
            ));
        }
        if cond.len() != spec.width {
            return Err(invalid("cond", "wrong conditioning width"));
        }
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
            return Err(invalid("sigma", "must be finite and in [0, 1]"));
        }

        let embedding = sigma_embedding(sigma, spec.width, 1000.0)?;
        let mut hidden = vec![0.0; spec.width];
        matmul_rows(
            &embedding,
            &self.weights.sigma_mlp_1,
            &mut hidden,
            1,
            spec.width,
            spec.width,
        )?;
        let activated: Vec<f32> = hidden.iter().map(|value| silu(*value)).collect();
        let mut sigma_cond = vec![0.0; spec.width];
        matmul_rows(
            &activated,
            &self.weights.sigma_mlp_2,
            &mut sigma_cond,
            1,
            spec.width,
            spec.width,
        )?;
        let combined: Vec<f32> = cond
            .iter()
            .zip(&sigma_cond)
            .map(|(text, time)| text + time)
            .collect();

        let tokens_flat = patch_flatten(latent, spec.latent_channels, &spatial, &self.patch)?;
        let x = patch_embed(
            &tokens_flat,
            &self.weights.embed,
            &self.weights.embed_bias,
            &spec,
        )?;
        let positions = position_grid(spec.axes)?;
        let hidden_block = dit_block(&x, &combined, &positions, &spec, &self.weights.block, None)?;
        let output_len = tokens
            .checked_mul(spec.patch_volume)
            .ok_or_else(|| invalid("velocity", "element count overflow"))?;
        let mut output = vec![0.0; output_len];
        matmul_rows(
            &hidden_block,
            &self.weights.head,
            &mut output,
            tokens,
            spec.width,
            spec.patch_volume,
        )?;
        for (value, bias) in output.iter_mut().zip(&self.weights.head_bias) {
            *value += bias;
        }
        Ok(output)
    }

    /// One exact Euler step using the predicted velocity.
    pub fn step(
        &self,
        latent: &[f32],
        sigma: f32,
        sigma_next: f32,
        cond: &[f32],
    ) -> Result<Vec<f32>, VkError> {
        let velocity = self.velocity(latent, sigma, cond)?;
        euler_step(latent, &velocity, sigma, sigma_next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> FlowPredictorWeights {
        FlowPredictorWeights {
            embed: vec![0.0; 8],
            embed_bias: vec![0.0; 4],
            sigma_mlp_1: vec![0.0; 16],
            sigma_mlp_2: vec![0.0; 16],
            block: DitBlockWeights {
                ada: vec![0.0; 96],
                qkv: vec![0.01; 48],
                qk_gain: vec![1.0, 1.0],
                out: vec![0.0; 16],
                ffn: vec![0.01; 32],
                ffn_down: vec![0.0; 16],
            },
            head: vec![0.0; 8],
            head_bias: vec![0.0; 2],
        }
    }

    fn spec() -> FlowDitSpec {
        FlowDitSpec {
            width: 4,
            heads: 2,
            head_dim: 2,
            ffn_width: 4,
            patch_volume: 2,
            latent_channels: 1,
            axes: PositionAxes::Image {
                height: 1,
                width: 2,
            },
        }
    }

    #[test]
    fn flow_dit_substrate_predictor_zero_velocity_preserves_latent() {
        let predictor = FlowPredictor::new(spec(), &[1, 2], weights()).unwrap();
        let latent = [0.25_f32, -0.5, 0.75, -1.0];
        let velocity = predictor.velocity(&latent, 0.5, &[0.0; 4]).unwrap();
        assert!(velocity.iter().all(|value| *value == 0.0));
        let stepped = predictor.step(&latent, 0.5, 0.25, &[0.0; 4]).unwrap();
        assert_eq!(stepped, latent);
    }

    #[test]
    fn flow_dit_substrate_predictor_rejects_bad_geometry_by_name() {
        let error = FlowPredictor::new(spec(), &[1, 1], weights())
            .unwrap_err()
            .to_string();
        assert!(error.contains("patch_volume"), "{error}");
    }
}
