//! Spatial self-attention + residual stage for the `UNet` path.
//!
//! NCHW → group norm → H·W-token sequence → single-head attention → residual.
//! ponytail: one head and no cross-attention or timestep embedding; add them
//! when a real transformer block checkpoint needs them.

use crate::{
    VkError,
    attention::{Attention, AttentionShape},
    compute::ElementwiseAdd,
    device::VulkanDevice,
    group_norm::{GroupNorm, GroupNormShape},
    matmul::MatMul,
    transpose::Transpose,
};
use std::sync::Arc;

/// NCHW geometry for the attention stage.
#[derive(Debug, Clone, Copy)]
pub struct UnetAttentionShape {
    /// Batch size.
    pub batch: usize,
    /// Channel count; also the token width.
    pub channels: usize,
    /// Spatial height.
    pub height: usize,
    /// Spatial width.
    pub width: usize,
    /// Group-norm group count; must divide `channels`.
    pub groups: usize,
}

/// Caller-supplied norm and projection weights.
pub struct UnetAttentionWeights<'a> {
    /// `[channels]` group-norm gain.
    pub gamma: &'a [f32],
    /// `[channels]` group-norm bias.
    pub beta: &'a [f32],
    /// `[channels, channels]` query projection.
    pub wq: &'a [f32],
    /// `[channels, channels]` key projection.
    pub wk: &'a [f32],
    /// `[channels, channels]` value projection.
    pub wv: &'a [f32],
    /// `[channels, channels]` output projection.
    pub wo: &'a [f32],
}

/// Group norm → spatial attention → residual add over NCHW tensors.
pub struct UnetAttentionBlock {
    norm: GroupNorm,
    query: MatMul,
    key: MatMul,
    value: MatMul,
    transpose: Transpose,
    attention: Attention,
    output: MatMul,
    add: ElementwiseAdd,
}

impl UnetAttentionBlock {
    /// Eight pipelines; no weights or tokens are bundled.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            norm: GroupNorm::new(Arc::clone(&device))?,
            query: MatMul::new(Arc::clone(&device))?,
            key: MatMul::new(Arc::clone(&device))?,
            value: MatMul::new(Arc::clone(&device))?,
            transpose: Transpose::new(Arc::clone(&device))?,
            attention: Attention::new(Arc::clone(&device))?,
            output: MatMul::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// Run the stage over `batch × channels × height × width` NCHW input.
    pub fn run(
        &self,
        input: &[f32],
        weights: &UnetAttentionWeights<'_>,
        shape: UnetAttentionShape,
        eps: f32,
    ) -> Result<Vec<f32>, VkError> {
        let spatial = shape
            .height
            .checked_mul(shape.width)
            .ok_or_else(|| VkError::Device("unet attention: spatial overflow".into()))?;
        let expected = spatial
            .checked_mul(shape.channels)
            .and_then(|per_batch| per_batch.checked_mul(shape.batch))
            .ok_or_else(|| VkError::Device("unet attention: tensor overflow".into()))?;
        if input.len() != expected || shape.channels == 0 || shape.groups == 0 {
            return Err(VkError::Device(
                "unet attention: input length and shape must agree".into(),
            ));
        }
        let normalized = self.norm.run(
            input,
            weights.gamma,
            weights.beta,
            GroupNormShape {
                batch: shape.batch,
                channels: shape.channels,
                height: shape.height,
                width: shape.width,
                groups: shape.groups,
            },
            eps,
        )?;

        let tokens = nchw_to_sequence(&normalized, shape.batch, shape.channels, spatial);
        let rows = shape.batch * spatial;
        let width = shape.channels;
        let q = self.query.run(&tokens, weights.wq, rows, width, width)?;
        let k = self.key.run(&tokens, weights.wk, rows, width, width)?;
        let k_t = self.transpose.run(&k, rows, width)?;
        let v = self.value.run(&tokens, weights.wv, rows, width, width)?;
        let attended = self.attention.run(
            &q,
            &k_t,
            &v,
            AttentionShape {
                seq_q: rows,
                seq_k: rows,
                head_dim: width,
                value_dim: width,
                heads: 1,
            },
        )?;
        let projected = self.output.run(&attended, weights.wo, rows, width, width)?;
        let residual = self.add.run(&tokens, &projected)?;
        Ok(sequence_to_nchw(
            &residual,
            shape.batch,
            shape.channels,
            spatial,
        ))
    }
}

fn nchw_to_sequence(values: &[f32], batch: usize, channels: usize, spatial: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * spatial * channels];
    for n in 0..batch {
        for position in 0..spatial {
            for channel in 0..channels {
                let nchw = (n * channels + channel) * spatial + position;
                out[(n * spatial + position) * channels + channel] = values[nchw];
            }
        }
    }
    out
}

fn sequence_to_nchw(values: &[f32], batch: usize, channels: usize, spatial: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * channels * spatial];
    for n in 0..batch {
        for position in 0..spatial {
            for channel in 0..channels {
                let sequence = (n * spatial + position) * channels + channel;
                out[(n * channels + channel) * spatial + position] = values[sequence];
            }
        }
    }
    out
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

    fn cpu_norm(
        values: &[f32],
        shape: UnetAttentionShape,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let spatial = shape.height * shape.width;
        let group_size = shape.channels / shape.groups * spatial;
        let count = f32::from(u16::try_from(group_size).expect("test group"));
        let mut out = values.to_vec();
        for group in 0..shape.batch * shape.groups {
            let start = group * group_size;
            let slice = &values[start..start + group_size];
            let mean = slice.iter().sum::<f32>() / count;
            let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count;
            for offset in 0..group_size {
                let channel = group % shape.groups + offset / spatial;
                out[start + offset] = (slice[offset] - mean) / (variance + eps).sqrt()
                    * gamma[channel]
                    + beta[channel];
            }
        }
        out
    }

    fn cpu_attention(
        tokens: &[f32],
        weights: &UnetAttentionWeights<'_>,
        rows: usize,
        width: usize,
    ) -> Vec<f32> {
        let project = |matrix: &[f32]| -> Vec<f32> {
            (0..rows)
                .flat_map(|row| {
                    (0..width).map(move |col| {
                        (0..width)
                            .map(|k| tokens[row * width + k] * matrix[k * width + col])
                            .sum::<f32>()
                    })
                })
                .collect()
        };
        let q = project(weights.wq);
        let k = project(weights.wk);
        let v = project(weights.wv);
        let scale = 1.0 / f32::from(u16::try_from(width).expect("test width")).sqrt();
        let mut out = Vec::with_capacity(rows * width);
        for row in 0..rows {
            let scores: Vec<f32> = (0..rows)
                .map(|col| {
                    (0..width)
                        .map(|d| q[row * width + d] * k[col * width + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights_row: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights_row.iter().sum();
            for d in 0..width {
                out.push(
                    (0..rows)
                        .map(|col| weights_row[col] / total * v[col * width + d])
                        .sum(),
                );
            }
        }
        out
    }

    #[test]
    fn attention_block_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let shape = UnetAttentionShape {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            groups: 1,
        };
        let input: Vec<f32> = (0..8)
            .map(|i| f32::from(u8::try_from(i).expect("0..8")) * 0.1 - 0.7)
            .collect();
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let weights = UnetAttentionWeights {
            gamma: &gamma,
            beta: &beta,
            wq: &identity,
            wk: &identity,
            wv: &identity,
            wo: &identity,
        };
        let eps = 1e-5_f32;
        let normalized = cpu_norm(&input, shape, &gamma, &beta, eps);
        let tokens = nchw_to_sequence(&normalized, shape.batch, shape.channels, 4);
        let attended = cpu_attention(&tokens, &weights, 4, shape.channels);
        let residual: Vec<f32> = tokens.iter().zip(&attended).map(|(a, b)| a + b).collect();
        let expected = sequence_to_nchw(&residual, shape.batch, shape.channels, 4);

        let block = UnetAttentionBlock::new(device).expect("pipeline");
        let actual = block.run(&input, &weights, shape, eps).expect("gpu block");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn attention_block_rejects_bad_shapes() {
        let Some(device) = device() else { return };
        let block = UnetAttentionBlock::new(device).expect("pipeline");
        let shape = UnetAttentionShape {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            groups: 1,
        };
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let weights = UnetAttentionWeights {
            gamma: &gamma,
            beta: &beta,
            wq: &identity,
            wk: &identity,
            wv: &identity,
            wo: &identity,
        };
        assert!(block.run(&[], &weights, shape, 1e-5).is_err());
        let short = UnetAttentionWeights {
            wq: &[1.0],
            ..weights
        };
        assert!(block.run(&[0.0; 8], &short, shape, 1e-5).is_err());
    }
}
