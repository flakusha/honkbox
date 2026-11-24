//! Multi-head transformer encoder block composed from shared Vulkan kernels.
//!
//! ponytail: split/concat runs on the host; no dedicated shader until a
//! profiled real checkpoint shows the copies dominate.

use crate::{
    VkError,
    attention::{Attention, AttentionShape},
    compute::ElementwiseAdd,
    device::VulkanDevice,
    matmul::MatMul,
    mlp::FeedForward,
    norm::LayerNorm,
    transpose::Transpose,
};
use std::sync::Arc;

/// All caller-supplied weights and norm parameters for one encoder block.
#[derive(Clone, Copy)]
pub struct TransformerWeights<'a> {
    /// `[dim, dim]` query projection.
    pub wq: &'a [f32],
    /// `[dim, dim]` key projection.
    pub wk: &'a [f32],
    /// `[dim, dim]` value projection.
    pub wv: &'a [f32],
    /// `[dim, dim]` attention output projection.
    pub wo: &'a [f32],
    /// `[dim, hidden]` MLP expansion.
    pub w1: &'a [f32],
    /// `[hidden, dim]` MLP contraction.
    pub w2: &'a [f32],
    /// Post-attention layer-norm gain.
    pub norm1_gamma: &'a [f32],
    /// Post-attention layer-norm bias.
    pub norm1_beta: &'a [f32],
    /// Post-MLP layer-norm gain.
    pub norm2_gamma: &'a [f32],
    /// Post-MLP layer-norm bias.
    pub norm2_beta: &'a [f32],
    /// Layer-norm epsilon.
    pub eps: f32,
}

/// Optional linear biases for one encoder block. Empty means "no bias".
#[derive(Clone, Copy, Default)]
pub struct TransformerBias<'a> {
    /// Query projection bias `[dim]`.
    pub bq: Option<&'a [f32]>,
    /// Key projection bias `[dim]`.
    pub bk: Option<&'a [f32]>,
    /// Value projection bias `[dim]`.
    pub bv: Option<&'a [f32]>,
    /// Attention output projection bias `[dim]`.
    pub bo: Option<&'a [f32]>,
    /// MLP expansion bias `[hidden]`.
    pub b1: Option<&'a [f32]>,
    /// MLP contraction bias `[dim]`.
    pub b2: Option<&'a [f32]>,
}

/// Pre-attention norm + attention + residual + MLP + residual + final norm.
pub struct TransformerBlock {
    heads: usize,
    query: MatMul,
    key: MatMul,
    value: MatMul,
    transpose: Transpose,
    attention: Attention,
    output: MatMul,
    add: ElementwiseAdd,
    norm1: LayerNorm,
    feed_forward: FeedForward,
    norm2: LayerNorm,
}

impl TransformerBlock {
    /// Ten pipelines; weights stay with the caller and no assets are bundled.
    pub fn new(device: Arc<VulkanDevice>, heads: usize) -> Result<Self, VkError> {
        if heads == 0 {
            return Err(VkError::Device(
                "transformer: heads must be positive".into(),
            ));
        }
        Ok(Self {
            heads,
            query: MatMul::new(Arc::clone(&device))?,
            key: MatMul::new(Arc::clone(&device))?,
            value: MatMul::new(Arc::clone(&device))?,
            transpose: Transpose::new(Arc::clone(&device))?,
            attention: Attention::new(Arc::clone(&device))?,
            output: MatMul::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(Arc::clone(&device))?,
            norm1: LayerNorm::new(Arc::clone(&device))?,
            feed_forward: FeedForward::new(Arc::clone(&device))?,
            norm2: LayerNorm::new(device)?,
        })
    }

    /// Run one encoder block over `seq × dim` row-major tokens with biases.
    pub fn run_with_bias(
        &self,
        input: &[f32],
        weights: &TransformerWeights<'_>,
        bias: &TransformerBias<'_>,
        seq: usize,
        dim: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, VkError> {
        if dim == 0 || !dim.is_multiple_of(self.heads) {
            return Err(VkError::Device(
                "transformer: dim must be positive and divisible by heads".into(),
            ));
        }
        let head_dim = dim / self.heads;
        let add_bias = |values: Vec<f32>, extra: Option<&[f32]>| -> Result<Vec<f32>, VkError> {
            match extra {
                Some(extra) if !extra.is_empty() => self.add.run_broadcast(&values, extra),
                _ => Ok(values),
            }
        };
        let q = add_bias(self.query.run(input, weights.wq, seq, dim, dim)?, bias.bq)?;
        let k = add_bias(self.key.run(input, weights.wk, seq, dim, dim)?, bias.bk)?;
        let k_t = self.transpose.run(&k, seq, dim)?;
        let v = add_bias(self.value.run(input, weights.wv, seq, dim, dim)?, bias.bv)?;
        let attended = self.attention.run(
            &q,
            &k_t,
            &v,
            AttentionShape {
                seq_q: seq,
                seq_k: seq,
                head_dim,
                value_dim: head_dim,
                heads: self.heads,
            },
        )?;
        let projected = add_bias(
            self.output.run(&attended, weights.wo, seq, dim, dim)?,
            bias.bo,
        )?;
        let residual = self.add.run(input, &projected)?;
        let normed = self.norm1.run(
            &residual,
            seq,
            dim,
            weights.norm1_gamma,
            weights.norm1_beta,
            weights.eps,
        )?;
        let fed = self.feed_forward.run_with_bias(
            &normed,
            weights.w1,
            bias.b1.unwrap_or(&[]),
            weights.w2,
            bias.b2.unwrap_or(&[]),
            seq,
            dim,
            hidden,
        )?;
        let residual2 = self.add.run(&normed, &fed)?;
        self.norm2.run(
            &residual2,
            seq,
            dim,
            weights.norm2_gamma,
            weights.norm2_beta,
            weights.eps,
        )
    }

    /// Run one encoder block over `seq × dim` row-major tokens.
    pub fn run(
        &self,
        input: &[f32],
        weights: &TransformerWeights<'_>,
        seq: usize,
        dim: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, VkError> {
        self.run_with_bias(
            input,
            weights,
            &TransformerBias::default(),
            seq,
            dim,
            hidden,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_layer_norm(values: &[f32], cols: usize, eps: f32) -> Vec<f32> {
        let count = f32::from(u16::try_from(cols).expect("test cols"));
        values
            .chunks(cols)
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / count;
                let variance = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / count;
                row.iter()
                    .map(|x| (x - mean) / (variance + eps).sqrt())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn block_rejects_mismatched_weights() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => return,
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let block = TransformerBlock::new(device, 1).expect("pipeline");
        let dim = 2usize;
        let matrix = vec![0.0; dim * dim];
        let bad_matrix = vec![0.0; dim * dim + 1];
        let norm = vec![1.0; dim];
        let weights = TransformerWeights {
            wq: &bad_matrix,
            wk: &matrix,
            wv: &matrix,
            wo: &matrix,
            w1: &matrix,
            w2: &matrix,
            norm1_gamma: &norm,
            norm1_beta: &norm,
            norm2_gamma: &norm,
            norm2_beta: &norm,
            eps: 1e-5,
        };
        assert!(block.run(&[0.0; 4], &weights, 2, dim, dim).is_err());
    }

    #[test]
    fn block_matches_cpu_reference() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let (seq, dim, hidden, eps) = (2usize, 2usize, 3usize, 1e-5_f32);
        let input = [0.4_f32, -0.2, 0.7, 0.1];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let w1 = [0.1_f32, 0.2, -0.3, 0.4, 0.5, -0.6];
        let w2 = [0.7_f32, -0.2, 0.3, 0.1, -0.1, 0.4];
        let ones = [1.0_f32, 1.0];
        let zeros = [0.0_f32, 0.0];
        let weights = TransformerWeights {
            wq: &identity,
            wk: &identity,
            wv: &identity,
            wo: &identity,
            w1: &w1,
            w2: &w2,
            norm1_gamma: &ones,
            norm1_beta: &zeros,
            norm2_gamma: &ones,
            norm2_beta: &zeros,
            eps,
        };

        let scale = 1.0 / f32::from(u16::try_from(dim).expect("test dim")).sqrt();
        let mut attended: Vec<f32> = Vec::with_capacity(seq * dim);
        for row in 0..seq {
            let scores: Vec<f32> = (0..seq)
                .map(|col| {
                    (0..dim)
                        .map(|d| input[row * dim + d] * input[col * dim + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights_row: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights_row.iter().sum();
            for d in 0..dim {
                attended.push(
                    (0..seq)
                        .map(|col| weights_row[col] / total * input[col * dim + d])
                        .sum(),
                );
            }
        }
        let residual: Vec<f32> = input.iter().zip(&attended).map(|(a, b)| a + b).collect();
        let normed = cpu_layer_norm(&residual, dim, eps);
        let expanded: Vec<f32> = normed
            .chunks(dim)
            .flat_map(|row| {
                (0..hidden)
                    .map(|h| row[0] * w1[h] + row[1] * w1[hidden + h])
                    .collect::<Vec<_>>()
            })
            .collect();
        let fed: Vec<f32> = expanded
            .chunks(hidden)
            .flat_map(|row| {
                let activated: Vec<f32> = row
                    .iter()
                    .map(|x| {
                        0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x)).tanh())
                    })
                    .collect();
                (0..dim)
                    .map(|d| (0..hidden).map(|h| activated[h] * w2[h * dim + d]).sum())
                    .collect::<Vec<_>>()
            })
            .collect();
        let residual2: Vec<f32> = normed.iter().zip(&fed).map(|(a, b)| a + b).collect();
        let expected = cpu_layer_norm(&residual2, dim, eps);

        let block = TransformerBlock::new(device, 1).expect("pipeline");
        let actual = block
            .run(&input, &weights, seq, dim, hidden)
            .expect("gpu block");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }
}
