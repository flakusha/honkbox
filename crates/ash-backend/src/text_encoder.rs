//! Prompt IDs → token/position embeddings → stacked encoder blocks.
//!
//! ponytail: no padding mask — every token is causal, so each query attends
//! only to itself and earlier tokens. The final layer norm is shared by all
//! token states, including the pooled EOS view.

use crate::{
    VkError,
    device::VulkanDevice,
    embedding::TextEncoder,
    norm::LayerNorm,
    transformer::{TransformerBias, TransformerBlock, TransformerWeights},
};
use std::sync::Arc;

/// Caller-supplied weights for the embedding tables and every encoder layer.
pub struct ClipTextEncoderWeights<'a> {
    /// `[vocab, dim]` token embedding table.
    pub token_table: &'a [f32],
    /// `[tokens, dim]` positional embedding table.
    pub position_table: &'a [f32],
    /// One entry per instantiated block, in execution order.
    pub layers: Vec<TransformerWeights<'a>>,
    /// Optional linear biases, one entry per block.
    pub biases: Vec<TransformerBias<'a>>,
    /// MLP expansion width shared by every block.
    pub ffn_hidden: usize,
    /// Final layer-norm gain `[dim]` applied to every token state.
    pub final_norm_gamma: &'a [f32],
    /// Final layer-norm bias `[dim]` applied to every token state.
    pub final_norm_beta: &'a [f32],
    /// Final layer-norm epsilon.
    pub final_norm_eps: f32,
}

/// Pooling selector for the EOS conditioning view.
///
/// The final layer norm now lives on [`ClipTextEncoderWeights`] and is applied
/// to every token; the affine fields stay for caller compatibility.
///
/// ponytail: no `text_projection` field — pruned SD 1.5 checkpoints omit it,
/// and the hidden width already matches the `UNet` cross-attention width. Add a
/// projection when a checkpoint family actually ships one.
pub struct ClipPooling<'a> {
    /// Final layer-norm gain `[dim]`.
    pub gamma: &'a [f32],
    /// Final layer-norm bias `[dim]`.
    pub beta: &'a [f32],
    /// Layer-norm epsilon.
    pub eps: f32,
    /// End-of-text token id (`49407` in the SD 1.5 CLIP tokenizer).
    pub eos: u32,
}

/// A fixed-depth stack over the shared embedding and transformer kernels.
pub struct ClipTextEncoder {
    embedding: TextEncoder,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
}

impl ClipTextEncoder {
    /// Instantiate `layers` blocks plus the embedding gather/add pipelines.
    pub fn new(device: Arc<VulkanDevice>, layers: usize, heads: usize) -> Result<Self, VkError> {
        if heads == 0 {
            return Err(VkError::Device(
                "clip encoder: heads must be positive".into(),
            ));
        }
        let mut blocks = Vec::with_capacity(layers);
        for _ in 0..layers {
            blocks.push(TransformerBlock::new(Arc::clone(&device), heads)?);
        }
        Ok(Self {
            embedding: TextEncoder::new(Arc::clone(&device))?,
            blocks,
            final_norm: LayerNorm::new(device)?,
        })
    }

    /// Run the stack with optional per-layer linear biases.
    pub fn run_with_bias(
        &self,
        ids: &[u32],
        weights: &ClipTextEncoderWeights<'_>,
        vocab: usize,
        dim: usize,
    ) -> Result<Vec<f32>, VkError> {
        if weights.layers.len() != self.blocks.len() {
            return Err(VkError::Device(format!(
                "clip text encoder: expected {} layer weight sets, got {}",
                self.blocks.len(),
                weights.layers.len()
            )));
        }
        if weights.biases.len() != self.blocks.len() {
            return Err(VkError::Device(format!(
                "clip text encoder: expected {} bias sets, got {}",
                self.blocks.len(),
                weights.biases.len()
            )));
        }
        let seq = ids.len();
        let mask_len = seq
            .checked_mul(seq)
            .ok_or_else(|| VkError::Device("clip encoder: sequence too long".into()))?;
        let mut mask = vec![0.0_f32; mask_len];
        for query in 0..seq {
            for key in query + 1..seq {
                mask[query * seq + key] = f32::NEG_INFINITY;
            }
        }
        let mut hidden =
            self.embedding
                .encode(ids, weights.token_table, weights.position_table, vocab, dim)?;
        for (block, (layer, bias)) in self
            .blocks
            .iter()
            .zip(weights.layers.iter().zip(weights.biases.iter()))
        {
            hidden = block.run_with_bias_masked(
                &hidden,
                layer,
                bias,
                Some(&mask),
                seq,
                dim,
                weights.ffn_hidden,
            )?;
        }
        self.final_norm.run(
            &hidden,
            seq,
            dim,
            weights.final_norm_gamma,
            weights.final_norm_beta,
            weights.final_norm_eps,
        )
    }

    /// Run the stack over `ids`, returning `tokens × dim` hidden states.
    pub fn run(
        &self,
        ids: &[u32],
        weights: &ClipTextEncoderWeights<'_>,
        vocab: usize,
        dim: usize,
    ) -> Result<Vec<f32>, VkError> {
        let biases = vec![TransformerBias::default(); self.blocks.len()];
        let weights = ClipTextEncoderWeights {
            token_table: weights.token_table,
            position_table: weights.position_table,
            layers: weights.layers.clone(),
            biases,
            ffn_hidden: weights.ffn_hidden,
            final_norm_gamma: weights.final_norm_gamma,
            final_norm_beta: weights.final_norm_beta,
            final_norm_eps: weights.final_norm_eps,
        };
        self.run_with_bias(ids, &weights, vocab, dim)
    }

    /// Return the first EOS token from the already-normalized context.
    pub fn run_pooled(
        &self,
        ids: &[u32],
        weights: &ClipTextEncoderWeights<'_>,
        vocab: usize,
        dim: usize,
        pooling: &ClipPooling<'_>,
    ) -> Result<Vec<f32>, VkError> {
        if ids.is_empty() {
            return Err(VkError::Device("clip pooling: empty token sequence".into()));
        }
        let hidden = self.run_with_bias(ids, weights, vocab, dim)?;
        let eos = ids
            .iter()
            .position(|&id| id == pooling.eos)
            .unwrap_or(ids.len() - 1);
        let row = &hidden[eos * dim..(eos + 1) * dim];
        Ok(row.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VkError;

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

    fn cpu_layer_norm(
        values: &[f32],
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let count = f32::from(u16::try_from(cols).expect("test cols"));
        values
            .chunks(cols)
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / count;
                let variance = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / count;
                row.iter()
                    .enumerate()
                    .map(|(i, x)| (x - mean) / (variance + eps).sqrt() * gamma[i] + beta[i])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn cpu_quick_gelu(x: f32) -> f32 {
        x / (1.0 + (-1.702 * x).exp())
    }

    fn causal_mask(seq: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; seq * seq];
        for query in 0..seq {
            for key in query + 1..seq {
                mask[query * seq + key] = f32::NEG_INFINITY;
            }
        }
        mask
    }

    fn layer_weights<'a>(
        matrix: &'a [f32],
        w2: &'a [f32],
        norm: &'a [f32],
        zero: &'a [f32],
    ) -> TransformerWeights<'a> {
        TransformerWeights {
            wq: matrix,
            wk: matrix,
            wv: matrix,
            wo: matrix,
            w1: matrix,
            w2,
            norm1_gamma: norm,
            norm1_beta: zero,
            norm2_gamma: norm,
            norm2_beta: zero,
            eps: 1e-5,
        }
    }

    #[test]
    fn zero_layers_apply_final_norm_to_every_token() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device, 0, 1).expect("encoder");
        let (dim, vocab, ids) = (2usize, 3usize, [2u32, 1, 0]);
        let token_table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let position_table = [0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06];
        let gamma = [2.0_f32, 3.0];
        let beta = [0.5_f32, -0.5];
        let eps = 1e-5_f32;
        let weights = ClipTextEncoderWeights {
            token_table: &token_table,
            position_table: &position_table,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &gamma,
            final_norm_beta: &beta,
            final_norm_eps: eps,
        };
        let actual = encoder.run(&ids, &weights, vocab, dim).expect("encode");

        let embedded: Vec<f32> = ids
            .iter()
            .zip(position_table.chunks(dim))
            .flat_map(|(id, position)| {
                let row = usize::try_from(*id).expect("test token id");
                (0..dim).map(move |d| token_table[row * dim + d] + position[d])
            })
            .collect();
        let expected = cpu_layer_norm(&embedded, dim, &gamma, &beta, eps);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn pooled_state_views_normalized_eos_context() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device, 0, 1).expect("encoder");
        let (dim, vocab, eos) = (2usize, 3usize, 2u32);
        let ids = [1u32, eos, eos];
        let token_table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let position_table = [0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06];
        let gamma = [2.0_f32, 3.0];
        let beta = [0.5_f32, -0.5];
        let eps = 1e-5_f32;
        let weights = ClipTextEncoderWeights {
            token_table: &token_table,
            position_table: &position_table,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &gamma,
            final_norm_beta: &beta,
            final_norm_eps: eps,
        };
        let pooling = ClipPooling {
            gamma: &[2.0_f32, 3.0],
            beta: &[0.5_f32, -0.5],
            eps: 1e-5,
            eos,
        };
        let context = encoder.run(&ids, &weights, vocab, dim).expect("encode");
        let actual = encoder
            .run_pooled(&ids, &weights, vocab, dim, &pooling)
            .expect("pooled");
        let expected = cpu_layer_norm(&[0.53_f32, 0.64], dim, &gamma, &beta, eps);
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
        for (got, want) in actual.iter().zip(&context[dim..2 * dim]) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn one_layer_matches_direct_transformer_block() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device.clone(), 1, 1).expect("encoder");
        let (dim, vocab) = (2usize, 3usize);
        let ids = [1u32, 2];
        let token_table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let position_table = [0.01_f32, 0.02, 0.03, 0.04];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let zero_matrix = [0.0_f32; 4];
        let layers = vec![layer_weights(&identity, &zero_matrix, &norm, &zero)];
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let eps = 1e-5_f32;
        let weights = ClipTextEncoderWeights {
            token_table: &token_table,
            position_table: &position_table,
            layers: layers.clone(),
            biases: vec![TransformerBias::default()],
            ffn_hidden: dim,
            final_norm_gamma: &gamma,
            final_norm_beta: &beta,
            final_norm_eps: eps,
        };

        let actual = encoder.run(&ids, &weights, vocab, dim).expect("encode");
        let embedded = crate::embedding::TextEncoder::new(device.clone())
            .expect("embedding")
            .encode(&ids, &token_table, &position_table, vocab, dim)
            .expect("embed");
        let mask = causal_mask(ids.len());
        let block_out = crate::transformer::TransformerBlock::new(device, 1)
            .expect("block")
            .run_with_bias_masked(
                &embedded,
                &layers[0],
                &TransformerBias::default(),
                Some(&mask),
                ids.len(),
                dim,
                dim,
            )
            .expect("block run");
        let expected = cpu_layer_norm(&block_out, dim, &gamma, &beta, eps);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    // One inline CPU reference keeps the causal-mask expectation auditable here.
    #[allow(clippy::too_many_lines)]
    fn causal_mask_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device, 1, 1).expect("encoder");
        let (seq, dim, hidden, eps) = (3usize, 2usize, 2usize, 1e-5_f32);
        let ids = [1u32, 2, 0];
        let vocab = 4usize;
        let token_table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let position_table = [0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06];
        let qk = [1.0_f32, 0.3, -0.2, 0.8];
        let v_matrix = [0.7_f32, -0.4, 0.2, 0.9];
        let o_matrix = [1.1_f32, 0.2, -0.3, 0.6];
        let w1 = [0.5_f32, -0.2, 0.3, 0.7];
        let w2 = [0.4_f32, 0.1, -0.3, 0.6];
        let norm1_gamma = [1.4_f32, 0.6];
        let norm1_beta = [0.1_f32, -0.1];
        let norm2_gamma = [0.8_f32, 1.2];
        let norm2_beta = [-0.2_f32, 0.2];
        let final_gamma = [1.3_f32, 0.7];
        let final_beta = [0.1_f32, -0.1];
        let layer = TransformerWeights {
            wq: &qk,
            wk: &qk,
            wv: &v_matrix,
            wo: &o_matrix,
            w1: &w1,
            w2: &w2,
            norm1_gamma: &norm1_gamma,
            norm1_beta: &norm1_beta,
            norm2_gamma: &norm2_gamma,
            norm2_beta: &norm2_beta,
            eps,
        };
        let weights = ClipTextEncoderWeights {
            token_table: &token_table,
            position_table: &position_table,
            layers: vec![layer],
            biases: vec![TransformerBias::default()],
            ffn_hidden: hidden,
            final_norm_gamma: &final_gamma,
            final_norm_beta: &final_beta,
            final_norm_eps: eps,
        };

        let embedded: Vec<f32> = ids
            .iter()
            .zip(position_table.chunks(dim))
            .flat_map(|(id, position)| {
                let row = usize::try_from(*id).expect("test token id");
                (0..dim).map(move |d| token_table[row * dim + d] + position[d])
            })
            .collect();
        let h1 = cpu_layer_norm(&embedded, dim, &norm1_gamma, &norm1_beta, eps);
        let scale = 1.0 / f32::from(u16::try_from(dim).expect("test dim")).sqrt();
        let mut attended = vec![0.0_f32; seq * dim];
        for query in 0..seq {
            let scores: Vec<f32> = (0..=query)
                .map(|key| {
                    (0..dim)
                        .map(|d| h1[query * dim + d] * h1[key * dim + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = probabilities.iter().sum();
            for d in 0..dim {
                attended[query * dim + d] = (0..=query)
                    .map(|key| probabilities[key] / total * h1[key * dim + d])
                    .sum();
            }
        }
        let projected: Vec<f32> = attended
            .chunks(dim)
            .flat_map(|row| {
                (0..dim)
                    .map(|d| (0..dim).map(|j| row[j] * o_matrix[j * dim + d]).sum())
                    .collect::<Vec<_>>()
            })
            .collect();
        let residual: Vec<f32> = embedded
            .iter()
            .zip(&projected)
            .map(|(a, b)| a + b)
            .collect();
        let h2 = cpu_layer_norm(&residual, dim, &norm2_gamma, &norm2_beta, eps);
        let fed: Vec<f32> = h2
            .chunks(dim)
            .flat_map(|row| {
                let expanded: Vec<f32> = (0..hidden)
                    .map(|h| (0..dim).map(|d| row[d] * w1[d * hidden + h]).sum())
                    .collect();
                (0..dim)
                    .map(|d| {
                        (0..hidden)
                            .map(|h| cpu_quick_gelu(expanded[h]) * w2[h * dim + d])
                            .sum()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let block_out: Vec<f32> = residual.iter().zip(&fed).map(|(a, b)| a + b).collect();
        let expected = cpu_layer_norm(&block_out, dim, &final_gamma, &final_beta, eps);

        let actual = encoder.run(&ids, &weights, vocab, dim).expect("encode");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }
    #[test]
    fn rejects_wrong_layer_count_and_oversized_sequence() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device.clone(), 1, 1).expect("encoder");
        let dim = 2usize;
        let matrix = [0.0_f32; 4];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let layers = vec![layer_weights(&matrix, &matrix, &norm, &zero)];
        let weights = ClipTextEncoderWeights {
            token_table: &matrix,
            position_table: &norm,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &norm,
            final_norm_beta: &zero,
            final_norm_eps: 1e-5,
        };
        assert!(encoder.run(&[0], &weights, 2, dim).is_err());

        let weights = ClipTextEncoderWeights {
            token_table: &matrix,
            position_table: &norm,
            layers: layers.clone(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &norm,
            final_norm_beta: &zero,
            final_norm_eps: 1e-5,
        };
        assert!(encoder.run(&[0, 1, 2], &weights, 2, dim).is_err());
    }
}
