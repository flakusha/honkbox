//! Encoder hidden states → separate conditional/unconditional token states.
//!
//! Cross-attention consumes `tokens × dim` states; the two prompt towers stay
//! distinct so the sampler can guide each `UNet` epsilon prediction instead of
//! blending embeddings. No EOS projection head: add one when a checkpoint's
//! text tower requires it.

use crate::{
    VkError,
    conditioning::TextConditioning,
    device::VulkanDevice,
    text_encoder::{ClipPooling, ClipTextEncoder, ClipTextEncoderWeights},
};
use std::sync::Arc;

/// Positive/negative text towers; both token-state sets stay separate.
pub struct TextConditioner {
    encoder: ClipTextEncoder,
}

impl TextConditioner {
    /// `layers` encoder blocks with `heads` attention heads.
    pub fn new(device: Arc<VulkanDevice>, layers: usize, heads: usize) -> Result<Self, VkError> {
        Ok(Self {
            encoder: ClipTextEncoder::new(device, layers, heads)?,
        })
    }

    /// Encode positive and negative token ID sequences without blending them.
    ///
    /// Tokenization stays with the caller so this module never touches asset
    /// files or mixes `io::Error` into the Vulkan error path.
    pub fn condition(
        &self,
        positive_ids: &[u32],
        negative_ids: &[u32],
        weights: &ClipTextEncoderWeights<'_>,
        vocab: usize,
        dim: usize,
    ) -> Result<TextConditioning, VkError> {
        let positive = self
            .encoder
            .run_with_bias(positive_ids, weights, vocab, dim)?;
        let negative = self
            .encoder
            .run_with_bias(negative_ids, weights, vocab, dim)?;
        let tokens = positive_ids.len();
        TextConditioning::new(positive, negative, tokens, dim)
    }

    /// Encode and pool the EOS hidden state for both prompts, unblended.
    pub fn condition_pooled(
        &self,
        positive_ids: &[u32],
        negative_ids: &[u32],
        weights: &ClipTextEncoderWeights<'_>,
        vocab: usize,
        dim: usize,
        pooling: &ClipPooling<'_>,
    ) -> Result<TextConditioning, VkError> {
        let positive = self
            .encoder
            .run_pooled(positive_ids, weights, vocab, dim, pooling)?;
        let negative = self
            .encoder
            .run_pooled(negative_ids, weights, vocab, dim, pooling)?;
        TextConditioning::new(positive, negative, 1, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_encoder::ClipTextEncoderWeights;

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

    fn embedded(ids: &[u32], tokens: &[f32], positions: &[f32], dim: usize) -> Vec<f32> {
        ids.iter()
            .zip(positions.chunks(dim))
            .flat_map(|(id, position)| {
                let row = usize::try_from(*id).expect("test token id");
                (0..dim).map(move |d| tokens[row * dim + d] + position[d])
            })
            .collect()
    }

    #[allow(clippy::cast_precision_loss)] // test dims are tiny
    fn final_norm(rows: &[f32], dim: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let cols = dim as f32;
        rows.chunks(dim)
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / cols;
                let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols;
                let scale = 1.0 / (variance + eps).sqrt();
                (0..dim).map(move |d| (row[d] - mean) * scale * gamma[d] + beta[d])
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn condition_keeps_both_normalized_token_sets() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0, 1).expect("conditioner");
        let (dim, vocab) = (2usize, 3usize);
        let positive = [1u32, 2];
        let negative = [2u32, 1];
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [0.01_f32, 0.02, 0.03, 0.04];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &norm,
            final_norm_beta: &zero,
            final_norm_eps: 1e-5,
        };
        let actual = conditioner
            .condition(&positive, &negative, &weights, vocab, dim)
            .expect("condition");
        assert_eq!((actual.tokens(), actual.dim()), (positive.len(), dim));
        assert_close(
            actual.positive(),
            &final_norm(
                &embedded(&positive, &tokens, &positions, dim),
                dim,
                &norm,
                &zero,
                1e-5,
            ),
        );
        assert_close(
            actual.negative(),
            &final_norm(
                &embedded(&negative, &tokens, &positions, dim),
                dim,
                &norm,
                &zero,
                1e-5,
            ),
        );
    }

    #[test]
    fn condition_does_not_blend_positive_and_negative_states() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0, 1).expect("conditioner");
        let (dim, vocab) = (3usize, 3usize);
        let positive = [1u32];
        let negative = [2u32];
        let tokens = [0.1_f32, 0.2, 0.3, 0.9, 0.1, 0.6, 0.7, 0.8, 0.2];
        let positions = [0.01_f32, 0.02, 0.03];
        let norm = [1.0_f32; 3];
        let zero = [0.0_f32; 3];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &norm,
            final_norm_beta: &zero,
            final_norm_eps: 1e-5,
        };
        let actual = conditioner
            .condition(&positive, &negative, &weights, vocab, dim)
            .expect("condition");
        let cond = final_norm(
            &embedded(&positive, &tokens, &positions, dim),
            dim,
            &norm,
            &zero,
            1e-5,
        );
        let uncond = final_norm(
            &embedded(&negative, &tokens, &positions, dim),
            dim,
            &norm,
            &zero,
            1e-5,
        );
        assert_ne!(actual.positive(), actual.negative());
        assert_close(actual.positive(), &cond);
        assert_close(actual.negative(), &uncond);
    }

    #[test]
    fn pooled_conditioning_returns_both_eos_states() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0, 1).expect("conditioner");
        let (dim, vocab, eos) = (2usize, 3usize, 2u32);
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [0.01_f32, 0.02, 0.03, 0.04];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: dim,
            final_norm_gamma: &norm,
            final_norm_beta: &zero,
            final_norm_eps: 1e-5,
        };
        let pooling = ClipPooling {
            gamma: &[1.0_f32, 1.0],
            beta: &[0.0_f32, 0.0],
            eps: 1e-5,
            eos,
        };
        let actual = conditioner
            .condition_pooled(&[1, eos], &[eos], &weights, vocab, dim, &pooling)
            .expect("pooled condition");
        assert_eq!(actual.tokens(), 1);
        // EOS token row after the CLIP final layer norm.
        let row = [0.53_f32, 0.64];
        let expected = final_norm(&row, dim, &norm, &zero, 1e-5);
        assert_close(actual.positive(), &expected);
        assert_close(actual.negative(), &expected);
    }

    #[test]
    fn rejects_different_positive_and_negative_token_counts() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0, 1).expect("conditioner");
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [0.01_f32, 0.02, 0.03, 0.04];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            biases: Vec::new(),
            ffn_hidden: 2,
            final_norm_gamma: &norm,
            final_norm_beta: &zero,
            final_norm_eps: 1e-5,
        };
        assert!(
            conditioner
                .condition(&[1, 2], &[2], &weights, 3, 2)
                .is_err()
        );
    }
}
