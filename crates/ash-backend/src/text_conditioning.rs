//! Encoder hidden states → classifier-free guidance over every token.
//!
//! Cross-attention consumes `tokens × dim` states, so guidance is applied
//! element-wise rather than to a mean-pooled vector. No EOS projection head:
//! add one when a checkpoint's text tower requires it.

use crate::{
    VkError,
    conditioning::{Cfg, TextConditioning},
    device::VulkanDevice,
    text_encoder::{ClipTextEncoder, ClipTextEncoderWeights},
};
use std::sync::Arc;

/// Positive/negative text towers and the guidance blend over their output.
pub struct TextConditioner {
    encoder: ClipTextEncoder,
    guidance: Cfg,
}

impl TextConditioner {
    /// `layers` encoder blocks plus the CFG pipeline.
    pub fn new(device: Arc<VulkanDevice>, layers: usize) -> Result<Self, VkError> {
        Ok(Self {
            encoder: ClipTextEncoder::new(Arc::clone(&device), layers)?,
            guidance: Cfg::new(device)?,
        })
    }

    /// Encode positive and negative token ID sequences and blend them.
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
        scale: f32,
    ) -> Result<Vec<f32>, VkError> {
        let positive = self.encoder.run(positive_ids, weights, vocab, dim)?;
        let negative = self.encoder.run(negative_ids, weights, vocab, dim)?;
        let tokens = positive_ids.len();
        TextConditioning::new(positive, negative, tokens, dim)?.guided(&self.guidance, scale)
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

    #[test]
    fn scale_one_returns_positive_token_states() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0).expect("conditioner");
        let (dim, vocab) = (2usize, 3usize);
        let positive = [1u32, 2];
        let negative = [2u32, 1];
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [0.01_f32, 0.02, 0.03, 0.04];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            ffn_hidden: dim,
        };
        let actual = conditioner
            .condition(&positive, &negative, &weights, vocab, dim, 1.0)
            .expect("condition");
        let expected = embedded(&positive, &tokens, &positions, dim);
        assert_eq!(actual.len(), positive.len() * dim);
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn guidance_scale_is_applied_per_token() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0).expect("conditioner");
        let (dim, vocab) = (2usize, 3usize);
        let positive = [1u32];
        let negative = [2u32];
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [0.01_f32, 0.02];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            ffn_hidden: dim,
        };
        let cond = embedded(&positive, &tokens, &positions, dim);
        let uncond = embedded(&negative, &tokens, &positions, dim);
        let scale = 2.5_f32;
        let expected: Vec<f32> = cond
            .iter()
            .zip(&uncond)
            .map(|(c, u)| u + scale * (c - u))
            .collect();
        let actual = conditioner
            .condition(&positive, &negative, &weights, vocab, dim, scale)
            .expect("condition");
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
        assert!(
            conditioner
                .condition(&positive, &negative, &weights, vocab, dim, 0.5)
                .is_err()
        );
    }

    #[test]
    fn rejects_different_positive_and_negative_token_counts() {
        let Some(device) = device() else { return };
        let conditioner = TextConditioner::new(device, 0).expect("conditioner");
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [0.01_f32, 0.02, 0.03, 0.04];
        let weights = ClipTextEncoderWeights {
            token_table: &tokens,
            position_table: &positions,
            layers: Vec::new(),
            ffn_hidden: 2,
        };
        assert!(
            conditioner
                .condition(&[1, 2], &[2], &weights, 3, 2, 7.5)
                .is_err()
        );
    }
}
