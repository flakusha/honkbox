//! Encoder hidden states → pooled embeddings → classifier-free guidance.
//!
//! ponytail: mean pooling on the host (77×768 is trivial) instead of a GPU
//! reduction kernel. Real CLIP pools the projected EOS token; add that head
//! and a pooling kernel when a checkpoint's text tower is wired in.

use crate::{
    VkError,
    conditioning::Cfg,
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

    /// Mean-pool each `tokens × dim` batch into one `dim` vector.
    fn pool(hidden: &[f32], dim: usize) -> Result<Vec<f32>, VkError> {
        if dim == 0 || hidden.is_empty() || !hidden.len().is_multiple_of(dim) {
            return Err(VkError::Device(
                "text conditioning: hidden states must be a non-empty multiple of dim".into(),
            ));
        }
        let rows = hidden.len() / dim;
        let rows_u16 = u16::try_from(rows)
            .map_err(|_| VkError::Device("text conditioning: too many tokens".into()))?;
        let count = f32::from(rows_u16);
        Ok((0..dim)
            .map(|d| (0..rows).map(|row| hidden[row * dim + d]).sum::<f32>() / count)
            .collect())
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
        self.guidance.run(
            &Self::pool(&positive, dim)?,
            &Self::pool(&negative, dim)?,
            scale,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_encoder::ClipTextEncoderWeights;
    use crate::transformer::TransformerWeights;

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

    fn cpu_pool(hidden: &[f32], dim: usize) -> Vec<f32> {
        let rows = hidden.len() / dim;
        let count = f32::from(u16::try_from(rows).expect("test rows"));
        (0..dim)
            .map(|d| (0..rows).map(|row| hidden[row * dim + d]).sum::<f32>() / count)
            .collect()
    }

    #[test]
    fn scale_one_returns_pooled_positive_embeddings() {
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
            layers: &[] as &[TransformerWeights<'_>],
            ffn_hidden: dim,
        };
        let actual = conditioner
            .condition(&positive, &negative, &weights, vocab, dim, 1.0)
            .expect("condition");
        let pooled = cpu_pool(
            &positive
                .iter()
                .zip(positions.chunks(dim))
                .flat_map(|(id, position)| {
                    let row = usize::try_from(*id).expect("test token id");
                    (0..dim).map(move |d| tokens[row * dim + d] + position[d])
                })
                .collect::<Vec<_>>(),
            dim,
        );
        for (got, want) in actual.iter().zip(&pooled) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn guidance_scale_is_applied_to_pooled_vectors() {
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
            layers: &[] as &[TransformerWeights<'_>],
            ffn_hidden: dim,
        };
        let cond = [tokens[dim] + positions[0], tokens[dim + 1] + positions[1]];
        let uncond = [
            tokens[2 * dim] + positions[0],
            tokens[2 * dim + 1] + positions[1],
        ];
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
}
