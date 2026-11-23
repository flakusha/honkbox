//! Prompt IDs → token/position embeddings → stacked encoder blocks.
//!
//! ponytail: no causal or padding mask — `Attention` is bidirectional here.
//! Callers supply exactly the tokens that should attend to each other; add a
//! mask to `Attention` when a real CLIP checkpoint needs padded prompts.

use crate::{
    VkError,
    device::VulkanDevice,
    embedding::TextEncoder,
    transformer::{TransformerBlock, TransformerWeights},
};
use std::sync::Arc;

/// Caller-supplied weights for the embedding tables and every encoder layer.
pub struct ClipTextEncoderWeights<'a> {
    /// `[vocab, dim]` token embedding table.
    pub token_table: &'a [f32],
    /// `[tokens, dim]` positional embedding table.
    pub position_table: &'a [f32],
    /// One entry per instantiated block, in execution order.
    pub layers: &'a [TransformerWeights<'a>],
    /// MLP expansion width shared by every block.
    pub ffn_hidden: usize,
}

/// A fixed-depth stack over the shared embedding and transformer kernels.
pub struct ClipTextEncoder {
    embedding: TextEncoder,
    blocks: Vec<TransformerBlock>,
}

impl ClipTextEncoder {
    /// Instantiate `layers` blocks plus the embedding gather/add pipelines.
    pub fn new(device: Arc<VulkanDevice>, layers: usize) -> Result<Self, VkError> {
        let mut blocks = Vec::with_capacity(layers);
        for _ in 0..layers {
            blocks.push(TransformerBlock::new(Arc::clone(&device))?);
        }
        Ok(Self {
            embedding: TextEncoder::new(device)?,
            blocks,
        })
    }

    /// Run the whole stack over `ids`, returning `tokens × dim` hidden states.
    pub fn run(
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
        let mut hidden =
            self.embedding
                .encode(ids, weights.token_table, weights.position_table, vocab, dim)?;
        for (block, layer) in self.blocks.iter().zip(weights.layers) {
            hidden = block.run(&hidden, layer, ids.len(), dim, weights.ffn_hidden)?;
        }
        Ok(hidden)
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
    fn zero_layers_return_plain_token_plus_position_embeddings() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device.clone(), 0).expect("encoder");
        let (dim, vocab, ids) = (2usize, 3usize, [1u32, 2]);
        let token_table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let position_table = [0.01_f32, 0.02, 0.03, 0.04];
        let weights = ClipTextEncoderWeights {
            token_table: &token_table,
            position_table: &position_table,
            layers: &[],
            ffn_hidden: dim,
        };
        let actual = encoder.run(&ids, &weights, vocab, dim).expect("encode");
        let expected: Vec<f32> = ids
            .iter()
            .zip(position_table.chunks(dim))
            .flat_map(|(id, position)| {
                (0..dim).map(move |d| {
                    let row = usize::try_from(*id).expect("test token id");
                    token_table[row * dim + d] + position[d]
                })
            })
            .collect();
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn one_layer_matches_direct_transformer_block() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device.clone(), 1).expect("encoder");
        let (dim, vocab) = (2usize, 3usize);
        let ids = [1u32, 2];
        let token_table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let position_table = [0.01_f32, 0.02, 0.03, 0.04];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let zero_matrix = [0.0_f32; 4];
        let layers = [layer_weights(&identity, &zero_matrix, &norm, &zero)];
        let weights = ClipTextEncoderWeights {
            token_table: &token_table,
            position_table: &position_table,
            layers: &layers,
            ffn_hidden: dim,
        };

        let actual = encoder.run(&ids, &weights, vocab, dim).expect("encode");
        let embedded = crate::embedding::TextEncoder::new(device.clone())
            .expect("embedding")
            .encode(&ids, &token_table, &position_table, vocab, dim)
            .expect("embed");
        let expected = crate::transformer::TransformerBlock::new(device)
            .expect("block")
            .run(&embedded, &layers[0], ids.len(), dim, dim)
            .expect("block run");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn rejects_wrong_layer_count_and_oversized_sequence() {
        let Some(device) = device() else { return };
        let encoder = ClipTextEncoder::new(device.clone(), 1).expect("encoder");
        let dim = 2usize;
        let matrix = [0.0_f32; 4];
        let norm = [1.0_f32, 1.0];
        let zero = [0.0_f32, 0.0];
        let layers = [layer_weights(&matrix, &matrix, &norm, &zero)];
        let weights = ClipTextEncoderWeights {
            token_table: &matrix,
            position_table: &norm,
            layers: &[],
            ffn_hidden: dim,
        };
        assert!(encoder.run(&[0], &weights, 2, dim).is_err());

        let weights = ClipTextEncoderWeights {
            token_table: &matrix,
            position_table: &norm,
            layers: &layers,
            ffn_hidden: dim,
        };
        assert!(encoder.run(&[0, 1, 2], &weights, 2, dim).is_err());
    }
}
