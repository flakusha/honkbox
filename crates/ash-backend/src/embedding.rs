//! Text-encoder embedding stage: token/positional row gather plus their sum.
//!
//! This is the compute half of text conditioning — the caller supplies token
//! IDs and embedding tables located in the checkpoint. CLIP BPE tokenization
//! needs vocab/merge assets and a license review, so no prompt→ID conversion
//! lives here; nothing is derived from a prompt string.

use crate::{
    VkError,
    compute::{
        EMBEDDING_GATHER_SPIRV, ElementwiseAdd, GpuBuffer, Kernel, f32_bytes, f32_from_bytes,
        spirv_words,
    },
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const LOCAL_SIZE: usize = 64;

/// Reusable pipeline for `out[token, :] = table[ids[token], :]`.
pub struct Embedding {
    kernel: Kernel,
}

impl Embedding {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(EMBEDDING_GATHER_SPIRV))?,
        })
    }

    /// Gather one row of a `vocab × dim` table per token ID.
    pub fn run(
        &self,
        table: &[f32],
        ids: &[u32],
        vocab: usize,
        dim: usize,
    ) -> Result<Vec<f32>, VkError> {
        let invalid = || {
            VkError::Device(
                "embedding: non-empty ids, positive dim and matching table required".into(),
            )
        };
        let table_len = vocab.checked_mul(dim).ok_or_else(invalid)?;
        let output_len = ids.len().checked_mul(dim).ok_or_else(invalid)?;
        if vocab == 0 || dim == 0 || ids.is_empty() || table.len() != table_len {
            return Err(invalid());
        }
        if ids
            .iter()
            .any(|id| usize::try_from(*id).is_ok_and(|id| id >= vocab))
        {
            return Err(VkError::Device(
                "embedding: token id outside the table".into(),
            ));
        }
        let words =
            |value: usize| -> Result<u32, VkError> { u32::try_from(value).map_err(|_| invalid()) };
        let dims = [words(ids.len())?, words(vocab)?, words(dim)?];
        let workgroups = words(output_len.div_ceil(LOCAL_SIZE))?;
        let table_bytes = table_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("embedding: table byte length overflow".into()))?;
        let ids_bytes = ids
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| VkError::Device("embedding: ids byte length overflow".into()))?;
        let output_bytes = output_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VkError::Device("embedding: output byte length overflow".into()))?;
        let table_u64 = u64::try_from(table_bytes)
            .map_err(|_| VkError::Device("embedding: table buffer too large".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: table_u64,
            ..CapacityRequirements::default()
        })?;

        let params = dims
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>();
        let ids_bytes_vec = ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<u8>>();
        let table_buf = GpuBuffer::new(&self.kernel.device, &f32_bytes(table))?;
        let ids_buf = GpuBuffer::new(&self.kernel.device, &ids_bytes_vec)?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; output_bytes])?;
        let params_buf = GpuBuffer::new(&self.kernel.device, &params)?;
        let result = self.kernel.dispatch(
            &[table_buf, ids_buf, output, params_buf],
            [
                table_u64,
                u64::try_from(ids_bytes)
                    .map_err(|_| VkError::Device("embedding: ids buffer too large".into()))?,
                u64::try_from(output_bytes)
                    .map_err(|_| VkError::Device("embedding: output buffer too large".into()))?,
                12,
            ],
            2,
            output_bytes,
            [workgroups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

/// Token embedding + positional embedding for a sequence of token IDs.
pub struct TextEncoder {
    gather: Embedding,
    add: ElementwiseAdd,
}

impl TextEncoder {
    /// Two pipelines: the shared gather kernel and the elementwise add.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            gather: Embedding::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// `token_table[ids[t]] + position_table[t]`, row-major `tokens × dim`.
    pub fn encode(
        &self,
        ids: &[u32],
        token_table: &[f32],
        position_table: &[f32],
        vocab: usize,
        dim: usize,
    ) -> Result<Vec<f32>, VkError> {
        let invalid = || {
            VkError::Device(
                "embedding: positional table must be a non-empty multiple of dim".into(),
            )
        };
        if dim == 0 || position_table.is_empty() || !position_table.len().is_multiple_of(dim) {
            return Err(invalid());
        }
        let positions = position_table.len() / dim;
        if ids.len() > positions {
            return Err(VkError::Device(
                "embedding: sequence exceeds positional table".into(),
            ));
        }
        let vectors = self.gather.run(token_table, ids, vocab, dim)?;
        let indices = (0..u32::try_from(ids.len()).map_err(|_| invalid())?).collect::<Vec<u32>>();
        let positional = self.gather.run(position_table, &indices, positions, dim)?;
        self.add.run(&vectors, &positional)
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
    fn embedding_gather_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let embedding = Embedding::new(device).expect("pipeline");
        let table = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let ids = [2u32, 0, 1];
        let actual = embedding.run(&table, &ids, 3, 2).expect("gather");
        let expected = [0.5_f32, 0.6, 0.1, 0.2, 0.3, 0.4];
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn embedding_rejects_bad_ids_and_shapes() {
        let Some(device) = device() else { return };
        let embedding = Embedding::new(device).expect("pipeline");
        let table = [0.1_f32, 0.2, 0.3, 0.4];
        assert!(embedding.run(&table, &[3], 3, 2).is_err());
        assert!(embedding.run(&table, &[0], 3, 3).is_err());
        assert!(embedding.run(&table, &[], 2, 2).is_err());
        assert!(embedding.run(&table, &[0], 0, 2).is_err());
    }

    #[test]
    fn text_encoder_adds_token_and_positional_tables() {
        let Some(device) = device() else { return };
        let encoder = TextEncoder::new(device).expect("pipeline");
        let tokens = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let positions = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ids = [2u32, 0];
        let actual = encoder
            .encode(&ids, &tokens, &positions, 3, 2)
            .expect("encode");
        let expected = [1.5_f32, 2.6, 3.1, 4.2];
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
        assert!(
            encoder
                .encode(&[0, 1, 2, 3], &tokens, &positions, 3, 2)
                .is_err()
        );
    }
}
