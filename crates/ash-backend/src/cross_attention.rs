//! Fused `to_qkv`/`to_kv` projections plus cross-attention for SD `UNet` blocks.
//!
//! Diffusers stores these weights `[out_features, in_features]`; the shared
//! `MatMul` wants `[in, out]`, so a transpose pass converts them before dispatch.
//! ponytail: one head, no attention mask, no scaling override.

use crate::{
    VkError,
    attention::{Attention, AttentionShape},
    compute::ElementwiseAdd,
    device::VulkanDevice,
    matmul::MatMul,
    transpose::Transpose,
};
use std::sync::Arc;

/// One fused `[3·dim, dim]` projection split into row-major Q/K/V matrices.
pub struct FusedQkv {
    reorder: Transpose,
    project: MatMul,
    add: ElementwiseAdd,
}

impl FusedQkv {
    /// Compile the transpose and projection pipelines.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            reorder: Transpose::new(Arc::clone(&device))?,
            project: MatMul::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// Apply a `[parts·out_dim, in_dim]` weight and split into `parts` outputs.
    #[allow(clippy::too_many_arguments)] // kernel signature: input/weight/bias + shape
    pub fn run_with_bias(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        parts: usize,
    ) -> Result<Vec<Vec<f32>>, VkError> {
        if parts == 0 {
            return Err(VkError::Device("fused qkv: parts must be positive".into()));
        }
        let weight_len = in_dim
            .checked_mul(out_dim)
            .and_then(|per| per.checked_mul(parts))
            .ok_or_else(|| VkError::Device("fused qkv: weight length overflow".into()))?;
        if weight.len() != weight_len {
            return Err(VkError::Device(
                "fused qkv: weight must be [parts*out_dim, in_dim]".into(),
            ));
        }
        let reordered = self.reorder.run(weight, parts * out_dim, in_dim)?;
        let projected = self
            .project
            .run(input, &reordered, rows, in_dim, out_dim * parts)?;
        let projected = match bias {
            Some(bias) if !bias.is_empty() => self.add.run_broadcast(&projected, bias)?,
            _ => projected,
        };
        let plane = out_dim
            .checked_mul(parts)
            .ok_or_else(|| VkError::Device("fused qkv: output overflow".into()))?;
        let mut split = vec![Vec::with_capacity(rows * out_dim); parts];
        for row in projected.chunks(plane) {
            for (index, part) in split.iter_mut().enumerate() {
                let start = index * out_dim;
                part.extend_from_slice(&row[start..start + out_dim]);
            }
        }
        Ok(split)
    }

    /// Apply a `[parts·out_dim, in_dim]` weight and split into `parts` outputs.
    pub fn run(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        parts: usize,
    ) -> Result<Vec<Vec<f32>>, VkError> {
        self.run_with_bias(input, weight, None, rows, in_dim, out_dim, parts)
    }
}

/// Weights for one cross-attention block.
pub struct CrossAttentionWeights<'a> {
    /// `[dim, dim]` hidden-state query projection (diffusers `to_q`).
    pub wq: &'a [f32],
    /// `[2·dim, dim]` fused context key/value projection (`to_kv`).
    pub wkv: &'a [f32],
    /// `[dim, dim]` attention output projection (`to_out.0`).
    pub wo: &'a [f32],
}

/// Optional linear biases for one cross-attention block.
#[derive(Clone, Copy, Default)]
pub struct CrossAttentionBias<'a> {
    /// Query projection bias `[dim]`.
    pub bq: Option<&'a [f32]>,
    /// Fused key/value bias `[2·dim]`, key rows then value rows.
    pub bkv: Option<&'a [f32]>,
    /// Output projection bias `[dim]`.
    pub bo: Option<&'a [f32]>,
}

/// Row counts and shared channel width for one cross-attention head.
#[derive(Debug, Clone, Copy)]
pub struct CrossAttentionShape {
    /// Hidden-state tokens (query rows).
    pub query_rows: usize,
    /// Context tokens (key/value rows).
    pub context_rows: usize,
    /// Query/key/value channel width.
    pub dim: usize,
}

/// `Attention(W_q·hidden, W_kv·context)` plus output projection and residual.
pub struct CrossAttention {
    query: MatMul,
    kv: FusedQkv,
    transpose: Transpose,
    attention: Attention,
    output: MatMul,
    add: ElementwiseAdd,
}

impl CrossAttention {
    /// Compile the six pipelines used by one block.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            query: MatMul::new(Arc::clone(&device))?,
            kv: FusedQkv::new(Arc::clone(&device))?,
            transpose: Transpose::new(Arc::clone(&device))?,
            attention: Attention::new(Arc::clone(&device))?,
            output: MatMul::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// Run cross-attention with optional projection biases.
    pub fn run_with_bias(
        &self,
        hidden: &[f32],
        context: &[f32],
        weights: &CrossAttentionWeights<'_>,
        bias: &CrossAttentionBias<'_>,
        shape: CrossAttentionShape,
    ) -> Result<Vec<f32>, VkError> {
        let mut q = self
            .query
            .run(hidden, weights.wq, shape.query_rows, shape.dim, shape.dim)?;
        if let Some(extra) = bias.bq {
            q = self.add.run_broadcast(&q, extra)?;
        }
        let mut kv = self.kv.run_with_bias(
            context,
            weights.wkv,
            bias.bkv,
            shape.context_rows,
            shape.dim,
            shape.dim,
            2,
        )?;
        if kv.len() != 2 {
            return Err(VkError::Device("cross attention: expected K and V".into()));
        }
        let value = kv.pop().expect("two parts");
        let key = kv.pop().expect("two parts");
        let k_t = self.transpose.run(&key, shape.context_rows, shape.dim)?;
        let attended = self.attention.run(
            &q,
            &k_t,
            &value,
            AttentionShape {
                seq_q: shape.query_rows,
                seq_k: shape.context_rows,
                head_dim: shape.dim,
                value_dim: shape.dim,
            },
        )?;
        let mut projected = self.output.run(
            &attended,
            weights.wo,
            shape.query_rows,
            shape.dim,
            shape.dim,
        )?;
        if let Some(extra) = bias.bo {
            projected = self.add.run_broadcast(&projected, extra)?;
        }
        self.add.run(hidden, &projected)
    }

    /// Run cross-attention with caller-owned weights and context tokens.
    pub fn run(
        &self,
        hidden: &[f32],
        context: &[f32],
        weights: &CrossAttentionWeights<'_>,
        shape: CrossAttentionShape,
    ) -> Result<Vec<f32>, VkError> {
        self.run_with_bias(
            hidden,
            context,
            weights,
            &CrossAttentionBias::default(),
            shape,
        )
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

    #[test]
    fn fused_qkv_splits_into_three_projections() {
        let Some(device) = device() else { return };
        let (rows, in_dim, out_dim) = (2usize, 3usize, 2usize);
        let input = [0.1_f32, -0.2, 0.3, 0.4, 0.5, -0.6];
        // [parts*out_dim, in_dim] rows are q0,q1,k0,k1,v0,v1.
        let weight = [
            0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -0.1, -0.2, -0.3, 0.2, -0.4, 0.6, 0.5,
            0.3, 0.1,
        ];
        let parts = FusedQkv::new(Arc::clone(&device))
            .expect("fused")
            .run(&input, &weight, rows, in_dim, out_dim, 3)
            .expect("run");
        assert_eq!(parts.len(), 3);
        let plain = MatMul::new(device).expect("matmul");
        for (index, expected) in weight.chunks(out_dim * in_dim).enumerate() {
            // Convert [out, in] to [in, out] for the plain matmul.
            let mut rhs = vec![0.0_f32; in_dim * out_dim];
            for out in 0..out_dim {
                for into in 0..in_dim {
                    rhs[into * out_dim + out] = expected[out * in_dim + into];
                }
            }
            let want = plain
                .run(&input, &rhs, rows, in_dim, out_dim)
                .expect("plain");
            assert_eq!(parts[index].len(), want.len());
            for (got, value) in parts[index].iter().zip(&want) {
                assert!(
                    (got - value).abs() < 1e-6,
                    "part {index}: got {got}, want {value}"
                );
            }
        }
    }

    #[test]
    fn fused_qkv_rejects_wrong_weight_length() {
        let Some(device) = device() else { return };
        assert!(
            FusedQkv::new(device)
                .expect("fused")
                .run(&[1.0, 2.0], &[0.5], 2, 2, 2, 3)
                .is_err()
        );
    }

    #[test]
    fn fused_qkv_bias_keeps_key_before_value() {
        let Some(device) = device() else { return };
        let fused = FusedQkv::new(device).expect("fused");
        // Rows: k0,k1,v0,v1. K is identity; V swaps the two inputs.
        let weight = [1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let bias = [0.1_f32, 0.2, 0.3, 0.4];
        let parts = fused
            .run_with_bias(&[1.0_f32, 2.0], &weight, Some(&bias), 1, 2, 2, 2)
            .expect("biased run");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], vec![1.1_f32, 2.2]);
        assert_eq!(parts[1], vec![2.3_f32, 1.4]);
    }

    #[test]
    fn cross_attention_matches_manual_decomposition() {
        let Some(device) = device() else { return };
        let (query_rows, context_rows, dim) = (2usize, 3usize, 2usize);
        let hidden = [0.1_f32, -0.2, 0.3, 0.4];
        let context = [0.5_f32, 0.1, -0.3, 0.2, 0.4, -0.1];
        let wq = [0.1_f32, 0.2, -0.3, 0.4];
        // [2*dim, dim] rows: k0,k1,v0,v1.
        let wkv = [0.5_f32, -0.1, 0.2, 0.3, -0.2, 0.4, 0.1, -0.3];
        let wo = [0.2_f32, 0.1, -0.1, 0.3];
        let weights = CrossAttentionWeights {
            wq: &wq,
            wkv: &wkv,
            wo: &wo,
        };
        let shape = CrossAttentionShape {
            query_rows,
            context_rows,
            dim,
        };
        let actual = CrossAttention::new(Arc::clone(&device))
            .expect("cross")
            .run(&hidden, &context, &weights, shape)
            .expect("run");

        let fused = FusedQkv::new(Arc::clone(&device))
            .expect("fused")
            .run(&context, &wkv, context_rows, dim, dim, 2)
            .expect("fused run");
        let key = &fused[0];
        let value = &fused[1];
        let plain = MatMul::new(Arc::clone(&device)).expect("matmul");
        let k_t = Transpose::new(Arc::clone(&device))
            .expect("transpose")
            .run(key, context_rows, dim)
            .expect("transpose run");
        let q = plain.run(&hidden, &wq, query_rows, dim, dim).expect("q");
        let attended = Attention::new(Arc::clone(&device))
            .expect("attention")
            .run(
                &q,
                &k_t,
                value,
                AttentionShape {
                    seq_q: query_rows,
                    seq_k: context_rows,
                    head_dim: dim,
                    value_dim: dim,
                },
            )
            .expect("attention run");
        let projected = plain
            .run(&attended, &wo, query_rows, dim, dim)
            .expect("project");
        let expected = ElementwiseAdd::new(device)
            .expect("add")
            .run(&hidden, &projected)
            .expect("add run");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }
}
