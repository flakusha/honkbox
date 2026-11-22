//! Scaled dot-product attention composed from the matmul + softmax kernels.
//!
//! `K` is supplied pre-transposed (`head_dim × seq_k`): a transpose kernel is
//! not worth a separate buffer pass yet. Callers own that layout decision.

use crate::{VkError, device::VulkanDevice, matmul::MatMul, softmax::Softmax};
use std::sync::Arc;

/// Matrix shapes for one attention head.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    /// Query rows.
    pub seq_q: usize,
    /// Key/value rows.
    pub seq_k: usize,
    /// Per-head query/key width.
    pub head_dim: usize,
    /// Per-head value width.
    pub value_dim: usize,
}

/// `softmax(Q · Kᵀ / sqrt(head_dim)) · V`.
pub struct Attention {
    scores: MatMul,
    probabilities: Softmax,
    output: MatMul,
}

fn validate_shapes(
    q: &[f32],
    k_t: &[f32],
    v: &[f32],
    shape: AttentionShape,
) -> Result<(), VkError> {
    let invalid =
        || VkError::Device("attention: non-empty shapes must match all input lengths".into());
    let q_len = shape
        .seq_q
        .checked_mul(shape.head_dim)
        .ok_or_else(invalid)?;
    let k_len = shape
        .head_dim
        .checked_mul(shape.seq_k)
        .ok_or_else(invalid)?;
    let v_len = shape
        .seq_k
        .checked_mul(shape.value_dim)
        .ok_or_else(invalid)?;
    if shape.seq_q == 0
        || shape.seq_k == 0
        || shape.head_dim == 0
        || shape.value_dim == 0
        || q.len() != q_len
        || k_t.len() != k_len
        || v.len() != v_len
    {
        return Err(invalid());
    }
    Ok(())
}

impl Attention {
    /// Three pipelines: score matmul, scaled softmax, output matmul.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            scores: MatMul::new(Arc::clone(&device))?,
            probabilities: Softmax::new(Arc::clone(&device))?,
            output: MatMul::new(device)?,
        })
    }

    /// Run one attention head and return `seq_q × value_dim` row-major output.
    pub fn run(
        &self,
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        shape: AttentionShape,
    ) -> Result<Vec<f32>, VkError> {
        validate_shapes(q, k_t, v, shape)?;
        // Head dimensions are far below u16::MAX in every model we target;
        // the bound keeps the f32 conversion lossless for clippy and math.
        let dim = u16::try_from(shape.head_dim)
            .map_err(|_| VkError::Device("attention: head_dim exceeds u16::MAX".into()))?;
        let scale = 1.0 / f32::from(dim).sqrt();
        let scores = self
            .scores
            .run(q, k_t, shape.seq_q, shape.head_dim, shape.seq_k)?;
        let weights = self
            .probabilities
            .run_scaled(&scores, shape.seq_q, shape.seq_k, scale)?;
        self.output
            .run(&weights, v, shape.seq_q, shape.seq_k, shape.value_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_attention(
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / f32::from(u16::try_from(head_dim).expect("test head_dim")).sqrt();
        let mut out = Vec::with_capacity(seq_q * value_dim);
        for row in 0..seq_q {
            let scores: Vec<f32> = (0..seq_k)
                .map(|col| {
                    (0..head_dim)
                        .map(|d| q[row * head_dim + d] * k_t[d * seq_k + col])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
            let total: f32 = weights.iter().sum();
            for d in 0..value_dim {
                let value: f32 = (0..seq_k)
                    .map(|col| weights[col] / total * v[col * value_dim + d])
                    .sum();
                out.push(value);
            }
        }
        out
    }

    #[test]
    fn attention_rejects_mismatched_shapes() {
        let q = vec![1.0; 4];
        let k_t = vec![1.0; 6];
        let v = vec![1.0; 6];
        assert!(
            validate_shapes(
                &q,
                &k_t,
                &v,
                AttentionShape {
                    seq_q: 2,
                    seq_k: 3,
                    head_dim: 2,
                    value_dim: 2,
                }
            )
            .is_ok()
        );
        assert!(
            validate_shapes(
                &q,
                &k_t,
                &v,
                AttentionShape {
                    seq_q: 2,
                    seq_k: 3,
                    head_dim: 2,
                    value_dim: 3,
                }
            )
            .is_err()
        );
        assert!(
            validate_shapes(
                &q,
                &k_t,
                &v,
                AttentionShape {
                    seq_q: 0,
                    seq_k: 3,
                    head_dim: 2,
                    value_dim: 2,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn attention_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let (seq_q, seq_k, head_dim, value_dim) = (3usize, 4usize, 5usize, 2usize);
        let value = |i: usize| f32::from(u8::try_from(i % 7).expect("0..7")) * 0.25;
        let q: Vec<f32> = (0..seq_q * head_dim).map(value).collect();
        let k_t: Vec<f32> = (0..head_dim * seq_k).map(|i| value(i + 2)).collect();
        let v: Vec<f32> = (0..seq_k * value_dim).map(|i| value(i + 5)).collect();

        let attention = Attention::new(device).expect("pipeline creation");
        let shape = AttentionShape {
            seq_q,
            seq_k,
            head_dim,
            value_dim,
        };
        let actual = attention.run(&q, &k_t, &v, shape).expect("gpu dispatch");
        let expected = cpu_attention(&q, &k_t, &v, seq_q, seq_k, head_dim, value_dim);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}
