//! Scaled dot-product attention composed from the matmul + softmax kernels.
//!
//! Q/V are token-major (`[token, head, width]`), K is head-blocked
//! (`[head, head_dim, seq_k]`), and output is token-major
//! (`[token, head, value_dim]`). K's transpose is the caller's decision.

use crate::{VkError, device::VulkanDevice, matmul::MatMul, softmax::Softmax};
use std::sync::Arc;

/// Matrix shapes for one or more attention heads.
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
    /// Number of heads concatenated in each input matrix.
    pub heads: usize,
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
    let dim = shape
        .head_dim
        .checked_mul(shape.heads)
        .ok_or_else(invalid)?;
    let value = shape
        .value_dim
        .checked_mul(shape.heads)
        .ok_or_else(invalid)?;
    let q_len = shape.seq_q.checked_mul(dim).ok_or_else(invalid)?;
    let k_len = dim.checked_mul(shape.seq_k).ok_or_else(invalid)?;
    let v_len = shape.seq_k.checked_mul(value).ok_or_else(invalid)?;
    if shape.seq_q == 0
        || shape.seq_k == 0
        || shape.head_dim == 0
        || shape.value_dim == 0
        || shape.heads == 0
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

    /// Run `heads` parallel heads over token-major Q/V and head-blocked Kᵀ.
    ///
    /// ponytail: split/concat on the host; no dedicated shader until a
    /// profiled real checkpoint shows the copies dominate.
    pub fn run(
        &self,
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        shape: AttentionShape,
    ) -> Result<Vec<f32>, VkError> {
        self.run_masked(q, k_t, v, shape, None)
    }

    /// Additive `seq_q × seq_k` mask applied to scaled scores before softmax;
    /// `-INFINITY` blocks a key. Each query needs at least one unmasked key.
    pub fn run_masked(
        &self,
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        shape: AttentionShape,
        mask: Option<&[f32]>,
    ) -> Result<Vec<f32>, VkError> {
        validate_shapes(q, k_t, v, shape)?;
        if let Some(mask) = mask {
            let len = shape
                .seq_q
                .checked_mul(shape.seq_k)
                .ok_or_else(|| VkError::Device("attention: mask length overflow".into()))?;
            if mask.len() != len {
                return Err(VkError::Device(
                    "attention: mask must be seq_q * seq_k".into(),
                ));
            }
        }
        if shape.heads == 1 {
            return self.run_single(q, k_t, v, shape, mask);
        }
        let k_stride = shape
            .head_dim
            .checked_mul(shape.seq_k)
            .ok_or_else(|| VkError::Device("attention: head stride overflow".into()))?;
        let capacity = shape
            .seq_q
            .checked_mul(shape.heads)
            .and_then(|tokens| tokens.checked_mul(shape.value_dim))
            .ok_or_else(|| VkError::Device("attention: output overflow".into()))?;
        let mut out = vec![0.0; capacity];
        for head in 0..shape.heads {
            let k_start = head * k_stride;
            let mut q_head = Vec::with_capacity(shape.seq_q * shape.head_dim);
            for token in 0..shape.seq_q {
                let start = token * shape.heads * shape.head_dim + head * shape.head_dim;
                q_head.extend_from_slice(&q[start..start + shape.head_dim]);
            }
            let mut v_head = Vec::with_capacity(shape.seq_k * shape.value_dim);
            for token in 0..shape.seq_k {
                let start = token * shape.heads * shape.value_dim + head * shape.value_dim;
                v_head.extend_from_slice(&v[start..start + shape.value_dim]);
            }
            let head_out = self.run_single(
                &q_head,
                &k_t[k_start..k_start + k_stride],
                &v_head,
                AttentionShape { heads: 1, ..shape },
                mask,
            )?;
            for token in 0..shape.seq_q {
                let src = token * shape.value_dim;
                let dst = token * shape.heads * shape.value_dim + head * shape.value_dim;
                out[dst..dst + shape.value_dim]
                    .copy_from_slice(&head_out[src..src + shape.value_dim]);
            }
        }
        Ok(out)
    }

    fn run_single(
        &self,
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        shape: AttentionShape,
        mask: Option<&[f32]>,
    ) -> Result<Vec<f32>, VkError> {
        // Head dimensions are far below u16::MAX in every model we target;
        // the bound keeps the f32 conversion lossless for clippy and math.
        let dim = u16::try_from(shape.head_dim)
            .map_err(|_| VkError::Device("attention: head_dim exceeds u16::MAX".into()))?;
        let scale = 1.0 / f32::from(dim).sqrt();
        let scores = self
            .scores
            .run(q, k_t, shape.seq_q, shape.head_dim, shape.seq_k)?;
        let weights = match mask {
            Some(mask) => {
                let masked: Vec<f32> = scores
                    .iter()
                    .zip(mask)
                    .map(|(score, bias)| score * scale + bias)
                    .collect();
                self.probabilities.run(&masked, shape.seq_q, shape.seq_k)?
            }
            None => self
                .probabilities
                .run_scaled(&scores, shape.seq_q, shape.seq_k, scale)?,
        };
        self.output
            .run(&weights, v, shape.seq_q, shape.seq_k, shape.value_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    fn cpu_attention(
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        shape: AttentionShape,
        mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let scale = 1.0 / f32::from(u16::try_from(shape.head_dim).expect("test head_dim")).sqrt();
        let mut out = vec![0.0; shape.seq_q * shape.heads * shape.value_dim];
        for head in 0..shape.heads {
            let k_base = head * shape.head_dim * shape.seq_k;
            for row in 0..shape.seq_q {
                // Token-major gather: [token, head, width].
                let q_base = row * shape.heads * shape.head_dim + head * shape.head_dim;
                let scores: Vec<f32> = (0..shape.seq_k)
                    .map(|col| {
                        let dot: f32 = (0..shape.head_dim)
                            .map(|d| q[q_base + d] * k_t[k_base + d * shape.seq_k + col])
                            .sum();
                        dot * scale + mask.map_or(0.0, |m| m[row * shape.seq_k + col])
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total: f32 = weights.iter().sum();
                for d in 0..shape.value_dim {
                    let value: f32 = (0..shape.seq_k)
                        .map(|col| {
                            weights[col] / total
                                * v[col * shape.heads * shape.value_dim
                                    + head * shape.value_dim
                                    + d]
                        })
                        .sum();
                    // Token-major scatter: [token, head, value_dim].
                    let dst = row * shape.heads * shape.value_dim + head * shape.value_dim + d;
                    out[dst] = value;
                }
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
                    heads: 1,
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
                    heads: 1,
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
                    heads: 1,
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
            heads: 1,
        };
        let actual = attention.run(&q, &k_t, &v, shape).expect("gpu dispatch");
        let expected = cpu_attention(&q, &k_t, &v, shape, None);
        assert_close(&actual, &expected);
    }

    #[test]
    fn multi_head_attention_matches_split_cpu_reference() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let (heads, seq_q, seq_k, head_dim, value_dim) = (3usize, 2usize, 4usize, 3usize, 2usize);
        let value = |i: usize| f32::from(u8::try_from(i % 11).expect("0..11")) * 0.125;
        let q: Vec<f32> = (0..heads * seq_q * head_dim).map(value).collect();
        let k_t: Vec<f32> = (0..heads * head_dim * seq_k)
            .map(|i| value(i + 3))
            .collect();
        let v: Vec<f32> = (0..heads * seq_k * value_dim)
            .map(|i| value(i + 7))
            .collect();
        let shape = AttentionShape {
            seq_q,
            seq_k,
            head_dim,
            value_dim,
            heads,
        };
        let expected = cpu_attention(&q, &k_t, &v, shape, None);
        let attention = Attention::new(device).expect("pipeline creation");
        let actual = attention.run(&q, &k_t, &v, shape).expect("gpu dispatch");
        assert_close(&actual, &expected);
        assert!(attention.run(&q[..q.len() - 1], &k_t, &v, shape).is_err());
    }

    #[test]
    fn causal_mask_blocks_future_keys() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let (heads, seq_q, seq_k, head_dim, value_dim) = (2usize, 3usize, 4usize, 2usize, 2usize);
        let value = |i: usize| f32::from(u8::try_from(i % 13).expect("0..13")) * 0.125;
        let q: Vec<f32> = (0..heads * seq_q * head_dim).map(value).collect();
        let k_t: Vec<f32> = (0..heads * head_dim * seq_k)
            .map(|i| value(i + 5))
            .collect();
        let v: Vec<f32> = (0..heads * seq_k * value_dim)
            .map(|i| value(i + 9))
            .collect();
        let mask: Vec<f32> = (0..seq_q)
            .flat_map(|row| {
                (0..seq_k).map(move |col| if col > row { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        let shape = AttentionShape {
            seq_q,
            seq_k,
            head_dim,
            value_dim,
            heads,
        };
        let attention = Attention::new(device).expect("pipeline creation");
        let actual = attention
            .run_masked(&q, &k_t, &v, shape, Some(&mask))
            .expect("gpu dispatch");
        assert_close(&actual, &cpu_attention(&q, &k_t, &v, shape, Some(&mask)));
        assert!(
            attention
                .run_masked(&q, &k_t, &v, shape, Some(&mask[..mask.len() - 1]))
                .is_err()
        );

        // Future keys are masked: perturbing them cannot change earlier queries.
        let (mut changed_k, mut changed_v) = (k_t.clone(), v.clone());
        for col in 1..seq_k {
            for d in 0..heads * head_dim {
                changed_k[d * seq_k + col] += 1.0;
            }
            for d in 0..heads * value_dim {
                changed_v[col * heads * value_dim + d] += 1.0;
            }
        }
        let changed = attention
            .run_masked(&q, &changed_k, &changed_v, shape, Some(&mask))
            .expect("gpu dispatch");
        for head in 0..heads {
            for d in 0..value_dim {
                let idx = head * value_dim + d;
                assert!(
                    (actual[idx] - changed[idx]).abs() < 1e-6,
                    "query 0 changed: got {}, want {}",
                    changed[idx],
                    actual[idx]
                );
            }
        }
    }
}
