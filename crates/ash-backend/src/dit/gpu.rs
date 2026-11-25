//! Vulkan path for one flow-DiT block: GPU projections and attention, host
//! elementwise gating.

use super::block::{
    DitBlockWeights, gate_residual, matmul_rows, modulation, rms_norm_rows, rope_axes,
};
use super::invalid;
use super::spec::FlowDitSpec;
use crate::VkError;
use crate::attention::{Attention, AttentionShape};
use crate::device::VulkanDevice;
use crate::matmul::MatMul;
use crate::model::decoder_ops::{qk_norm, swiglu};
use std::sync::Arc;

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;

fn area(a: usize, b: usize, field: &str) -> Result<usize, VkError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(field, "size overflow"))
}

fn require_len(actual: usize, expected: usize, field: &str) -> Result<(), VkError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(
            field,
            &format!("expected {expected}, got {actual}"),
        ))
    }
}

/// Reusable GPU pipelines for the single-stream flow-DiT substrate.
pub struct DitGpu {
    matmul: MatMul,
    attention: Attention,
}

impl DitGpu {
    /// Build the matmul and attention pipelines from embedded SPIR-V.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            matmul: MatMul::new(Arc::clone(&device))?,
            attention: Attention::new(device)?,
        })
    }

    /// GPU counterpart of [`super::block::dit_block`], element-for-element.
    // ponytail: head splits, gates and SWiGLU stay on the host; fuse them into
    // one shader only if profiling shows these copies dominate throughput.
    #[allow(clippy::too_many_lines)] // one transparent composition of existing kernels
    pub fn block(
        &self,
        x: &[f32],
        cond: &[f32],
        positions: &[[f32; 3]],
        spec: &FlowDitSpec,
        weights: &DitBlockWeights,
        mask: Option<&[f32]>,
    ) -> Result<Vec<f32>, VkError> {
        spec.validate()?;
        let tokens = spec.tokens()?;
        let width = spec.width;
        let heads = spec.heads;
        let head_dim = spec.head_dim;
        let ffn_width = spec.ffn_width;
        let rows = area(tokens, width, "x")?;
        require_len(x.len(), rows, "x")?;
        require_len(cond.len(), width, "cond")?;
        require_len(positions.len(), tokens, "positions")?;
        if let Some(mask) = mask {
            require_len(mask.len(), area(tokens, tokens, "mask")?, "mask")?;
        }
        let ada_cols = area(width, 6, "ada")?;
        let qkv_cols = area(width, 3, "qkv")?;
        let ffn_cols = area(ffn_width, 2, "ffn")?;
        require_len(weights.ada.len(), area(width, ada_cols, "ada")?, "ada")?;
        require_len(weights.qkv.len(), area(qkv_cols, width, "qkv")?, "qkv")?;
        require_len(weights.qk_gain.len(), head_dim, "qk_gain")?;
        require_len(weights.out.len(), area(width, width, "out")?, "out")?;
        require_len(weights.ffn.len(), area(ffn_cols, width, "ffn")?, "ffn")?;
        require_len(
            weights.ffn_down.len(),
            area(width, ffn_width, "ffn_down")?,
            "ffn_down",
        )?;

        let mut ada = vec![0.0; ada_cols];
        matmul_rows(cond, &weights.ada, &mut ada, 1, width, ada_cols)?;
        let (attn_shift, rest) = ada.split_at(width);
        let (attn_scale, rest) = rest.split_at(width);
        let (attn_gate, rest) = rest.split_at(width);
        let (ffn_shift, rest) = rest.split_at(width);
        let (ffn_scale, ffn_gate) = rest.split_at(width);

        let mut norm = vec![0.0; rows];
        rms_norm_rows(x, &mut norm, tokens, width, RMS_EPS)?;
        let mut modulated = vec![0.0; rows];
        modulation(&norm, attn_shift, attn_scale, &mut modulated, tokens, width)?;
        let qkv = self
            .matmul
            .run(&modulated, &weights.qkv, tokens, width, qkv_cols)?;

        let per = area(tokens, width, "qkv")?;
        let per2 = area(2, per, "qkv")?;
        let q = &qkv[..per];
        let k = &qkv[per..per2];
        let v = &qkv[per2..];
        let mut q_norm = vec![0.0; per];
        let mut k_norm = vec![0.0; per];
        qk_norm(
            q,
            &mut q_norm,
            tokens,
            heads,
            head_dim,
            &weights.qk_gain,
            RMS_EPS,
        )?;
        qk_norm(
            k,
            &mut k_norm,
            tokens,
            heads,
            head_dim,
            &weights.qk_gain,
            RMS_EPS,
        )?;
        let mut q_rot = vec![0.0; per];
        let mut k_rot = vec![0.0; per];
        rope_axes(
            &q_norm, positions, &mut q_rot, tokens, heads, head_dim, spec.axes, ROPE_THETA,
        )?;
        rope_axes(
            &k_norm, positions, &mut k_rot, tokens, heads, head_dim, spec.axes, ROPE_THETA,
        )?;

        // Attention takes head-blocked Kᵀ; the CPU block uses token-major K.
        let k_stride = area(head_dim, tokens, "k_t")?;
        let mut k_t = vec![0.0; area(heads, k_stride, "k_t")?];
        for head in 0..heads {
            for token in 0..tokens {
                for dim in 0..head_dim {
                    let src = token * width + head * head_dim + dim;
                    k_t[head * k_stride + dim * tokens + token] = k_rot[src];
                }
            }
        }
        let attended = self.attention.run_masked(
            &q_rot,
            &k_t,
            v,
            AttentionShape {
                seq_q: tokens,
                seq_k: tokens,
                head_dim,
                value_dim: head_dim,
                heads,
            },
            mask,
        )?;
        let projected = self
            .matmul
            .run(&attended, &weights.out, tokens, width, width)?;
        let mut residual = vec![0.0; rows];
        gate_residual(&projected, x, attn_gate, &mut residual, tokens, width)?;

        let mut norm = vec![0.0; rows];
        rms_norm_rows(&residual, &mut norm, tokens, width, RMS_EPS)?;
        let mut modulated = vec![0.0; rows];
        modulation(&norm, ffn_shift, ffn_scale, &mut modulated, tokens, width)?;
        let ffn = self
            .matmul
            .run(&modulated, &weights.ffn, tokens, width, ffn_cols)?;

        let half = area(tokens, ffn_width, "ffn")?;
        let mut gate = vec![0.0; half];
        let mut up = vec![0.0; half];
        for token in 0..tokens {
            let row = token * ffn_cols;
            gate[token * ffn_width..(token + 1) * ffn_width]
                .copy_from_slice(&ffn[row..row + ffn_width]);
            up[token * ffn_width..(token + 1) * ffn_width]
                .copy_from_slice(&ffn[row + ffn_width..row + ffn_cols]);
        }
        let mut activated = vec![0.0; half];
        swiglu(&gate, &up, &mut activated)?;
        let down = self
            .matmul
            .run(&activated, &weights.ffn_down, tokens, ffn_width, width)?;
        let mut out = vec![0.0; rows];
        gate_residual(&down, &residual, ffn_gate, &mut out, tokens, width)?;
        Ok(out)
    }
}
