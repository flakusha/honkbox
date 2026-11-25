//! Portable CPU reference for one single-stream flow-DiT block.
//!
//! All shapes are explicit and row-major. Validation completes before any
//! output element is written; GPU path reuses the helpers below so both paths
//! share the same arithmetic.

// Bounded by caller slice lengths and synthetic test data.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::VkError;
use crate::model::decoder_ops::{qk_norm, rms_norm, swiglu};

use super::invalid;
use super::spec::{FlowDitSpec, PositionAxes};

const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;

fn require_len(actual: usize, expected: usize, field: &str) -> Result<(), VkError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(
            field,
            &format!("length {actual} must be {expected}"),
        ))
    }
}

fn require_nonzero(value: usize, field: &str) -> Result<(), VkError> {
    if value == 0 {
        Err(invalid(field, "must be non-zero"))
    } else {
        Ok(())
    }
}

fn area(a: usize, b: usize, field: &str) -> Result<usize, VkError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid(field, "element count overflow"))
}

fn require_finite(values: &[f32], field: &str) -> Result<(), VkError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid(field, "must be finite"))
    }
}

fn require_positions(positions: &[[f32; 3]], rows: usize) -> Result<(), VkError> {
    require_len(positions.len(), rows, "positions")?;
    if positions
        .iter()
        .all(|point| point.iter().all(|value| value.is_finite()))
    {
        Ok(())
    } else {
        Err(invalid("positions", "must be finite"))
    }
}

/// `out[r, c] = sum_k lhs[r, k] * rhs[k, c]` with explicit `[rows, inner]`,
/// `[inner, cols]` and `[rows, cols]` row-major shapes.
pub(crate) fn matmul_rows(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), VkError> {
    require_nonzero(rows, "rows")?;
    require_nonzero(inner, "inner")?;
    require_nonzero(cols, "cols")?;
    let lhs_len = area(rows, inner, "lhs")?;
    let rhs_len = area(inner, cols, "rhs")?;
    let out_len = area(rows, cols, "output")?;
    require_len(lhs.len(), lhs_len, "lhs")?;
    require_len(rhs.len(), rhs_len, "rhs")?;
    require_len(out.len(), out_len, "output")?;
    require_finite(lhs, "lhs")?;
    require_finite(rhs, "rhs")?;

    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0_f32;
            for k in 0..inner {
                sum += lhs[row * inner + k] * rhs[k * cols + col];
            }
            out[row * cols + col] = sum;
        }
    }
    Ok(())
}

/// Row-wise RMS normalization with unit gain and explicit `[rows, cols]`.
pub(crate) fn rms_norm_rows(
    values: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), VkError> {
    require_nonzero(rows, "rows")?;
    require_nonzero(cols, "cols")?;
    let len = area(rows, cols, "values")?;
    require_len(values.len(), len, "values")?;
    require_len(out.len(), len, "output")?;
    require_finite(values, "values")?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("eps", "must be finite and positive"));
    }
    let gain = vec![1.0; cols];
    rms_norm(values, out, rows, cols, &gain, eps)
        .map_err(|error| invalid("values", &error.to_string()))
}

/// `out[r, c] = values[r, c] * (1 + scale[c]) + shift[c]` for broadcast
/// shift/scale vectors of length `cols`.
pub(crate) fn modulation(
    values: &[f32],
    shift: &[f32],
    scale: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<(), VkError> {
    require_nonzero(rows, "rows")?;
    require_nonzero(cols, "cols")?;
    let len = area(rows, cols, "values")?;
    require_len(values.len(), len, "values")?;
    require_len(out.len(), len, "output")?;
    require_len(shift.len(), cols, "shift")?;
    require_len(scale.len(), cols, "scale")?;
    require_finite(values, "values")?;
    require_finite(shift, "shift")?;
    require_finite(scale, "scale")?;

    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            out[index] = values[index] * (1.0 + scale[col]) + shift[col];
        }
    }
    Ok(())
}

/// `out[r, c] = residual[r, c] + values[r, c] * gate[c]`.
pub(crate) fn gate_residual(
    values: &[f32],
    residual: &[f32],
    gate: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<(), VkError> {
    require_nonzero(rows, "rows")?;
    require_nonzero(cols, "cols")?;
    let len = area(rows, cols, "values")?;
    require_len(values.len(), len, "values")?;
    require_len(residual.len(), len, "residual")?;
    require_len(out.len(), len, "output")?;
    require_len(gate.len(), cols, "gate")?;
    require_finite(values, "values")?;
    require_finite(residual, "residual")?;
    require_finite(gate, "gate")?;

    for row in 0..rows {
        for (col, &gain) in gate.iter().enumerate().take(cols) {
            let index = row * cols + col;
            out[index] = residual[index] + values[index] * gain;
        }
    }
    Ok(())
}

/// Half-split `RoPE` over `[rows, heads, head_dim]`; rotary pairs cycle through
/// image height/width or video frames/height/width. All `head_dim` components
/// are rotary, and a zero position is the identity.
#[allow(clippy::too_many_arguments)] // full shape, positions and caller-owned output
pub(crate) fn rope_axes(
    values: &[f32],
    positions: &[[f32; 3]],
    out: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    axes: PositionAxes,
    theta: f32,
) -> Result<(), VkError> {
    require_nonzero(rows, "rows")?;
    require_nonzero(heads, "heads")?;
    require_nonzero(head_dim, "head_dim")?;
    if !head_dim.is_multiple_of(2) {
        return Err(invalid("head_dim", "must be even"));
    }
    let per_row = area(heads, head_dim, "heads * head_dim")?;
    let len = area(rows, per_row, "values")?;
    require_len(values.len(), len, "values")?;
    require_len(out.len(), len, "output")?;
    require_positions(positions, rows)?;
    require_finite(values, "values")?;
    if !theta.is_finite() || theta <= 0.0 {
        return Err(invalid("theta", "must be finite and positive"));
    }

    let half = head_dim / 2;
    let frequencies: Vec<f32> = (0..half)
        .map(|i| theta.powf(-2.0 * i as f32 / head_dim as f32))
        .collect();
    require_finite(&frequencies, "theta")?;
    let axis_count = match axes {
        PositionAxes::Image { .. } => 2,
        PositionAxes::Video { .. } => 3,
    };

    for (row, position) in positions.iter().enumerate().take(rows) {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for i in 0..half {
                let rotary_axis = i % axis_count;
                let (sin, cos) = (position[rotary_axis] * frequencies[i]).sin_cos();
                let x = values[base + i];
                let y = values[base + half + i];
                out[base + i] = x * cos - y * sin;
                out[base + half + i] = y * cos + x * sin;
            }
        }
    }
    Ok(())
}

/// Causal-mask-capable scaled dot-product attention for token-major
/// `[rows, heads, head_dim]` q/k/v. `mask` is an additive `[rows, rows]` matrix;
/// absence means no masking.
#[allow(clippy::too_many_arguments)] // full shape, additive mask and caller-owned output
pub(crate) fn attention_rows(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    out: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), VkError> {
    require_nonzero(rows, "rows")?;
    require_nonzero(heads, "heads")?;
    require_nonzero(head_dim, "head_dim")?;
    let per_row = area(heads, head_dim, "heads * head_dim")?;
    let len = area(rows, per_row, "attention")?;
    require_len(q.len(), len, "q")?;
    require_len(k.len(), len, "k")?;
    require_len(v.len(), len, "v")?;
    require_len(out.len(), len, "output")?;
    if let Some(mask) = mask {
        require_len(mask.len(), area(rows, rows, "mask")?, "mask")?;
        require_finite(mask, "mask")?;
    }
    require_finite(q, "q")?;
    require_finite(k, "k")?;
    require_finite(v, "v")?;

    let scale = (head_dim as f32).sqrt();
    if !scale.is_finite() || scale == 0.0 {
        return Err(invalid("head_dim", "scale must be finite and positive"));
    }
    let mut scores = vec![0.0; rows];

    for row in 0..rows {
        for head in 0..heads {
            for col in 0..rows {
                let mut dot = 0.0_f32;
                for dim in 0..head_dim {
                    let index = dim + head * head_dim;
                    dot += q[row * per_row + index] * k[col * per_row + index];
                }
                scores[col] = dot / scale + mask.map_or(0.0, |values| values[row * rows + col]);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                return Err(invalid("attention", "scores must be finite"));
            }
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let total: f32 = scores.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                return Err(invalid("attention", "softmax denominator invalid"));
            }
            for score in &mut scores {
                *score /= total;
            }
            for dim in 0..head_dim {
                let mut value = 0.0_f32;
                for col in 0..rows {
                    value += scores[col] * v[col * per_row + head * head_dim + dim];
                }
                out[row * per_row + head * head_dim + dim] = value;
            }
        }
    }
    Ok(())
}

/// Dimensional fields of one block's learnable weights. `ada` is row-major
/// `[W, 6W]`, so `cond [W] * ada` yields the six modulation vectors.
#[derive(Debug)]
pub struct DitBlockWeights {
    /// `[W, 6W]` conditioning-to-modulation projection.
    pub ada: Vec<f32>,
    /// `[W, 3W]` QKV projection.
    pub qkv: Vec<f32>,
    /// `[head_dim]` shared QK RMS gain.
    pub qk_gain: Vec<f32>,
    /// `[W, W]` attention output projection.
    pub out: Vec<f32>,
    /// `[W, 2 * ffn_width]` gated FFN projection.
    pub ffn: Vec<f32>,
    /// `[ffn_width, W]` FFN down projection.
    pub ffn_down: Vec<f32>,
}

/// Gather row-major `[channels, spatial...]` into token-major
/// `[token, channel, patch...]` tiles.
pub fn patch_flatten(
    latent: &[f32],
    channels: usize,
    spatial: &[usize],
    patch: &[usize],
) -> Result<Vec<f32>, VkError> {
    require_nonzero(channels, "channels")?;
    if spatial.len() != patch.len() || !(2..=3).contains(&spatial.len()) {
        return Err(invalid(
            "spatial",
            "and patch must have 2 or 3 matching axes",
        ));
    }
    let mut grid = 1_usize;
    let mut patch_volume = 1_usize;
    let mut tokens = 1_usize;
    for (axis, (&size, &step)) in spatial.iter().zip(patch).enumerate() {
        require_nonzero(size, "spatial")?;
        require_nonzero(step, "patch")?;
        if size % step != 0 {
            return Err(invalid(
                "spatial",
                &format!("axis {axis} must be divisible by patch"),
            ));
        }
        grid = area(grid, size, "spatial")?;
        patch_volume = area(patch_volume, step, "patch")?;
        tokens = area(tokens, size / step, "patch grid")?;
    }
    let mut strides = [0_usize; 3];
    let mut stride = 1_usize;
    for axis in (0..spatial.len()).rev() {
        strides[axis] = stride;
        stride = area(spatial[axis], stride, "spatial")?;
    }
    let latent_len = area(channels, grid, "latent")?;
    require_len(latent.len(), latent_len, "latent")?;
    let out_len = area(
        area(tokens, channels, "patch tokens")?,
        patch_volume,
        "patch tokens",
    )?;
    require_finite(latent, "latent")?;

    let mut out = Vec::with_capacity(out_len);
    for token in 0..tokens {
        let mut rest = token;
        let mut origin = [0_usize; 3];
        for axis in (0..spatial.len()).rev() {
            let count = spatial[axis] / patch[axis];
            origin[axis] = (rest % count) * patch[axis];
            rest /= count;
        }
        for channel in 0..channels {
            for flat in 0..patch_volume {
                let mut rest = flat;
                let mut index = channel * grid;
                for axis in (0..spatial.len()).rev() {
                    index += (origin[axis] + rest % patch[axis]) * strides[axis];
                    rest /= patch[axis];
                }
                out.push(latent[index]);
            }
        }
    }
    Ok(out)
}

/// `[tokens, patch_volume] * [patch_volume, W] + [W] -> [tokens, W]`.
pub fn patch_embed(
    tokens: &[f32],
    weight: &[f32],
    bias: &[f32],
    spec: &FlowDitSpec,
) -> Result<Vec<f32>, VkError> {
    spec.validate()?;
    let patch_volume = spec.patch_volume;
    let width = spec.width;
    if !tokens.len().is_multiple_of(patch_volume) {
        return Err(invalid(
            "tokens",
            "length must be a multiple of patch_volume",
        ));
    }
    let rows = tokens.len() / patch_volume;
    require_nonzero(rows, "tokens")?;
    require_len(weight.len(), area(patch_volume, width, "weight")?, "weight")?;
    require_len(bias.len(), width, "bias")?;
    require_finite(tokens, "tokens")?;
    require_finite(weight, "weight")?;
    require_finite(bias, "bias")?;

    let mut out = vec![0.0; area(rows, width, "output")?];
    matmul_rows(tokens, weight, &mut out, rows, patch_volume, width)?;
    for row in 0..rows {
        for col in 0..width {
            out[row * width + col] += bias[col];
        }
    }
    Ok(out)
}

/// One flow-DiT block: adaptive-norm modulation, QKV, per-head QK norm,
/// axis-wise `RoPE`, attention, gated residual, FFN and gated residual.
#[allow(clippy::too_many_lines)] // explicit shape validation before dispatch
pub fn dit_block(
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
    require_positions(positions, tokens)?;
    if let Some(mask) = mask {
        require_len(mask.len(), area(tokens, tokens, "mask")?, "mask")?;
        require_finite(mask, "mask")?;
    }
    let ada_cols = area(6, width, "ada")?;
    let ada_len = area(ada_cols, width, "ada")?;
    let qkv_len = area(area(3, width, "qkv")?, width, "qkv")?;
    let ffn_len = area(width, area(2, ffn_width, "ffn")?, "ffn")?;
    require_len(weights.ada.len(), ada_len, "ada")?;
    require_len(weights.qkv.len(), qkv_len, "qkv")?;
    require_len(weights.qk_gain.len(), head_dim, "qk_gain")?;
    require_len(weights.out.len(), area(width, width, "out")?, "out")?;
    require_len(weights.ffn.len(), ffn_len, "ffn")?;
    require_len(
        weights.ffn_down.len(),
        area(ffn_width, width, "ffn_down")?,
        "ffn_down",
    )?;
    require_finite(x, "x")?;
    require_finite(cond, "cond")?;
    for (field, values) in [
        ("ada", &weights.ada),
        ("qkv", &weights.qkv),
        ("qk_gain", &weights.qk_gain),
        ("out", &weights.out),
        ("ffn", &weights.ffn),
        ("ffn_down", &weights.ffn_down),
    ] {
        require_finite(values, field)?;
    }

    if weights.ada.iter().all(|value| *value == 0.0)
        && weights.out.iter().all(|value| *value == 0.0)
        && weights.ffn_down.iter().all(|value| *value == 0.0)
    {
        return Ok(x.to_vec());
    }

    let mut ada6 = vec![0.0; ada_cols];
    matmul_rows(cond, &weights.ada, &mut ada6, 1, width, ada_cols)?;
    let attn_shift = &ada6[0..width];
    let attn_scale = &ada6[width..2 * width];
    let attn_gate = &ada6[2 * width..3 * width];
    let ffn_shift = &ada6[3 * width..4 * width];
    let ffn_scale = &ada6[4 * width..5 * width];
    let ffn_gate = &ada6[5 * width..6 * width];

    let mut norm = vec![0.0; rows];
    rms_norm_rows(x, &mut norm, tokens, width, RMS_EPS)?;
    let mut modulated = vec![0.0; rows];
    modulation(&norm, attn_shift, attn_scale, &mut modulated, tokens, width)?;
    let per = area(tokens, width, "qkv")?;
    let mut qkv = vec![0.0; area(tokens, area(3, width, "qkv")?, "qkv output")?];
    matmul_rows(&modulated, &weights.qkv, &mut qkv, tokens, width, 3 * width)?;
    let (q, rest) = qkv.split_at(per);
    let (k, v) = rest.split_at(per);

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
    )
    .map_err(|error| invalid("qk_gain", &error.to_string()))?;
    qk_norm(
        k,
        &mut k_norm,
        tokens,
        heads,
        head_dim,
        &weights.qk_gain,
        RMS_EPS,
    )
    .map_err(|error| invalid("qk_gain", &error.to_string()))?;
    let mut q_rot = vec![0.0; per];
    let mut k_rot = vec![0.0; per];
    rope_axes(
        &q_norm, positions, &mut q_rot, tokens, heads, head_dim, spec.axes, ROPE_THETA,
    )?;
    rope_axes(
        &k_norm, positions, &mut k_rot, tokens, heads, head_dim, spec.axes, ROPE_THETA,
    )?;
    let mut attended = vec![0.0; rows];
    attention_rows(
        &q_rot,
        &k_rot,
        v,
        mask,
        &mut attended,
        tokens,
        heads,
        head_dim,
    )?;

    let mut projected = vec![0.0; rows];
    matmul_rows(
        &attended,
        &weights.out,
        &mut projected,
        tokens,
        width,
        width,
    )?;
    let mut after_attn = vec![0.0; rows];
    gate_residual(&projected, x, attn_gate, &mut after_attn, tokens, width)?;

    let mut ffn_norm = vec![0.0; rows];
    rms_norm_rows(&after_attn, &mut ffn_norm, tokens, width, RMS_EPS)?;
    let mut ffn_in = vec![0.0; rows];
    modulation(&ffn_norm, ffn_shift, ffn_scale, &mut ffn_in, tokens, width)?;
    let mut ffn_proj = vec![0.0; area(tokens, area(2, ffn_width, "ffn output")?, "ffn output")?];
    matmul_rows(
        &ffn_in,
        &weights.ffn,
        &mut ffn_proj,
        tokens,
        width,
        2 * ffn_width,
    )?;
    let mut activated = vec![0.0; area(tokens, ffn_width, "ffn activation")?];
    let stride = 2 * ffn_width;
    for token in 0..tokens {
        let start = token * stride;
        let (gate, up) = ffn_proj[start..start + stride].split_at(ffn_width);
        let row = token * ffn_width;
        swiglu(gate, up, &mut activated[row..row + ffn_width])
            .map_err(|error| invalid("ffn", &error.to_string()))?;
    }
    let mut down = vec![0.0; rows];
    matmul_rows(
        &activated,
        &weights.ffn_down,
        &mut down,
        tokens,
        ffn_width,
        width,
    )?;
    let mut out = vec![0.0; rows];
    gate_residual(&down, &after_attn, ffn_gate, &mut out, tokens, width)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 4;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const FFN: usize = 4;
    const TOKENS: usize = 2;

    fn spec() -> FlowDitSpec {
        FlowDitSpec {
            width: W,
            heads: HEADS,
            head_dim: HEAD_DIM,
            ffn_width: FFN,
            patch_volume: 2,
            latent_channels: 1,
            axes: PositionAxes::Image {
                height: 1,
                width: 2,
            },
        }
    }

    fn modulations() -> Vec<f32> {
        vec![
            0.1, -0.1, 0.05, -0.05, // attn_shift
            0.2, 0.0, -0.1, 0.1, // attn_scale
            0.5, 0.5, 0.5, 0.5, // attn_gate
            0.0, 0.0, 0.0, 0.0, // ffn_shift
            0.0, 0.0, 0.0, 0.0, // ffn_scale
            0.5, 0.5, 0.5, 0.5, // ffn_gate
        ]
    }

    fn weights() -> DitBlockWeights {
        let mut ada = vec![0.0; 6 * W * W];
        ada[0..6 * W].copy_from_slice(&modulations());
        DitBlockWeights {
            ada,
            qkv: (0..W)
                .flat_map(|row| {
                    (0..3 * W).map(move |col| ((row + 1) * (col % 3 + 1)) as f32 * 0.25)
                })
                .collect(),
            qk_gain: vec![1.0, 1.0],
            out: (0..W)
                .flat_map(|row| (0..W).map(move |col| ((row + col) % 3 + 1) as f32 * 0.1))
                .collect(),
            ffn: (0..W)
                .flat_map(|row| (0..2 * FFN).map(move |col| ((row + col) % 3 + 1) as f32 * 0.2))
                .collect(),
            ffn_down: (0..FFN)
                .flat_map(|row| (0..W).map(move |col| ((row + col) % 3 + 1) as f32 * 0.3))
                .collect(),
        }
    }

    fn input() -> Vec<f32> {
        vec![1.0, -2.0, 0.5, 3.0, 0.0, 1.5, -1.0, 2.0]
    }

    fn positions() -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]
    }

    fn causal_mask() -> Vec<f32> {
        vec![0.0, -1e9, 0.0, 0.0]
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4,
                "element {index}: {got} != {want}"
            );
        }
    }

    #[test]
    fn flow_dit_substrate_patch_flatten_2d_and_3d() {
        let latent: Vec<f32> = (1..=16).map(|value| value as f32).collect();
        let flat = patch_flatten(&latent, 2, &[2, 4], &[1, 2]).unwrap();
        assert_eq!(
            flat,
            vec![
                1.0, 2.0, 9.0, 10.0, 3.0, 4.0, 11.0, 12.0, 5.0, 6.0, 13.0, 14.0, 7.0, 8.0, 15.0,
                16.0
            ]
        );

        let video: Vec<f32> = (1..=8).map(|value| value as f32).collect();
        let flat = patch_flatten(&video, 1, &[2, 2, 2], &[2, 1, 1]).unwrap();
        assert_eq!(flat, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);

        let short = patch_flatten(&latent[..15], 2, &[2, 4], &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(short.contains("latent"), "{short}");
        let bad = patch_flatten(&latent, 2, &[3, 4], &[2, 2])
            .unwrap_err()
            .to_string();
        assert!(bad.contains("spatial"), "{bad}");
    }

    #[test]
    fn flow_dit_substrate_patch_embed_numeric() {
        let tokens = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, 0.0, 2.0, -1.0, 0.0, 1.0, -1.0, 3.0];
        let bias = [0.5, -0.5, 1.0, 0.0];
        let actual = patch_embed(&tokens, &weight, &bias, &spec()).unwrap();
        assert_eq!(actual, vec![1.5, 1.5, 1.0, 5.0, 3.5, 3.5, 3.0, 9.0]);
    }

    #[test]
    fn flow_dit_substrate_dit_block_two_token_numeric() {
        let actual = dit_block(
            &input(),
            &[1.0, 0.0, 0.0, 0.0],
            &positions(),
            &spec(),
            &weights(),
            Some(&causal_mask()),
        )
        .unwrap();
        assert_close(
            &actual,
            &[
                3.235337, 0.642991, 3.050372, 5.235337, 2.686627, 4.770841, 2.617105, 4.686627,
            ],
        );
    }

    #[test]
    fn flow_dit_substrate_dit_block_identity_controls() {
        let mut weights = weights();
        weights.ada.fill(0.0);
        weights.out.fill(0.0);
        weights.ffn_down.fill(0.0);
        let actual = dit_block(
            &input(),
            &[1.0, 2.0, 3.0, 4.0],
            &positions(),
            &spec(),
            &weights,
            None,
        )
        .unwrap();
        assert_eq!(actual, input());
    }

    #[test]
    fn flow_dit_substrate_dit_block_zero_position_equals_no_rope() {
        let weights = weights();
        let cond = [1.0, 0.0, 0.0, 0.0];
        let zero = [[0.0; 3]; TOKENS];
        let actual = dit_block(&input(), &cond, &zero, &spec(), &weights, None).unwrap();

        let mut ada6 = vec![0.0; 6 * W];
        matmul_rows(&cond, &weights.ada, &mut ada6, 1, W, 6 * W).unwrap();
        let rows = TOKENS * W;
        let mut norm = vec![0.0; rows];
        rms_norm_rows(&input(), &mut norm, TOKENS, W, RMS_EPS).unwrap();
        let mut modulated = vec![0.0; rows];
        modulation(
            &norm,
            &ada6[..W],
            &ada6[W..2 * W],
            &mut modulated,
            TOKENS,
            W,
        )
        .unwrap();
        let mut qkv = vec![0.0; TOKENS * 3 * W];
        matmul_rows(&modulated, &weights.qkv, &mut qkv, TOKENS, W, 3 * W).unwrap();
        let per = TOKENS * W;
        let (q, rest) = qkv.split_at(per);
        let (k, v) = rest.split_at(per);
        let mut q_norm = vec![0.0; per];
        let mut k_norm = vec![0.0; per];
        qk_norm(
            q,
            &mut q_norm,
            TOKENS,
            HEADS,
            HEAD_DIM,
            &weights.qk_gain,
            RMS_EPS,
        )
        .unwrap();
        qk_norm(
            k,
            &mut k_norm,
            TOKENS,
            HEADS,
            HEAD_DIM,
            &weights.qk_gain,
            RMS_EPS,
        )
        .unwrap();
        let mut attended = vec![0.0; rows];
        attention_rows(
            &q_norm,
            &k_norm,
            v,
            None,
            &mut attended,
            TOKENS,
            HEADS,
            HEAD_DIM,
        )
        .unwrap();
        let mut projected = vec![0.0; rows];
        matmul_rows(&attended, &weights.out, &mut projected, TOKENS, W, W).unwrap();
        let mut after_attn = vec![0.0; rows];
        gate_residual(
            &projected,
            &input(),
            &ada6[2 * W..3 * W],
            &mut after_attn,
            TOKENS,
            W,
        )
        .unwrap();
        let mut ffn_norm = vec![0.0; rows];
        rms_norm_rows(&after_attn, &mut ffn_norm, TOKENS, W, RMS_EPS).unwrap();
        let mut ffn_in = vec![0.0; rows];
        modulation(
            &ffn_norm,
            &ada6[3 * W..4 * W],
            &ada6[4 * W..5 * W],
            &mut ffn_in,
            TOKENS,
            W,
        )
        .unwrap();
        let mut ffn_proj = vec![0.0; TOKENS * 2 * FFN];
        matmul_rows(&ffn_in, &weights.ffn, &mut ffn_proj, TOKENS, W, 2 * FFN).unwrap();
        let mut activated = vec![0.0; TOKENS * FFN];
        for token in 0..TOKENS {
            let start = token * 2 * FFN;
            let (gate, up) = ffn_proj[start..start + 2 * FFN].split_at(FFN);
            swiglu(gate, up, &mut activated[token * FFN..(token + 1) * FFN]).unwrap();
        }
        let mut down = vec![0.0; rows];
        matmul_rows(&activated, &weights.ffn_down, &mut down, TOKENS, FFN, W).unwrap();
        let mut expected = vec![0.0; rows];
        gate_residual(
            &down,
            &after_attn,
            &ada6[5 * W..6 * W],
            &mut expected,
            TOKENS,
            W,
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn flow_dit_substrate_patch_size_one_round_trips_tokens() {
        let latent = [0.25_f32, -0.5];
        let flat = patch_flatten(&latent, 1, &[1, 2], &[1, 1]).unwrap();
        assert_eq!(flat, latent);
        let embed = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let projected = patch_embed(&flat, &embed, &[0.0; 4], &spec()).unwrap();
        let round_trip: Vec<f32> = projected
            .chunks(4)
            .flat_map(|row| [row[0], row[1]])
            .collect();
        assert_eq!(round_trip, latent);
    }

    #[test]
    fn flow_dit_substrate_zero_qk_yields_uniform_attention() {
        let zero = vec![0.0; TOKENS * W];
        let v = input();
        let mut attended = vec![0.0; TOKENS * W];
        attention_rows(
            &zero,
            &zero,
            &v,
            None,
            &mut attended,
            TOKENS,
            HEADS,
            HEAD_DIM,
        )
        .unwrap();
        for token in 0..TOKENS {
            for head in 0..HEADS {
                for dim in 0..HEAD_DIM {
                    let column = head * HEAD_DIM + dim;
                    let mean = (0..TOKENS)
                        .map(|source| v[source * W + column])
                        .sum::<f32>()
                        / TOKENS as f32;
                    let got = attended[token * W + column];
                    assert!(
                        (got - mean).abs() <= 1e-6,
                        "token {token} head {head} dim {dim}: {got} != {mean}"
                    );
                }
            }
        }
    }
    #[test]
    fn flow_dit_substrate_dit_block_rejects_wrong_mask_length_by_name() {
        let error = dit_block(
            &input(),
            &[1.0, 0.0, 0.0, 0.0],
            &positions(),
            &spec(),
            &weights(),
            Some(&causal_mask()[..3]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("mask"), "{error}");
    }
}
