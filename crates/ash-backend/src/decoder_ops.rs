//! Portable f32 primitives for decoder-only transformer families.
//!
//! Every function is stateless: values, gains/weights and the output slice stay
//! caller-owned. No device, shader, checkpoint or cross-call state is touched,
//! so the Vulkan path in [`crate::decoder_ops_gpu`] stays optional. Shapes are
//! row-major, and all validation completes before any element is written.

// Shapes are bounded by caller slice lengths and test data is synthetic.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::VkError;

fn require_nonzero(value: usize, op: &str, field: &str) -> Result<(), VkError> {
    if value == 0 {
        Err(VkError::Device(format!("{op}: {field} must be non-zero")))
    } else {
        Ok(())
    }
}

fn require_len(actual: usize, expected: usize, op: &str, field: &str) -> Result<(), VkError> {
    if actual == expected {
        Ok(())
    } else {
        Err(VkError::Device(format!(
            "{op}: {field} length {actual} must be {expected}"
        )))
    }
}

fn require_finite(values: &[f32], op: &str, field: &str) -> Result<(), VkError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(VkError::Device(format!("{op}: {field} must be finite")))
    }
}

fn require_positive_eps(eps: f32, op: &str) -> Result<(), VkError> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(VkError::Device(format!(
            "{op}: eps must be finite and positive"
        )))
    }
}

fn checked_area(a: usize, b: usize, op: &str, field: &str) -> Result<usize, VkError> {
    a.checked_mul(b)
        .ok_or_else(|| VkError::Device(format!("{op}: {field} element count overflow")))
}

fn rms_rows(values: &[f32], out: &mut [f32], width: usize, gain: &[f32], eps: f32) {
    for (row, dst) in values.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let sum_sq: f32 = row.iter().map(|value| value * value).sum();
        let inv = (sum_sq / width as f32 + eps).sqrt().recip();
        for (target, (&source, &scale)) in dst.iter_mut().zip(row.iter().zip(gain)) {
            *target = source * inv * scale;
        }
    }
}

/// `out[r, c] = values[r, c] * rsqrt(mean_c(values[r, c]^2) + eps) * gain[c]`
/// for a `[rows, cols]` row-major input.
pub fn rms_norm(
    values: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gain: &[f32],
    eps: f32,
) -> Result<(), VkError> {
    require_nonzero(rows, "rms_norm", "rows")?;
    require_nonzero(cols, "rms_norm", "cols")?;
    let len = checked_area(rows, cols, "rms_norm", "rows * cols")?;
    require_len(values.len(), len, "rms_norm", "values")?;
    require_len(out.len(), len, "rms_norm", "output")?;
    require_len(gain.len(), cols, "rms_norm", "gain")?;
    require_positive_eps(eps, "rms_norm")?;
    require_finite(values, "rms_norm", "values")?;
    require_finite(gain, "rms_norm", "gain")?;
    rms_rows(values, out, cols, gain, eps);
    Ok(())
}

/// Apply [`rms_norm`] independently to each head of `[rows, heads, head_dim]`;
/// `gain` has `head_dim` entries and no value crosses a head boundary.
#[allow(clippy::too_many_arguments)] // full shape plus caller-owned gain/output
pub fn qk_norm(
    values: &[f32],
    out: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    gain: &[f32],
    eps: f32,
) -> Result<(), VkError> {
    require_nonzero(rows, "qk_norm", "rows")?;
    require_nonzero(heads, "qk_norm", "heads")?;
    require_nonzero(head_dim, "qk_norm", "head_dim")?;
    let per_row = checked_area(heads, head_dim, "qk_norm", "heads * head_dim")?;
    let len = checked_area(rows, per_row, "qk_norm", "rows * heads * head_dim")?;
    require_len(values.len(), len, "qk_norm", "values")?;
    require_len(out.len(), len, "qk_norm", "output")?;
    require_len(gain.len(), head_dim, "qk_norm", "gain")?;
    require_positive_eps(eps, "qk_norm")?;
    require_finite(values, "qk_norm", "values")?;
    require_finite(gain, "qk_norm", "gain")?;
    rms_rows(values, out, head_dim, gain, eps);
    Ok(())
}

/// Half-split `RoPE` over `[rows, heads, head_dim]`. Pairs `(i, i + half)` are
/// rotated by `position / theta^(2i / rotary_dim)`; `[rotary_dim, head_dim)`
/// is copied unchanged. `position` is a scalar offset for the whole tensor.
#[allow(clippy::too_many_arguments)] // full shape plus rotary parameters
pub fn rope(
    values: &[f32],
    out: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    position: u64,
    rotary_dim: usize,
    theta: f32,
) -> Result<(), VkError> {
    require_nonzero(rows, "rope", "rows")?;
    require_nonzero(heads, "rope", "heads")?;
    require_nonzero(head_dim, "rope", "head_dim")?;
    let per_row = checked_area(heads, head_dim, "rope", "heads * head_dim")?;
    let len = checked_area(rows, per_row, "rope", "rows * heads * head_dim")?;
    require_len(values.len(), len, "rope", "values")?;
    require_len(out.len(), len, "rope", "output")?;
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err(VkError::Device(
            "rope: rotary_dim must be even and non-zero".into(),
        ));
    }
    if rotary_dim > head_dim {
        return Err(VkError::Device(
            "rope: rotary_dim must not exceed head_dim".into(),
        ));
    }
    let position = u32::try_from(position)
        .map_err(|_| VkError::Device("rope: position exceeds u32 range".into()))?;
    if !theta.is_finite() || theta <= 0.0 {
        return Err(VkError::Device(
            "rope: theta must be finite and positive".into(),
        ));
    }
    require_finite(values, "rope", "values")?;

    let half = rotary_dim / 2;
    let frequencies: Vec<f32> = (0..half)
        .map(|i| theta.powf(-2.0 * i as f32 / rotary_dim as f32))
        .collect();
    require_finite(&frequencies, "rope", "rotary frequencies")?;
    // u32 -> f32 never overflows (max ~4.3e9 is finite in f32); it only loses
    // integer precision above 2^24, which the angle math tolerates.
    let position = position as f32;

    for (source, target) in values
        .chunks_exact(head_dim)
        .zip(out.chunks_exact_mut(head_dim))
    {
        target[rotary_dim..].copy_from_slice(&source[rotary_dim..]);
        for (i, &frequency) in frequencies.iter().enumerate() {
            let (sin, cos) = (position * frequency).sin_cos();
            let x = source[i];
            let y = source[half + i];
            target[i] = x * cos - y * sin;
            target[half + i] = y * cos + x * sin;
        }
    }
    Ok(())
}

/// Elementwise `out[i] = silu(gate[i]) * up[i]` for equal-length vectors.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), VkError> {
    require_nonzero(gate.len(), "swiglu", "gate")?;
    require_len(up.len(), gate.len(), "swiglu", "up")?;
    require_len(out.len(), gate.len(), "swiglu", "output")?;
    require_finite(gate, "swiglu", "gate")?;
    require_finite(up, "swiglu", "up")?;
    for (target, (&input, &activation)) in out.iter_mut().zip(gate.iter().zip(up)) {
        *target = (input / (1.0 + (-input).exp())) * activation;
    }
    Ok(())
}

/// One-shot depthwise causal convolution for `[rows, channels, time]`.
/// Weights are row-major `[channels, kernel]`, stride/dilation are 1 and the
/// left padding is `kernel - 1`, so output shape equals input shape. No
/// rolling state is retained between calls.
#[allow(clippy::too_many_arguments)] // full shape plus caller-owned weights/output
pub fn short_conv(
    values: &[f32],
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    channels: usize,
    time: usize,
    kernel: usize,
) -> Result<(), VkError> {
    require_nonzero(rows, "short_conv", "rows")?;
    require_nonzero(channels, "short_conv", "channels")?;
    require_nonzero(time, "short_conv", "time")?;
    require_nonzero(kernel, "short_conv", "kernel")?;
    let per_row = checked_area(channels, time, "short_conv", "channels * time")?;
    let len = checked_area(rows, per_row, "short_conv", "rows * channels * time")?;
    let weight_len = checked_area(channels, kernel, "short_conv", "channels * kernel")?;
    time.checked_add(kernel)
        .ok_or_else(|| VkError::Device("short_conv: time + kernel index overflow".into()))?;
    require_len(values.len(), len, "short_conv", "values")?;
    require_len(out.len(), len, "short_conv", "output")?;
    require_len(weights.len(), weight_len, "short_conv", "weights")?;
    require_finite(values, "short_conv", "values")?;
    require_finite(weights, "short_conv", "weights")?;

    let shift = kernel - 1;
    for row in 0..rows {
        for channel in 0..channels {
            let base = row * per_row + channel * time;
            let taps = &weights[channel * kernel..(channel + 1) * kernel];
            for (t, target) in out[base..base + time].iter_mut().enumerate() {
                let mut sum = 0.0_f32;
                for (k, &weight) in taps.iter().enumerate() {
                    let index = t + k;
                    if index >= shift {
                        sum += weight * values[base + index - shift];
                    }
                }
                *target = sum;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Reference helpers round-trip through f64 and narrow back to f32.
    #![allow(clippy::cast_possible_truncation)]

    use super::*;

    const TOL: f32 = 1e-4;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= TOL,
                "element {i}: got {got}, want {want}"
            );
        }
    }

    fn expect_error(result: Result<(), VkError>, op: &str, field: &str) {
        match result {
            Err(VkError::Device(message)) => assert!(
                message.contains(op) && message.contains(field),
                "error {message:?} must name {op} and {field}"
            ),
            other => panic!("expected {op} device error for {field}, got {other:?}"),
        }
    }

    fn reference_rms(values: &[f32], width: usize, gain: &[f32], eps: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(values.len());
        for row in values.chunks_exact(width) {
            let sum: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let rms = (sum / width as f64 + f64::from(eps)).sqrt();
            for (i, &value) in row.iter().enumerate() {
                out.push((f64::from(value) / rms * f64::from(gain[i])) as f32);
            }
        }
        out
    }

    fn reference_rope(
        values: &[f32],
        head_dim: usize,
        position: u64,
        rotary_dim: usize,
        theta: f32,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut out = values.to_vec();
        for head in out.chunks_exact_mut(head_dim) {
            for i in 0..half {
                // Match the kernel's frequency rounding: at large positions a
                // 1-ulp difference here amplifies into a large angle error.
                let frequency = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                let angle = position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let x = f64::from(head[i]);
                let y = f64::from(head[half + i]);
                head[i] = (x * f64::from(cos) - y * f64::from(sin)) as f32;
                head[half + i] = (y * f64::from(cos) + x * f64::from(sin)) as f32;
            }
        }
        out
    }

    fn reference_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(&g, &u)| {
                let sigmoid = 1.0 / (1.0 + (-f64::from(g)).exp());
                (f64::from(g) * sigmoid * f64::from(u)) as f32
            })
            .collect()
    }

    fn reference_short_conv(
        values: &[f32],
        rows: usize,
        channels: usize,
        time: usize,
        weights: &[f32],
        kernel: usize,
    ) -> Vec<f32> {
        let row_stride = channels * time;
        let mut out = vec![0.0_f32; values.len()];
        for row in 0..rows {
            for channel in 0..channels {
                let start = row * row_stride + channel * time;
                for (t, target) in out[start..start + time].iter_mut().enumerate() {
                    let mut sum = 0.0_f64;
                    for k in 0..kernel {
                        let source = t as isize + k as isize - kernel as isize + 1;
                        if source >= 0 {
                            sum += f64::from(weights[channel * kernel + k])
                                * f64::from(values[start + source as usize]);
                        }
                    }
                    *target = sum as f32;
                }
            }
        }
        out
    }

    #[test]
    fn ash_decoder_ops_rms_norm_matches_scalar_reference() {
        let cols = 4;
        let values = [
            1.0_f32, -2.0, 0.5, 3.0, 0.0, 4.0, -1.0, 2.0, 7.0, -3.0, 0.25, 1.5,
        ];
        let gain = [0.8_f32, 1.2, -0.5, 2.0];
        let eps = 1e-5;
        let mut out = vec![0.0; values.len()];
        rms_norm(&values, &mut out, 3, cols, &gain, eps).expect("multiple rows");
        assert_close(&out, &reference_rms(&values, cols, &gain, eps));

        let mut single = vec![0.0; cols];
        rms_norm(&values[..cols], &mut single, 1, cols, &gain, eps).expect("single row");
        assert_close(&single, &reference_rms(&values[..cols], cols, &gain, eps));
    }

    #[test]
    fn ash_decoder_ops_rms_norm_rejects_invalid_inputs() {
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let gain = [1.0_f32, 1.0];
        let mut out = [0.0_f32; 4];
        expect_error(
            rms_norm(&values, &mut out, 0, 2, &gain, 1e-5),
            "rms_norm",
            "rows",
        );
        expect_error(
            rms_norm(&values, &mut out, 2, 0, &gain, 1e-5),
            "rms_norm",
            "cols",
        );
        expect_error(
            rms_norm(&values, &mut out, 2, 4, &gain, 1e-5),
            "rms_norm",
            "values",
        );
        expect_error(
            rms_norm(&values, &mut out[..3], 2, 2, &gain, 1e-5),
            "rms_norm",
            "output",
        );
        expect_error(
            rms_norm(&values, &mut out, 2, 2, &gain[..1], 1e-5),
            "rms_norm",
            "gain",
        );
        for eps in [0.0, -1e-5, f32::NAN, f32::INFINITY] {
            expect_error(
                rms_norm(&values, &mut out, 2, 2, &gain, eps),
                "rms_norm",
                "eps",
            );
        }
        let mut nan = values;
        nan[1] = f32::NAN;
        expect_error(
            rms_norm(&nan, &mut out, 2, 2, &gain, 1e-5),
            "rms_norm",
            "values",
        );
        let mut bad_gain = gain;
        bad_gain[0] = f32::INFINITY;
        expect_error(
            rms_norm(&values, &mut out, 2, 2, &bad_gain, 1e-5),
            "rms_norm",
            "gain",
        );
        expect_error(
            rms_norm(&values, &mut out, usize::MAX, 2, &gain, 1e-5),
            "rms_norm",
            "overflow",
        );
    }

    #[test]
    fn ash_decoder_ops_qk_norm_matches_scalar_reference() {
        let (rows, heads, head_dim) = (2, 3, 4);
        let values: Vec<f32> = (0..rows * heads * head_dim)
            .map(|i| (i as f32 - 5.0) * 0.75)
            .collect();
        let gain = [1.0_f32, 0.5, -1.5, 2.0];
        let eps = 1e-5;
        let mut out = vec![0.0; values.len()];
        qk_norm(&values, &mut out, rows, heads, head_dim, &gain, eps).expect("qk norm");
        assert_close(&out, &reference_rms(&values, head_dim, &gain, eps));

        let mut single = vec![0.0; head_dim];
        qk_norm(&values[..head_dim], &mut single, 1, 1, head_dim, &gain, eps).expect("one head");
        assert_close(
            &single,
            &reference_rms(&values[..head_dim], head_dim, &gain, eps),
        );
    }

    #[test]
    fn ash_decoder_ops_qk_norm_rejects_invalid_inputs() {
        let values = [1.0_f32; 8];
        let gain = [1.0_f32, 1.0];
        let mut out = [0.0_f32; 8];
        expect_error(
            qk_norm(&values, &mut out, 0, 2, 2, &gain, 1e-5),
            "qk_norm",
            "rows",
        );
        expect_error(
            qk_norm(&values, &mut out, 2, 0, 2, &gain, 1e-5),
            "qk_norm",
            "heads",
        );
        expect_error(
            qk_norm(&values, &mut out, 2, 2, 0, &gain, 1e-5),
            "qk_norm",
            "head_dim",
        );
        expect_error(
            qk_norm(&values, &mut out, 2, 2, 3, &gain, 1e-5),
            "qk_norm",
            "values",
        );
        expect_error(
            qk_norm(&values, &mut out[..7], 2, 2, 2, &gain, 1e-5),
            "qk_norm",
            "output",
        );
        expect_error(
            qk_norm(&values, &mut out, 2, 2, 2, &gain[..1], 1e-5),
            "qk_norm",
            "gain",
        );
        expect_error(
            qk_norm(&values, &mut out, 2, 2, 2, &gain, -1.0),
            "qk_norm",
            "eps",
        );
        let mut nan = values;
        nan[3] = f32::NAN;
        expect_error(
            qk_norm(&nan, &mut out, 2, 2, 2, &gain, 1e-5),
            "qk_norm",
            "values",
        );
        expect_error(
            qk_norm(&values, &mut out, usize::MAX, 2, 2, &gain, 1e-5),
            "qk_norm",
            "overflow",
        );
    }

    #[test]
    fn ash_decoder_ops_qk_norm_isolates_heads() {
        let head_dim = 4;
        let gain = [1.0_f32; 4];
        let mut values = vec![0.0_f32; 2 * head_dim];
        values[0] = 1.0;
        values[head_dim] = 50.0;
        let mut out = vec![0.0_f32; values.len()];
        qk_norm(&values, &mut out, 1, 2, head_dim, &gain, 1e-5).expect("two heads");
        // RMSNorm is scale-invariant: two differently scaled heads normalize
        // to the same vector only if each head is normalized on its own.
        assert_close(&out[..head_dim], &out[head_dim..]);

        let mut perturbed = values.clone();
        perturbed[head_dim + 1] = 7.0;
        let mut second = vec![0.0_f32; values.len()];
        qk_norm(&perturbed, &mut second, 1, 2, head_dim, &gain, 1e-5).expect("perturbed head");
        assert_eq!(&second[..head_dim], &out[..head_dim]);
        assert_ne!(&second[head_dim..], &out[head_dim..]);
    }

    #[test]
    fn ash_decoder_ops_rope_matches_scalar_reference() {
        let (rows, heads, head_dim, rotary_dim) = (2, 2, 8, 4);
        let values: Vec<f32> = (0..rows * heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.25)
            .collect();
        for position in [0_u64, 1, 1_000_000] {
            let mut out = vec![0.0; values.len()];
            rope(
                &values, &mut out, rows, heads, head_dim, position, rotary_dim, 10_000.0,
            )
            .expect("rope");
            assert_close(
                &out,
                &reference_rope(&values, head_dim, position, rotary_dim, 10_000.0),
            );
        }

        let mut single = vec![0.0; head_dim];
        rope(
            &values[..head_dim],
            &mut single,
            1,
            1,
            head_dim,
            1,
            rotary_dim,
            10_000.0,
        )
        .expect("single head");
        assert_close(
            &single,
            &reference_rope(&values[..head_dim], head_dim, 1, rotary_dim, 10_000.0),
        );
    }

    #[test]
    fn ash_decoder_ops_rope_handles_tails_and_positions() {
        let head_dim = 8;
        let values: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();
        for rotary_dim in [2, head_dim] {
            for position in [0_u64, 1, 65_536] {
                let mut out = vec![0.0; head_dim];
                rope(
                    &values, &mut out, 1, 1, head_dim, position, rotary_dim, 500.0,
                )
                .expect("rope");
                assert_close(
                    &out,
                    &reference_rope(&values, head_dim, position, rotary_dim, 500.0),
                );
                assert_eq!(&out[rotary_dim..], &values[rotary_dim..], "non-rotary tail");
            }
        }

        // theta = 1, rotary_dim = 2, position = 1 gives an exact 1 rad rotation.
        let pair = [1.0_f32, 2.0];
        let mut rotated = [0.0_f32; 2];
        rope(&pair, &mut rotated, 1, 1, 2, 1, 2, 1.0).expect("unit theta");
        let (sin, cos) = 1.0_f32.sin_cos();
        assert_close(&rotated, &[cos - 2.0 * sin, 2.0 * cos + sin]);
    }

    #[test]
    fn ash_decoder_ops_rope_rejects_invalid_inputs() {
        let values = [1.0_f32; 8];
        let mut out = [0.0_f32; 8];
        for rotary_dim in [0, 3, 9] {
            expect_error(
                rope(&values, &mut out, 1, 1, 8, 0, rotary_dim, 100.0),
                "rope",
                "rotary_dim",
            );
        }
        expect_error(
            rope(&values, &mut out, 0, 1, 8, 0, 4, 100.0),
            "rope",
            "rows",
        );
        expect_error(
            rope(&values, &mut out, 1, 0, 8, 0, 4, 100.0),
            "rope",
            "heads",
        );
        expect_error(
            rope(&values, &mut out, 1, 1, 0, 0, 0, 100.0),
            "rope",
            "head_dim",
        );
        expect_error(
            rope(&values, &mut out, 1, 1, 7, 0, 4, 100.0),
            "rope",
            "values",
        );
        expect_error(
            rope(&values, &mut out[..7], 1, 1, 8, 0, 4, 100.0),
            "rope",
            "output",
        );
        expect_error(
            rope(&values, &mut out, 1, 1, 8, u64::MAX, 4, 100.0),
            "rope",
            "position",
        );
        for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            expect_error(
                rope(&values, &mut out, 1, 1, 8, 0, 4, theta),
                "rope",
                "theta",
            );
        }
        let mut nan = values;
        nan[2] = f32::NAN;
        expect_error(rope(&nan, &mut out, 1, 1, 8, 0, 4, 100.0), "rope", "values");
        expect_error(
            rope(&values, &mut out, usize::MAX, 1, 8, 0, 4, 100.0),
            "rope",
            "overflow",
        );
    }

    #[test]
    fn ash_decoder_ops_swiglu_matches_scalar_reference() {
        let gate = [1.5_f32, -2.0, 0.0, 8.0, -0.25];
        let up = [0.5_f32, 1.25, -3.0, 2.0, 4.0];
        let mut out = vec![0.0; gate.len()];
        swiglu(&gate, &up, &mut out).expect("swiglu");
        assert_close(&out, &reference_swiglu(&gate, &up));

        let mut single = [0.0_f32];
        swiglu(&gate[..1], &up[..1], &mut single).expect("single value");
        assert_close(&single, &reference_swiglu(&gate[..1], &up[..1]));
    }

    #[test]
    fn ash_decoder_ops_swiglu_rejects_invalid_inputs() {
        let gate = [1.0_f32, 2.0];
        let up = [3.0_f32, 4.0];
        let mut out = [0.0_f32; 2];
        expect_error(swiglu(&[], &[], &mut []), "swiglu", "gate");
        expect_error(swiglu(&gate, &up[..1], &mut out), "swiglu", "up");
        expect_error(swiglu(&gate, &up, &mut out[..1]), "swiglu", "output");
        let mut nan = gate;
        nan[0] = f32::NAN;
        expect_error(swiglu(&nan, &up, &mut out), "swiglu", "gate");
        let mut inf = up;
        inf[1] = f32::NEG_INFINITY;
        expect_error(swiglu(&gate, &inf, &mut out), "swiglu", "up");
    }

    #[test]
    fn ash_decoder_ops_short_conv_matches_scalar_reference() {
        let (rows, channels, time, kernel) = (2, 2, 5, 3);
        let values: Vec<f32> = (0..rows * channels * time)
            .map(|i| (i as f32 - 4.0) * 0.5)
            .collect();
        let weights: Vec<f32> = (0..channels * kernel)
            .map(|i| (i as f32 + 1.0) * 0.25)
            .collect();
        let mut out = vec![0.0; values.len()];
        short_conv(&values, &mut out, &weights, rows, channels, time, kernel).expect("short conv");
        assert_close(
            &out,
            &reference_short_conv(&values, rows, channels, time, &weights, kernel),
        );
    }

    #[test]
    fn ash_decoder_ops_short_conv_handles_kernel_lengths() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let identity = [1.0_f32];
        let mut out = vec![0.0; values.len()];
        short_conv(&values, &mut out, &identity, 1, 1, 5, 1).expect("kernel 1");
        assert_close(&out, &reference_short_conv(&values, 1, 1, 5, &identity, 1));
        assert_eq!(out, values);

        for (time, kernel) in [(1, 5), (2, 7), (3, 5)] {
            let slice = &values[..time];
            let weights: Vec<f32> = (0..kernel).map(|i| (i as f32 + 1.0) * 0.2).collect();
            let mut result = vec![0.0; time];
            short_conv(slice, &mut result, &weights, 1, 1, time, kernel)
                .expect("kernel longer than sequence");
            assert_close(
                &result,
                &reference_short_conv(slice, 1, 1, time, &weights, kernel),
            );
        }
    }

    #[test]
    fn ash_decoder_ops_short_conv_rejects_invalid_inputs() {
        let values = [1.0_f32; 8];
        let weights = [0.5_f32; 4];
        let mut out = [0.0_f32; 8];
        expect_error(
            short_conv(&values, &mut out, &weights, 0, 2, 4, 2),
            "short_conv",
            "rows",
        );
        expect_error(
            short_conv(&values, &mut out, &weights, 1, 0, 4, 2),
            "short_conv",
            "channels",
        );
        expect_error(
            short_conv(&values, &mut out, &weights, 1, 2, 0, 2),
            "short_conv",
            "time",
        );
        expect_error(
            short_conv(&values, &mut out, &weights, 1, 2, 4, 0),
            "short_conv",
            "kernel",
        );
        expect_error(
            short_conv(&values, &mut out, &weights, 2, 2, 4, 2),
            "short_conv",
            "values",
        );
        expect_error(
            short_conv(&values, &mut out[..7], &weights, 1, 2, 4, 2),
            "short_conv",
            "output",
        );
        expect_error(
            short_conv(&values, &mut out, &weights[..3], 1, 2, 4, 2),
            "short_conv",
            "weights",
        );
        let mut nan = values;
        nan[1] = f32::NAN;
        expect_error(
            short_conv(&nan, &mut out, &weights, 1, 2, 4, 2),
            "short_conv",
            "values",
        );
        let mut bad_weights = weights;
        bad_weights[0] = f32::INFINITY;
        expect_error(
            short_conv(&values, &mut out, &bad_weights, 1, 2, 4, 2),
            "short_conv",
            "weights",
        );
        expect_error(
            short_conv(&values, &mut out, &weights, usize::MAX, 2, 4, 2),
            "short_conv",
            "overflow",
        );
    }
}
