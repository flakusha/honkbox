#![allow(clippy::cast_possible_truncation)]

use super::*;
use crate::{
    VkError,
    device::VulkanDevice,
    model::decoder_ops::{qk_norm, rms_norm, rope, short_conv, swiglu},
};

const TOL: f32 = 1e-4;

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "element {index}: got {got}, want {want}"
        );
    }
}

fn expect_error<T>(result: Result<T, VkError>, op: &str, field: &str) {
    match result {
        Err(VkError::Device(message)) => assert!(
            message.contains(op) && message.contains(field),
            "error {message:?} must name {op} and {field}"
        ),
        _ => panic!("expected {op} device error for {field}"),
    }
}

fn device() -> Option<Arc<VulkanDevice>> {
    match VulkanDevice::headless() {
        Ok(device) => Some(Arc::new(device)),
        Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => None,
        Err(error) => panic!("vulkan api failed: {error:?}"),
    }
}

fn reference_rms(values: &[f32], width: usize, gain: &[f32], eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for row in values.chunks_exact(width) {
        let sum_sq: f64 = row
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum();
        let rms = (sum_sq / width as f64 + f64::from(eps)).sqrt();
        for (index, &value) in row.iter().enumerate() {
            out.push((f64::from(value) / rms * f64::from(gain[index])) as f32);
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
        for index in 0..half {
            let frequency = theta.powf(-2.0 * index as f32 / rotary_dim as f32);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let x = f64::from(head[index]);
            let y = f64::from(head[half + index]);
            head[index] = (x * f64::from(cos) - y * f64::from(sin)) as f32;
            head[half + index] = (y * f64::from(cos) + x * f64::from(sin)) as f32;
        }
    }
    out
}

fn reference_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&gated, &value)| {
            let sigmoid = 1.0 / (1.0 + (-f64::from(gated)).exp());
            (f64::from(gated) * sigmoid * f64::from(value)) as f32
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
                let shift = kernel - 1;
                let mut sum = 0.0_f64;
                for k in 0..kernel {
                    let source = t + k;
                    if source >= shift {
                        sum += f64::from(weights[channel * kernel + k])
                            * f64::from(values[start + source - shift]);
                    }
                }
                *target = sum as f32;
            }
        }
    }
    out
}

#[test]
fn ash_decoder_ops_gpu_rms_cpu_reference() {
    let values = [
        1.0_f32, -2.0, 0.5, 3.0, 0.0, 4.0, -1.0, 2.0, 7.0, -3.0, 0.25, 1.5,
    ];
    let gain = [0.8_f32, 1.2, -0.5, 2.0];
    let mut out = vec![0.0; values.len()];
    rms_norm(&values, &mut out, 3, 4, &gain, 1e-5).expect("portable rms norm");
    assert_close(&out, &reference_rms(&values, 4, &gain, 1e-5));
}

#[test]
fn ash_decoder_ops_gpu_qk_cpu_reference() {
    let values: Vec<f32> = (0..24).map(|index| (index as f32 - 8.0) * 0.375).collect();
    let gain = [1.0_f32, 0.5, -1.5, 2.0];
    let mut out = vec![0.0; values.len()];
    qk_norm(&values, &mut out, 2, 3, 4, &gain, 1e-5).expect("portable qk norm");
    assert_close(&out, &reference_rms(&values, 4, &gain, 1e-5));
}

#[test]
fn ash_decoder_ops_gpu_rope_cpu_reference() {
    let values: Vec<f32> = (0..32).map(|index| (index as f32 + 1.0) * 0.25).collect();
    for rotary_dim in [2, 8] {
        for position in [0_u64, 1, 65_536] {
            let mut out = vec![0.0; values.len()];
            rope(&values, &mut out, 2, 2, 8, position, rotary_dim, 500.0).expect("portable rope");
            assert_close(
                &out,
                &reference_rope(&values, 8, position, rotary_dim, 500.0),
            );
            for (index, head) in out.chunks_exact(8).enumerate() {
                let source = &values[index * 8..index * 8 + 8];
                assert_eq!(
                    &head[rotary_dim..],
                    &source[rotary_dim..],
                    "non-rotary tail"
                );
            }
        }
    }
}

#[test]
fn ash_decoder_ops_gpu_swiglu_cpu_reference() {
    let gate = [1.5_f32, -2.0, 0.0, 8.0, -0.25];
    let up = [0.5_f32, 1.25, -3.0, 2.0, 4.0];
    let mut out = vec![0.0; gate.len()];
    swiglu(&gate, &up, &mut out).expect("portable swiglu");
    assert_close(&out, &reference_swiglu(&gate, &up));
}

#[test]
fn ash_decoder_ops_gpu_short_conv_cpu_reference() {
    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    for (time, kernel) in [(1, 5), (2, 7), (3, 5), (5, 1)] {
        let weights: Vec<f32> = (0..kernel)
            .map(|index| (index as f32 + 1.0) * 0.2)
            .collect();
        let mut out = vec![0.0; time];
        short_conv(&values[..time], &mut out, &weights, 1, 1, time, kernel)
            .expect("portable short conv");
        assert_close(
            &out,
            &reference_short_conv(&values[..time], 1, 1, time, &weights, kernel),
        );
    }
}

#[test]
fn ash_decoder_ops_gpu_rms_validation_is_named() {
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let gain = [1.0_f32, 1.0];
    expect_error(
        validate_rms(&values, 0, 2, &gain, 1e-5, "rms_norm"),
        "rms_norm",
        "rows",
    );
    expect_error(
        validate_rms(&values, 2, 0, &gain, 1e-5, "rms_norm"),
        "rms_norm",
        "cols",
    );
    expect_error(
        validate_rms(&values, 2, 4, &gain, 1e-5, "rms_norm"),
        "rms_norm",
        "values",
    );
    expect_error(
        validate_rms(&values, 2, 2, &gain[..1], 1e-5, "rms_norm"),
        "rms_norm",
        "gain",
    );
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        expect_error(
            validate_rms(&values, 2, 2, &gain, eps, "rms_norm"),
            "rms_norm",
            "eps",
        );
    }
    let mut nan = values;
    nan[0] = f32::NAN;
    expect_error(
        validate_rms(&nan, 2, 2, &gain, 1e-5, "rms_norm"),
        "rms_norm",
        "values",
    );
    let bad_gain = [1.0_f32, f32::INFINITY];
    expect_error(
        validate_rms(&values, 2, 2, &bad_gain, 1e-5, "rms_norm"),
        "rms_norm",
        "gain",
    );
    expect_error(
        validate_rms(&values, usize::MAX, 2, &gain, 1e-5, "rms_norm"),
        "rms_norm",
        "overflow",
    );
}

#[test]
fn ash_decoder_ops_gpu_qk_validation_is_named() {
    let values = [1.0_f32; 8];
    let gain = [1.0_f32, 1.0];
    expect_error(
        validate_qk(&values, 0, 2, 2, &gain, 1e-5),
        "qk_norm",
        "rows",
    );
    expect_error(
        validate_qk(&values, 2, 0, 2, &gain, 1e-5),
        "qk_norm",
        "heads",
    );
    expect_error(
        validate_qk(&values, 2, 2, 0, &gain, 1e-5),
        "qk_norm",
        "head_dim",
    );
    expect_error(
        validate_qk(&values, 2, 2, 3, &gain, 1e-5),
        "qk_norm",
        "values",
    );
    expect_error(
        validate_qk(&values, 2, 2, 2, &gain[..1], 1e-5),
        "qk_norm",
        "gain",
    );
    expect_error(validate_qk(&values, 2, 2, 2, &gain, -1.0), "qk_norm", "eps");
    expect_error(
        validate_qk(&values, 2, 2, 2, &gain, f32::NAN),
        "qk_norm",
        "eps",
    );
    let bad_gain = [1.0_f32, f32::INFINITY];
    expect_error(
        validate_qk(&values, 2, 2, 2, &bad_gain, 1e-5),
        "qk_norm",
        "gain",
    );
    expect_error(
        validate_qk(&values, 1, usize::MAX, 2, &gain, 1e-5),
        "qk_norm",
        "overflow",
    );
    expect_error(
        validate_qk(&values, usize::MAX, 2, 2, &gain, 1e-5),
        "qk_norm",
        "overflow",
    );
    let mut nan = values;
    nan[3] = f32::NAN;
    expect_error(validate_qk(&nan, 2, 2, 2, &gain, 1e-5), "qk_norm", "values");
}

#[test]
fn ash_decoder_ops_gpu_rope_validation_is_named() {
    let values = [1.0_f32; 8];
    for rotary_dim in [0, 3, 9] {
        expect_error(
            validate_rope(&values, 1, 1, 8, 0, rotary_dim, 100.0),
            "rope",
            "rotary_dim",
        );
    }
    expect_error(validate_rope(&values, 0, 1, 8, 0, 4, 100.0), "rope", "rows");
    expect_error(
        validate_rope(&values, 1, 0, 8, 0, 4, 100.0),
        "rope",
        "heads",
    );
    expect_error(
        validate_rope(&values, 1, 1, 0, 0, 0, 100.0),
        "rope",
        "head_dim",
    );
    expect_error(
        validate_rope(&values, 1, 1, 7, 0, 4, 100.0),
        "rope",
        "values",
    );
    expect_error(
        validate_rope(&values, 1, 1, 8, u64::MAX, 4, 100.0),
        "rope",
        "position",
    );
    for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        expect_error(
            validate_rope(&values, 1, 1, 8, 0, 4, theta),
            "rope",
            "theta",
        );
    }

    let mut nan = values;
    nan[2] = f32::NAN;
    expect_error(validate_rope(&nan, 1, 1, 8, 0, 4, 100.0), "rope", "values");
    expect_error(
        validate_rope(&values, usize::MAX, 1, 8, 0, 4, 100.0),
        "rope",
        "overflow",
    );
}

#[test]
fn ash_decoder_ops_gpu_swiglu_validation_is_named() {
    let gate = [1.0_f32, 2.0];
    let up = [3.0_f32, 4.0];
    expect_error(validate_swiglu(&[], &[]), "swiglu", "gate");
    expect_error(validate_swiglu(&gate, &up[..1]), "swiglu", "up");
    let mut nan = gate;
    nan[0] = f32::NAN;
    expect_error(validate_swiglu(&nan, &up), "swiglu", "gate");
    let inf = [1.0_f32, f32::INFINITY];
    expect_error(validate_swiglu(&gate, &inf), "swiglu", "up");
}

#[test]
fn ash_decoder_ops_gpu_short_conv_validation_is_named() {
    let values = [1.0_f32; 8];
    let weights = [0.5_f32; 4];
    expect_error(
        validate_short_conv(&values, &weights, 0, 2, 4, 2),
        "short_conv",
        "rows",
    );
    expect_error(
        validate_short_conv(&values, &weights, 1, 0, 4, 2),
        "short_conv",
        "channels",
    );
    expect_error(
        validate_short_conv(&values, &weights, 1, 2, 0, 2),
        "short_conv",
        "time",
    );
    expect_error(
        validate_short_conv(&values, &weights, 1, 2, 4, 0),
        "short_conv",
        "kernel",
    );
    expect_error(
        validate_short_conv(&values, &weights, 2, 2, 4, 2),
        "short_conv",
        "values",
    );
    expect_error(
        validate_short_conv(&values, &weights[..3], 1, 2, 4, 2),
        "short_conv",
        "weights",
    );
    expect_error(
        validate_short_conv(&values, &weights, 1, 1, usize::MAX, 2),
        "short_conv",
        "time + kernel",
    );
    expect_error(
        validate_short_conv(&values, &weights, 1, 1, u32::MAX as usize - 1, 2),
        "short_conv",
        "time + kernel",
    );
    expect_error(
        validate_short_conv(&values, &weights, usize::MAX, 2, 4, 2),
        "short_conv",
        "overflow",
    );
    let mut nan = values;
    nan[1] = f32::NAN;
    expect_error(
        validate_short_conv(&nan, &weights, 1, 2, 4, 2),
        "short_conv",
        "values",
    );
    let bad_weights = [0.5_f32, f32::NEG_INFINITY, 0.5, 0.5];
    expect_error(
        validate_short_conv(&values, &bad_weights, 1, 2, 4, 2),
        "short_conv",
        "weights",
    );
}

#[test]
fn ash_decoder_ops_gpu_rms_and_qk_dispatch_on_vulkan_device() {
    let Some(device) = device() else { return };
    let norm = RmsNorm::new(device).expect("rms pipeline");
    let values: Vec<f32> = (0..15).map(|index| (index as f32 - 6.0) * 0.5).collect();
    let gain = [0.8_f32, 1.2, -0.5, 2.0, 0.25];
    let actual = norm.run(&values, 3, 5, &gain, 1e-5).expect("rms dispatch");
    assert_close(&actual, &reference_rms(&values, 5, &gain, 1e-5));

    let qk: Vec<f32> = (0..24).map(|index| (index as f32 - 9.0) * 0.25).collect();
    let head_gain = [1.0_f32, 0.5, -1.5, 2.0];
    let actual = norm
        .qk_norm(&qk, 2, 3, 4, &head_gain, 1e-5)
        .expect("qk dispatch");
    assert_close(&actual, &reference_rms(&qk, 4, &head_gain, 1e-5));

    let mut scaled = vec![0.0_f32; 8];
    scaled[..4].copy_from_slice(&[1.0, -2.0, 0.5, 3.0]);
    scaled[4..].copy_from_slice(&[50.0, -100.0, 25.0, 150.0]);
    let unity = [1.0_f32; 4];
    let isolated = norm
        .qk_norm(&scaled, 1, 2, 4, &unity, 1e-5)
        .expect("isolated heads");
    assert_close(&isolated[..4], &isolated[4..]);
    scaled[5] = 7.0;
    let perturbed = norm
        .qk_norm(&scaled, 1, 2, 4, &unity, 1e-5)
        .expect("perturbed heads");
    assert_close(&perturbed[..4], &isolated[..4]);
    assert_ne!(perturbed[4..], isolated[4..]);
}

#[test]
fn ash_decoder_ops_gpu_rope_dispatch_on_vulkan_device() {
    let Some(device) = device() else { return };
    let rotary = Rope::new(device).expect("rope pipeline");
    let values: Vec<f32> = (0..48).map(|index| (index as f32 + 1.0) * 0.125).collect();
    for rotary_dim in [2, 8] {
        for position in [0_u64, 1, 1_000_000] {
            let actual = rotary
                .run(&values, 2, 3, 8, position, rotary_dim, 10_000.0)
                .expect("rope dispatch");
            assert_close(
                &actual,
                &reference_rope(&values, 8, position, rotary_dim, 10_000.0),
            );
            for (index, head) in actual.chunks_exact(8).enumerate() {
                let source = &values[index * 8..index * 8 + 8];
                assert_eq!(&head[rotary_dim..], &source[rotary_dim..], "tail changed");
            }
        }
    }
}

#[test]
fn ash_decoder_ops_gpu_swiglu_dispatch_on_vulkan_device() {
    let Some(device) = device() else { return };
    let gated = SwiGlu::new(device).expect("swiglu pipeline");
    let gate: Vec<f32> = (0..129)
        .map(|index| (index as f32 % 17.0 - 8.0) * 0.5)
        .collect();
    let up: Vec<f32> = (0..129)
        .map(|index| (index as f32 % 11.0 - 5.0) * 0.25)
        .collect();
    let actual = gated.run(&gate, &up).expect("swiglu dispatch");
    assert_close(&actual, &reference_swiglu(&gate, &up));
}

#[test]
fn ash_decoder_ops_gpu_short_conv_dispatch_on_vulkan_device() {
    let Some(device) = device() else { return };
    let convolution = ShortConv::new(device).expect("short conv pipeline");
    for (rows, channels, time, kernel) in [
        (1, 1, 1, 5),
        (2, 2, 2, 7),
        (2, 3, 3, 5),
        (2, 2, 5, 3),
        (2, 5, 7, 3),
    ] {
        let values: Vec<f32> = (0..rows * channels * time)
            .map(|index| (index as f32 - 7.0) * 0.375)
            .collect();
        let weights: Vec<f32> = (0..channels * kernel)
            .map(|index| (index as f32 + 1.0) * 0.0625)
            .collect();
        let actual = convolution
            .run(&values, &weights, rows, channels, time, kernel)
            .expect("short conv dispatch");
        assert_close(
            &actual,
            &reference_short_conv(&values, rows, channels, time, &weights, kernel),
        );
    }
}
