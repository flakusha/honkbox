//! CPU/GPU parity for one synthetic flow-DiT block. No model files are read.

#![cfg(feature = "ash-sd")]
#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]

use honkbox_ash_backend::VkError;
use honkbox_ash_backend::device::VulkanDevice;
use honkbox_ash_backend::dit::gpu::DitGpu;
use honkbox_ash_backend::dit::{DitBlockWeights, FlowDitSpec, PositionAxes, dit_block};
use std::sync::Arc;

const TOL: f32 = 1e-4;

/// Stable lock path: a file is never removed, so no two inodes can be
/// locked concurrently by sibling sessions. The guard unlocks on drop.
struct DeviceLease(std::fs::File);

impl DeviceLease {
    fn acquire() -> Self {
        let path = std::env::temp_dir().join("honkbox-ash-flow-dit-device.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)
            .expect("device lease file");
        file.lock().expect("exclusive device lease");
        Self(file)
    }
}

impl Drop for DeviceLease {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn device() -> Option<Arc<VulkanDevice>> {
    match VulkanDevice::headless() {
        Ok(device) => Some(Arc::new(device)),
        Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
            if std::env::var("HONKBOX_ASH_REQUIRE_VULKAN").as_deref() == Ok("1") {
                panic!("HONKBOX_ASH_REQUIRE_VULKAN=1 but no usable Vulkan device");
            }
            eprintln!("skipping flow-DiT parity: no usable Vulkan device");
            None
        }
        Err(error) => panic!("vulkan initialization failed: {error}"),
    }
}

fn spec() -> FlowDitSpec {
    FlowDitSpec {
        width: 4,
        heads: 2,
        head_dim: 2,
        ffn_width: 4,
        patch_volume: 2,
        latent_channels: 1,
        axes: PositionAxes::Image {
            height: 1,
            width: 2,
        },
    }
}

/// Exact f32 fractions: no rounding drift between host and GPU arithmetic.
fn synth(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let step = (i * 37 + seed * 17) % 33;
            (step as i32 - 16) as f32 / 64.0
        })
        .collect()
}

struct Fixture {
    spec: FlowDitSpec,
    x: Vec<f32>,
    cond: Vec<f32>,
    positions: Vec<[f32; 3]>,
    weights: DitBlockWeights,
    mask: Vec<f32>,
}

fn fixture() -> Fixture {
    let spec = spec();
    let width = spec.width;
    let ffn = spec.ffn_width;
    Fixture {
        spec,
        x: synth(2 * width, 7),
        cond: synth(width, 8),
        positions: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        weights: DitBlockWeights {
            ada: synth(width * 6 * width, 1),
            qkv: synth(width * 3 * width, 2),
            qk_gain: synth(spec.head_dim, 3),
            out: synth(width * width, 4),
            ffn: synth(width * 2 * ffn, 5),
            ffn_down: synth(ffn * width, 6),
        },
        // Zero biases still exercise the masked dispatch without depending on
        // additive-mask scaling order.
        mask: vec![0.0; 4],
    }
}

#[test]
fn flow_dit_substrate_parity() {
    let _lease = DeviceLease::acquire();
    let Some(device) = device() else {
        return;
    };
    eprintln!("flow-DiT parity device: {}", device.info.name);
    let fixture = fixture();
    let cpu = dit_block(
        &fixture.x,
        &fixture.cond,
        &fixture.positions,
        &fixture.spec,
        &fixture.weights,
        Some(&fixture.mask),
    )
    .expect("CPU block");
    let gpu = DitGpu::new(device)
        .expect("flow-DiT GPU pipelines")
        .block(
            &fixture.x,
            &fixture.cond,
            &fixture.positions,
            &fixture.spec,
            &fixture.weights,
            Some(&fixture.mask),
        )
        .expect("GPU block");
    assert_eq!(gpu.len(), cpu.len());
    for (i, (got, want)) in gpu.iter().zip(&cpu).enumerate() {
        assert!(
            (got - want).abs() <= TOL,
            "element {i}: GPU {got}, CPU {want}"
        );
    }
}

#[test]
fn flow_dit_substrate_parity_cpu_golden() {
    let fixture = fixture();
    let out = dit_block(
        &fixture.x,
        &fixture.cond,
        &fixture.positions,
        &fixture.spec,
        &fixture.weights,
        Some(&fixture.mask),
    )
    .expect("CPU block");
    assert!(out.iter().all(|value| value.is_finite()), "{out:?}");

    let width = fixture.spec.width;
    let identity = DitBlockWeights {
        ada: vec![0.0; width * 6 * width],
        qkv: fixture.weights.qkv.clone(),
        qk_gain: fixture.weights.qk_gain.clone(),
        out: vec![0.0; width * width],
        ffn: fixture.weights.ffn.clone(),
        ffn_down: vec![0.0; width * fixture.spec.ffn_width],
    };
    let preserved = dit_block(
        &fixture.x,
        &fixture.cond,
        &fixture.positions,
        &fixture.spec,
        &identity,
        Some(&fixture.mask),
    )
    .expect("CPU block with zero gates");
    assert_eq!(preserved, fixture.x);
}

#[test]
fn flow_dit_substrate_parity_rejects_bad_shapes() {
    let fixture = fixture();
    let error = dit_block(
        &fixture.x,
        &fixture.cond,
        &fixture.positions,
        &fixture.spec,
        &fixture.weights,
        Some(&fixture.mask[..3]),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("mask"), "{error}");

    let mut bad_weights = DitBlockWeights {
        ada: fixture.weights.ada.clone(),
        qkv: fixture.weights.qkv.clone(),
        qk_gain: fixture.weights.qk_gain.clone(),
        out: fixture.weights.out.clone(),
        ffn: fixture.weights.ffn.clone(),
        ffn_down: fixture.weights.ffn_down.clone(),
    };
    bad_weights.qkv.pop();
    let error = dit_block(
        &fixture.x,
        &fixture.cond,
        &fixture.positions,
        &fixture.spec,
        &bad_weights,
        Some(&fixture.mask),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("qkv"), "{error}");

    let Some(device) = device() else {
        return;
    };
    let gpu = DitGpu::new(device).expect("flow-DiT GPU pipelines");
    let error = gpu
        .block(
            &fixture.x,
            &fixture.cond,
            &fixture.positions,
            &fixture.spec,
            &fixture.weights,
            Some(&fixture.mask[..3]),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("mask"), "{error}");
    let error = gpu
        .block(
            &fixture.x,
            &fixture.cond,
            &fixture.positions,
            &fixture.spec,
            &bad_weights,
            Some(&fixture.mask),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("qkv"), "{error}");
}
