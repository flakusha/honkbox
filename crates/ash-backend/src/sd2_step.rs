//! Tiny SD2-base one-step fixture: declared spec roles → `UnetPredictor` →
//! `DdimSampler`, each compared with an independent CPU reference.
//!
//! The safetensors header is generated in a unique temp file. No weights are
//! bundled, no checkpoint is downloaded, and no image-quality claim is made.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    cross_attention::CrossAttentionWeights,
    ddim_sampler::DdimSampler,
    device::VulkanDevice,
    model_spec::{SD2_BASE_REDUCED, SD2_BASE_TEXT_KEYS, tensor_keys, validate},
    sampler::DdimAlphas,
    timestep::TimestepWeights,
    unet_attention::{UnetAttentionShape, UnetAttentionWeights},
    unet_checkpoint::UnetInputKeys,
    unet_input::UnetInputShape,
    unet_predict::{UnetCrossConfig, UnetPredictConfig, UnetPredictor},
    unet_stack::UnetStackBlockSpec,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
    map: CheckpointMap,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn replace(
    tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    key: &str,
    shape: Vec<usize>,
    values: Vec<f32>,
) {
    tensors.retain(|(existing, _, _)| existing != key);
    tensors.push((key.to_owned(), shape, values));
}

fn fixture() -> Fixture {
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = tensor_keys(&SD2_BASE_REDUCED)
        .into_iter()
        .map(|(_, key)| (key, vec![1], vec![0.0_f32]))
        .collect();

    // Identity centre-tap convolution: each output channel copies its input.
    let mut kernel = vec![0.0_f32; 2 * 2 * 3 * 3];
    for channel in 0..2 {
        kernel[channel * 18 + channel * 9 + 4] = 1.0;
    }
    replace(
        &mut tensors,
        SD2_BASE_REDUCED.conv_in.weight,
        vec![2, 2, 3, 3],
        kernel,
    );
    replace(
        &mut tensors,
        SD2_BASE_REDUCED.conv_in.bias,
        vec![2],
        vec![0.0; 2],
    );
    replace(
        &mut tensors,
        SD2_BASE_REDUCED.norm_out.weight,
        vec![2],
        vec![1.0; 2],
    );
    replace(
        &mut tensors,
        SD2_BASE_REDUCED.norm_out.bias,
        vec![2],
        vec![0.0; 2],
    );
    replace(
        &mut tensors,
        SD2_BASE_TEXT_KEYS[1],
        vec![77, 1024],
        vec![0.0; 77 * 1024],
    );

    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (key, shape, values) in tensors {
        let start = payload.len();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            key,
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&header).expect("header");
    let mut bytes = u64::try_from(header_bytes.len())
        .expect("header length")
        .to_le_bytes()
        .to_vec();
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "honkbox-ash-sd2-step-{}-{}-{stamp}.safetensors",
        std::process::id(),
        TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).expect("write fixture");
    Fixture {
        map: CheckpointMap::read(&path).expect("map"),
        path,
    }
}

fn device() -> Option<Arc<VulkanDevice>> {
    match VulkanDevice::headless() {
        Ok(device) => Some(Arc::new(device)),
        Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
            eprintln!("skipping SD2 step fixture: no usable Vulkan environment");
            None
        }
        Err(error) => panic!("vulkan api failed: {error:?}"),
    }
}

fn group_norm(values: &[f32], eps: f32) -> Vec<f32> {
    let mean = values.iter().sum::<f32>() / 8.0;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / 8.0;
    let scale = 1.0 / (variance + eps).sqrt();
    values.iter().map(|value| (value - mean) * scale).collect()
}

fn cpu_predict(latent: &[f32], conditioning: &[f32]) -> Vec<f32> {
    let activated: Vec<f32> = group_norm(latent, 1e-5)
        .iter()
        .map(|value| value / (1.0 + (-value).exp()))
        .collect();
    group_norm(&activated, 1e-5)
        .iter()
        .enumerate()
        .map(|(index, value)| value + conditioning[index / 4 % 2])
        .collect()
}

fn run_fixture() {
    let Some(device) = device() else { return };
    let fixture = fixture();
    validate(&SD2_BASE_REDUCED, &fixture.map).expect("declared SD2 roles");

    let block = UnetStackBlockSpec {
        keys: UnetInputKeys {
            kernel: SD2_BASE_REDUCED.conv_in.weight,
            bias: SD2_BASE_REDUCED.conv_in.bias,
            gamma: SD2_BASE_REDUCED.norm_out.weight,
            beta: SD2_BASE_REDUCED.norm_out.bias,
        },
        shape: UnetInputShape {
            batch: 1,
            in_c: 2,
            out_c: 2,
            in_h: 2,
            in_w: 2,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
        },
    };
    let attention_shape = UnetAttentionShape {
        batch: 1,
        channels: 2,
        height: 2,
        width: 2,
        groups: 1,
    };
    let zero_projection = [0.0_f32; 4];
    let identity = [1.0_f32, 0.0, 0.0, 1.0];
    let wkv = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let w1 = [0.0_f32; 8];
    let w2 = [0.0_f32; 4];
    let gamma = [1.0_f32, 1.0];
    let beta = [0.0_f32, 0.0];
    let config = UnetPredictConfig {
        timestep: TimestepWeights { w1: &w1, w2: &w2 },
        timestep_dim: 4,
        timestep_hidden: 2,
        blocks: &[block],
        attention: UnetAttentionWeights {
            gamma: &gamma,
            beta: &beta,
            wq: &zero_projection,
            wk: &zero_projection,
            wv: &zero_projection,
            wo: &zero_projection,
        },
        attention_shape,
        cross: UnetCrossConfig {
            weights: CrossAttentionWeights {
                wq: &identity,
                wkv: &wkv,
                wo: &identity,
            },
            context_rows: 1,
            heads: 1,
        },
        eps: 1e-5,
    };
    let latent = [0.4_f32, -0.2, 0.7, 0.1, -0.3, 0.6, 0.2, -0.5];
    let conditioning = [0.3_f32, -0.2];
    let prediction = UnetPredictor::new(&device, 1)
        .expect("predictor")
        .predict(
            &latent,
            1,
            &conditioning,
            &fixture.map,
            &fixture.path,
            &config,
        )
        .expect("SD2 step prediction");
    let expected_prediction = cpu_predict(&latent, &conditioning);
    assert_eq!(prediction.len(), expected_prediction.len());
    for (got, want) in prediction.iter().zip(&expected_prediction) {
        assert!(
            (got - want).abs() < 1e-4,
            "predictor: got {got}, want {want}"
        );
    }

    let alphas = DdimAlphas::from_cumprod(0.1, 0.3).expect("test alphas");
    let expected_step: Vec<f32> = latent
        .iter()
        .zip(&prediction)
        .map(|(value, epsilon)| {
            alphas.sqrt_alpha_prev
                * ((value - alphas.sqrt_one_minus_alpha * epsilon) / alphas.sqrt_alpha)
                + alphas.sqrt_one_minus_alpha_prev * epsilon
        })
        .collect();
    let sampler = DdimSampler::new(Arc::clone(&device)).expect("sampler");
    let mut calls = Vec::new();
    let stepped = sampler
        .run(&latent, &[0.1, 0.3], |current, index| {
            calls.push(index);
            assert_eq!(current, latent);
            Ok(prediction.clone())
        })
        .expect("DDIM step");
    assert_eq!(calls, vec![0]);
    assert_eq!(stepped.len(), expected_step.len());
    for (got, want) in stepped.iter().zip(&expected_step) {
        assert!((got - want).abs() < 1e-4, "DDIM: got {got}, want {want}");
    }
}

#[test]
fn sd2_unet_predictor_step_matches_cpu_reference() {
    run_fixture();
}

#[test]
#[ignore = "exclusive Vulkan device; same fixture as the CPU-reference test"]
fn sd2_one_step_on_exclusive_device() {
    run_fixture();
}
