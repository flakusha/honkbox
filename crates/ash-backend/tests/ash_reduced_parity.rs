//! Tensor parity for the reduced SD 1.5 topology against independent CPU math.
//!
//! Every fixture tensor is synthesized by the test; no model assets are read.
//! Vulkan-less machines skip cleanly. The CPU references duplicate the shader
//! arithmetic rather than calling any Vulkan kernel.

//! Resource contract: each test owns a unique safetensors file (PID + atomic
//! counter) removed by `Fixture::drop`, plus its own headless Vulkan device;
//! tests share no mutable global state and are safe to run in parallel.

#![cfg(feature = "ash-sd")]
#![allow(
    unused_mut,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value,
    clippy::chunks_exact_to_as_chunks,
    clippy::too_many_arguments,
    clippy::too_many_lines
)] // independent CPU reference fixture mirrors production shapes and arithmetic

use honkbox_ash_backend::clip_checkpoint::{self, ClipGeometry, ClipTensors};
use honkbox_ash_backend::device::VulkanDevice;
use honkbox_ash_backend::text_encoder::ClipTextEncoder;
use honkbox_ash_backend::unet_real::{UnetReal, UnetRealWeights};
use honkbox_ash_backend::vae_real::{VaeReal, VaeRealWeights};
use honkbox_ash_backend::{VkError, checkpoint::CheckpointMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const TOLERANCE: f32 = 1e-3;
const GN_EPS: f32 = 1e-6;
const CLIP_EPS: f32 = 1e-5;
const UNET_ROOT: &str = "model.diffusion_model.";
const VAE_ROOT: &str = "first_stage_model.decoder.";

fn device() -> Option<Arc<VulkanDevice>> {
    match VulkanDevice::headless() {
        Ok(device) => Some(Arc::new(device)),
        Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
            eprintln!("skipping reduced-topology parity: no usable Vulkan device");
            None
        }
        Err(error) => panic!("vulkan initialization failed: {error}"),
    }
}

// ---------------------------------------------------------------------------
// Synthetic safetensors fixture
// ---------------------------------------------------------------------------

struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Fixture {
    fn path(&self) -> &Path {
        &self.0
    }
}

static FIXTURE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Small exact f32 pattern: integer multiples of 1/256 in [-8/256, 8/256].
fn synth(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let code = (index.wrapping_mul(37).wrapping_add(seed.wrapping_mul(101))) % 17;
            (code as f32 - 8.0) / 256.0
        })
        .collect()
}

fn synth_norm(len: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let raw = synth(len, seed);
    let gain = raw.iter().map(|value| 0.75 + *value * 2.0).collect();
    (gain, raw)
}

fn push_tensor(tensors: &mut Vec<TensorSpec>, name: String, shape: Vec<usize>, seed: usize) {
    let len = shape.iter().product();
    tensors.push(TensorSpec {
        name,
        shape,
        values: synth(len, seed),
    });
}

fn push_norm(tensors: &mut Vec<TensorSpec>, prefix: &str, channels: usize, seed: usize) {
    let (gain, bias) = synth_norm(channels, seed);
    tensors.push(TensorSpec {
        name: format!("{prefix}.weight"),
        shape: vec![channels],
        values: gain,
    });
    tensors.push(TensorSpec {
        name: format!("{prefix}.bias"),
        shape: vec![channels],
        values: bias,
    });
}

fn push_conv(
    tensors: &mut Vec<TensorSpec>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    kernel: usize,
    seed: usize,
) {
    push_tensor(
        tensors,
        format!("{prefix}.weight"),
        vec![out_c, in_c, kernel, kernel],
        seed,
    );
    push_tensor(tensors, format!("{prefix}.bias"), vec![out_c], seed + 1);
}

fn push_linear(
    tensors: &mut Vec<TensorSpec>,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
    seed: usize,
) {
    push_tensor(
        tensors,
        format!("{prefix}.weight"),
        vec![out_dim, in_dim],
        seed,
    );
    push_tensor(tensors, format!("{prefix}.bias"), vec![out_dim], seed + 1);
}

fn write_fixture(label: &str, tensors: Vec<TensorSpec>) -> Fixture {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for tensor in &tensors {
        let len =
            u64::try_from(tensor.values.len() * size_of::<f32>()).expect("tensor byte length");
        let start = payload.len();
        header.insert(
            tensor.name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, start + len as usize],
            }),
        );
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    let header_bytes = serde_json::to_vec(&header).expect("safetensors header");
    let mut bytes = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    bytes.extend_from_slice(
        &u64::try_from(header_bytes.len())
            .expect("header length")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&payload);

    let path = std::env::temp_dir().join(format!(
        "honkbox-ash-parity-{label}-{}-{}.safetensors",
        std::process::id(),
        FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).expect("write safetensors fixture");
    Fixture(path)
}

fn open_fixture(fixture: &Fixture) -> CheckpointMap {
    CheckpointMap::read(fixture.path()).expect("read safetensors header")
}

fn f32_tensor(map: &CheckpointMap, path: &Path, name: &str, shape: &[usize]) -> Vec<f32> {
    let meta = map
        .get(name)
        .unwrap_or_else(|| panic!("fixture tensor '{name}' is missing"));
    assert_eq!(meta.shape.as_slice(), shape, "fixture shape of '{name}'");
    let bytes = map.read_tensor(path, name).expect("read fixture tensor");
    let expected_bytes = shape.iter().product::<usize>() * size_of::<f32>();
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "fixture byte length of '{name}'"
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

fn assert_tensor(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: tensor length: gpu {}, cpu {}",
        actual.len(),
        expected.len()
    );
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let delta = (got - want).abs();
        assert!(
            delta <= TOLERANCE,
            "{label}: mismatch at [{index}]: gpu {got}, cpu {want}, |delta| {delta}"
        );
    }
}

// ---------------------------------------------------------------------------
// Independent CPU kernels (shader arithmetic, no Vulkan)
// ---------------------------------------------------------------------------

fn cpu_transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    out
}

fn cpu_matmul(lhs: &[f32], rhs: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
    assert_eq!(lhs.len(), rows * inner, "cpu matmul lhs length");
    assert_eq!(rhs.len(), inner * cols, "cpu matmul rhs length");
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0_f32;
            for k in 0..inner {
                sum += lhs[row * inner + k] * rhs[k * cols + col];
            }
            out[row * cols + col] = sum;
        }
    }
    out
}

fn cpu_add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    assert_eq!(lhs.len(), rhs.len(), "cpu add lengths");
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left + right)
        .collect()
}

fn cpu_broadcast(values: &[f32], bias: &[f32]) -> Vec<f32> {
    assert!(!bias.is_empty(), "cpu broadcast empty bias");
    assert_eq!(values.len() % bias.len(), 0, "cpu broadcast length");
    values
        .iter()
        .enumerate()
        .map(|(index, value)| value + bias[index % bias.len()])
        .collect()
}

fn cpu_silu(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| *value / (1.0 + (-*value).exp()))
        .collect()
}

fn cpu_quick_gelu(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| *value / (1.0 + (-1.702 * *value).exp()))
        .collect()
}

fn cpu_layer_norm(
    values: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert_eq!(values.len(), rows * cols, "cpu layer norm length");
    assert_eq!(gamma.len(), cols, "cpu layer norm gamma");
    assert_eq!(beta.len(), cols, "cpu layer norm beta");
    let mut out = Vec::with_capacity(values.len());
    for row in values.chunks(cols) {
        let count = cols as f32;
        let mean: f32 = row.iter().sum::<f32>() / count;
        let variance: f32 = row
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / count;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for (index, value) in row.iter().enumerate() {
            out.push((value - mean) * inv_std * gamma[index] + beta[index]);
        }
    }
    out
}

fn cpu_group_norm(
    values: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    groups: usize,
    eps: f32,
) -> Vec<f32> {
    let spatial = height * width;
    assert_eq!(values.len(), channels * spatial, "cpu group norm length");
    assert_eq!(gamma.len(), channels, "cpu group norm gamma");
    assert_eq!(beta.len(), channels, "cpu group norm beta");
    let channels_per_group = channels / groups;
    let group_elems = channels_per_group * spatial;
    let mut out = vec![0.0_f32; values.len()];
    for channel in 0..channels {
        let group = channel / channels_per_group;
        let first_channel = group * channels_per_group;
        let mut mean = 0.0_f32;
        for local_channel in 0..channels_per_group {
            let base = (first_channel + local_channel) * spatial;
            for position in 0..spatial {
                mean += values[base + position];
            }
        }
        mean /= group_elems as f32;
        let mut variance = 0.0_f32;
        for local_channel in 0..channels_per_group {
            let base = (first_channel + local_channel) * spatial;
            for position in 0..spatial {
                let delta = values[base + position] - mean;
                variance += delta * delta;
            }
        }
        variance /= group_elems as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for position in 0..spatial {
            let index = channel * spatial + position;
            out[index] = (values[index] - mean) * inv_std * gamma[channel] + beta[channel];
        }
    }
    out
}

#[derive(Clone, Copy)]
struct ConvGeometry {
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    kernel: usize,
    stride: usize,
}

fn cpu_conv(input: &[f32], weight: &[f32], bias: &[f32], shape: ConvGeometry) -> Vec<f32> {
    let padding = shape.kernel / 2;
    let out_h = (shape.in_h + 2 * padding - shape.kernel) / shape.stride + 1;
    let out_w = (shape.in_w + 2 * padding - shape.kernel) / shape.stride + 1;
    let plane = out_h * out_w;
    assert_eq!(
        input.len(),
        shape.in_c * shape.in_h * shape.in_w,
        "cpu conv input"
    );
    assert_eq!(
        weight.len(),
        shape.out_c * shape.in_c * shape.kernel * shape.kernel,
        "cpu conv weight"
    );
    assert_eq!(bias.len(), shape.out_c, "cpu conv bias");
    let mut out = vec![0.0_f32; shape.out_c * plane];
    for oc in 0..shape.out_c {
        for pos in 0..plane {
            let oh = pos / out_w;
            let ow = pos % out_w;
            let mut sum = bias[oc];
            let weight_base = oc * shape.in_c * shape.kernel * shape.kernel;
            for ic in 0..shape.in_c {
                for kh in 0..shape.kernel {
                    let ih = oh * shape.stride + kh;
                    if ih < padding || ih >= padding + shape.in_h {
                        continue;
                    }
                    for kw in 0..shape.kernel {
                        let iw = ow * shape.stride + kw;
                        if iw < padding || iw >= padding + shape.in_w {
                            continue;
                        }
                        let in_offset =
                            (ic * shape.in_h + ih - padding) * shape.in_w + iw - padding;
                        let weight_offset =
                            weight_base + (ic * shape.kernel + kh) * shape.kernel + kw;
                        sum += input[in_offset] * weight[weight_offset];
                    }
                }
            }
            out[oc * plane + pos] = sum;
        }
    }
    out
}

fn cpu_nearest_upsample(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    factor: usize,
) -> Vec<f32> {
    let out_h = height * factor;
    let out_w = width * factor;
    let mut out = vec![0.0_f32; channels * out_h * out_w];
    for channel in 0..channels {
        for y in 0..out_h {
            let source_y = y / factor;
            for x in 0..out_w {
                let source_x = x / factor;
                out[(channel * out_h + y) * out_w + x] =
                    input[(channel * height + source_y) * width + source_x];
            }
        }
    }
    out
}

/// `softmax(Q·Kᵀ/√d)·V` with token-major Q/V and K as `[context_rows, dim]`.
fn cpu_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_rows: usize,
    context_rows: usize,
    dim: usize,
    heads: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; query_rows * dim];
    for head in 0..heads {
        for row in 0..query_rows {
            let q_base = row * dim + head * head_dim;
            let scores: Vec<f32> = (0..context_rows)
                .map(|column| {
                    let dot: f32 = (0..head_dim)
                        .map(|offset| {
                            query[q_base + offset] * key[column * dim + head * head_dim + offset]
                        })
                        .sum();
                    dot * scale + mask.map_or(0.0, |mask| mask[row * context_rows + column])
                })
                .collect();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f32 = weights.iter().sum();
            let weights: Vec<f32> = weights.iter().map(|weight| weight / total).collect();
            for offset in 0..head_dim {
                let result: f32 = (0..context_rows)
                    .map(|column| weights[column] * value[column * dim + head * head_dim + offset])
                    .sum();
                out[row * dim + head * head_dim + offset] = result;
            }
        }
    }
    out
}

fn cpu_sinusoidal(step: usize, dim: usize, max_period: usize) -> Vec<f32> {
    let half = dim / 2;
    let log_period = (max_period as f32).ln();
    let step_value = step as f32;
    let mut out = vec![0.0_f32; dim];
    for index in 0..half {
        let exponent = -2.0 * (index as f32) * log_period / (half as f32);
        let angle = step_value * exponent.exp();
        out[index] = angle.cos();
        out[half + index] = angle.sin();
    }
    out
}

// ---------------------------------------------------------------------------
// Test 1: CLIP token states
// ---------------------------------------------------------------------------

fn clip_name(suffix: &str) -> String {
    const PREFIX: &str = "cond_stage_model.transformer.text_model.";
    if suffix.starts_with("layers.") {
        format!("{PREFIX}encoder.{suffix}")
    } else {
        format!("{PREFIX}{suffix}")
    }
}

fn clip_fixture(geometry: ClipGeometry) -> Fixture {
    let specs = clip_checkpoint::required_tensors(geometry);
    let mut tensors = Vec::new();
    for (index, (name, shape)) in specs.into_iter().enumerate() {
        if name.ends_with(".weight") && name.contains("layer_norm") {
            let channels = shape[0];
            let (gain, _) = synth_norm(channels, index + 1);
            tensors.push(TensorSpec {
                name,
                shape,
                values: gain,
            });
        } else {
            let len = shape.iter().product();
            tensors.push(TensorSpec {
                name,
                shape,
                values: synth(len, index + 1),
            });
        }
    }
    write_fixture("clip", tensors)
}

fn cpu_clip(
    map: &CheckpointMap,
    path: &Path,
    geometry: ClipGeometry,
    eps: f32,
    ids: &[u32],
) -> Vec<f32> {
    let ClipGeometry {
        vocab,
        tokens: positions,
        dim,
        ffn_hidden: hidden,
        layers,
        heads,
    } = geometry;
    let token_table = f32_tensor(
        map,
        path,
        &clip_name("embeddings.token_embedding.weight"),
        &[vocab, dim],
    );
    let position_table = f32_tensor(
        map,
        path,
        &clip_name("embeddings.position_embedding.weight"),
        &[positions, dim],
    );
    let mut states = vec![0.0_f32; ids.len() * dim];
    for (row, id) in ids.iter().enumerate() {
        for column in 0..dim {
            states[row * dim + column] = token_table
                [usize::try_from(*id).expect("token id") * dim + column]
                + position_table[row * dim + column];
        }
    }
    if layers > 0 {
        let mask_len = ids.len() * ids.len();
        let mut mask = vec![0.0_f32; mask_len];
        for query in 0..ids.len() {
            for key in query + 1..ids.len() {
                mask[query * ids.len() + key] = f32::NEG_INFINITY;
            }
        }
        for layer in 0..layers {
            let suffix = |field: &str| clip_name(&format!("layers.{layer}.{field}"));
            let norm1_gain = f32_tensor(map, path, &suffix("layer_norm1.weight"), &[dim]);
            let norm1_bias = f32_tensor(map, path, &suffix("layer_norm1.bias"), &[dim]);
            let normalized = cpu_layer_norm(&states, ids.len(), dim, &norm1_gain, &norm1_bias, eps);
            let query = cpu_matmul(
                &normalized,
                &cpu_transpose(
                    &f32_tensor(map, path, &suffix("self_attn.q_proj.weight"), &[dim, dim]),
                    dim,
                    dim,
                ),
                ids.len(),
                dim,
                dim,
            );
            let query = cpu_broadcast(
                &query,
                &f32_tensor(map, path, &suffix("self_attn.q_proj.bias"), &[dim]),
            );
            let key = cpu_matmul(
                &normalized,
                &cpu_transpose(
                    &f32_tensor(map, path, &suffix("self_attn.k_proj.weight"), &[dim, dim]),
                    dim,
                    dim,
                ),
                ids.len(),
                dim,
                dim,
            );
            let key = cpu_broadcast(
                &key,
                &f32_tensor(map, path, &suffix("self_attn.k_proj.bias"), &[dim]),
            );
            let value = cpu_matmul(
                &normalized,
                &cpu_transpose(
                    &f32_tensor(map, path, &suffix("self_attn.v_proj.weight"), &[dim, dim]),
                    dim,
                    dim,
                ),
                ids.len(),
                dim,
                dim,
            );
            let value = cpu_broadcast(
                &value,
                &f32_tensor(map, path, &suffix("self_attn.v_proj.bias"), &[dim]),
            );
            let attended = cpu_attention(
                &query,
                &key,
                &value,
                ids.len(),
                ids.len(),
                dim,
                heads,
                Some(&mask),
            );
            let projected = cpu_matmul(
                &attended,
                &cpu_transpose(
                    &f32_tensor(map, path, &suffix("self_attn.out_proj.weight"), &[dim, dim]),
                    dim,
                    dim,
                ),
                ids.len(),
                dim,
                dim,
            );
            let projected = cpu_broadcast(
                &projected,
                &f32_tensor(map, path, &suffix("self_attn.out_proj.bias"), &[dim]),
            );
            let residual = cpu_add(&states, &projected);
            let norm2_gain = f32_tensor(map, path, &suffix("layer_norm2.weight"), &[dim]);
            let norm2_bias = f32_tensor(map, path, &suffix("layer_norm2.bias"), &[dim]);
            let normalized =
                cpu_layer_norm(&residual, ids.len(), dim, &norm2_gain, &norm2_bias, eps);
            let expanded = cpu_matmul(
                &normalized,
                &cpu_transpose(
                    &f32_tensor(map, path, &suffix("mlp.fc1.weight"), &[hidden, dim]),
                    hidden,
                    dim,
                ),
                ids.len(),
                dim,
                hidden,
            );
            let expanded = cpu_broadcast(
                &expanded,
                &f32_tensor(map, path, &suffix("mlp.fc1.bias"), &[hidden]),
            );
            let activated = cpu_quick_gelu(&expanded);
            let fed = cpu_matmul(
                &activated,
                &cpu_transpose(
                    &f32_tensor(map, path, &suffix("mlp.fc2.weight"), &[dim, hidden]),
                    dim,
                    hidden,
                ),
                ids.len(),
                hidden,
                dim,
            );
            let fed = cpu_broadcast(
                &fed,
                &f32_tensor(map, path, &suffix("mlp.fc2.bias"), &[dim]),
            );
            states = cpu_add(&residual, &fed);
        }
    }
    let gain = f32_tensor(map, path, &clip_name("final_layer_norm.weight"), &[dim]);
    let bias = f32_tensor(map, path, &clip_name("final_layer_norm.bias"), &[dim]);
    cpu_layer_norm(&states, ids.len(), dim, &gain, &bias, eps)
}

#[test]
fn clip_token_states_match_cpu_reference() {
    let Some(device) = device() else { return };
    let geometry = ClipGeometry {
        vocab: 16,
        tokens: 4,
        dim: 8,
        ffn_hidden: 16,
        layers: 2,
        heads: 2,
    };
    let fixture = clip_fixture(geometry);
    let map = open_fixture(&fixture);
    let tensors =
        ClipTensors::load(&map, fixture.path(), geometry, CLIP_EPS).expect("clip tensors");
    let encoder =
        ClipTextEncoder::new(device, geometry.layers, geometry.heads).expect("clip encoder");
    let ids = [1_u32, 3, 2];
    let actual = encoder
        .run_with_bias(&ids, &tensors.weights(), geometry.vocab, geometry.dim)
        .expect("clip encoder run");
    let expected = cpu_clip(&map, fixture.path(), geometry, CLIP_EPS, &ids);
    assert_tensor("clip token states", &actual, &expected);
}

// ---------------------------------------------------------------------------
// Test 2: one reduced UNet step
// ---------------------------------------------------------------------------

fn push_unet_resnet(
    tensors: &mut Vec<TensorSpec>,
    seed: &mut usize,
    block: &str,
    in_c: usize,
    out_c: usize,
) {
    let key = |suffix: &str| format!("{UNET_ROOT}{block}.{suffix}");
    push_norm(tensors, &key("in_layers.0"), in_c, *seed);
    *seed += 2;
    push_conv(tensors, &key("in_layers.2"), in_c, out_c, 3, *seed);
    *seed += 2;
    push_linear(tensors, &key("emb_layers.1"), out_c, 1280, *seed);
    *seed += 2;
    push_norm(tensors, &key("out_layers.0"), out_c, *seed);
    *seed += 2;
    push_conv(tensors, &key("out_layers.3"), out_c, out_c, 3, *seed);
    *seed += 2;
    if in_c != out_c {
        push_conv(tensors, &key("skip_connection"), in_c, out_c, 1, *seed);
        *seed += 2;
    }
}

fn unet_fixture() -> Fixture {
    let mut tensors = Vec::new();
    let mut seed = 1_usize;
    push_conv(
        &mut tensors,
        &format!("{UNET_ROOT}input_blocks.0.0"),
        4,
        320,
        3,
        seed,
    );
    seed += 2;
    push_unet_resnet(&mut tensors, &mut seed, "input_blocks.1.0", 320, 320);
    push_conv(
        &mut tensors,
        &format!("{UNET_ROOT}input_blocks.3.0.op"),
        320,
        320,
        3,
        seed,
    );
    seed += 2;
    push_unet_resnet(&mut tensors, &mut seed, "input_blocks.4.0", 320, 640);
    push_conv(
        &mut tensors,
        &format!("{UNET_ROOT}input_blocks.6.0.op"),
        640,
        640,
        3,
        seed,
    );
    seed += 2;
    push_unet_resnet(&mut tensors, &mut seed, "input_blocks.7.0", 640, 1280);
    push_conv(
        &mut tensors,
        &format!("{UNET_ROOT}input_blocks.9.0.op"),
        1280,
        1280,
        3,
        seed,
    );
    seed += 2;
    push_unet_resnet(&mut tensors, &mut seed, "middle_block.0", 1280, 1280);

    let attention = format!("{UNET_ROOT}middle_block.1.transformer_blocks.0.attn2.");
    push_tensor(
        &mut tensors,
        format!("{attention}to_q.weight"),
        vec![1280, 1280],
        seed,
    );
    seed += 1;
    push_tensor(
        &mut tensors,
        format!("{attention}to_q.bias"),
        vec![1280],
        seed,
    );
    seed += 1;
    push_tensor(
        &mut tensors,
        format!("{attention}to_k.weight"),
        vec![1280, 768],
        seed,
    );
    seed += 1;
    push_tensor(
        &mut tensors,
        format!("{attention}to_k.bias"),
        vec![1280],
        seed,
    );
    seed += 1;
    push_tensor(
        &mut tensors,
        format!("{attention}to_v.weight"),
        vec![1280, 768],
        seed,
    );
    seed += 1;
    push_tensor(
        &mut tensors,
        format!("{attention}to_v.bias"),
        vec![1280],
        seed,
    );
    seed += 1;
    push_linear(
        &mut tensors,
        &format!("{attention}to_out.0"),
        1280,
        1280,
        seed,
    );
    seed += 2;
    push_unet_resnet(&mut tensors, &mut seed, "middle_block.2", 1280, 1280);

    push_conv(
        &mut tensors,
        &format!("{UNET_ROOT}output_blocks.7.0.skip_connection"),
        1280,
        640,
        1,
        seed,
    );
    seed += 2;
    push_conv(
        &mut tensors,
        &format!("{UNET_ROOT}output_blocks.10.0.skip_connection"),
        640,
        320,
        1,
        seed,
    );
    seed += 2;
    push_norm(&mut tensors, &format!("{UNET_ROOT}out.0"), 320, seed);
    seed += 2;
    push_conv(&mut tensors, &format!("{UNET_ROOT}out.2"), 320, 4, 3, seed);
    seed += 2;
    push_linear(
        &mut tensors,
        &format!("{UNET_ROOT}time_embed.0"),
        1280,
        320,
        seed,
    );
    seed += 2;
    push_linear(
        &mut tensors,
        &format!("{UNET_ROOT}time_embed.2"),
        1280,
        1280,
        seed,
    );

    write_fixture("unet", tensors)
}

struct CpuConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct CpuNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct CpuLinear {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct CpuResnet {
    norm1: CpuNorm,
    conv1: CpuConv,
    time: CpuLinear,
    norm2: CpuNorm,
    conv2: CpuConv,
    shortcut: Option<CpuConv>,
}

fn cpu_load_norm(map: &CheckpointMap, path: &Path, prefix: &str, channels: usize) -> CpuNorm {
    CpuNorm {
        gamma: f32_tensor(map, path, &format!("{prefix}.weight"), &[channels]),
        beta: f32_tensor(map, path, &format!("{prefix}.bias"), &[channels]),
    }
}

fn cpu_load_conv(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    kernel: usize,
) -> CpuConv {
    CpuConv {
        weight: f32_tensor(
            map,
            path,
            &format!("{prefix}.weight"),
            &[out_c, in_c, kernel, kernel],
        ),
        bias: f32_tensor(map, path, &format!("{prefix}.bias"), &[out_c]),
    }
}

fn cpu_load_linear(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
) -> CpuLinear {
    CpuLinear {
        weight: cpu_transpose(
            &f32_tensor(map, path, &format!("{prefix}.weight"), &[out_dim, in_dim]),
            out_dim,
            in_dim,
        ),
        bias: f32_tensor(map, path, &format!("{prefix}.bias"), &[out_dim]),
    }
}

fn cpu_load_resnet(
    map: &CheckpointMap,
    path: &Path,
    block: &str,
    in_c: usize,
    out_c: usize,
) -> CpuResnet {
    let key = |suffix: &str| format!("{UNET_ROOT}{block}.{suffix}");
    CpuResnet {
        norm1: cpu_load_norm(map, path, &key("in_layers.0"), in_c),
        conv1: cpu_load_conv(map, path, &key("in_layers.2"), in_c, out_c, 3),
        time: cpu_load_linear(map, path, &key("emb_layers.1"), out_c, 1280),
        norm2: cpu_load_norm(map, path, &key("out_layers.0"), out_c),
        conv2: cpu_load_conv(map, path, &key("out_layers.3"), out_c, out_c, 3),
        shortcut: (in_c != out_c)
            .then(|| cpu_load_conv(map, path, &key("skip_connection"), in_c, out_c, 1)),
    }
}

fn cpu_resnet(
    input: &[f32],
    time: &[f32],
    weights: &CpuResnet,
    in_c: usize,
    out_c: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let hidden = cpu_group_norm(
        input,
        &weights.norm1.gamma,
        &weights.norm1.beta,
        in_c,
        height,
        width,
        32,
        GN_EPS,
    );
    let hidden = cpu_silu(&hidden);
    let mut out = cpu_conv(
        &hidden,
        &weights.conv1.weight,
        &weights.conv1.bias,
        ConvGeometry {
            in_c,
            out_c,
            in_h: height,
            in_w: width,
            kernel: 3,
            stride: 1,
        },
    );
    let activated_time = cpu_silu(time);
    let projected = cpu_matmul(&activated_time, &weights.time.weight, 1, 1280, out_c);
    let projected = cpu_broadcast(&projected, &weights.time.bias);
    out = cpu_broadcast(&out, &projected);
    out = match &weights.shortcut {
        Some(shortcut) => {
            let residual = cpu_conv(
                input,
                &shortcut.weight,
                &shortcut.bias,
                ConvGeometry {
                    in_c,
                    out_c,
                    in_h: height,
                    in_w: width,
                    kernel: 1,
                    stride: 1,
                },
            );
            cpu_add(&out, &residual)
        }
        None => cpu_add(&out, input),
    };
    let skip = out;
    let hidden = cpu_group_norm(
        &skip,
        &weights.norm2.gamma,
        &weights.norm2.beta,
        out_c,
        height,
        width,
        32,
        GN_EPS,
    );
    let hidden = cpu_silu(&hidden);
    let out = cpu_conv(
        &hidden,
        &weights.conv2.weight,
        &weights.conv2.bias,
        ConvGeometry {
            in_c: out_c,
            out_c,
            in_h: height,
            in_w: width,
            kernel: 3,
            stride: 1,
        },
    );
    cpu_add(&out, &skip)
}

fn cpu_unet_time(map: &CheckpointMap, path: &Path, timestep: usize) -> Vec<f32> {
    let first = cpu_load_linear(map, path, &format!("{UNET_ROOT}time_embed.0"), 1280, 320);
    let second = cpu_load_linear(map, path, &format!("{UNET_ROOT}time_embed.2"), 1280, 1280);
    let encoded = cpu_sinusoidal(timestep, 320, 10_000);
    let hidden = cpu_matmul(&encoded, &first.weight, 1, 320, 1280);
    let hidden = cpu_broadcast(&hidden, &first.bias);
    let hidden = cpu_silu(&hidden);
    let hidden = cpu_matmul(&hidden, &second.weight, 1, 1280, 1280);
    cpu_broadcast(&hidden, &second.bias)
}

fn cpu_cross_attention(
    hidden: &[f32],
    context: &[f32],
    map: &CheckpointMap,
    path: &Path,
) -> Vec<f32> {
    let prefix = format!("{UNET_ROOT}middle_block.1.transformer_blocks.0.attn2.");
    let tokens = hidden.len() / 1280;
    let query = cpu_broadcast(
        &cpu_matmul(
            hidden,
            &cpu_transpose(
                &f32_tensor(map, path, &format!("{prefix}to_q.weight"), &[1280, 1280]),
                1280,
                1280,
            ),
            tokens,
            1280,
            1280,
        ),
        &f32_tensor(map, path, &format!("{prefix}to_q.bias"), &[1280]),
    );
    let key = cpu_broadcast(
        &cpu_matmul(
            context,
            &cpu_transpose(
                &f32_tensor(map, path, &format!("{prefix}to_k.weight"), &[1280, 768]),
                1280,
                768,
            ),
            77,
            768,
            1280,
        ),
        &f32_tensor(map, path, &format!("{prefix}to_k.bias"), &[1280]),
    );
    let value = cpu_broadcast(
        &cpu_matmul(
            context,
            &cpu_transpose(
                &f32_tensor(map, path, &format!("{prefix}to_v.weight"), &[1280, 768]),
                1280,
                768,
            ),
            77,
            768,
            1280,
        ),
        &f32_tensor(map, path, &format!("{prefix}to_v.bias"), &[1280]),
    );
    let attended = cpu_attention(&query, &key, &value, tokens, 77, 1280, 8, None);
    let projected = cpu_matmul(
        &attended,
        &cpu_transpose(
            &f32_tensor(
                map,
                path,
                &format!("{prefix}to_out.0.weight"),
                &[1280, 1280],
            ),
            1280,
            1280,
        ),
        tokens,
        1280,
        1280,
    );
    let projected = cpu_broadcast(
        &projected,
        &f32_tensor(map, path, &format!("{prefix}to_out.0.bias"), &[1280]),
    );
    cpu_add(hidden, &projected)
}

fn cpu_unet(
    map: &CheckpointMap,
    path: &Path,
    latent: &[f32],
    timestep: usize,
    context: &[f32],
    side: usize,
) -> Vec<f32> {
    let time = cpu_unet_time(map, path, timestep);
    let mut x = cpu_conv(
        latent,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.0.0.weight"),
            &[320, 4, 3, 3],
        ),
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.0.0.bias"),
            &[320],
        ),
        ConvGeometry {
            in_c: 4,
            out_c: 320,
            in_h: side,
            in_w: side,
            kernel: 3,
            stride: 1,
        },
    );
    let mut run = |block: &str, in_c: usize, out_c: usize, height: usize, value: Vec<f32>| {
        let weights = cpu_load_resnet(map, path, block, in_c, out_c);
        cpu_resnet(&value, &time, &weights, in_c, out_c, height, height)
    };
    x = run("input_blocks.1.0", 320, 320, side, x);
    let half = side / 2;
    x = cpu_conv(
        &x,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.3.0.op.weight"),
            &[320, 320, 3, 3],
        ),
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.3.0.op.bias"),
            &[320],
        ),
        ConvGeometry {
            in_c: 320,
            out_c: 320,
            in_h: side,
            in_w: side,
            kernel: 3,
            stride: 2,
        },
    );
    x = run("input_blocks.4.0", 320, 640, half, x);
    let quarter = side / 4;
    x = cpu_conv(
        &x,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.6.0.op.weight"),
            &[640, 640, 3, 3],
        ),
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.6.0.op.bias"),
            &[640],
        ),
        ConvGeometry {
            in_c: 640,
            out_c: 640,
            in_h: half,
            in_w: half,
            kernel: 3,
            stride: 2,
        },
    );
    x = run("input_blocks.7.0", 640, 1280, quarter, x);
    let eighth = side / 8;
    x = cpu_conv(
        &x,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.9.0.op.weight"),
            &[1280, 1280, 3, 3],
        ),
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}input_blocks.9.0.op.bias"),
            &[1280],
        ),
        ConvGeometry {
            in_c: 1280,
            out_c: 1280,
            in_h: quarter,
            in_w: quarter,
            kernel: 3,
            stride: 2,
        },
    );
    x = run("middle_block.0", 1280, 1280, eighth, x);
    // NCHW [1280,1,1] → token rows → cross-attention → NCHW.
    let hidden = cpu_transpose(&x, 1280, eighth * eighth);
    x = cpu_transpose(
        &cpu_cross_attention(&hidden, context, map, path),
        eighth * eighth,
        1280,
    );
    x = run("middle_block.2", 1280, 1280, eighth, x);
    x = cpu_nearest_upsample(&x, 1280, eighth, eighth, 8);
    x = cpu_conv(
        &x,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}output_blocks.7.0.skip_connection.weight"),
            &[640, 1280, 1, 1],
        ),
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}output_blocks.7.0.skip_connection.bias"),
            &[640],
        ),
        ConvGeometry {
            in_c: 1280,
            out_c: 640,
            in_h: side,
            in_w: side,
            kernel: 1,
            stride: 1,
        },
    );
    x = cpu_conv(
        &x,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}output_blocks.10.0.skip_connection.weight"),
            &[320, 640, 1, 1],
        ),
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}output_blocks.10.0.skip_connection.bias"),
            &[320],
        ),
        ConvGeometry {
            in_c: 640,
            out_c: 320,
            in_h: side,
            in_w: side,
            kernel: 1,
            stride: 1,
        },
    );
    let norm = cpu_load_norm(map, path, &format!("{UNET_ROOT}out.0"), 320);
    let hidden = cpu_group_norm(&x, &norm.gamma, &norm.beta, 320, side, side, 32, GN_EPS);
    let hidden = cpu_silu(&hidden);
    cpu_conv(
        &hidden,
        &f32_tensor(
            map,
            path,
            &format!("{UNET_ROOT}out.2.weight"),
            &[4, 320, 3, 3],
        ),
        &f32_tensor(map, path, &format!("{UNET_ROOT}out.2.bias"), &[4]),
        ConvGeometry {
            in_c: 320,
            out_c: 4,
            in_h: side,
            in_w: side,
            kernel: 3,
            stride: 1,
        },
    )
}

#[test]
fn unet_predict_matches_cpu_reference_at_side_eight() {
    let Some(device) = device() else { return };
    let fixture = unet_fixture();
    let map = open_fixture(&fixture);
    let weights = UnetRealWeights::load(&map, fixture.path()).expect("unet weights");
    let unet = UnetReal::new(device).expect("unet pipelines");
    let side = 8_usize;
    let latent = synth(4 * side * side, 31);
    let context = synth(77 * 768, 67);
    let timestep = 100_usize;
    let actual = unet
        .predict(&latent, timestep, &context, &weights)
        .expect("unet predict");
    let expected = cpu_unet(&map, fixture.path(), &latent, timestep, &context, side);
    assert_tensor("unet prediction", &actual, &expected);
}

// ---------------------------------------------------------------------------
// Test 3: VAE decode
// ---------------------------------------------------------------------------

fn push_vae_resnet(
    tensors: &mut Vec<TensorSpec>,
    seed: &mut usize,
    prefix: &str,
    in_c: usize,
    out_c: usize,
) {
    push_norm(tensors, &format!("{prefix}norm1"), in_c, *seed);
    *seed += 2;
    push_conv(tensors, &format!("{prefix}conv1"), in_c, out_c, 3, *seed);
    *seed += 2;
    push_norm(tensors, &format!("{prefix}norm2"), out_c, *seed);
    *seed += 2;
    push_conv(tensors, &format!("{prefix}conv2"), out_c, out_c, 3, *seed);
    *seed += 2;
    if in_c != out_c {
        push_conv(
            tensors,
            &format!("{prefix}nin_shortcut"),
            in_c,
            out_c,
            1,
            *seed,
        );
        *seed += 2;
    }
}

fn vae_fixture() -> Fixture {
    let mut tensors = Vec::new();
    let mut seed = 1_usize;
    push_conv(
        &mut tensors,
        "first_stage_model.post_quant_conv",
        4,
        4,
        1,
        seed,
    );
    seed += 2;
    push_conv(&mut tensors, &format!("{VAE_ROOT}conv_in"), 4, 512, 3, seed);
    seed += 2;
    for (block, in_c, out_c) in [
        ("mid.block_1.", 512, 512),
        ("mid.block_2.", 512, 512),
        ("up.3.block.0.", 512, 512),
        ("up.2.block.0.", 512, 512),
        ("up.1.block.0.", 512, 256),
        ("up.0.block.0.", 256, 128),
    ] {
        push_vae_resnet(
            &mut tensors,
            &mut seed,
            &format!("{VAE_ROOT}{block}"),
            in_c,
            out_c,
        );
    }
    push_norm(&mut tensors, &format!("{VAE_ROOT}norm_out"), 128, seed);
    seed += 2;
    push_conv(
        &mut tensors,
        &format!("{VAE_ROOT}conv_out"),
        128,
        3,
        3,
        seed,
    );
    write_fixture("vae", tensors)
}

struct CpuVaeResnet {
    norm1: CpuNorm,
    conv1: CpuConv,
    norm2: CpuNorm,
    conv2: CpuConv,
    shortcut: Option<CpuConv>,
}

fn cpu_load_vae_resnet(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    in_c: usize,
    out_c: usize,
) -> CpuVaeResnet {
    CpuVaeResnet {
        norm1: cpu_load_norm(map, path, &format!("{prefix}norm1"), in_c),
        conv1: cpu_load_conv(map, path, &format!("{prefix}conv1"), in_c, out_c, 3),
        norm2: cpu_load_norm(map, path, &format!("{prefix}norm2"), out_c),
        conv2: cpu_load_conv(map, path, &format!("{prefix}conv2"), out_c, out_c, 3),
        shortcut: (in_c != out_c)
            .then(|| cpu_load_conv(map, path, &format!("{prefix}nin_shortcut"), in_c, out_c, 1)),
    }
}

fn cpu_vae_resnet(
    input: &[f32],
    weights: &CpuVaeResnet,
    in_c: usize,
    out_c: usize,
    side: usize,
) -> Vec<f32> {
    let normed = cpu_group_norm(
        input,
        &weights.norm1.gamma,
        &weights.norm1.beta,
        in_c,
        side,
        side,
        32,
        GN_EPS,
    );
    let activated = cpu_silu(&normed);
    let hidden = cpu_conv(
        &activated,
        &weights.conv1.weight,
        &weights.conv1.bias,
        ConvGeometry {
            in_c,
            out_c,
            in_h: side,
            in_w: side,
            kernel: 3,
            stride: 1,
        },
    );
    let normed = cpu_group_norm(
        &hidden,
        &weights.norm2.gamma,
        &weights.norm2.beta,
        out_c,
        side,
        side,
        32,
        GN_EPS,
    );
    let activated = cpu_silu(&normed);
    let hidden = cpu_conv(
        &activated,
        &weights.conv2.weight,
        &weights.conv2.bias,
        ConvGeometry {
            in_c: out_c,
            out_c,
            in_h: side,
            in_w: side,
            kernel: 3,
            stride: 1,
        },
    );
    match &weights.shortcut {
        Some(shortcut) => {
            let projected = cpu_conv(
                input,
                &shortcut.weight,
                &shortcut.bias,
                ConvGeometry {
                    in_c,
                    out_c,
                    in_h: side,
                    in_w: side,
                    kernel: 1,
                    stride: 1,
                },
            );
            cpu_add(&hidden, &projected)
        }
        None => cpu_add(&hidden, input),
    }
}

fn cpu_vae_decode(map: &CheckpointMap, path: &Path, latent: &[f32]) -> Vec<f32> {
    const LATENT_SCALE: f32 = 1.0 / 0.18215;
    let side = 1_usize;
    let scaled: Vec<f32> = latent.iter().map(|value| value * LATENT_SCALE).collect();
    let mut x = cpu_conv(
        &scaled,
        &f32_tensor(
            map,
            path,
            "first_stage_model.post_quant_conv.weight",
            &[4, 4, 1, 1],
        ),
        &f32_tensor(map, path, "first_stage_model.post_quant_conv.bias", &[4]),
        ConvGeometry {
            in_c: 4,
            out_c: 4,
            in_h: side,
            in_w: side,
            kernel: 1,
            stride: 1,
        },
    );
    x = cpu_conv(
        &x,
        &f32_tensor(
            map,
            path,
            &format!("{VAE_ROOT}conv_in.weight"),
            &[512, 4, 3, 3],
        ),
        &f32_tensor(map, path, &format!("{VAE_ROOT}conv_in.bias"), &[512]),
        ConvGeometry {
            in_c: 4,
            out_c: 512,
            in_h: side,
            in_w: side,
            kernel: 3,
            stride: 1,
        },
    );
    for (block, in_c, out_c) in [("mid.block_1.", 512, 512), ("mid.block_2.", 512, 512)] {
        let weights = cpu_load_vae_resnet(map, path, &format!("{VAE_ROOT}{block}"), in_c, out_c);
        x = cpu_vae_resnet(&x, &weights, in_c, out_c, side);
    }
    let mut current = side;
    for (block, in_c, out_c) in [
        ("up.3.block.0.", 512, 512),
        ("up.2.block.0.", 512, 512),
        ("up.1.block.0.", 512, 256),
    ] {
        x = cpu_nearest_upsample(&x, in_c, current, current, 2);
        current *= 2;
        let weights = cpu_load_vae_resnet(map, path, &format!("{VAE_ROOT}{block}"), in_c, out_c);
        x = cpu_vae_resnet(&x, &weights, in_c, out_c, current);
    }
    let weights = cpu_load_vae_resnet(map, path, &format!("{VAE_ROOT}up.0.block.0."), 256, 128);
    x = cpu_vae_resnet(&x, &weights, 256, 128, current);
    let norm = cpu_load_norm(map, path, &format!("{VAE_ROOT}norm_out"), 128);
    let normed = cpu_group_norm(
        &x,
        &norm.gamma,
        &norm.beta,
        128,
        current,
        current,
        32,
        GN_EPS,
    );
    let activated = cpu_silu(&normed);
    cpu_conv(
        &activated,
        &f32_tensor(
            map,
            path,
            &format!("{VAE_ROOT}conv_out.weight"),
            &[3, 128, 3, 3],
        ),
        &f32_tensor(map, path, &format!("{VAE_ROOT}conv_out.bias"), &[3]),
        ConvGeometry {
            in_c: 128,
            out_c: 3,
            in_h: current,
            in_w: current,
            kernel: 3,
            stride: 1,
        },
    )
}

#[test]
fn vae_decode_matches_cpu_reference_at_latent_side_one() {
    let Some(device) = device() else { return };
    let fixture = vae_fixture();
    let map = open_fixture(&fixture);
    let weights = VaeRealWeights::load(&map, fixture.path()).expect("vae weights");
    let vae = VaeReal::new(device).expect("vae pipelines");
    let latent = synth(4, 101);
    let actual = vae.decode(&latent, &weights).expect("vae decode");
    let expected = cpu_vae_decode(&map, fixture.path(), &latent);
    assert_tensor("vae decode", &actual, &expected);
}
