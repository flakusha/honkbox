//! Reduced real-weight SD 1.5 `UNet` over the shared Vulkan kernels.
//!
//! Every tensor comes from the caller's safetensors file; this module bundles
//! no weights and synthesises no fake tensors. The topology is deliberately
//! shortened: three down levels, mid block, one cross-attention layer, a host
//! nearest upsample and two 1×1 projections.
//!
//! ponytail: up-block resnets, their skip connections and attention layers are
//! the upgrade path. Adding them needs the full traversal, not more flags.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    compute::ElementwiseAdd,
    conv2d::{Conv2d, ConvShape},
    cross_attention::{
        CrossAttention, CrossAttentionBias, CrossAttentionShape, CrossAttentionWeights,
    },
    device::VulkanDevice,
    group_norm::{GroupNorm, GroupNormShape},
    matmul::MatMul,
    silu::Silu,
    timestep::sinusoidal,
    unet_checkpoint::load_tensor,
};
use std::io;
use std::path::Path;
use std::sync::Arc;

const ROOT: &str = "model.diffusion_model.";
const LATENT_CHANNELS: usize = 4;
const TIME_DIM: usize = 320;
const TIME_CHANNELS: usize = 1280;
const CONTEXT_ROWS: usize = 77;
const CONTEXT_DIM: usize = 768;
/// SD 1.5 cross-attention uses `head_dim` 64: 1280 / 64 = 20 heads.
const CROSS_HEADS: usize = TIME_CHANNELS / 64;
const GROUPS: usize = 32;
const EPS: f32 = 1e-5;
const UPSAMPLE: usize = 8;

/// `[out,in,k,k]` kernel plus `[out]` bias, in checkpoint order.
struct ConvWeights {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// Group-norm affine pair `[channels]`.
struct NormWeights {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

/// Linear layer with the weight stored in `MatMul` order `[in,out]`.
struct LinearWeights {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// One SD 1.5 `ResBlock`: GN → `SiLU` → conv → time → skip → GN → `SiLU` → conv → skip.
struct ResnetWeights {
    norm1: NormWeights,
    conv1: ConvWeights,
    time: LinearWeights,
    norm2: NormWeights,
    conv2: ConvWeights,
    /// Real `skip_connection` projection when the channel count changes.
    shortcut: Option<ConvWeights>,
}

/// Cross-attention projections. `wq`/`wo` are transposed for `MatMul`;
/// `wkv` stays `[2·dim, context_dim]` because `FusedQkv` transposes it
/// internally (K rows then V rows).
struct CrossWeights {
    wq: Vec<f32>,
    wkv: Vec<f32>,
    wo: Vec<f32>,
    bo: Vec<f32>,
}

/// Two-layer timestep MLP, both weights transposed for `MatMul`.
struct TimeWeights {
    first: LinearWeights,
    second: LinearWeights,
}

/// All decoded f32 tensors the reduced `UNet` needs, owned and validated at load.
pub struct UnetRealWeights {
    input_conv: ConvWeights,
    down1: ResnetWeights,
    downsample1: ConvWeights,
    down2: ResnetWeights,
    downsample2: ConvWeights,
    down3: ResnetWeights,
    downsample3: ConvWeights,
    mid0: ResnetWeights,
    cross: CrossWeights,
    mid2: ResnetWeights,
    project1: ConvWeights,
    project2: ConvWeights,
    out_norm: NormWeights,
    out_conv: ConvWeights,
    time: TimeWeights,
}

/// Reusable Vulkan pipelines for one reduced `UNet` pass.
pub struct UnetReal {
    conv: Conv2d,
    group_norm: GroupNorm,
    silu: Silu,
    matmul: MatMul,
    add: ElementwiseAdd,
    cross: CrossAttention,
}

/// Transpose a row-major `rows × cols` matrix into `cols × rows`.
fn transpose_2d(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // ponytail: host transpose at load/reshape time; move to the GPU only if
    // profiling shows these copies matter.
    let mut out = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    out
}

/// Nearest-neighbour upsample of NCHW `[channels,height,width]` by `factor`.
fn nearest_upsample(
    values: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    factor: usize,
) -> Vec<f32> {
    // ponytail: no learned up-block; real upsamplers plus skips are the
    // upgrade path when fidelity matters.
    let out_h = height * factor;
    let out_w = width * factor;
    let mut out = vec![0.0_f32; channels * out_h * out_w];
    for channel in 0..channels {
        for y in 0..out_h {
            let source_y = y / factor;
            for x in 0..out_w {
                let source_x = x / factor;
                let from = (channel * height + source_y) * width + source_x;
                let to = (channel * out_h + y) * out_w + x;
                out[to] = values[from];
            }
        }
    }
    out
}

fn invalid(message: impl Into<String>) -> VkError {
    VkError::Device(message.into())
}

// ponytail: f64 round trip is exact here; the checked side² == spatial guard
// below rejects any imprecise or negative result before it can be used.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn square_side(len: usize, channels: usize) -> Result<usize, VkError> {
    if len == 0 || !len.is_multiple_of(channels) {
        return Err(invalid("unet_real: latent must be NCHW [1,4,H,W]"));
    }
    let spatial = len / channels;
    let side = f64::sqrt(spatial as f64).round() as usize;
    if side == 0 || side.checked_mul(side) != Some(spatial) {
        return Err(invalid("unet_real: latent spatial size must be square"));
    }
    Ok(side)
}

#[allow(clippy::too_many_arguments)] // explicit NCHW shape; no config for one call shape
fn conv_shape(
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    kernel: usize,
    stride: usize,
) -> ConvShape {
    ConvShape {
        batch: 1,
        in_c,
        out_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel,
        stride,
        padding: kernel / 2,
    }
}

fn load_conv(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    kernel: usize,
) -> io::Result<ConvWeights> {
    Ok(ConvWeights {
        weight: load_tensor(
            map,
            path,
            &format!("{prefix}.weight"),
            &[out_c, in_c, kernel, kernel],
        )?,
        bias: load_tensor(map, path, &format!("{prefix}.bias"), &[out_c])?,
    })
}

fn load_norm(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    channels: usize,
) -> io::Result<NormWeights> {
    Ok(NormWeights {
        gamma: load_tensor(map, path, &format!("{prefix}.weight"), &[channels])?,
        beta: load_tensor(map, path, &format!("{prefix}.bias"), &[channels])?,
    })
}

fn load_linear(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
) -> io::Result<LinearWeights> {
    let raw = load_tensor(map, path, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
    Ok(LinearWeights {
        weight: transpose_2d(&raw, out_dim, in_dim),
        bias: load_tensor(map, path, &format!("{prefix}.bias"), &[out_dim])?,
    })
}

fn load_resnet(
    map: &CheckpointMap,
    path: &Path,
    block: &str,
    in_c: usize,
    out_c: usize,
) -> io::Result<ResnetWeights> {
    let key = |suffix: &str| format!("{ROOT}{block}.{suffix}");
    let shortcut = if in_c == out_c {
        None
    } else {
        Some(load_conv(
            map,
            path,
            &key("skip_connection"),
            in_c,
            out_c,
            1,
        )?)
    };
    Ok(ResnetWeights {
        norm1: load_norm(map, path, &key("in_layers.0"), in_c)?,
        conv1: load_conv(map, path, &key("in_layers.2"), in_c, out_c, 3)?,
        time: load_linear(map, path, &key("emb_layers.1"), TIME_CHANNELS, out_c)?,
        norm2: load_norm(map, path, &key("out_layers.0"), out_c)?,
        conv2: load_conv(map, path, &key("out_layers.3"), out_c, out_c, 3)?,
        shortcut,
    })
}

fn load_time(map: &CheckpointMap, path: &Path) -> io::Result<TimeWeights> {
    Ok(TimeWeights {
        first: load_linear(
            map,
            path,
            &format!("{ROOT}time_embed.0"),
            TIME_DIM,
            TIME_CHANNELS,
        )?,
        second: load_linear(
            map,
            path,
            &format!("{ROOT}time_embed.2"),
            TIME_CHANNELS,
            TIME_CHANNELS,
        )?,
    })
}

fn load_cross(map: &CheckpointMap, path: &Path) -> io::Result<CrossWeights> {
    let prefix = format!("{ROOT}middle_block.1.transformer_blocks.0.attn2.");
    let key = |suffix: &str| format!("{prefix}{suffix}");
    let q = load_tensor(
        map,
        path,
        &key("to_q.weight"),
        &[TIME_CHANNELS, TIME_CHANNELS],
    )?;
    let k = load_tensor(
        map,
        path,
        &key("to_k.weight"),
        &[TIME_CHANNELS, CONTEXT_DIM],
    )?;
    let v = load_tensor(
        map,
        path,
        &key("to_v.weight"),
        &[TIME_CHANNELS, CONTEXT_DIM],
    )?;
    let out = load_tensor(
        map,
        path,
        &key("to_out.0.weight"),
        &[TIME_CHANNELS, TIME_CHANNELS],
    )?;
    // Original SD1.5 has no Q/K/V biases; only `to_out.0` carries one.
    let mut wkv = k;
    wkv.extend_from_slice(&v);
    Ok(CrossWeights {
        wq: transpose_2d(&q, TIME_CHANNELS, TIME_CHANNELS),
        wkv,
        wo: transpose_2d(&out, TIME_CHANNELS, TIME_CHANNELS),
        bo: load_tensor(map, path, &key("to_out.0.bias"), &[TIME_CHANNELS])?,
    })
}

impl UnetRealWeights {
    /// Load and shape-check every tensor used by the reduced topology.
    pub fn load(map: &CheckpointMap, path: &Path) -> io::Result<Self> {
        Ok(Self {
            input_conv: load_conv(
                map,
                path,
                &format!("{ROOT}input_blocks.0.0"),
                LATENT_CHANNELS,
                TIME_DIM,
                3,
            )?,
            down1: load_resnet(map, path, "input_blocks.1.0", TIME_DIM, TIME_DIM)?,
            downsample1: load_conv(
                map,
                path,
                &format!("{ROOT}input_blocks.3.0.op"),
                TIME_DIM,
                TIME_DIM,
                3,
            )?,
            down2: load_resnet(map, path, "input_blocks.4.0", TIME_DIM, 640)?,
            downsample2: load_conv(
                map,
                path,
                &format!("{ROOT}input_blocks.6.0.op"),
                640,
                640,
                3,
            )?,
            down3: load_resnet(map, path, "input_blocks.7.0", 640, TIME_CHANNELS)?,
            downsample3: load_conv(
                map,
                path,
                &format!("{ROOT}input_blocks.9.0.op"),
                TIME_CHANNELS,
                TIME_CHANNELS,
                3,
            )?,
            mid0: load_resnet(map, path, "middle_block.0", TIME_CHANNELS, TIME_CHANNELS)?,
            cross: load_cross(map, path)?,
            mid2: load_resnet(map, path, "middle_block.2", TIME_CHANNELS, TIME_CHANNELS)?,
            // Real single-file SD1.5 uses `skip_connection`, not
            // `nin_shortcut`; these two carry the exact 1280→640/640→320
            // 1×1 projections the reduced topology needs.
            project1: load_conv(
                map,
                path,
                &format!("{ROOT}output_blocks.7.0.skip_connection"),
                TIME_CHANNELS,
                640,
                1,
            )?,
            project2: load_conv(
                map,
                path,
                &format!("{ROOT}output_blocks.10.0.skip_connection"),
                640,
                TIME_DIM,
                1,
            )?,
            out_norm: load_norm(map, path, &format!("{ROOT}out.0"), TIME_DIM)?,
            out_conv: load_conv(
                map,
                path,
                &format!("{ROOT}out.2"),
                TIME_DIM,
                LATENT_CHANNELS,
                3,
            )?,
            time: load_time(map, path)?,
        })
    }
}

impl UnetReal {
    /// Compile the pipelines used by one reduced `UNet` pass.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            conv: Conv2d::new(Arc::clone(&device))?,
            group_norm: GroupNorm::new(Arc::clone(&device))?,
            silu: Silu::new(Arc::clone(&device))?,
            matmul: MatMul::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(Arc::clone(&device))?,
            cross: CrossAttention::new(device)?,
        })
    }

    fn time_embedding(&self, timestep: usize, weights: &TimeWeights) -> Result<Vec<f32>, VkError> {
        let encoded = sinusoidal(timestep, TIME_DIM, 10_000)?;
        let mut hidden =
            self.matmul
                .run(&encoded, &weights.first.weight, 1, TIME_DIM, TIME_CHANNELS)?;
        hidden = self.add.run_broadcast(&hidden, &weights.first.bias)?;
        hidden = self.silu.run(&hidden)?;
        hidden = self.matmul.run(
            &hidden,
            &weights.second.weight,
            1,
            TIME_CHANNELS,
            TIME_CHANNELS,
        )?;
        self.add.run_broadcast(&hidden, &weights.second.bias)
    }

    #[allow(clippy::too_many_arguments)]
    fn resnet(
        &self,
        input: &[f32],
        time: &[f32],
        weights: &ResnetWeights,
        in_c: usize,
        out_c: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, VkError> {
        let shape = GroupNormShape {
            batch: 1,
            channels: out_c,
            height,
            width,
            groups: GROUPS,
        };
        let hidden = self.group_norm.run(
            input,
            &weights.norm1.gamma,
            &weights.norm1.beta,
            GroupNormShape {
                channels: in_c,
                ..shape
            },
            EPS,
        )?;
        let hidden = self.silu.run(&hidden)?;
        let mut out = self.conv.run(
            &hidden,
            &weights.conv1.weight,
            &weights.conv1.bias,
            conv_shape(in_c, out_c, height, width, height, width, 3, 1),
        )?;
        let time = self.silu.run(time)?;
        let projected = self
            .matmul
            .run(&time, &weights.time.weight, 1, TIME_CHANNELS, out_c)?;
        let projected = self.add.run_broadcast(&projected, &weights.time.bias)?;
        out = self.add.run_broadcast(&out, &projected)?;
        out = match &weights.shortcut {
            Some(shortcut) => {
                let residual = self.conv.run(
                    input,
                    &shortcut.weight,
                    &shortcut.bias,
                    conv_shape(in_c, out_c, height, width, height, width, 1, 1),
                )?;
                self.add.run(&out, &residual)?
            }
            None => self.add.run(&out, input)?,
        };
        let skip = out;
        let hidden =
            self.group_norm
                .run(&skip, &weights.norm2.gamma, &weights.norm2.beta, shape, EPS)?;
        let hidden = self.silu.run(&hidden)?;
        let out = self.conv.run(
            &hidden,
            &weights.conv2.weight,
            &weights.conv2.bias,
            conv_shape(out_c, out_c, height, width, height, width, 3, 1),
        )?;
        self.add.run(&out, &skip)
    }

    /// Predict noise for latent NCHW `[1,4,H,W]` with square H=W divisible by 8.
    #[allow(clippy::too_many_lines)] // one linear reduced topology; no speculative graph engine
    pub fn predict(
        &self,
        latent: &[f32],
        timestep: usize,
        context: &[f32],
        weights: &UnetRealWeights,
    ) -> Result<Vec<f32>, VkError> {
        let side = square_side(latent.len(), LATENT_CHANNELS)?;
        if !side.is_multiple_of(UPSAMPLE) {
            return Err(invalid(
                "unet_real: latent side must be divisible by the 8× downsample",
            ));
        }
        if context.len() != CONTEXT_ROWS * CONTEXT_DIM {
            return Err(invalid("unet_real: context must be [77,768] row-major"));
        }
        let time = self.time_embedding(timestep, &weights.time)?;

        let mut x = self.conv.run(
            latent,
            &weights.input_conv.weight,
            &weights.input_conv.bias,
            conv_shape(LATENT_CHANNELS, TIME_DIM, side, side, side, side, 3, 1),
        )?;
        x = self.resnet(&x, &time, &weights.down1, TIME_DIM, TIME_DIM, side, side)?;

        let half = side / 2;
        x = self.conv.run(
            &x,
            &weights.downsample1.weight,
            &weights.downsample1.bias,
            conv_shape(TIME_DIM, TIME_DIM, side, side, half, half, 3, 2),
        )?;
        x = self.resnet(&x, &time, &weights.down2, TIME_DIM, 640, half, half)?;

        let quarter = side / 4;
        x = self.conv.run(
            &x,
            &weights.downsample2.weight,
            &weights.downsample2.bias,
            conv_shape(640, 640, half, half, quarter, quarter, 3, 2),
        )?;
        x = self.resnet(
            &x,
            &time,
            &weights.down3,
            640,
            TIME_CHANNELS,
            quarter,
            quarter,
        )?;

        let eighth = side / 8;
        x = self.conv.run(
            &x,
            &weights.downsample3.weight,
            &weights.downsample3.bias,
            conv_shape(
                TIME_CHANNELS,
                TIME_CHANNELS,
                quarter,
                quarter,
                eighth,
                eighth,
                3,
                2,
            ),
        )?;
        x = self.resnet(
            &x,
            &time,
            &weights.mid0,
            TIME_CHANNELS,
            TIME_CHANNELS,
            eighth,
            eighth,
        )?;

        // NCHW [1280,eighth,eighth] → token rows for the attention pipeline →
        // back to NCHW.
        let tokens = eighth * eighth;
        let hidden = transpose_2d(&x, TIME_CHANNELS, tokens);
        let attended = self.cross.run_with_bias(
            &hidden,
            context,
            &CrossAttentionWeights {
                wq: &weights.cross.wq,
                wkv: &weights.cross.wkv,
                wo: &weights.cross.wo,
            },
            &CrossAttentionBias {
                bo: Some(&weights.cross.bo),
                ..CrossAttentionBias::default()
            },
            CrossAttentionShape {
                query_rows: tokens,
                context_rows: CONTEXT_ROWS,
                dim: TIME_CHANNELS,
                context_dim: CONTEXT_DIM,
                heads: CROSS_HEADS,
            },
        )?;
        x = transpose_2d(&attended, tokens, TIME_CHANNELS);

        x = self.resnet(
            &x,
            &time,
            &weights.mid2,
            TIME_CHANNELS,
            TIME_CHANNELS,
            eighth,
            eighth,
        )?;
        x = nearest_upsample(&x, TIME_CHANNELS, eighth, eighth, UPSAMPLE);

        x = self.conv.run(
            &x,
            &weights.project1.weight,
            &weights.project1.bias,
            conv_shape(TIME_CHANNELS, 640, side, side, side, side, 1, 1),
        )?;
        x = self.conv.run(
            &x,
            &weights.project2.weight,
            &weights.project2.bias,
            conv_shape(640, TIME_DIM, side, side, side, side, 1, 1),
        )?;

        let hidden = self.group_norm.run(
            &x,
            &weights.out_norm.gamma,
            &weights.out_norm.beta,
            GroupNormShape {
                batch: 1,
                channels: TIME_DIM,
                height: side,
                width: side,
                groups: GROUPS,
            },
            EPS,
        )?;
        let hidden = self.silu.run(&hidden)?;
        self.conv.run(
            &hidden,
            &weights.out_conv.weight,
            &weights.out_conv.bias,
            conv_shape(TIME_DIM, LATENT_CHANNELS, side, side, side, side, 3, 1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_2d_swaps_rows_and_columns() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            transpose_2d(&values, 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert_eq!(transpose_2d(&transpose_2d(&values, 2, 3), 3, 2), values);
    }

    #[test]
    fn nearest_upsample_replicates_each_pixel() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            nearest_upsample(&values, 1, 2, 2, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0
            ]
        );
    }

    #[test]
    fn mid_cross_attention_heads_derive_from_channel_width() {
        assert_eq!(CROSS_HEADS, TIME_CHANNELS / 64);
        assert_eq!(CROSS_HEADS, 20);
    }

    #[test]
    fn resblock_time_projection_matches_silu_then_linear() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let unet = UnetReal::new(device).expect("pipelines");
        let channels = GROUPS;
        let as_f32 = |value: usize| f32::from(u16::try_from(value).expect("value fits u16"));
        let time: Vec<f32> = (0..TIME_CHANNELS)
            .map(|i| as_f32(i % 17) * 0.25 - 2.0)
            .collect();
        let weight: Vec<f32> = (0..TIME_CHANNELS * channels)
            .map(|i| as_f32(i % 11) * 0.1 - 0.5)
            .collect();
        let bias: Vec<f32> = (0..channels).map(|c| as_f32(c) * 0.05 - 0.5).collect();
        let norm = || NormWeights {
            gamma: vec![1.0; channels],
            beta: vec![0.0; channels],
        };
        let conv = || ConvWeights {
            weight: vec![0.0; channels * channels * 9],
            bias: vec![0.0; channels],
        };
        let weights = ResnetWeights {
            norm1: norm(),
            conv1: conv(),
            time: LinearWeights {
                weight: weight.clone(),
                bias: bias.clone(),
            },
            norm2: norm(),
            conv2: conv(),
            shortcut: None,
        };
        let expected: Vec<f32> = (0..channels)
            .map(|c| {
                (0..TIME_CHANNELS)
                    .map(|i| {
                        let x = time[i];
                        (x / (1.0 + (-x).exp())) * weight[i * channels + c]
                    })
                    .sum::<f32>()
                    + bias[c]
            })
            .collect();
        let actual = unet
            .resnet(
                &vec![0.0; channels],
                &time,
                &weights,
                channels,
                channels,
                1,
                1,
            )
            .expect("resnet");
        assert_eq!(actual.len(), channels);
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }
}
