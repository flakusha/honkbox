//! Reduced real-weight SD 1.5 VAE decoder on the shared Vulkan kernels.
//!
//! Weights come from a caller-supplied safetensors checkpoint; nothing is
//! bundled. The topology is the latency-critical slice of the SD 1.5 VAE
//! decoder: post-quant 1x1, `conv_in`, two mid resnets, three nearest-upsample
//! stages with one or two resnets each, and the norm/SiLU/conv output head.
//! Mid attention and the spare blocks in each up level are deliberately
//! absent — see the `ponytail:` notes below.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    compute::ElementwiseAdd,
    conv2d::{Conv2d, ConvShape},
    device::VulkanDevice,
    group_norm::{GroupNorm, GroupNormShape},
    silu::Silu,
    unet_checkpoint::load_tensor,
};
use std::io;
use std::path::Path;
use std::sync::Arc;

/// SD 1.5 latent scale; SDXL shares the constant.
const LATENT_SCALE: f32 = 1.0 / 0.18215;
const LATENT_CHANNELS: usize = 4;
const WIDTH: usize = 512;
const GROUPS: usize = 32;
const EPS: f32 = 1e-6;

fn invalid(message: impl Into<String>) -> VkError {
    VkError::Device(message.into())
}

/// Conv weight shapes and the optional 1x1 residual projection of a block.
fn resnet_shapes(in_c: usize, out_c: usize) -> ([usize; 4], [usize; 4], Option<[usize; 4]>) {
    (
        [out_c, in_c, 3, 3],
        [out_c, out_c, 3, 3],
        (in_c != out_c).then_some([out_c, in_c, 1, 1]),
    )
}

/// Load by original single-file key, falling back to the bare spec spelling.
/// ponytail: diffusers `up_blocks/resnets` renaming is not covered; target is
/// the original single-file SD 1.5 checkpoint.
fn load_owned(
    map: &CheckpointMap,
    path: &Path,
    original: &str,
    shape: &[usize],
) -> io::Result<Vec<f32>> {
    let spec = original
        .strip_prefix("first_stage_model.")
        .unwrap_or(original);
    let name = if map.get_aliased(original).is_some() {
        original
    } else {
        spec
    };
    load_tensor(map, path, name, shape)
}

/// Decoded tensors for one resnet block. Channel widths were checked against
/// the checkpoint header at load time, so `run_resnet` only builds shapes.
struct ResnetWeights {
    norm1_gamma: Vec<f32>,
    norm1_beta: Vec<f32>,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    norm2_gamma: Vec<f32>,
    norm2_beta: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    /// Channel-changing blocks carry a 1x1 projection of the residual.
    shortcut: Option<(Vec<f32>, Vec<f32>)>,
}

fn load_resnet(
    map: &CheckpointMap,
    path: &Path,
    prefix: &str,
    in_c: usize,
    out_c: usize,
) -> io::Result<ResnetWeights> {
    let (conv1_shape, conv2_shape, shortcut_shape) = resnet_shapes(in_c, out_c);
    let key = |suffix: &str| format!("{prefix}{suffix}");
    let shortcut = match shortcut_shape {
        Some(shape) => Some((
            load_owned(map, path, &key("nin_shortcut.weight"), &shape)?,
            load_owned(map, path, &key("nin_shortcut.bias"), &[out_c])?,
        )),
        None => None,
    };
    Ok(ResnetWeights {
        norm1_gamma: load_owned(map, path, &key("norm1.weight"), &[in_c])?,
        norm1_beta: load_owned(map, path, &key("norm1.bias"), &[in_c])?,
        conv1_weight: load_owned(map, path, &key("conv1.weight"), &conv1_shape)?,
        conv1_bias: load_owned(map, path, &key("conv1.bias"), &[out_c])?,
        norm2_gamma: load_owned(map, path, &key("norm2.weight"), &[out_c])?,
        norm2_beta: load_owned(map, path, &key("norm2.bias"), &[out_c])?,
        conv2_weight: load_owned(map, path, &key("conv2.weight"), &conv2_shape)?,
        conv2_bias: load_owned(map, path, &key("conv2.bias"), &[out_c])?,
        shortcut,
    })
}

/// Owned f32 tensors for the reduced SD 1.5 VAE decoder.
pub struct VaeRealWeights {
    post_quant_weight: Vec<f32>,
    post_quant_bias: Vec<f32>,
    conv_in_weight: Vec<f32>,
    conv_in_bias: Vec<f32>,
    mid1: ResnetWeights,
    mid2: ResnetWeights,
    up3: ResnetWeights,
    up2: ResnetWeights,
    up1: ResnetWeights,
    up0: ResnetWeights,
    norm_out_gamma: Vec<f32>,
    norm_out_beta: Vec<f32>,
    conv_out_weight: Vec<f32>,
    conv_out_bias: Vec<f32>,
}

impl VaeRealWeights {
    /// Decode and validate every tensor the reduced decoder touches.
    /// A mismatch fails before any GPU work is queued.
    pub fn load(map: &CheckpointMap, path: &Path) -> io::Result<Self> {
        let decoder = "first_stage_model.decoder.";
        Ok(Self {
            post_quant_weight: load_owned(
                map,
                path,
                "first_stage_model.post_quant_conv.weight",
                &[4, 4, 1, 1],
            )?,
            post_quant_bias: load_owned(map, path, "first_stage_model.post_quant_conv.bias", &[4])?,
            conv_in_weight: load_owned(
                map,
                path,
                &format!("{decoder}conv_in.weight"),
                &[WIDTH, LATENT_CHANNELS, 3, 3],
            )?,
            conv_in_bias: load_owned(map, path, &format!("{decoder}conv_in.bias"), &[WIDTH])?,
            mid1: load_resnet(map, path, &format!("{decoder}mid.block_1."), WIDTH, WIDTH)?,
            mid2: load_resnet(map, path, &format!("{decoder}mid.block_2."), WIDTH, WIDTH)?,
            up3: load_resnet(map, path, &format!("{decoder}up.3.block.0."), WIDTH, WIDTH)?,
            up2: load_resnet(map, path, &format!("{decoder}up.2.block.0."), WIDTH, WIDTH)?,
            up1: load_resnet(map, path, &format!("{decoder}up.1.block.0."), WIDTH, 256)?,
            up0: load_resnet(map, path, &format!("{decoder}up.0.block.0."), 256, 128)?,
            norm_out_gamma: load_owned(map, path, &format!("{decoder}norm_out.weight"), &[128])?,
            norm_out_beta: load_owned(map, path, &format!("{decoder}norm_out.bias"), &[128])?,
            conv_out_weight: load_owned(
                map,
                path,
                &format!("{decoder}conv_out.weight"),
                &[3, 128, 3, 3],
            )?,
            conv_out_bias: load_owned(map, path, &format!("{decoder}conv_out.bias"), &[3])?,
        })
    }
}

fn conv3x3(in_c: usize, out_c: usize, h: usize, w: usize) -> ConvShape {
    ConvShape {
        batch: 1,
        in_c,
        out_c,
        in_h: h,
        in_w: w,
        out_h: h,
        out_w: w,
        kernel: 3,
        stride: 1,
        padding: 1,
    }
}

fn conv1x1(in_c: usize, out_c: usize, h: usize, w: usize) -> ConvShape {
    ConvShape {
        batch: 1,
        in_c,
        out_c,
        in_h: h,
        in_w: w,
        out_h: h,
        out_w: w,
        kernel: 1,
        stride: 1,
        padding: 0,
    }
}

fn group_shape(channels: usize, h: usize, w: usize) -> GroupNormShape {
    GroupNormShape {
        batch: 1,
        channels,
        height: h,
        width: w,
        groups: GROUPS,
    }
}

/// Side length of a square NCHW plane, or `None` for a non-square plane.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn square_side(plane: usize) -> Option<usize> {
    if plane == 0 {
        return None;
    }
    let side = (plane as f64).sqrt() as usize;
    side.checked_mul(side)
        .filter(|sq| *sq == plane)
        .map(|_| side)
}

/// Nearest-neighbour 2x upsample of a single NCHW [1, C, h, w] tensor.
fn nearest_upsample(input: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let out_h = h * 2;
    let out_w = w * 2;
    let mut out = vec![0.0_f32; channels * out_h * out_w];
    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let value = input[c * h * w + y * w + x];
                let base = c * out_h * out_w;
                for dy in 0..2 {
                    for dx in 0..2 {
                        out[base + (2 * y + dy) * out_w + 2 * x + dx] = value;
                    }
                }
            }
        }
    }
    out
}

/// Reduced SD 1.5 VAE decoder composed from the shared Vulkan kernels.
pub struct VaeReal {
    conv: Conv2d,
    norm: GroupNorm,
    silu: Silu,
    add: ElementwiseAdd,
    /// Device the pipelines were built on.
    pub device: Arc<VulkanDevice>,
}

impl VaeReal {
    /// Bind the pipelines used by the reduced decoder to one Vulkan device.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            conv: Conv2d::new(Arc::clone(&device))?,
            norm: GroupNorm::new(Arc::clone(&device))?,
            silu: Silu::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(Arc::clone(&device))?,
            device,
        })
    }

    /// GN -> `SiLU` -> conv1 -> GN -> `SiLU` -> conv2 -> shortcut -> residual add.
    fn run_resnet(
        &self,
        input: &[f32],
        in_c: usize,
        out_c: usize,
        h: usize,
        w: usize,
        weights: &ResnetWeights,
    ) -> Result<Vec<f32>, VkError> {
        let normed = self.norm.run(
            input,
            &weights.norm1_gamma,
            &weights.norm1_beta,
            group_shape(in_c, h, w),
            EPS,
        )?;
        let activated = self.silu.run(&normed)?;
        let hidden = self.conv.run(
            &activated,
            &weights.conv1_weight,
            &weights.conv1_bias,
            conv3x3(in_c, out_c, h, w),
        )?;
        let normed = self.norm.run(
            &hidden,
            &weights.norm2_gamma,
            &weights.norm2_beta,
            group_shape(out_c, h, w),
            EPS,
        )?;
        let activated = self.silu.run(&normed)?;
        let hidden = self.conv.run(
            &activated,
            &weights.conv2_weight,
            &weights.conv2_bias,
            conv3x3(out_c, out_c, h, w),
        )?;
        match &weights.shortcut {
            Some((weight, bias)) => {
                let projected = self
                    .conv
                    .run(input, weight, bias, conv1x1(in_c, out_c, h, w))?;
                self.add.run(&hidden, &projected)
            }
            // Identity shortcut: borrow the residual, no host copy.
            None => self.add.run(&hidden, input),
        }
    }

    /// Decode a latent NCHW [1, 4, h8, w8] into image NCHW [1, 3, 8h, 8w].
    ///
    /// ponytail: the flat slice carries no spatial dimensions, so the plane
    /// must be square (SD 1.5's 64x64 default). Add explicit h/w arguments
    /// when non-square images are needed. Mid attention and spare up blocks
    /// are skipped; add them when full-fidelity parity matters.
    pub fn decode(&self, latent: &[f32], weights: &VaeRealWeights) -> Result<Vec<f32>, VkError> {
        if !latent.len().is_multiple_of(LATENT_CHANNELS) {
            return Err(invalid("vae_real: latent length is not a multiple of 4"));
        }
        let side = square_side(latent.len() / LATENT_CHANNELS)
            .ok_or_else(|| invalid("vae_real: latent plane is not square"))?;
        let side2 = side
            .checked_mul(2)
            .ok_or_else(|| invalid("vae_real: latent size overflow"))?;
        let side4 = side2
            .checked_mul(2)
            .ok_or_else(|| invalid("vae_real: latent size overflow"))?;
        let side8 = side4
            .checked_mul(2)
            .ok_or_else(|| invalid("vae_real: latent size overflow"))?;

        let scaled: Vec<f32> = latent.iter().map(|v| v * LATENT_SCALE).collect();
        let x = self.conv.run(
            &scaled,
            &weights.post_quant_weight,
            &weights.post_quant_bias,
            conv1x1(LATENT_CHANNELS, LATENT_CHANNELS, side, side),
        )?;
        let x = self.conv.run(
            &x,
            &weights.conv_in_weight,
            &weights.conv_in_bias,
            conv3x3(LATENT_CHANNELS, WIDTH, side, side),
        )?;
        let x = self.run_resnet(&x, WIDTH, WIDTH, side, side, &weights.mid1)?;
        let x = self.run_resnet(&x, WIDTH, WIDTH, side, side, &weights.mid2)?;

        let x = nearest_upsample(&x, WIDTH, side, side);
        let x = self.run_resnet(&x, WIDTH, WIDTH, side2, side2, &weights.up3)?;

        let x = nearest_upsample(&x, WIDTH, side2, side2);
        let x = self.run_resnet(&x, WIDTH, WIDTH, side4, side4, &weights.up2)?;

        let x = nearest_upsample(&x, WIDTH, side4, side4);
        let x = self.run_resnet(&x, WIDTH, 256, side8, side8, &weights.up1)?;
        let x = self.run_resnet(&x, 256, 128, side8, side8, &weights.up0)?;

        let normed = self.norm.run(
            &x,
            &weights.norm_out_gamma,
            &weights.norm_out_beta,
            group_shape(128, side8, side8),
            EPS,
        )?;
        let activated = self.silu.run(&normed)?;
        self.conv.run(
            &activated,
            &weights.conv_out_weight,
            &weights.conv_out_bias,
            conv3x3(128, 3, side8, side8),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_upsample_doubles_every_pixel() {
        let src = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            nearest_upsample(&src, 1, 2, 2),
            vec![
                1., 1., 2., 2., //
                1., 1., 2., 2., //
                3., 3., 4., 4., //
                3., 3., 4., 4., //
            ]
        );
    }

    #[test]
    fn nearest_upsample_keeps_channel_major_order() {
        let src = [1.0_f32, 2.0, 10.0, 20.0];
        assert_eq!(
            nearest_upsample(&src, 2, 1, 2),
            vec![
                1., 1., 2., 2., //
                1., 1., 2., 2., //
                10., 10., 20., 20., //
                10., 10., 20., 20., //
            ]
        );
    }

    #[test]
    fn shortcut_shapes_follow_channel_changes() {
        let (c1, c2, shortcut) = resnet_shapes(512, 512);
        assert_eq!(c1, [512, 512, 3, 3]);
        assert_eq!(c2, [512, 512, 3, 3]);
        assert_eq!(shortcut, None);

        let (c1, c2, shortcut) = resnet_shapes(512, 256);
        assert_eq!(c1, [256, 512, 3, 3]);
        assert_eq!(c2, [256, 256, 3, 3]);
        assert_eq!(shortcut, Some([256, 512, 1, 1]));

        let (c1, c2, shortcut) = resnet_shapes(256, 128);
        assert_eq!(c1, [128, 256, 3, 3]);
        assert_eq!(c2, [128, 128, 3, 3]);
        assert_eq!(shortcut, Some([128, 256, 1, 1]));
    }

    #[test]
    fn latent_plane_must_be_square() {
        assert_eq!(square_side(64 * 64), Some(64));
        assert_eq!(square_side(48 * 64), None);
        assert_eq!(square_side(0), None);
    }
}
