//! One VAE decoder block: transposed conv → group norm → `SiLU` → 1×1 projection.

use crate::{
    VkError,
    conv_transpose::{ConvTranspose2d, ConvTransposeShape},
    conv2d::{Conv2d, ConvShape},
    device::VulkanDevice,
    group_norm::{GroupNorm, GroupNormShape},
    silu::Silu,
};
use std::sync::Arc;

/// Weights for one decoder block, in execution order.
pub struct VaeWeights<'a> {
    /// `[in_c, out_c, k, k]` upsampling weights.
    pub up_weights: &'a [f32],
    /// `[out_c]` upsampling bias.
    pub up_bias: &'a [f32],
    /// `[out_c]` group-norm scale.
    pub norm_gamma: &'a [f32],
    /// `[out_c]` group-norm shift.
    pub norm_beta: &'a [f32],
    /// `[1, out_c, proj_out_c, 1]` projection weights.
    pub proj_weights: &'a [f32],
    /// `[proj_out_c]` projection bias.
    pub proj_bias: &'a [f32],
}

/// Composes the existing kernels into one decoder block. No model weights are
/// bundled: every buffer comes from the caller.
pub struct VaeDecode {
    up: ConvTranspose2d,
    norm: GroupNorm,
    activation: Silu,
    projection: Conv2d,
}

impl VaeDecode {
    /// Build all four pipelines on the selected device.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            up: ConvTranspose2d::new(Arc::clone(&device))?,
            norm: GroupNorm::new(Arc::clone(&device))?,
            activation: Silu::new(Arc::clone(&device))?,
            projection: Conv2d::new(device)?,
        })
    }

    /// Decode one latent batch into `proj_out_c × out_h × out_w` feature maps.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        latent: &[f32],
        up_shape: ConvTransposeShape,
        groups: usize,
        eps: f32,
        proj_shape: ConvShape,
        weights: &VaeWeights<'_>,
    ) -> Result<Vec<f32>, VkError> {
        let hidden = self
            .up
            .run(latent, weights.up_weights, weights.up_bias, up_shape)?;
        let norm_shape = GroupNormShape {
            batch: up_shape.batch,
            channels: up_shape.out_c,
            height: up_shape.out_h,
            width: up_shape.out_w,
            groups,
        };
        let normalized = self.norm.run(
            &hidden,
            weights.norm_gamma,
            weights.norm_beta,
            norm_shape,
            eps,
        )?;
        let activated = self.activation.run(&normalized)?;
        self.projection.run(
            &activated,
            weights.proj_weights,
            weights.proj_bias,
            proj_shape,
        )
    }
}

/// Independent CPU reference for one decoder block.
///
/// Signed index math is checked by the shape guards, so the pedantic cast
/// lints add no safety here.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn cpu_decode_reference(
    latent: &[f32],
    up_shape: ConvTransposeShape,
    groups: usize,
    eps: f32,
    proj_shape: ConvShape,
    weights: &VaeWeights<'_>,
) -> Vec<f32> {
    let in_plane = up_shape.in_h * up_shape.in_w;
    let mut hidden = Vec::new();
    for n in 0..up_shape.batch {
        for oc in 0..up_shape.out_c {
            for oh in 0..up_shape.out_h {
                for ow in 0..up_shape.out_w {
                    let mut sum = weights.up_bias[oc];
                    for ic in 0..up_shape.in_c {
                        for kh in 0..up_shape.kernel {
                            let ih = oh as isize + up_shape.padding as isize
                                - (kh * up_shape.stride) as isize;
                            if ih < 0 || ih % up_shape.stride as isize != 0 {
                                continue;
                            }
                            let ih = ih / up_shape.stride as isize;
                            if ih >= up_shape.in_h as isize {
                                continue;
                            }
                            for kw in 0..up_shape.kernel {
                                let iw = ow as isize + up_shape.padding as isize
                                    - (kw * up_shape.stride) as isize;
                                if iw < 0 || iw % up_shape.stride as isize != 0 {
                                    continue;
                                }
                                let iw = iw / up_shape.stride as isize;
                                if iw >= up_shape.in_w as isize {
                                    continue;
                                }
                                let in_index = (n * up_shape.in_c + ic) * in_plane
                                    + ih as usize * up_shape.in_w
                                    + iw as usize;
                                let w = ((ic * up_shape.out_c + oc) * up_shape.kernel + kh)
                                    * up_shape.kernel
                                    + kw;
                                sum += latent[in_index] * weights.up_weights[w];
                            }
                        }
                    }
                    hidden.push(sum);
                }
            }
        }
    }

    let plane = up_shape.out_h * up_shape.out_w;
    let cpg = up_shape.out_c / groups;
    let count = (cpg * plane) as f32;
    let mut normalized = vec![0.0_f32; hidden.len()];
    for n in 0..up_shape.batch {
        for group in 0..groups {
            let mut mean = 0.0;
            for c in group * cpg..(group + 1) * cpg {
                for s in 0..plane {
                    mean += hidden[n * up_shape.out_c * plane + c * plane + s];
                }
            }
            mean /= count;
            let mut variance = 0.0;
            for c in group * cpg..(group + 1) * cpg {
                for s in 0..plane {
                    let delta = hidden[n * up_shape.out_c * plane + c * plane + s] - mean;
                    variance += delta * delta;
                }
            }
            variance /= count;
            let inv_std = 1.0 / (variance + eps).sqrt();
            for c in group * cpg..(group + 1) * cpg {
                for s in 0..plane {
                    let index = n * up_shape.out_c * plane + c * plane + s;
                    let x = (hidden[index] - mean) * inv_std;
                    normalized[index] = x * weights.norm_gamma[c] + weights.norm_beta[c];
                }
            }
        }
    }
    let activated: Vec<f32> = normalized.iter().map(|x| x / (1.0 + (-x).exp())).collect();

    let mut out = Vec::new();
    let out_plane = proj_shape.out_h * proj_shape.out_w;
    for n in 0..proj_shape.batch {
        for oc in 0..proj_shape.out_c {
            for pos in 0..out_plane {
                let mut sum = weights.proj_bias[oc];
                for ic in 0..proj_shape.in_c {
                    sum += activated[(n * proj_shape.in_c + ic) * plane + pos]
                        * weights.proj_weights[oc * proj_shape.in_c + ic];
                }
                out.push(sum);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_decode_slice_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let up_shape = ConvTransposeShape {
            batch: 1,
            in_c: 2,
            out_c: 4,
            in_h: 1,
            in_w: 1,
            out_h: 2,
            out_w: 2,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        let proj_shape = ConvShape {
            batch: 1,
            in_c: 4,
            out_c: 3,
            in_h: 2,
            in_w: 2,
            out_h: 2,
            out_w: 2,
            kernel: 1,
            stride: 1,
            padding: 0,
        };
        let latent: Vec<f32> = (0u8..2).map(|i| f32::from(i) * 0.5 - 0.25).collect();
        let up_weights: Vec<f32> = (0u8..(2 * 4 * 4))
            .map(|i| f32::from(i % 5) * 0.2 - 0.4)
            .collect();
        let up_bias: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];
        let norm_gamma: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
        let norm_beta: Vec<f32> = vec![0.05, -0.05, 0.1, -0.1];
        let proj_weights: Vec<f32> = (0u8..12).map(|i| f32::from(i) * 0.05 - 0.3).collect();
        let proj_bias: Vec<f32> = vec![0.1, -0.1, 0.2];
        let weights = VaeWeights {
            up_weights: &up_weights,
            up_bias: &up_bias,
            norm_gamma: &norm_gamma,
            norm_beta: &norm_beta,
            proj_weights: &proj_weights,
            proj_bias: &proj_bias,
        };

        let decoder = VaeDecode::new(device).expect("pipeline creation");
        let actual = decoder
            .run(&latent, up_shape, 2, 1e-5, proj_shape, &weights)
            .expect("gpu dispatch");
        let expected = cpu_decode_reference(&latent, up_shape, 2, 1e-5, proj_shape, &weights);
        assert_eq!(actual.len(), 3 * 2 * 2);
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}
