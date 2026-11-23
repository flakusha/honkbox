//! First `UNet` block: `Conv2d → GroupNorm → SiLU` over caller-supplied weights.
//!
//! This is one real block of the diffusion backbone, not a `predict`
//! implementation — no timestep conditioning, attention or residual path yet.

use crate::{
    VkError,
    conv2d::{Conv2d, ConvShape},
    device::VulkanDevice,
    group_norm::{GroupNorm, GroupNormShape},
    silu::Silu,
};
use std::sync::Arc;

/// NCHW geometry for one convolution followed by group norm.
#[derive(Debug, Clone, Copy)]
pub struct UnetInputShape {
    /// NCHW batch size.
    pub batch: usize,
    /// Input channel count.
    pub in_c: usize,
    /// Output channel count.
    pub out_c: usize,
    /// Input height.
    pub in_h: usize,
    /// Input width.
    pub in_w: usize,
    /// Square convolution kernel side.
    pub kernel: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Zero padding on each side.
    pub padding: usize,
    /// Group-norm group count; must divide `out_c`.
    pub groups: usize,
}

impl UnetInputShape {
    /// Output spatial side for one dimension of a square kernel/padding.
    pub fn out_side(input: usize, kernel: usize, stride: usize, padding: usize) -> Option<usize> {
        if kernel == 0 || stride == 0 || input + 2 * padding < kernel {
            return None;
        }
        Some((input + 2 * padding - kernel).checked_add(stride - 1)? / stride + 1)
    }

    fn conv(&self, out_h: usize, out_w: usize) -> ConvShape {
        ConvShape {
            batch: self.batch,
            in_c: self.in_c,
            out_c: self.out_c,
            in_h: self.in_h,
            in_w: self.in_w,
            out_h,
            out_w,
            kernel: self.kernel,
            stride: self.stride,
            padding: self.padding,
        }
    }
}

/// Weights for the convolution and the following affine group norm.
pub struct UnetInputWeights<'a> {
    /// `[out_c, in_c, kernel, kernel]` row-major convolution kernel.
    pub kernel: &'a [f32],
    /// `[out_c]` convolution bias.
    pub bias: &'a [f32],
    /// `[out_c]` group-norm gain.
    pub gamma: &'a [f32],
    /// `[out_c]` group-norm bias.
    pub beta: &'a [f32],
}

/// One `UNet` input block, built from three shared kernels.
pub struct UnetInputBlock {
    conv: Conv2d,
    norm: GroupNorm,
    silu: Silu,
}

impl UnetInputBlock {
    /// Three pipelines: convolution, group norm, `SiLU`.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            conv: Conv2d::new(Arc::clone(&device))?,
            norm: GroupNorm::new(Arc::clone(&device))?,
            silu: Silu::new(device)?,
        })
    }

    /// Run the block over NCHW input, returning NCHW output with `out_c`
    /// channels and `(out_h, out_w)` spatial size.
    pub fn run(
        &self,
        input: &[f32],
        weights: &UnetInputWeights<'_>,
        shape: UnetInputShape,
        eps: f32,
    ) -> Result<Vec<f32>, VkError> {
        let out_h = UnetInputShape::out_side(shape.in_h, shape.kernel, shape.stride, shape.padding)
            .ok_or_else(|| VkError::Device("unet input: invalid output height".into()))?;
        let out_w = UnetInputShape::out_side(shape.in_w, shape.kernel, shape.stride, shape.padding)
            .ok_or_else(|| VkError::Device("unet input: invalid output width".into()))?;
        let spatial = out_h
            .checked_mul(out_w)
            .ok_or_else(|| VkError::Device("unet input: spatial overflow".into()))?;
        let convolved = self.conv.run(
            input,
            weights.kernel,
            weights.bias,
            shape.conv(out_h, out_w),
        )?;
        let normalized = self.norm.run(
            &convolved,
            weights.gamma,
            weights.beta,
            GroupNormShape {
                batch: shape.batch,
                channels: shape.out_c,
                height: out_h,
                width: out_w,
                groups: shape.groups,
            },
            eps,
        )?;
        if normalized.len() != shape.batch * shape.out_c * spatial {
            return Err(VkError::Device(
                "unet input: normalized tensor has unexpected length".into(),
            ));
        }
        self.silu.run(&normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<Arc<VulkanDevice>> {
        match VulkanDevice::headless() {
            Ok(device) => Some(Arc::new(device)),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                None
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        }
    }

    fn cpu_block(
        input: &[f32],
        weights: &UnetInputWeights<'_>,
        shape: UnetInputShape,
        eps: f32,
    ) -> Vec<f32> {
        let out_h = UnetInputShape::out_side(shape.in_h, shape.kernel, shape.stride, shape.padding)
            .expect("test height");
        let out_w = UnetInputShape::out_side(shape.in_w, shape.kernel, shape.stride, shape.padding)
            .expect("test width");
        let spatial = shape.in_h * shape.in_w;
        let mut convolved = vec![0.0_f32; shape.batch * shape.out_c * out_h * out_w];
        for n in 0..shape.batch {
            for oc in 0..shape.out_c {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let mut sum = 0.0_f32;
                        for ic in 0..shape.in_c {
                            for ky in 0..shape.kernel {
                                for kx in 0..shape.kernel {
                                    let iy = y * shape.stride + ky;
                                    let ix = x * shape.stride + kx;
                                    if iy < shape.padding
                                        || ix < shape.padding
                                        || iy >= shape.padding + shape.in_h
                                        || ix >= shape.padding + shape.in_w
                                    {
                                        continue;
                                    }
                                    let pixel = n * shape.in_c * spatial
                                        + ic * spatial
                                        + (iy - shape.padding) * shape.in_w
                                        + ix
                                        - shape.padding;
                                    let weight = ((oc * shape.in_c + ic) * shape.kernel + ky)
                                        * shape.kernel
                                        + kx;
                                    sum += input[pixel] * weights.kernel[weight];
                                }
                            }
                        }
                        let index = ((n * shape.out_c + oc) * out_h + y) * out_w + x;
                        convolved[index] = sum + weights.bias[oc];
                    }
                }
            }
        }

        let group_size = shape.out_c / shape.groups * out_h * out_w;
        let count = f32::from(u16::try_from(group_size).expect("test group"));
        let mut normalized = convolved.clone();
        for group in 0..shape.batch * shape.groups {
            let start = group * group_size;
            let slice = &convolved[start..start + group_size];
            let mean = slice.iter().sum::<f32>() / count;
            let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count;
            for offset in 0..group_size {
                let channel = group % shape.groups + offset / (out_h * out_w);
                normalized[start + offset] = (slice[offset] - mean) / (variance + eps).sqrt()
                    * weights.gamma[channel]
                    + weights.beta[channel];
            }
        }
        normalized.iter().map(|x| x / (1.0 + (-x).exp())).collect()
    }

    #[test]
    fn block_matches_cpu_reference() {
        let Some(device) = device() else { return };
        let block = UnetInputBlock::new(device).expect("pipeline");
        let shape = UnetInputShape {
            batch: 1,
            in_c: 2,
            out_c: 3,
            in_h: 3,
            in_w: 3,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
        };
        let input: Vec<f32> = (0..18)
            .map(|i| f32::from(u8::try_from(i).expect("0..18")) * 0.1 - 0.9)
            .collect();
        let kernel: Vec<f32> = (0..54)
            .map(|i| f32::from(u8::try_from(i % 7).expect("0..7")) * 0.1 - 0.3)
            .collect();
        let bias = [0.05_f32, -0.1, 0.02];
        let gamma = [1.0_f32, 0.9, 1.1];
        let beta = [0.0_f32, 0.1, -0.1];
        let weights = UnetInputWeights {
            kernel: &kernel,
            bias: &bias,
            gamma: &gamma,
            beta: &beta,
        };
        let expected = cpu_block(&input, &weights, shape, 1e-5);
        let actual = block.run(&input, &weights, shape, 1e-5).expect("gpu block");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }

    #[test]
    fn block_rejects_bad_shapes() {
        let Some(device) = device() else { return };
        let block = UnetInputBlock::new(device).expect("pipeline");
        let shape = UnetInputShape {
            batch: 1,
            in_c: 2,
            out_c: 3,
            in_h: 3,
            in_w: 3,
            kernel: 3,
            stride: 1,
            padding: 0,
            groups: 1,
        };
        let kernel = vec![0.1_f32; 54];
        let bias = [0.0_f32; 3];
        let gamma = [1.0_f32; 3];
        let beta = [0.0_f32; 3];
        let weights = UnetInputWeights {
            kernel: &kernel,
            bias: &bias,
            gamma: &gamma,
            beta: &beta,
        };
        assert!(block.run(&[], &weights, shape, 1e-5).is_err());
        let bad_groups = UnetInputShape { groups: 2, ..shape };
        assert!(block.run(&[0.0; 18], &weights, bad_groups, 1e-5).is_err());
        let bad_kernel = UnetInputShape { kernel: 5, ..shape };
        assert!(block.run(&[0.0; 18], &weights, bad_kernel, 1e-5).is_err());
    }
}
