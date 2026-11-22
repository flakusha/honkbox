//! NCHW transposed convolution for VAE decoder upsampling.

use crate::{
    VkError,
    compute::{CONV_TRANSPOSE_SPIRV, GpuBuffer, Kernel, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const OUTPUTS_PER_GROUP: u32 = 64;

/// Shapes for one transposed convolution. `out = (in - 1) × stride - 2 × pad + kernel`.
#[derive(Debug, Clone, Copy)]
pub struct ConvTransposeShape {
    /// Batch size (`N`).
    pub batch: usize,
    /// Input channels (`C_in`).
    pub in_c: usize,
    /// Output channels (`C_out`).
    pub out_c: usize,
    /// Input height.
    pub in_h: usize,
    /// Input width.
    pub in_w: usize,
    /// Output height.
    pub out_h: usize,
    /// Output width.
    pub out_w: usize,
    /// Square kernel side.
    pub kernel: usize,
    /// Stride.
    pub stride: usize,
    /// Zero padding on each side.
    pub padding: usize,
}

/// Direct transposed convolution; weights are `[in_c, out_c, k, k]`.
pub struct ConvTranspose2d {
    kernel: Kernel,
}

fn validate(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    shape: ConvTransposeShape,
) -> Result<(usize, usize, usize), VkError> {
    let invalid = || VkError::Device("conv_transpose: invalid shape or buffer length".into());
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(invalid);
    let input_len = mul(mul(mul(shape.batch, shape.in_c)?, shape.in_h)?, shape.in_w)?;
    let weight_len = mul(
        mul(mul(shape.in_c, shape.out_c)?, shape.kernel)?,
        shape.kernel,
    )?;
    let out_len = mul(
        mul(mul(shape.batch, shape.out_c)?, shape.out_h)?,
        shape.out_w,
    )?;
    if shape.batch == 0
        || shape.in_c == 0
        || shape.out_c == 0
        || shape.in_h == 0
        || shape.in_w == 0
        || shape.kernel == 0
        || shape.stride == 0
        || input.len() != input_len
        || weights.len() != weight_len
        || bias.len() != shape.out_c
    {
        return Err(invalid());
    }
    let expected = |size: usize| {
        let span = (size - 1)
            .checked_mul(shape.stride)
            .and_then(|n| n.checked_add(shape.kernel))
            .ok_or_else(invalid)?;
        let removed = shape.padding.checked_mul(2).ok_or_else(invalid)?;
        span.checked_sub(removed).ok_or_else(invalid)
    };
    if expected(shape.in_h)? != shape.out_h || expected(shape.in_w)? != shape.out_w {
        return Err(invalid());
    }
    Ok((input_len, weight_len, out_len))
}

impl ConvTranspose2d {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(CONV_TRANSPOSE_SPIRV))?,
        })
    }

    /// Run the transpose convolution and return `batch × out_c × out_h × out_w` f32.
    pub fn run(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        shape: ConvTransposeShape,
    ) -> Result<Vec<f32>, VkError> {
        let (input_len, weight_len, out_len) = validate(input, weights, bias, shape)?;
        let dim = |value: usize| {
            u32::try_from(value)
                .map_err(|_| VkError::Device("conv_transpose: dimension exceeds u32".into()))
        };
        let dims = [
            dim(shape.batch)?,
            dim(shape.in_c)?,
            dim(shape.out_c)?,
            dim(shape.in_h)?,
            dim(shape.in_w)?,
            dim(shape.out_h)?,
            dim(shape.out_w)?,
            dim(shape.kernel)?,
            dim(shape.stride)?,
            dim(shape.padding)?,
        ];
        let bytes = |len: usize| {
            len.checked_mul(size_of::<f32>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| VkError::Device("conv_transpose: buffer too large".into()))
        };
        let input_bytes = bytes(input_len)?;
        let param_bytes = bytes(weight_len + bias.len())?;
        let out_bytes = bytes(out_len)?;
        let workgroups = u32::try_from(out_len.div_ceil(OUTPUTS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("conv_transpose: too many outputs".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: input_bytes.max(param_bytes).max(out_bytes).max(40),
            ..CapacityRequirements::default()
        })?;

        let mut params = Vec::with_capacity(weight_len + bias.len());
        params.extend_from_slice(weights);
        params.extend_from_slice(bias);
        let input_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(input))?;
        let param_buffer = GpuBuffer::new(&self.kernel.device, &f32_bytes(&params))?;
        let output = GpuBuffer::new(&self.kernel.device, &vec![0u8; out_len * size_of::<f32>()])?;
        let shape_buffer = GpuBuffer::new(
            &self.kernel.device,
            &dims
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        let result = self.kernel.dispatch(
            &[input_buffer, param_buffer, output, shape_buffer],
            [input_bytes, param_bytes, out_bytes, 40],
            2,
            out_len * size_of::<f32>(),
            [workgroups, 1, 1],
        )?;
        Ok(f32_from_bytes(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test-only reference over tiny fixed shapes; signed index math is checked
    // by the asserts, so the pedantic cast lints add no safety here.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::needless_range_loop
    )]
    fn cpu_transpose(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        shape: ConvTransposeShape,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(shape.batch * shape.out_c * shape.out_h * shape.out_w);
        let in_plane = shape.in_h * shape.in_w;
        for n in 0..shape.batch {
            for oc in 0..shape.out_c {
                for oh in 0..shape.out_h {
                    for ow in 0..shape.out_w {
                        let mut sum = bias[oc];
                        for ic in 0..shape.in_c {
                            for kh in 0..shape.kernel {
                                let ih = oh as isize + shape.padding as isize
                                    - (kh * shape.stride) as isize;
                                if ih < 0 || ih % shape.stride as isize != 0 {
                                    continue;
                                }
                                let ih = ih / shape.stride as isize;
                                if ih >= shape.in_h as isize {
                                    continue;
                                }
                                for kw in 0..shape.kernel {
                                    let iw = ow as isize + shape.padding as isize
                                        - (kw * shape.stride) as isize;
                                    if iw < 0 || iw % shape.stride as isize != 0 {
                                        continue;
                                    }
                                    let iw = iw / shape.stride as isize;
                                    if iw >= shape.in_w as isize {
                                        continue;
                                    }
                                    let in_index = (n * shape.in_c + ic) * in_plane
                                        + ih as usize * shape.in_w
                                        + iw as usize;
                                    let weight_index = ((ic * shape.out_c + oc) * shape.kernel
                                        + kh)
                                        * shape.kernel
                                        + kw;
                                    sum += input[in_index] * weights[weight_index];
                                }
                            }
                        }
                        out.push(sum);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn conv_transpose_rejects_bad_shapes() {
        let shape = ConvTransposeShape {
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
        let input = vec![1.0; 2];
        let weights = vec![0.5; 2 * 4 * 2 * 2];
        let bias = vec![0.0; 4];
        assert!(validate(&input, &weights, &bias, shape).is_ok());
        assert!(
            validate(
                &input,
                &weights,
                &bias,
                ConvTransposeShape { out_w: 3, ..shape }
            )
            .is_err()
        );
        assert!(validate(&input, &weights, &bias[..3], shape).is_err());
    }

    #[test]
    fn conv_transpose_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let shape = ConvTransposeShape {
            batch: 1,
            in_c: 2,
            out_c: 4,
            in_h: 2,
            in_w: 2,
            out_h: 4,
            out_w: 4,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        let input: Vec<f32> = (0u8..8).map(|i| f32::from(i) * 0.25 - 0.5).collect();
        let weights: Vec<f32> = (0u8..(2 * 4 * 2 * 2))
            .map(|i| f32::from(i % 7) * 0.125 - 0.25)
            .collect();
        let bias: Vec<f32> = (0u8..4).map(|i| f32::from(i) * 0.05).collect();

        let kernel = ConvTranspose2d::new(device).expect("pipeline creation");
        let actual = kernel
            .run(&input, &weights, &bias, shape)
            .expect("gpu dispatch");
        let expected = cpu_transpose(&input, &weights, &bias, shape);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}
