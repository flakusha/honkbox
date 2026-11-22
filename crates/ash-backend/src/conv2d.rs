//! Direct NCHW convolution on the shared Vulkan compute scaffold.

use crate::{
    VkError,
    compute::{CONV2D_SPIRV, GpuBuffer, Kernel, f32_bytes, f32_from_bytes, spirv_words},
    device::{CapacityRequirements, VulkanDevice},
};
use std::sync::Arc;

const OUTPUTS_PER_GROUP: u32 = 64;

/// Shapes for one 2D convolution. All sizes are in elements, not bytes.
#[derive(Debug, Clone, Copy)]
pub struct ConvShape {
    /// Batch size (`N`).
    pub batch: usize,
    /// Input channels (`C`).
    pub in_c: usize,
    /// Output channels.
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

/// Direct convolution; `weights` are `[out_c, in_c, k, k]`, `bias` is `[out_c]`.
pub struct Conv2d {
    kernel: Kernel,
}

#[allow(clippy::too_many_arguments)]
fn validate(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    shape: ConvShape,
) -> Result<(usize, usize, usize), VkError> {
    let invalid = || VkError::Device("conv2d: invalid shape or buffer length".into());
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(invalid);
    let input_len = mul(mul(mul(shape.batch, shape.in_c)?, shape.in_h)?, shape.in_w)?;
    let weight_len = mul(
        mul(mul(shape.out_c, shape.in_c)?, shape.kernel)?,
        shape.kernel,
    )?;
    let out_len = mul(
        mul(mul(shape.batch, shape.out_c)?, shape.out_h)?,
        shape.out_w,
    )?;
    if shape.batch == 0
        || shape.in_c == 0
        || shape.out_c == 0
        || shape.kernel == 0
        || shape.stride == 0
        || input.len() != input_len
        || weights.len() != weight_len
        || bias.len() != shape.out_c
    {
        return Err(invalid());
    }
    let padded_h = shape
        .in_h
        .checked_add(shape.padding.checked_mul(2).ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
    let padded_w = shape
        .in_w
        .checked_add(shape.padding.checked_mul(2).ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
    if shape.kernel > padded_h || shape.kernel > padded_w {
        return Err(invalid());
    }
    if (padded_h - shape.kernel) / shape.stride + 1 != shape.out_h
        || (padded_w - shape.kernel) / shape.stride + 1 != shape.out_w
    {
        return Err(invalid());
    }
    Ok((input_len, weight_len, out_len))
}

impl Conv2d {
    /// Compile-time-embedded SPIR-V → shader module, layout and pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            kernel: Kernel::new(device, &spirv_words(CONV2D_SPIRV))?,
        })
    }

    /// Run the convolution and return `batch × out_c × out_h × out_w` f32.
    pub fn run(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        shape: ConvShape,
    ) -> Result<Vec<f32>, VkError> {
        let (input_len, weight_len, out_len) = validate(input, weights, bias, shape)?;
        let dim = |value: usize| {
            u32::try_from(value)
                .map_err(|_| VkError::Device("conv2d: dimension exceeds u32".into()))
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
                .ok_or_else(|| VkError::Device("conv2d: buffer too large".into()))
        };
        let input_bytes = bytes(input_len)?;
        let weight_bytes = bytes(weight_len)?;
        let bias_bytes = bytes(bias.len())?;
        let out_bytes = bytes(out_len)?;
        let workgroups = u32::try_from(out_len.div_ceil(OUTPUTS_PER_GROUP as usize))
            .map_err(|_| VkError::Device("conv2d: too many outputs".into()))?;

        self.kernel.device.info.admit(&CapacityRequirements {
            workgroup_count: [workgroups, 1, 1],
            storage_range: input_bytes
                .max(weight_bytes + bias_bytes)
                .max(out_bytes)
                .max(40),
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
            [input_bytes, weight_bytes + bias_bytes, out_bytes, 40],
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

    fn cpu_conv(input: &[f32], weights: &[f32], bias: &[f32], shape: ConvShape) -> Vec<f32> {
        let mut out = Vec::with_capacity(shape.batch * shape.out_c * shape.out_h * shape.out_w);
        let plane = shape.out_h * shape.out_w;
        for n in 0..shape.batch {
            for (oc, &bias_oc) in bias.iter().enumerate().take(shape.out_c) {
                for pos in 0..plane {
                    let (oh, ow) = (pos / shape.out_w, pos % shape.out_w);
                    let mut sum = bias_oc;
                    for ic in 0..shape.in_c {
                        for kh in 0..shape.kernel {
                            let ih = oh * shape.stride + kh;
                            if ih < shape.padding || ih >= shape.padding + shape.in_h {
                                continue;
                            }
                            for kw in 0..shape.kernel {
                                let iw = ow * shape.stride + kw;
                                if iw < shape.padding || iw >= shape.padding + shape.in_w {
                                    continue;
                                }
                                let in_offset = ((n * shape.in_c + ic) * shape.in_h + ih
                                    - shape.padding)
                                    * shape.in_w
                                    + iw
                                    - shape.padding;
                                let weight_offset = ((oc * shape.in_c + ic) * shape.kernel + kh)
                                    * shape.kernel
                                    + kw;
                                sum += input[in_offset] * weights[weight_offset];
                            }
                        }
                    }
                    out.push(sum);
                }
            }
        }
        out
    }

    #[test]
    fn conv2d_rejects_bad_shapes() {
        let shape = ConvShape {
            batch: 1,
            in_c: 1,
            out_c: 1,
            in_h: 3,
            in_w: 3,
            out_h: 2,
            out_w: 2,
            kernel: 2,
            stride: 1,
            padding: 0,
        };
        let input = vec![0.0; 9];
        let weights = vec![0.0; 4];
        let bias = vec![0.0];
        assert!(validate(&input, &weights, &bias, shape).is_ok());
        assert!(validate(&input, &weights, &bias, ConvShape { out_w: 3, ..shape }).is_err());
        assert!(validate(&input, &weights, &[], shape).is_err());
    }

    #[test]
    fn conv2d_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let shape = ConvShape {
            batch: 1,
            in_c: 2,
            out_c: 2,
            in_h: 4,
            in_w: 4,
            out_h: 4,
            out_w: 4,
            kernel: 3,
            stride: 1,
            padding: 1,
        };
        let value = |i: usize| f32::from(u8::try_from(i % 9).expect("0..9")) * 0.125 - 0.5;
        let input: Vec<f32> = (0..32).map(value).collect();
        let weights: Vec<f32> = (0..36).map(|i| value(i + 3)).collect();
        let bias = vec![0.25_f32, -0.5];

        let kernel = Conv2d::new(device).expect("pipeline creation");
        let actual = kernel
            .run(&input, &weights, &bias, shape)
            .expect("gpu dispatch");
        let expected = cpu_conv(&input, &weights, &bias, shape);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }
}
