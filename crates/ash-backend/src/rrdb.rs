// size-allow: 400
//! One residual-dense block (RRDB) spike over the shared Ash compute ops.
//!
//! This is not a full RRDBNet or Real-ESRGAN implementation. PixelShuffle is
//! an independent test-only geometry reference, not an upscaler component.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    compute::ElementwiseAdd,
    conv2d::{Conv2d, ConvShape},
    device::VulkanDevice,
    leaky_relu::LeakyRelu,
    unet_checkpoint::load_tensor,
};
use std::{io, path::Path, sync::Arc};

const CHANNELS: usize = 64;
const GROWTH: usize = 32;
const CONVOLUTIONS: usize = 5;
const RESIDUAL_SCALE: f32 = 0.2;

const fn conv_out_channels(index: usize) -> usize {
    if index + 1 == CONVOLUTIONS {
        CHANNELS
    } else {
        GROWTH
    }
}

/// Fixed `[1, 64, H, W]` geometry for the spike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RrdbShape {
    /// Batch size; the spike accepts only one.
    pub batch: usize,
    /// Spatial height.
    pub height: usize,
    /// Spatial width.
    pub width: usize,
}

/// Weight and bias tensors for one 3×3 convolution.
#[derive(Debug, Clone)]
pub struct ConvWeights {
    /// Row-major `[out_c, in_c, 3, 3]`.
    pub weights: Vec<f32>,
    /// `[out_c]`.
    pub bias: Vec<f32>,
}

/// Five convolutions for `body.0.rdb1`.
#[derive(Debug, Clone)]
pub struct RrdbWeights {
    /// Dense convolutions 1 through 5.
    pub convs: [ConvWeights; CONVOLUTIONS],
}

impl RrdbWeights {
    /// Load and validate the original-layout Real-ESRGAN RRDB keys.
    pub fn from_checkpoint(map: &CheckpointMap, path: &Path) -> io::Result<Self> {
        let mut convs = Vec::with_capacity(CONVOLUTIONS);
        for index in 0..CONVOLUTIONS {
            let in_c = CHANNELS + index * GROWTH;
            let out_c = conv_out_channels(index);
            let stem = format!("body.0.rdb1.conv{}", index + 1);
            let weight_key = format!("{stem}.weight");
            let bias_key = format!("{stem}.bias");
            convs.push(ConvWeights {
                weights: load_tensor(map, path, &weight_key, &[out_c, in_c, 3, 3])?,
                bias: load_tensor(map, path, &bias_key, &[out_c])?,
            });
        }
        Ok(Self {
            convs: convs.try_into().expect("five RRDB convolutions"),
        })
    }
}

/// Reusable pipelines for one RRDB block.
pub struct RrdbBlock {
    conv: Conv2d,
    activation: LeakyRelu,
    add: ElementwiseAdd,
}

impl RrdbBlock {
    /// Build the shared convolution pipeline plus activation and residual add.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            conv: Conv2d::new(Arc::clone(&device))?,
            activation: LeakyRelu::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// Run one RRDB block and return `[1, 64, H, W]` f32.
    pub fn run(
        &self,
        input: &[f32],
        shape: RrdbShape,
        weights: &RrdbWeights,
    ) -> Result<Vec<f32>, VkError> {
        let plane = validate(input, shape, weights)?;
        let mut dense = input.to_vec();
        dense
            .try_reserve_exact((CHANNELS + (CONVOLUTIONS - 1) * GROWTH) * plane)
            .map_err(|_| VkError::Device("rrdb: dense allocation failed".into()))?;
        let mut output = Vec::new();
        for (index, tensors) in weights.convs.iter().enumerate() {
            let out_c = conv_out_channels(index);
            let conv_shape = ConvShape {
                batch: shape.batch,
                in_c: CHANNELS + index * GROWTH,
                out_c,
                in_h: shape.height,
                in_w: shape.width,
                out_h: shape.height,
                out_w: shape.width,
                kernel: 3,
                stride: 1,
                padding: 1,
            };
            let current = self
                .conv
                .run(&dense, &tensors.weights, &tensors.bias, conv_shape)?;
            let current = self.activation.run(&current)?;
            if index + 1 == CONVOLUTIONS {
                output = current;
            } else {
                dense.extend_from_slice(&current);
            }
        }
        let residual = input
            .iter()
            .map(|value| value * RESIDUAL_SCALE)
            .collect::<Vec<_>>();
        self.add.run(&residual, &output)
    }
}

fn validate(input: &[f32], shape: RrdbShape, weights: &RrdbWeights) -> Result<usize, VkError> {
    let invalid = |message: &str| VkError::Device(format!("rrdb: {message}"));
    if shape.batch != 1 || shape.height == 0 || shape.width == 0 {
        return Err(invalid("expected shape [1, 64, H, W] with H/W non-zero"));
    }
    let plane = shape
        .height
        .checked_mul(shape.width)
        .ok_or_else(|| invalid("spatial element overflow"))?;
    let input_len = CHANNELS
        .checked_mul(plane)
        .ok_or_else(|| invalid("input element overflow"))?;
    if input.len() != input_len {
        return Err(invalid("input length does not match [1, 64, H, W]"));
    }
    for (index, tensors) in weights.convs.iter().enumerate() {
        let in_c = CHANNELS + index * GROWTH;
        let out_c = conv_out_channels(index);
        let expected = out_c * in_c * 9;
        if tensors.weights.len() != expected || tensors.bias.len() != out_c {
            return Err(invalid(&format!(
                "conv{} expects weights [{out_c},{in_c},3,3] and bias [{out_c}]",
                index + 1
            )));
        }
    }
    Ok(plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> RrdbWeights {
        let convs = std::array::from_fn(|index| {
            let in_c = CHANNELS + index * GROWTH;
            let out_c = conv_out_channels(index);
            ConvWeights {
                weights: (0..out_c * in_c * 9)
                    .map(|i| {
                        f32::from(u8::try_from((i + index) % 13).expect("0..13")) * 0.002 - 0.012
                    })
                    .collect(),
                bias: (0..out_c)
                    .map(|i| f32::from(u8::try_from((i + index) % 7).expect("0..7")) * 0.01 - 0.03)
                    .collect(),
            }
        });
        RrdbWeights { convs }
    }

    fn cpu_conv(input: &[f32], tensors: &ConvWeights, in_c: usize, h: usize, w: usize) -> Vec<f32> {
        let out_c = tensors.bias.len();
        let mut out = vec![0.0; out_c * h * w];
        for oc in 0..out_c {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = tensors.bias[oc];
                    for ic in 0..in_c {
                        for ky in 0..3 {
                            let iy = y + ky;
                            if iy < 1 || iy > h {
                                continue;
                            }
                            for kx in 0..3 {
                                let ix = x + kx;
                                if ix < 1 || ix > w {
                                    continue;
                                }
                                let input_index = (ic * h + iy - 1) * w + ix - 1;
                                let weight_index = ((oc * in_c + ic) * 3 + ky) * 3 + kx;
                                sum += input[input_index] * tensors.weights[weight_index];
                            }
                        }
                    }
                    out[(oc * h + y) * w + x] = sum;
                }
            }
        }
        out
    }

    fn cpu_rrdb(input: &[f32], shape: RrdbShape, weights: &RrdbWeights) -> Vec<f32> {
        let mut dense = input.to_vec();
        let mut output = Vec::new();
        for (index, tensors) in weights.convs.iter().enumerate() {
            let mut current = cpu_conv(
                &dense,
                tensors,
                CHANNELS + index * GROWTH,
                shape.height,
                shape.width,
            );
            for value in &mut current {
                *value = if *value < 0.0 { 0.2 * *value } else { *value };
            }
            if index + 1 == CONVOLUTIONS {
                output = current;
            } else {
                dense.extend_from_slice(&current);
            }
        }
        input.iter().zip(output).map(|(x, d)| 0.2 * x + d).collect()
    }

    #[test]
    fn rrdb_spike_cpu_vectors_prove_residual_and_dense_channels() {
        let shape = RrdbShape {
            batch: 1,
            height: 2,
            width: 2,
        };
        let input = (0..CHANNELS * 4)
            .map(|i| f32::from(u8::try_from(i % 9).expect("0..9")) - 4.0)
            .collect::<Vec<_>>();
        let mut zero = weights();
        for tensors in &mut zero.convs {
            tensors.weights.fill(0.0);
            tensors.bias.fill(0.0);
        }
        let output = cpu_rrdb(&input, shape, &zero);
        assert_eq!(output.len(), input.len());
        for (got, want) in output.iter().zip(&input) {
            assert_eq!(*got, 0.2 * want);
        }

        let mut one = weights();
        for tensors in &mut one.convs {
            tensors.weights.fill(0.0);
            tensors.bias.fill(0.0);
        }
        one.convs[0].bias.fill(1.0);
        for oc in 0..CHANNELS {
            let weight_index = ((oc * (CHANNELS + 4 * GROWTH) + CHANNELS) * 3 + 1) * 3 + 1;
            one.convs[4].weights[weight_index] = 1.0;
        }
        let output = cpu_rrdb(&input, shape, &one);
        for (got, want) in output.iter().zip(&input) {
            assert_eq!(*got, 0.2 * want + 1.0);
        }
    }

    #[test]
    fn rrdb_spike_rejects_wrong_shapes_and_weights_before_dispatch() {
        let shape = RrdbShape {
            batch: 1,
            height: 2,
            width: 2,
        };
        let input = vec![0.0; CHANNELS * 4];
        let valid = weights();
        assert_eq!(validate(&input, shape, &valid), Ok(4));
        assert!(validate(&input[..1], shape, &valid).is_err());
        assert!(validate(&input, RrdbShape { batch: 2, ..shape }, &valid).is_err());
        let mut bad = weights();
        bad.convs[2].weights.pop();
        assert!(validate(&input, shape, &bad).is_err());
    }

    #[test]
    fn rrdb_spike_dispatches_on_vulkan_device() {
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let shape = RrdbShape {
            batch: 1,
            height: 2,
            width: 2,
        };
        let input = (0..CHANNELS * 4)
            .map(|i| f32::from(u8::try_from(i % 9).expect("0..9")) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let weights = weights();
        let block = RrdbBlock::new(device).expect("rrdb pipelines");
        let actual = block.run(&input, shape, &weights).expect("rrdb dispatch");
        let expected = cpu_rrdb(&input, shape, &weights);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 2e-4, "got {got}, want {want}");
        }
    }

    fn pixel_shuffle(
        input: &[f32],
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        factor: usize,
    ) -> Result<Vec<f32>, VkError> {
        let scale = factor
            .checked_mul(factor)
            .ok_or_else(|| VkError::Device("pixel_shuffle: overflow".into()))?;
        if batch == 0 || height == 0 || width == 0 || scale == 0 || channels % scale != 0 {
            return Err(VkError::Device(
                "pixel_shuffle: channels must divide by factor²".into(),
            ));
        }
        let out_c = channels / scale;
        let out_h = height * factor;
        let out_w = width * factor;
        let expected = batch * channels * height * width;
        if input.len() != expected {
            return Err(VkError::Device(
                "pixel_shuffle: input length mismatch".into(),
            ));
        }
        let mut output = Vec::with_capacity(batch * out_c * out_h * out_w);
        for n in 0..batch {
            for c in 0..out_c {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let sub = (y % factor) * factor + x % factor;
                        let src_c = c * scale + sub;
                        let src =
                            ((n * channels + src_c) * height + y / factor) * width + x / factor;
                        output.push(input[src]);
                    }
                }
            }
        }
        Ok(output)
    }

    #[test]
    fn pixel_shuffle_spike_factors_two_and_four_preserve_batch() {
        for factor in [2, 4] {
            let input = (0..2 * 16 * 2 * 3)
                .map(|i| f32::from(u16::try_from(i).expect("small index")))
                .collect::<Vec<_>>();
            let output = pixel_shuffle(&input, 2, 16, 2, 3, factor).expect("shuffle");
            assert_eq!(
                output.len(),
                2 * (16 / (factor * factor)) * 2 * factor * 3 * factor
            );
            for n in 0..2 {
                for c in 0..16 / (factor * factor) {
                    for y in 0..2 * factor {
                        for x in 0..3 * factor {
                            let sub = (y % factor) * factor + x % factor;
                            let src_c = c * factor * factor + sub;
                            let src = ((n * 16 + src_c) * 2 + y / factor) * 3 + x / factor;
                            let dst =
                                ((n * (16 / (factor * factor)) + c) * 2 * factor + y) * 3 * factor
                                    + x;
                            assert_eq!(output[dst], input[src]);
                        }
                    }
                }
            }
        }
        assert!(pixel_shuffle(&[0.0; 4], 1, 3, 1, 1, 2).is_err());
    }
}
