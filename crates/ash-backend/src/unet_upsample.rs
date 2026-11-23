//! Transposed-convolution upsample with skip-connection add for the decoder.
//!
//! ponytail: one conv-transpose + elementwise add per stage; no attention or
//! additional projections here. VAE decode is a separate existing component.

use crate::{
    checkpoint::CheckpointMap,
    compute::ElementwiseAdd,
    conv_transpose::{ConvTranspose2d, ConvTransposeShape},
    device::VulkanDevice,
    unet_checkpoint::load_tensor,
    unet_stack::UnetStackError,
};
use std::path::Path;
use std::sync::Arc;

/// Geometry for one transposed-convolution upsample.
#[derive(Debug, Clone, Copy)]
pub struct UnetUpsampleShape {
    /// Batch size.
    pub batch: usize,
    /// Input channels.
    pub in_c: usize,
    /// Output channels (must match the skip tensor).
    pub out_c: usize,
    /// Input height.
    pub in_h: usize,
    /// Input width.
    pub in_w: usize,
    /// Square kernel side.
    pub kernel: usize,
    /// Stride (typically 2).
    pub stride: usize,
    /// Zero padding on each side.
    pub padding: usize,
}

fn output_axis(input: usize, shape: UnetUpsampleShape) -> Result<usize, UnetStackError> {
    let invalid = || UnetStackError::Shape("upsample: invalid output geometry".into());
    if shape.stride == 0 || input == 0 {
        return Err(invalid());
    }
    let base = (input - 1).checked_mul(shape.stride).ok_or_else(invalid)?;
    let unpadded = base
        .checked_add(shape.kernel)
        .ok_or_else(invalid)?
        .checked_sub(shape.padding.checked_mul(2).ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
    if unpadded == 0 {
        return Err(invalid());
    }
    Ok(unpadded)
}

impl UnetUpsampleShape {
    /// `(out_h, out_w)` implied by the input, stride, padding and kernel.
    pub fn output(&self) -> Result<(usize, usize), UnetStackError> {
        Ok((
            output_axis(self.in_h, *self)?,
            output_axis(self.in_w, *self)?,
        ))
    }
}

/// Checkpoint keys for the two upsample tensors.
#[derive(Debug, Clone, Copy)]
pub struct UnetUpsampleKeys<'a> {
    /// `[in_c, out_c, k, k]` transposed-convolution weights.
    pub kernel: &'a str,
    /// `[out_c]` bias.
    pub bias: &'a str,
}

/// Transposed convolution followed by a skip-connection add.
pub struct UnetUpsample {
    up: ConvTranspose2d,
    add: ElementwiseAdd,
}

impl UnetUpsample {
    /// Compile the transposed-convolution and add pipelines.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, crate::VkError> {
        Ok(Self {
            up: ConvTranspose2d::new(Arc::clone(&device))?,
            add: ElementwiseAdd::new(device)?,
        })
    }

    /// Load weights, upsample NCHW input, then add the matching skip tensor.
    pub fn run(
        &self,
        input: &[f32],
        skip: &[f32],
        map: &CheckpointMap,
        path: &Path,
        keys: &UnetUpsampleKeys<'_>,
        shape: UnetUpsampleShape,
    ) -> Result<Vec<f32>, UnetStackError> {
        let kernel_shape = [shape.in_c, shape.out_c, shape.kernel, shape.kernel];
        let bias_shape = [shape.out_c];
        for (name, expected) in [
            (keys.kernel, &kernel_shape[..]),
            (keys.bias, &bias_shape[..]),
        ] {
            let meta = map
                .get(name)
                .ok_or_else(|| UnetStackError::Shape(format!("missing tensor '{name}'")))?;
            if meta.shape != expected {
                return Err(UnetStackError::Shape(format!(
                    "tensor '{name}' has shape {:?}, expected {expected:?}",
                    meta.shape
                )));
            }
        }
        let kernel = load_tensor(map, path, keys.kernel, &kernel_shape)?;
        let bias = load_tensor(map, path, keys.bias, &bias_shape)?;
        let (out_h, out_w) = shape.output()?;
        let skip_len = shape
            .batch
            .checked_mul(shape.out_c)
            .and_then(|n| n.checked_mul(out_h))
            .and_then(|n| n.checked_mul(out_w))
            .ok_or_else(|| UnetStackError::Shape("upsample: skip length overflow".into()))?;
        if skip.len() != skip_len {
            return Err(UnetStackError::Shape(format!(
                "upsample: skip must be {}x{}x{}x{}",
                shape.batch, shape.out_c, out_h, out_w
            )));
        }
        let upsampled = self.up.run(
            input,
            &kernel,
            &bias,
            ConvTransposeShape {
                batch: shape.batch,
                in_c: shape.in_c,
                out_c: shape.out_c,
                in_h: shape.in_h,
                in_w: shape.in_w,
                out_h,
                out_w,
                kernel: shape.kernel,
                stride: shape.stride,
                padding: shape.padding,
            },
        )?;
        Ok(self.add.run(&upsampled, skip)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VkError;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn checkpoint(kernel: &[f32], bias: &[f32]) -> (std::path::PathBuf, CheckpointMap) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, values, dims) in [
            ("up.kernel", kernel, vec![1usize, 1, 2, 2]),
            ("up.bias", bias, vec![1]),
        ] {
            let start = payload.len();
            for value in values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let end = payload.len();
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": dims,
                    "data_offsets": [start, end],
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
            "honkbox-ash-upsample-{}-{stamp}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        (path.clone(), CheckpointMap::read(&path).expect("map"))
    }

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
        let mut out = Vec::new();
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
    fn upsample_adds_skip_and_matches_cpu() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint(&[0.5, -0.25, 0.75, 1.0], &[0.1]);
        let shape = UnetUpsampleShape {
            batch: 1,
            in_c: 1,
            out_c: 1,
            in_h: 2,
            in_w: 2,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let skip: Vec<f32> = (0..16)
            .map(|i| f32::from(u8::try_from(i).expect("0..16")) * 0.05)
            .collect();
        let actual = UnetUpsample::new(device)
            .expect("pipeline")
            .run(
                &input,
                &skip,
                &map,
                &path,
                &UnetUpsampleKeys {
                    kernel: "up.kernel",
                    bias: "up.bias",
                },
                shape,
            )
            .expect("run");

        let weights = [0.5_f32, -0.25, 0.75, 1.0];
        let bias = [0.1_f32];
        let expected: Vec<f32> = cpu_transpose(
            &input,
            &weights,
            &bias,
            ConvTransposeShape {
                batch: shape.batch,
                in_c: shape.in_c,
                out_c: shape.out_c,
                in_h: shape.in_h,
                in_w: shape.in_w,
                out_h: 4,
                out_w: 4,
                kernel: shape.kernel,
                stride: shape.stride,
                padding: shape.padding,
            },
        )
        .into_iter()
        .zip(&skip)
        .map(|(up, skip)| up + skip)
        .collect();
        assert_eq!(shape.output().expect("output"), (4, 4));
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn upsample_rejects_bad_skip_and_keys() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint(&[1.0; 4], &[0.0]);
        let pipeline = UnetUpsample::new(device).expect("pipeline");
        let keys = UnetUpsampleKeys {
            kernel: "up.kernel",
            bias: "missing.bias",
        };
        let shape = UnetUpsampleShape {
            batch: 1,
            in_c: 1,
            out_c: 1,
            in_h: 2,
            in_w: 2,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        assert!(
            pipeline
                .run(&[1.0; 4], &[0.0; 16], &map, &path, &keys, shape)
                .is_err()
        );
        let valid_keys = UnetUpsampleKeys {
            kernel: "up.kernel",
            bias: "up.bias",
        };
        assert!(
            pipeline
                .run(&[1.0; 4], &[0.0; 9], &map, &path, &valid_keys, shape)
                .is_err()
        );
        std::fs::remove_file(path).expect("cleanup");
    }
}
