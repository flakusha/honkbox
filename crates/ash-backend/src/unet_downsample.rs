//! Strided-convolution downsample block for the `UNet` encoder path.
//!
//! ponytail: plain `Conv2d(stride=2)`, no separate pooling or norm — SD
//! downsamplers use this shape. Upsample + VAE decode are still missing.

use crate::{
    checkpoint::CheckpointMap,
    conv2d::{Conv2d, ConvShape},
    device::VulkanDevice,
    unet_checkpoint::load_tensor,
    unet_stack::UnetStackError,
};
use std::path::Path;
use std::sync::Arc;

/// Geometry for one strided downsample convolution.
#[derive(Debug, Clone, Copy)]
pub struct UnetDownsampleShape {
    /// Batch size.
    pub batch: usize,
    /// Input channels.
    pub in_c: usize,
    /// Output channels.
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

fn output_axis(input: usize, shape: UnetDownsampleShape) -> Result<usize, UnetStackError> {
    let invalid = || UnetStackError::Shape("downsample: invalid output geometry".into());
    if shape.stride == 0 || shape.kernel == 0 {
        return Err(invalid());
    }
    let padded = input.checked_add(shape.padding.checked_mul(2).ok_or_else(invalid)?);
    let padded = padded.ok_or_else(invalid)?;
    if padded < shape.kernel {
        return Err(invalid());
    }
    Ok((padded - shape.kernel) / shape.stride + 1)
}

impl UnetDownsampleShape {
    /// `(out_h, out_w)` implied by the input and stride/padding.
    pub fn output(&self) -> Result<(usize, usize), UnetStackError> {
        Ok((
            output_axis(self.in_h, *self)?,
            output_axis(self.in_w, *self)?,
        ))
    }
}

/// Checkpoint keys for the two downsample tensors.
#[derive(Debug, Clone, Copy)]
pub struct UnetDownsampleKeys<'a> {
    /// `[out_c, in_c, k, k]` convolution kernel.
    pub kernel: &'a str,
    /// `[out_c]` bias.
    pub bias: &'a str,
}

/// Strided convolution over one checkpoint.
pub struct UnetDownsample {
    conv: Conv2d,
}

impl UnetDownsample {
    /// Compile the shared convolution pipeline.
    pub fn new(device: Arc<VulkanDevice>) -> Result<Self, crate::VkError> {
        Ok(Self {
            conv: Conv2d::new(device)?,
        })
    }

    /// Load weights from `map`/`path` and downsample NCHW input.
    pub fn run(
        &self,
        input: &[f32],
        map: &CheckpointMap,
        path: &Path,
        keys: &UnetDownsampleKeys<'_>,
        shape: UnetDownsampleShape,
    ) -> Result<Vec<f32>, UnetStackError> {
        let kernel_shape = [shape.out_c, shape.in_c, shape.kernel, shape.kernel];
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
        let conv_shape = ConvShape {
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
        };
        Ok(self.conv.run(input, &kernel, &bias, conv_shape)?)
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
            ("down.kernel", kernel, vec![1usize, 1, 2, 2]),
            ("down.bias", bias, vec![1]),
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
            "honkbox-ash-downsample-{}-{stamp}.safetensors",
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

    #[test]
    fn downsample_halves_spatial_size_and_matches_cpu() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint(&[0.5, -0.25, 0.75, 1.0], &[0.1]);
        let shape = UnetDownsampleShape {
            batch: 1,
            in_c: 1,
            out_c: 1,
            in_h: 4,
            in_w: 4,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        let input: Vec<f32> = (0..16)
            .map(|i| f32::from(u8::try_from(i).expect("0..16")) * 0.25)
            .collect();
        let actual = UnetDownsample::new(device)
            .expect("pipeline")
            .run(
                &input,
                &map,
                &path,
                &UnetDownsampleKeys {
                    kernel: "down.kernel",
                    bias: "down.bias",
                },
                shape,
            )
            .expect("run");
        let weights = [0.5_f32, -0.25, 0.75, 1.0];
        let expected: Vec<f32> = [0usize, 2, 8, 10]
            .iter()
            .map(|&origin| {
                let row = origin / 4;
                let col = origin % 4;
                input[row * 4 + col] * weights[0]
                    + input[row * 4 + col + 1] * weights[1]
                    + input[(row + 1) * 4 + col] * weights[2]
                    + input[(row + 1) * 4 + col + 1] * weights[3]
                    + 0.1
            })
            .collect();
        assert_eq!(shape.output().expect("output"), (2, 2));
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn downsample_checks_keys_and_geometry() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint(&[1.0; 4], &[0.0]);
        let pipeline = UnetDownsample::new(device).expect("pipeline");
        let keys = UnetDownsampleKeys {
            kernel: "down.kernel",
            bias: "missing.bias",
        };
        let shape = UnetDownsampleShape {
            batch: 1,
            in_c: 1,
            out_c: 1,
            in_h: 4,
            in_w: 4,
            kernel: 2,
            stride: 2,
            padding: 0,
        };
        assert!(pipeline.run(&[0.0; 16], &map, &path, &keys, shape).is_err());
        assert!(UnetDownsampleShape { stride: 0, ..shape }.output().is_err());
        std::fs::remove_file(path).expect("cleanup");
    }
}
