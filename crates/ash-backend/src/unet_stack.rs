//! Stack `UnetInputBlock`s with per-block checkpoint keys and chained shapes.
//!
//! Still one stage type only: no timestep embedding, attention or residual
//! path. A real `predict` closure needs those before it can be honest.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    device::VulkanDevice,
    unet_checkpoint::{UnetInputKeys, load_unet_input},
    unet_input::{UnetInputBlock, UnetInputShape, UnetInputWeights},
};
use std::error::Error;
use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// Checkpoint keys plus geometry for one stacked block.
#[derive(Debug, Clone, Copy)]
pub struct UnetStackBlockSpec<'a> {
    /// Four tensor names for this block.
    pub keys: UnetInputKeys<'a>,
    /// NCHW geometry; must chain with the preceding block.
    pub shape: UnetInputShape,
}

/// Failure while loading or running the stack.
#[derive(Debug)]
pub enum UnetStackError {
    /// Checkpoint read, key lookup or dtype failure.
    Io(io::Error),
    /// Vulkan dispatch or capacity failure.
    Vk(VkError),
    /// Block count or adjacent shape mismatch.
    Shape(String),
}

impl fmt::Display for UnetStackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "unet stack checkpoint: {e}"),
            Self::Vk(e) => write!(f, "unet stack dispatch: {e}"),
            Self::Shape(msg) => write!(f, "unet stack shape: {msg}"),
        }
    }
}

impl Error for UnetStackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Vk(e) => Some(e),
            Self::Shape(_) => None,
        }
    }
}

impl From<UnetStackError> for VkError {
    fn from(error: UnetStackError) -> Self {
        Self::Device(error.to_string())
    }
}

impl From<io::Error> for UnetStackError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<VkError> for UnetStackError {
    fn from(value: VkError) -> Self {
        Self::Vk(value)
    }
}

/// Fixed-depth stack of input-style blocks over one checkpoint.
pub struct UnetStack {
    blocks: Vec<UnetInputBlock>,
}

impl UnetStack {
    /// Instantiate `blocks` pipelines.
    pub fn new(device: &Arc<VulkanDevice>, blocks: usize) -> Result<Self, VkError> {
        let mut pipelines = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            pipelines.push(UnetInputBlock::new(Arc::clone(device))?);
        }
        Ok(Self { blocks: pipelines })
    }

    /// Validate the chain, load each block's tensors, then run in order.
    pub fn run(
        &self,
        input: &[f32],
        map: &CheckpointMap,
        path: &Path,
        specs: &[UnetStackBlockSpec<'_>],
        eps: f32,
    ) -> Result<Vec<f32>, UnetStackError> {
        if specs.len() != self.blocks.len() || specs.is_empty() {
            return Err(UnetStackError::Shape(format!(
                "expected {} block specs, got {}",
                self.blocks.len(),
                specs.len()
            )));
        }
        for (index, spec) in specs.iter().enumerate() {
            let Some(previous) = index.checked_sub(1).map(|i| &specs[i].shape) else {
                continue;
            };
            let prev_h = UnetInputShape::out_side(
                previous.in_h,
                previous.kernel,
                previous.stride,
                previous.padding,
            )
            .ok_or_else(|| UnetStackError::Shape("previous block: invalid height".into()))?;
            let prev_w = UnetInputShape::out_side(
                previous.in_w,
                previous.kernel,
                previous.stride,
                previous.padding,
            )
            .ok_or_else(|| UnetStackError::Shape("previous block: invalid width".into()))?;
            if previous.out_c != spec.shape.in_c
                || prev_h != spec.shape.in_h
                || prev_w != spec.shape.in_w
            {
                return Err(UnetStackError::Shape(format!(
                    "block {index}: input {}x{}x{} does not follow block {} output {prev_h}x{prev_w}x{}",
                    spec.shape.in_c,
                    spec.shape.in_h,
                    spec.shape.in_w,
                    index - 1,
                    previous.out_c
                )));
            }
        }

        let mut hidden = input.to_vec();
        for (block, spec) in self.blocks.iter().zip(specs) {
            let tensors = load_unet_input(map, path, &spec.keys, spec.shape)?;
            let weights = UnetInputWeights {
                kernel: &tensors.kernel,
                bias: &tensors.bias,
                gamma: &tensors.gamma,
                beta: &tensors.beta,
            };
            hidden = block.run(&hidden, &weights, spec.shape, eps)?;
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn shape(in_c: usize, out_c: usize) -> UnetInputShape {
        UnetInputShape {
            batch: 1,
            in_c,
            out_c,
            in_h: 3,
            in_w: 3,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
        }
    }

    fn keys(prefix: &str) -> (String, String, String, String) {
        (
            format!("{prefix}.kernel"),
            format!("{prefix}.bias"),
            format!("{prefix}.gamma"),
            format!("{prefix}.beta"),
        )
    }

    fn checkpoint_with_two_blocks() -> (std::path::PathBuf, CheckpointMap) {
        let first = keys("block0");
        let second = keys("block1");
        let entries = [
            (first.0.as_str(), vec![3usize, 2, 3, 3], 54),
            (first.1.as_str(), vec![3], 3),
            (first.2.as_str(), vec![3], 3),
            (first.3.as_str(), vec![3], 3),
            (second.0.as_str(), vec![2, 3, 3, 3], 54),
            (second.1.as_str(), vec![2], 2),
            (second.2.as_str(), vec![2], 2),
            (second.3.as_str(), vec![2], 2),
        ];
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape, count) in entries {
            let start = payload.len();
            for index in 0..count {
                let value = f32::from(u8::try_from(index % 7).expect("0..7")) * 0.2 - 0.3;
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let end = payload.len();
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
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
            "honkbox-ash-stack-{}-{stamp}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        let map = CheckpointMap::read(&path).expect("map");
        (path, map)
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
    fn stack_matches_direct_sequential_blocks() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint_with_two_blocks();
        let first = keys("block0");
        let second = keys("block1");
        let specs = [
            UnetStackBlockSpec {
                keys: UnetInputKeys {
                    kernel: &first.0,
                    bias: &first.1,
                    gamma: &first.2,
                    beta: &first.3,
                },
                shape: shape(2, 3),
            },
            UnetStackBlockSpec {
                keys: UnetInputKeys {
                    kernel: &second.0,
                    bias: &second.1,
                    gamma: &second.2,
                    beta: &second.3,
                },
                shape: shape(3, 2),
            },
        ];
        let input: Vec<f32> = (0..18)
            .map(|i| f32::from(u8::try_from(i).expect("0..18")) * 0.05 - 0.4)
            .collect();

        let stack = UnetStack::new(&device, 2).expect("stack");
        let actual = stack.run(&input, &map, &path, &specs, 1e-5).expect("run");

        let mut expected = input;
        for spec in &specs {
            let tensors = load_unet_input(&map, &path, &spec.keys, spec.shape).expect("load");
            let weights = UnetInputWeights {
                kernel: &tensors.kernel,
                bias: &tensors.bias,
                gamma: &tensors.gamma,
                beta: &tensors.beta,
            };
            expected = UnetInputBlock::new(device.clone())
                .expect("block")
                .run(&expected, &weights, spec.shape, 1e-5)
                .expect("direct run");
        }
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn stack_rejects_bad_chains_and_counts() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint_with_two_blocks();
        let first = keys("block0");
        let second = keys("block1");
        let good = [
            UnetStackBlockSpec {
                keys: UnetInputKeys {
                    kernel: &first.0,
                    bias: &first.1,
                    gamma: &first.2,
                    beta: &first.3,
                },
                shape: shape(2, 3),
            },
            UnetStackBlockSpec {
                keys: UnetInputKeys {
                    kernel: &second.0,
                    bias: &second.1,
                    gamma: &second.2,
                    beta: &second.3,
                },
                shape: shape(3, 2),
            },
        ];
        let stack = UnetStack::new(&device, 2).expect("stack");
        assert!(matches!(
            stack.run(&[0.0; 18], &map, &path, &good[..1], 1e-5),
            Err(UnetStackError::Shape(_))
        ));
        let broken = [
            good[0],
            UnetStackBlockSpec {
                shape: shape(2, 2),
                ..good[1]
            },
        ];
        assert!(matches!(
            stack.run(&[0.0; 18], &map, &path, &broken, 1e-5),
            Err(UnetStackError::Shape(_))
        ));
        std::fs::remove_file(path).expect("cleanup");
    }
}
