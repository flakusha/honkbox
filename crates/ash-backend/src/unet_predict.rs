//! Diffusion `predict` stage: timestep + text conditioning → `UNet` stages.
//!
//! ponytail: conditioning must already be projected to latent channels. Real
//! SD injects text via cross-attention; broadcasting here is an explicit
//! caller contract, not a claim of checkpoint parity. No down/up path or VAE
//! decode yet — this returns a noise tensor for `DdimSampler`.

use crate::{
    checkpoint::CheckpointMap,
    device::VulkanDevice,
    timestep::{TimestepEmbed, TimestepWeights},
    unet_attention::{UnetAttentionBlock, UnetAttentionShape, UnetAttentionWeights},
    unet_stack::{UnetStack, UnetStackBlockSpec, UnetStackError},
};
use std::path::Path;
use std::sync::Arc;

/// Caller-supplied weights and geometry for one prediction.
pub struct UnetPredictConfig<'a> {
    /// Sinusoidal MLP timestep weights.
    pub timestep: TimestepWeights<'a>,
    /// Sinusoidal embedding width.
    pub timestep_dim: usize,
    /// Timestep MLP hidden width.
    pub timestep_hidden: usize,
    /// Per-block checkpoint keys and shapes for the input stack.
    pub blocks: &'a [UnetStackBlockSpec<'a>],
    /// Spatial attention weights.
    pub attention: UnetAttentionWeights<'a>,
    /// Spatial attention geometry; channels must equal the final block output.
    pub attention_shape: UnetAttentionShape,
    /// Group-norm epsilon for every stage.
    pub eps: f32,
}

/// Noise predictor over one checkpoint and caller-supplied conditioning.
pub struct UnetPredictor {
    timestep: TimestepEmbed,
    stack: UnetStack,
    attention: UnetAttentionBlock,
}

impl UnetPredictor {
    /// Instantiate the timestep, block-stack and attention pipelines.
    pub fn new(device: &Arc<VulkanDevice>, blocks: usize) -> Result<Self, crate::VkError> {
        Ok(Self {
            timestep: TimestepEmbed::new(Arc::clone(device))?,
            stack: UnetStack::new(device, blocks)?,
            attention: UnetAttentionBlock::new(Arc::clone(device))?,
        })
    }

    /// Predict noise for one latent, timestep and channel-aligned conditioning.
    pub fn predict(
        &self,
        latent: &[f32],
        timestep: usize,
        conditioning: &[f32],
        map: &CheckpointMap,
        path: &Path,
        config: &UnetPredictConfig<'_>,
    ) -> Result<Vec<f32>, UnetStackError> {
        let first = config
            .blocks
            .first()
            .ok_or_else(|| UnetStackError::Shape("predict: no blocks".into()))?;
        let channels = first.shape.in_c;
        let spatial = config
            .attention_shape
            .height
            .checked_mul(config.attention_shape.width)
            .ok_or_else(|| UnetStackError::Shape("predict: spatial overflow".into()))?;
        let expected = spatial
            .checked_mul(channels)
            .and_then(|per_batch| per_batch.checked_mul(config.attention_shape.batch))
            .ok_or_else(|| UnetStackError::Shape("predict: tensor overflow".into()))?;
        if latent.len() != expected || conditioning.len() != channels {
            return Err(UnetStackError::Shape(format!(
                "predict: latent must be {}x{}x{}x{} and conditioning {channels}-wide",
                config.attention_shape.batch,
                channels,
                config.attention_shape.height,
                config.attention_shape.width
            )));
        }
        let last = config
            .blocks
            .last()
            .ok_or_else(|| UnetStackError::Shape("predict: no blocks".into()))?;
        if last.shape.out_c != config.attention_shape.channels {
            return Err(UnetStackError::Shape(
                "predict: attention channels must match final block output".into(),
            ));
        }

        let time = self.timestep.run(
            timestep,
            &config.timestep,
            config.timestep_dim,
            config.timestep_hidden,
            channels,
        )?;
        let injected: Vec<f32> = time.iter().zip(conditioning).map(|(t, c)| t + c).collect();
        let mut conditioned_latent = latent.to_vec();
        for n in 0..config.attention_shape.batch {
            for position in 0..spatial {
                for (channel, injection) in injected.iter().enumerate() {
                    let index = (n * channels + channel) * spatial + position;
                    conditioned_latent[index] += injection;
                }
            }
        }
        let hidden = self
            .stack
            .run(&conditioned_latent, map, path, config.blocks, config.eps)?;
        Ok(self.attention.run(
            &hidden,
            &config.attention,
            config.attention_shape,
            config.eps,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VkError;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn block_keys(prefix: &str) -> (String, String, String, String) {
        (
            format!("{prefix}.kernel"),
            format!("{prefix}.bias"),
            format!("{prefix}.gamma"),
            format!("{prefix}.beta"),
        )
    }

    fn checkpoint() -> (std::path::PathBuf, CheckpointMap) {
        let keys = block_keys("block0");
        let entries = [
            (keys.0.as_str(), vec![2usize, 2, 3, 3], 36usize),
            (keys.1.as_str(), vec![2], 2),
            (keys.2.as_str(), vec![2], 2),
            (keys.3.as_str(), vec![2], 2),
        ];
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape, count) in entries {
            let start = payload.len();
            for index in 0..count {
                let value = f32::from(u8::try_from(index % 7).expect("0..7")) * 0.1 - 0.2;
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
            "honkbox-ash-predict-{}-{stamp}.safetensors",
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
    fn predict_matches_manual_stage_composition() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint();
        let keys = block_keys("block0");
        let spec = UnetStackBlockSpec {
            keys: crate::unet_checkpoint::UnetInputKeys {
                kernel: &keys.0,
                bias: &keys.1,
                gamma: &keys.2,
                beta: &keys.3,
            },
            shape: crate::unet_input::UnetInputShape {
                batch: 1,
                in_c: 2,
                out_c: 2,
                in_h: 2,
                in_w: 2,
                kernel: 3,
                stride: 1,
                padding: 1,
                groups: 1,
            },
        };
        let blocks = [spec];
        let attention_shape = UnetAttentionShape {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            groups: 1,
        };
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let w1 = [0.1_f32, 0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8];
        let w2 = [0.5_f32, -0.6, 0.7, 0.8];
        let config = UnetPredictConfig {
            timestep: TimestepWeights { w1: &w1, w2: &w2 },
            timestep_dim: 4,
            timestep_hidden: w1.len() / 4, // 4 embedding columns, 2 hidden rows
            blocks: &blocks,
            attention: UnetAttentionWeights {
                gamma: &gamma,
                beta: &beta,
                wq: &identity,
                wk: &identity,
                wv: &identity,
                wo: &identity,
            },
            attention_shape,
            eps: 1e-5,
        };
        let latent: Vec<f32> = (0..8)
            .map(|i| f32::from(u8::try_from(i).expect("0..8")) * 0.1 - 0.4)
            .collect();
        let conditioning = [0.3_f32, -0.2];

        let predictor = UnetPredictor::new(&device, 1).expect("predictor");
        let actual = predictor
            .predict(&latent, 5, &conditioning, &map, &path, &config)
            .expect("predict");

        let time = TimestepEmbed::new(Arc::clone(&device))
            .expect("timestep")
            .run(5, &TimestepWeights { w1: &w1, w2: &w2 }, 4, 2, 2)
            .expect("time");
        let injected: Vec<f32> = time.iter().zip(&conditioning).map(|(t, c)| t + c).collect();
        let mut conditioned_latent = latent.clone();
        for n in 0..1 {
            for position in 0..4 {
                for (channel, injection) in injected.iter().enumerate() {
                    let index = (n * 2 + channel) * 4 + position;
                    conditioned_latent[index] += injection;
                }
            }
        }
        let hidden = UnetStack::new(&device, 1)
            .expect("stack")
            .run(&conditioned_latent, &map, &path, &blocks, 1e-5)
            .expect("stack run");
        let expected = UnetAttentionBlock::new(device)
            .expect("attention")
            .run(
                &hidden,
                &UnetAttentionWeights {
                    gamma: &gamma,
                    beta: &beta,
                    wq: &identity,
                    wk: &identity,
                    wv: &identity,
                    wo: &identity,
                },
                attention_shape,
                1e-5,
            )
            .expect("attention run");
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn predict_rejects_misaligned_conditioning() {
        let Some(device) = device() else { return };
        let (path, map) = checkpoint();
        let predictor = UnetPredictor::new(&device, 1).expect("predictor");
        let keys = block_keys("block0");
        let spec = UnetStackBlockSpec {
            keys: crate::unet_checkpoint::UnetInputKeys {
                kernel: &keys.0,
                bias: &keys.1,
                gamma: &keys.2,
                beta: &keys.3,
            },
            shape: crate::unet_input::UnetInputShape {
                batch: 1,
                in_c: 2,
                out_c: 2,
                in_h: 2,
                in_w: 2,
                kernel: 3,
                stride: 1,
                padding: 1,
                groups: 1,
            },
        };
        let blocks = [spec];
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let attention = UnetAttentionWeights {
            gamma: &gamma,
            beta: &beta,
            wq: &identity,
            wk: &identity,
            wv: &identity,
            wo: &identity,
        };
        let w1 = [0.1_f32, 0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8];
        let w2 = [0.5_f32, -0.6, 0.7, 0.8];
        let config = UnetPredictConfig {
            timestep: TimestepWeights { w1: &w1, w2: &w2 },
            timestep_dim: 4,
            timestep_hidden: 2,
            blocks: &blocks,
            attention,
            attention_shape: UnetAttentionShape {
                batch: 1,
                channels: 2,
                height: 2,
                width: 2,
                groups: 1,
            },
            eps: 1e-5,
        };
        assert!(matches!(
            predictor.predict(&[0.0; 8], 5, &[0.0; 3], &map, &path, &config),
            Err(UnetStackError::Shape(_))
        ));
        std::fs::remove_file(path).expect("cleanup");
    }
}
