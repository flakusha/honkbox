//! One-level `UNet` traversal: downsample → predict → upsample with skip → VAE.
//!
//! ponytail: a single down/up level and one VAE decoder block. Multi-scale
//! skips, stacked VAE blocks and checkpoint parity need real checkpoint specs.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    conv_transpose::ConvTransposeShape,
    conv2d::ConvShape,
    ddim_sampler::DdimSampler,
    device::VulkanDevice,
    unet_downsample::{UnetDownsample, UnetDownsampleKeys, UnetDownsampleShape},
    unet_predict::{UnetPredictConfig, UnetPredictor},
    unet_stack::UnetStackError,
    unet_upsample::{UnetUpsample, UnetUpsampleKeys, UnetUpsampleShape},
    vae_decode::{VaeDecode, VaeWeights},
};
use std::path::Path;
use std::sync::Arc;

/// All caller-supplied weights and geometry for one traversal.
pub struct UnetTraversalConfig<'a> {
    /// Encoder downsample block.
    pub down_keys: UnetDownsampleKeys<'a>,
    /// Encoder geometry; output must match the predictor latent.
    pub down_shape: UnetDownsampleShape,
    /// Bottleneck predictor configuration.
    pub predict: UnetPredictConfig<'a>,
    /// Decoder upsample block; skip is the encoder output.
    pub up_keys: UnetUpsampleKeys<'a>,
    /// Decoder geometry; output must match the VAE input.
    pub up_shape: UnetUpsampleShape,
    /// VAE transposed-convolution geometry.
    pub vae_up_shape: ConvTransposeShape,
    /// VAE projection geometry.
    pub vae_proj_shape: ConvShape,
    /// VAE group count.
    pub vae_groups: usize,
    /// VAE block weights.
    pub vae_weights: VaeWeights<'a>,
    /// Shared group-norm epsilon.
    pub eps: f32,
}

fn axes(actual: [usize; 3], expected: [usize; 3], label: &str) -> Result<(), UnetStackError> {
    if actual == expected {
        Ok(())
    } else {
        Err(UnetStackError::Shape(format!(
            "{label}: got {actual:?}, expected {expected:?}"
        )))
    }
}

/// Schedule, timestep mapping and text tokens for one sampling walk.
pub struct SampleConfig<'a> {
    /// One timestep index per interval in `cumprod`.
    pub timesteps: &'a [usize],
    /// `[context_rows, channels]` token matrix.
    pub context: &'a [f32],
    /// Cumulative alphas, non-increasing, with `steps + 1` entries.
    pub cumprod: &'a [f32],
}

fn validate(config: &UnetTraversalConfig<'_>) -> Result<(), UnetStackError> {
    let (down_h, down_w) = config.down_shape.output()?;
    let attention = config.predict.attention_shape;
    let first = config
        .predict
        .blocks
        .first()
        .ok_or_else(|| UnetStackError::Shape("traversal: no bottleneck blocks".into()))?;
    axes(
        [
            config.down_shape.batch,
            config.down_shape.out_c,
            down_h * down_w,
        ],
        [
            attention.batch,
            attention.channels,
            attention.height * attention.width,
        ],
        "traversal: encoder output vs bottleneck",
    )?;
    if first.shape.in_c != attention.channels {
        return Err(UnetStackError::Shape(
            "traversal: bottleneck block channels mismatch".into(),
        ));
    }
    let (up_h, up_w) = config.up_shape.output()?;
    axes(
        [config.up_shape.batch, config.up_shape.out_c, up_h * up_w],
        [
            config.down_shape.batch,
            config.down_shape.in_c,
            config.down_shape.in_h * config.down_shape.in_w,
        ],
        "traversal: decoder output vs skip",
    )?;
    if config.up_shape.in_c != attention.channels {
        return Err(UnetStackError::Shape(
            "traversal: decoder input channels mismatch".into(),
        ));
    }
    axes(
        [
            config.vae_up_shape.batch,
            config.vae_up_shape.in_c,
            config.vae_up_shape.in_h * config.vae_up_shape.in_w,
        ],
        [config.up_shape.batch, config.up_shape.out_c, up_h * up_w],
        "traversal: VAE input vs decoder output",
    )?;
    axes(
        [
            config.vae_proj_shape.batch,
            config.vae_proj_shape.in_c,
            config.vae_proj_shape.in_h * config.vae_proj_shape.in_w,
        ],
        [
            config.vae_up_shape.batch,
            config.vae_up_shape.out_c,
            config.vae_up_shape.out_h * config.vae_up_shape.out_w,
        ],
        "traversal: VAE projection vs upsample",
    )
}

/// Composed one-level `UNet` traversal over one checkpoint.
pub struct UnetTraversal {
    device: Arc<VulkanDevice>,
    down: UnetDownsample,
    predict: UnetPredictor,
    up: UnetUpsample,
    vae: VaeDecode,
}

impl UnetTraversal {
    /// Instantiate every stage on the selected device.
    pub fn new(device: &Arc<VulkanDevice>, blocks: usize) -> Result<Self, VkError> {
        Ok(Self {
            device: Arc::clone(device),
            down: UnetDownsample::new(Arc::clone(device))?,
            predict: UnetPredictor::new(device, blocks)?,
            up: UnetUpsample::new(Arc::clone(device))?,
            vae: VaeDecode::new(Arc::clone(device))?,
        })
    }

    /// Run latent through the encoder, bottleneck, decoder and VAE block.
    pub fn run(
        &self,
        latent: &[f32],
        timestep: usize,
        conditioning: &[f32],
        map: &CheckpointMap,
        path: &Path,
        config: &UnetTraversalConfig<'_>,
    ) -> Result<Vec<f32>, UnetStackError> {
        validate(config)?;

        let downsampled = self
            .down
            .run(latent, map, path, &config.down_keys, config.down_shape)?;
        let predicted = self.predict.predict(
            &downsampled,
            timestep,
            conditioning,
            map,
            path,
            &config.predict,
        )?;
        let upsampled = self.up.run(
            &predicted,
            latent,
            map,
            path,
            &config.up_keys,
            config.up_shape,
        )?;
        Ok(self.vae.run(
            &upsampled,
            config.vae_up_shape,
            config.vae_groups,
            config.eps,
            config.vae_proj_shape,
            &config.vae_weights,
        )?)
    }

    /// Walk an eta=0 DDIM schedule, predicting through the same traversal.
    ///
    /// `timesteps` has one entry per interval in `cumprod`; decoding happens
    /// once after the last step. The closure reuses down/predict/up so the
    /// diffusion state keeps the latent's full spatial shape.
    pub fn sample(
        &self,
        initial: &[f32],
        schedule: &SampleConfig<'_>,
        map: &CheckpointMap,
        path: &Path,
        config: &UnetTraversalConfig<'_>,
    ) -> Result<Vec<f32>, UnetStackError> {
        validate(config)?;
        if schedule.timesteps.len() + 1 != schedule.cumprod.len() {
            return Err(UnetStackError::Shape(
                "sample: need one timestep per schedule interval".into(),
            ));
        }
        let sampler = DdimSampler::new(Arc::clone(&self.device))?;
        let latent = sampler.run(initial, schedule.cumprod, |current, index| {
            let timestep = schedule.timesteps[index];
            let downsampled =
                self.down
                    .run(current, map, path, &config.down_keys, config.down_shape)?;
            let predicted = self.predict.predict(
                &downsampled,
                timestep,
                schedule.context,
                map,
                path,
                &config.predict,
            )?;
            Ok(self.up.run(
                &predicted,
                current,
                map,
                path,
                &config.up_keys,
                config.up_shape,
            )?)
        })?;
        Ok(self.vae.run(
            &latent,
            config.vae_up_shape,
            config.vae_groups,
            config.eps,
            config.vae_proj_shape,
            &config.vae_weights,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VkError;
    use crate::unet_checkpoint::UnetInputKeys;
    use crate::unet_input::UnetInputShape;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn checkpoint() -> (std::path::PathBuf, CheckpointMap) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, count, dims) in [
            ("down.kernel", 4, vec![1usize, 1, 2, 2]),
            ("down.bias", 1, vec![1]),
            ("block0.kernel", 9, vec![1, 1, 3, 3]),
            ("block0.bias", 1, vec![1]),
            ("block0.gamma", 1, vec![1]),
            ("block0.beta", 1, vec![1]),
            ("up.kernel", 4, vec![1, 1, 2, 2]),
            ("up.bias", 1, vec![1]),
        ] {
            let start = payload.len();
            for index in 0..count {
                let value = f32::from(u8::try_from(index % 5).expect("0..5")) * 0.1 - 0.1;
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
            "honkbox-ash-traversal-{}-{stamp}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        (path.clone(), CheckpointMap::read(&path).expect("map"))
    }

    static BLOCKS: [crate::unet_stack::UnetStackBlockSpec<'static>; 1] =
        [crate::unet_stack::UnetStackBlockSpec {
            keys: UnetInputKeys {
                kernel: "block0.kernel",
                bias: "block0.bias",
                gamma: "block0.gamma",
                beta: "block0.beta",
            },
            shape: UnetInputShape {
                batch: 1,
                in_c: 1,
                out_c: 1,
                in_h: 4,
                in_w: 4,
                kernel: 3,
                stride: 1,
                padding: 1,
                groups: 1,
            },
        }];
    static NORM: [f32; 1] = [1.0];
    static ZERO: [f32; 1] = [0.0];
    static PROJECTION: [f32; 1] = [0.5];
    static W1: [f32; 4] = [0.1, 0.2, -0.3, 0.4];
    static W2: [f32; 2] = [0.5, -0.6];
    static ONE: [f32; 1] = [1.0];

    struct Fixture {
        path: std::path::PathBuf,
        map: CheckpointMap,
        down_shape: UnetDownsampleShape,
        up_shape: UnetUpsampleShape,
        vae_up_shape: ConvTransposeShape,
        vae_proj_shape: ConvShape,
        vae_weights_storage: Vec<f32>,
    }

    impl Fixture {
        fn new() -> Self {
            let (path, map) = checkpoint();
            Self {
                path,
                map,
                down_shape: UnetDownsampleShape {
                    batch: 1,
                    in_c: 1,
                    out_c: 1,
                    in_h: 8,
                    in_w: 8,
                    kernel: 2,
                    stride: 2,
                    padding: 0,
                },
                up_shape: UnetUpsampleShape {
                    batch: 1,
                    in_c: 1,
                    out_c: 1,
                    in_h: 4,
                    in_w: 4,
                    kernel: 2,
                    stride: 2,
                    padding: 0,
                },
                vae_up_shape: ConvTransposeShape {
                    batch: 1,
                    in_c: 1,
                    out_c: 1,
                    in_h: 8,
                    in_w: 8,
                    out_h: 16,
                    out_w: 16,
                    kernel: 2,
                    stride: 2,
                    padding: 0,
                },
                vae_proj_shape: ConvShape {
                    batch: 1,
                    in_c: 1,
                    out_c: 1,
                    in_h: 16,
                    in_w: 16,
                    out_h: 16,
                    out_w: 16,
                    kernel: 1,
                    stride: 1,
                    padding: 0,
                },
                vae_weights_storage: vec![0.5_f32, -0.25, 0.75, 1.0],
            }
        }

        fn config(&self) -> UnetTraversalConfig<'_> {
            UnetTraversalConfig {
                down_keys: UnetDownsampleKeys {
                    kernel: "down.kernel",
                    bias: "down.bias",
                },
                down_shape: self.down_shape,
                predict: UnetPredictConfig {
                    timestep: crate::timestep::TimestepWeights { w1: &W1, w2: &W2 },
                    timestep_dim: 2,
                    timestep_hidden: 2,
                    blocks: &BLOCKS,
                    attention: crate::unet_attention::UnetAttentionWeights {
                        gamma: &NORM,
                        beta: &ZERO,
                        wq: &ONE,
                        wk: &ONE,
                        wv: &ONE,
                        wo: &ONE,
                    },
                    attention_shape: crate::unet_attention::UnetAttentionShape {
                        batch: 1,
                        channels: 1,
                        height: 4,
                        width: 4,
                        groups: 1,
                    },
                    cross: crate::unet_predict::UnetCrossConfig {
                        weights: crate::cross_attention::CrossAttentionWeights {
                            wq: &ONE,
                            wkv: &[1.0_f32, 1.0],
                            wo: &ONE,
                        },
                        context_rows: 1,
                    },
                    eps: 1e-5,
                },
                up_keys: UnetUpsampleKeys {
                    kernel: "up.kernel",
                    bias: "up.bias",
                },
                up_shape: self.up_shape,
                vae_up_shape: self.vae_up_shape,
                vae_proj_shape: self.vae_proj_shape,
                vae_groups: 1,
                vae_weights: VaeWeights {
                    up_weights: &self.vae_weights_storage,
                    up_bias: &ZERO,
                    norm_gamma: &NORM,
                    norm_beta: &ZERO,
                    proj_weights: &PROJECTION,
                    proj_bias: &ZERO,
                },
                eps: 1e-5,
            }
        }
    }

    #[test]
    fn traversal_matches_manual_stage_sequence() {
        let Some(device) = device() else { return };
        let fixture = Fixture::new();
        let config = fixture.config();
        let latent: Vec<f32> = (0..64)
            .map(|i| f32::from(u8::try_from(i).expect("0..64")) * 0.05 - 0.3)
            .collect();
        let conditioning = [0.2_f32];

        let actual = UnetTraversal::new(&device, 1)
            .expect("traversal")
            .run(
                &latent,
                5,
                &conditioning,
                &fixture.map,
                &fixture.path,
                &config,
            )
            .expect("run");

        let down = UnetDownsample::new(Arc::clone(&device))
            .expect("down")
            .run(
                &latent,
                &fixture.map,
                &fixture.path,
                &config.down_keys,
                config.down_shape,
            )
            .expect("down run");
        let predicted = UnetPredictor::new(&device, 1)
            .expect("predictor")
            .predict(
                &down,
                5,
                &conditioning,
                &fixture.map,
                &fixture.path,
                &config.predict,
            )
            .expect("predict");
        let up = UnetUpsample::new(Arc::clone(&device))
            .expect("up")
            .run(
                &predicted,
                &latent,
                &fixture.map,
                &fixture.path,
                &config.up_keys,
                config.up_shape,
            )
            .expect("up run");
        let expected = VaeDecode::new(device)
            .expect("vae")
            .run(
                &up,
                config.vae_up_shape,
                1,
                1e-5,
                config.vae_proj_shape,
                &config.vae_weights,
            )
            .expect("vae run");
        assert_eq!(actual.len(), 256);
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        std::fs::remove_file(&fixture.path).expect("cleanup");
    }

    #[test]
    fn sample_uses_each_timestep_and_decodes_once() {
        let Some(device) = device() else { return };
        let fixture = Fixture::new();
        let config = fixture.config();
        let latent: Vec<f32> = (0..64)
            .map(|i| f32::from(u8::try_from(i).expect("0..64")) * 0.05 - 0.3)
            .collect();
        let cumprod = [0.9_f32, 0.6, 0.3];
        let context = [0.2_f32];
        let traversal = UnetTraversal::new(&device, 1).expect("traversal");
        let stepped = traversal
            .sample(
                &latent,
                &SampleConfig {
                    timesteps: &[5, 4],
                    context: &context,
                    cumprod: &cumprod,
                },
                &fixture.map,
                &fixture.path,
                &config,
            )
            .expect("stepped sample");
        let constant = traversal
            .sample(
                &latent,
                &SampleConfig {
                    timesteps: &[5, 5],
                    context: &context,
                    cumprod: &cumprod,
                },
                &fixture.map,
                &fixture.path,
                &config,
            )
            .expect("constant sample");
        assert_eq!(stepped.len(), 256);
        assert_ne!(
            stepped, constant,
            "each timestep must change the prediction"
        );
        assert!(matches!(
            traversal.sample(
                &latent,
                &SampleConfig {
                    timesteps: &[5],
                    context: &context,
                    cumprod: &cumprod
                },
                &fixture.map,
                &fixture.path,
                &config
            ),
            Err(UnetStackError::Shape(_))
        ));
        std::fs::remove_file(&fixture.path).expect("cleanup");
    }

    #[test]
    fn traversal_rejects_mismatched_skip_shape() {
        let Some(device) = device() else { return };
        let mut fixture = Fixture::new();
        fixture.up_shape.stride = 1;
        let config = fixture.config();
        let traversal = UnetTraversal::new(&device, 1).expect("traversal");
        assert!(matches!(
            traversal.run(&[0.0; 64], 5, &[0.0], &fixture.map, &fixture.path, &config),
            Err(UnetStackError::Shape(_))
        ));
        std::fs::remove_file(&fixture.path).expect("cleanup");
    }
}
