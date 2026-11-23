//! Declarative tensor-name map for a checkpoint family and `UNet` level topology.
//!
//! ponytail: names and presence only. Shape inference, key remapping and
//! server dispatch come after a real checkpoint confirms this naming.

use crate::checkpoint::CheckpointMap;
use std::error::Error;
use std::fmt;

/// Convolution kernel + bias keys.
#[derive(Debug, Clone, Copy)]
pub struct ConvKeys<'a> {
    /// `[out_c, in_c, k, k]` weight name.
    pub weight: &'a str,
    /// `[out_c]` bias name.
    pub bias: &'a str,
}

/// Group-norm gain/shift keys.
#[derive(Debug, Clone, Copy)]
pub struct NormKeys<'a> {
    /// Gain name.
    pub weight: &'a str,
    /// Shift name.
    pub bias: &'a str,
}

/// One `ResNet` block: two conv+norm pairs.
#[derive(Debug, Clone, Copy)]
pub struct ResnetKeys<'a> {
    /// First conv.
    pub conv1: ConvKeys<'a>,
    /// First group norm.
    pub norm1: NormKeys<'a>,
    /// Second conv.
    pub conv2: ConvKeys<'a>,
    /// Second group norm.
    pub norm2: NormKeys<'a>,
}

/// One level: resnets plus an optional downsampler or upsampler.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec<'a> {
    /// Human-readable level name for errors.
    pub name: &'a str,
    /// `ResNet` blocks at this level.
    pub resnets: &'a [ResnetKeys<'a>],
    /// Downsampler conv, if this level has one.
    pub downsample: Option<ConvKeys<'a>>,
    /// Upsampler conv, if this level has one.
    pub upsample: Option<ConvKeys<'a>>,
}

/// Timestep MLP names.
#[derive(Debug, Clone, Copy)]
pub struct TimestepKeys<'a> {
    /// First linear layer.
    pub linear1: ConvKeys<'a>,
    /// Second linear layer.
    pub linear2: ConvKeys<'a>,
}

/// VAE decoder entry/exit names (one block).
#[derive(Debug, Clone, Copy)]
pub struct VaeKeys<'a> {
    /// Decoder entry convolution.
    pub conv_in: ConvKeys<'a>,
    /// Decoder output norm and convolution.
    pub norm_out: NormKeys<'a>,
    /// Decoder output convolution.
    pub conv_out: ConvKeys<'a>,
}

/// Tensor names and topology for one checkpoint family.
#[derive(Debug, Clone, Copy)]
pub struct UnetModelSpec<'a> {
    /// Checkpoint family identifier.
    pub family: &'a str,
    /// `UNet` entry convolution.
    pub conv_in: ConvKeys<'a>,
    /// Encoder levels, in execution order.
    pub down_levels: &'a [LevelSpec<'a>],
    /// Bottleneck level.
    pub mid_level: LevelSpec<'a>,
    /// Decoder levels, in execution order.
    pub up_levels: &'a [LevelSpec<'a>],
    /// `UNet` output norm and convolution.
    pub norm_out: NormKeys<'a>,
    /// `UNet` output convolution.
    pub conv_out: ConvKeys<'a>,
    /// Timestep embedding MLP.
    pub timestep: TimestepKeys<'a>,
    /// VAE decoder block.
    pub vae: VaeKeys<'a>,
}

/// Missing or unexpected checkpoint metadata.
#[derive(Debug)]
pub enum SpecError {
    /// A named tensor is absent.
    Missing {
        /// Semantic role from the spec.
        role: String,
        /// Checkpoint key that was not found.
        key: String,
    },
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { role, key } => write!(f, "{role}: checkpoint missing '{key}'"),
        }
    }
}

impl Error for SpecError {}

type KeyList = Vec<(String, String)>;

fn conv_keys(role: &str, keys: ConvKeys<'_>, out: &mut KeyList) {
    out.push((role.to_owned(), keys.weight.to_owned()));
    out.push((role.to_owned(), keys.bias.to_owned()));
}

fn norm_keys(role: &str, keys: NormKeys<'_>, out: &mut KeyList) {
    out.push((role.to_owned(), keys.weight.to_owned()));
    out.push((role.to_owned(), keys.bias.to_owned()));
}

/// All `(role, key)` pairs declared by a spec, in traversal order.
pub fn tensor_keys(spec: &UnetModelSpec<'_>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    conv_keys("entry", spec.conv_in, &mut out);
    for level in spec
        .down_levels
        .iter()
        .chain(std::iter::once(&spec.mid_level))
        .chain(spec.up_levels.iter())
    {
        for (index, resnet) in level.resnets.iter().enumerate() {
            conv_keys(
                &format!("{} resnet {index} conv1", level.name),
                resnet.conv1,
                &mut out,
            );
            norm_keys(
                &format!("{} resnet {index} norm1", level.name),
                resnet.norm1,
                &mut out,
            );
            conv_keys(
                &format!("{} resnet {index} conv2", level.name),
                resnet.conv2,
                &mut out,
            );
            norm_keys(
                &format!("{} resnet {index} norm2", level.name),
                resnet.norm2,
                &mut out,
            );
        }
        if let Some(keys) = level.downsample {
            conv_keys(&format!("{} downsample", level.name), keys, &mut out);
        }
        if let Some(keys) = level.upsample {
            conv_keys(&format!("{} upsample", level.name), keys, &mut out);
        }
    }
    norm_keys("output norm", spec.norm_out, &mut out);
    conv_keys("output conv", spec.conv_out, &mut out);
    conv_keys("timestep linear 1", spec.timestep.linear1, &mut out);
    conv_keys("timestep linear 2", spec.timestep.linear2, &mut out);
    conv_keys("vae entry", spec.vae.conv_in, &mut out);
    norm_keys("vae output norm", spec.vae.norm_out, &mut out);
    conv_keys("vae output conv", spec.vae.conv_out, &mut out);
    out
}

/// Check that every declared tensor exists in `map`.
pub fn validate(spec: &UnetModelSpec<'_>, map: &CheckpointMap) -> Result<(), SpecError> {
    for (role, key) in tensor_keys(spec) {
        if map.get_aliased(&key).is_none() {
            return Err(SpecError::Missing { role, key });
        }
    }
    Ok(())
}

static DOWN_RESNETS: [ResnetKeys<'static>; 2] = [
    ResnetKeys {
        conv1: ConvKeys {
            weight: "down_blocks.0.resnets.0.conv1.weight",
            bias: "down_blocks.0.resnets.0.conv1.bias",
        },
        norm1: NormKeys {
            weight: "down_blocks.0.resnets.0.norm1.weight",
            bias: "down_blocks.0.resnets.0.norm1.bias",
        },
        conv2: ConvKeys {
            weight: "down_blocks.0.resnets.0.conv2.weight",
            bias: "down_blocks.0.resnets.0.conv2.bias",
        },
        norm2: NormKeys {
            weight: "down_blocks.0.resnets.0.norm2.weight",
            bias: "down_blocks.0.resnets.0.norm2.bias",
        },
    },
    ResnetKeys {
        conv1: ConvKeys {
            weight: "down_blocks.0.resnets.1.conv1.weight",
            bias: "down_blocks.0.resnets.1.conv1.bias",
        },
        norm1: NormKeys {
            weight: "down_blocks.0.resnets.1.norm1.weight",
            bias: "down_blocks.0.resnets.1.norm1.bias",
        },
        conv2: ConvKeys {
            weight: "down_blocks.0.resnets.1.conv2.weight",
            bias: "down_blocks.0.resnets.1.conv2.bias",
        },
        norm2: NormKeys {
            weight: "down_blocks.0.resnets.1.norm2.weight",
            bias: "down_blocks.0.resnets.1.norm2.bias",
        },
    },
];

static MID_RESNETS: [ResnetKeys<'static>; 1] = [ResnetKeys {
    conv1: ConvKeys {
        weight: "mid_block.resnets.0.conv1.weight",
        bias: "mid_block.resnets.0.conv1.bias",
    },
    norm1: NormKeys {
        weight: "mid_block.resnets.0.norm1.weight",
        bias: "mid_block.resnets.0.norm1.bias",
    },
    conv2: ConvKeys {
        weight: "mid_block.resnets.0.conv2.weight",
        bias: "mid_block.resnets.0.conv2.bias",
    },
    norm2: NormKeys {
        weight: "mid_block.resnets.0.norm2.weight",
        bias: "mid_block.resnets.0.norm2.bias",
    },
}];

static UP_RESNETS: [ResnetKeys<'static>; 2] = [
    ResnetKeys {
        conv1: ConvKeys {
            weight: "up_blocks.1.resnets.0.conv1.weight",
            bias: "up_blocks.1.resnets.0.conv1.bias",
        },
        norm1: NormKeys {
            weight: "up_blocks.1.resnets.0.norm1.weight",
            bias: "up_blocks.1.resnets.0.norm1.bias",
        },
        conv2: ConvKeys {
            weight: "up_blocks.1.resnets.0.conv2.weight",
            bias: "up_blocks.1.resnets.0.conv2.bias",
        },
        norm2: NormKeys {
            weight: "up_blocks.1.resnets.0.norm2.weight",
            bias: "up_blocks.1.resnets.0.norm2.bias",
        },
    },
    ResnetKeys {
        conv1: ConvKeys {
            weight: "up_blocks.1.resnets.1.conv1.weight",
            bias: "up_blocks.1.resnets.1.conv1.bias",
        },
        norm1: NormKeys {
            weight: "up_blocks.1.resnets.1.norm1.weight",
            bias: "up_blocks.1.resnets.1.norm1.bias",
        },
        conv2: ConvKeys {
            weight: "up_blocks.1.resnets.1.conv2.weight",
            bias: "up_blocks.1.resnets.1.conv2.bias",
        },
        norm2: NormKeys {
            weight: "up_blocks.1.resnets.1.norm2.weight",
            bias: "up_blocks.1.resnets.1.norm2.bias",
        },
    },
];

static DOWN_LEVELS: [LevelSpec<'static>; 1] = [LevelSpec {
    name: "down level 0",
    resnets: &DOWN_RESNETS,
    downsample: Some(ConvKeys {
        weight: "down_blocks.0.downsamplers.0.conv.weight",
        bias: "down_blocks.0.downsamplers.0.conv.bias",
    }),
    upsample: None,
}];

static UP_LEVELS: [LevelSpec<'static>; 1] = [LevelSpec {
    name: "up level 0",
    resnets: &UP_RESNETS,
    downsample: None,
    upsample: Some(ConvKeys {
        weight: "up_blocks.1.upsamplers.0.conv.weight",
        bias: "up_blocks.1.upsamplers.0.conv.bias",
    }),
}];

/// SD 1.5-style naming for a one-level topology. New model families add a
/// sibling constant; no runtime branching on names is needed.
pub static SD15_SINGLE_LEVEL: UnetModelSpec<'static> = UnetModelSpec {
    family: "sd15-single-level",
    conv_in: ConvKeys {
        weight: "conv_in.weight",
        bias: "conv_in.bias",
    },
    down_levels: &DOWN_LEVELS,
    mid_level: LevelSpec {
        name: "mid level",
        resnets: &MID_RESNETS,
        downsample: None,
        upsample: None,
    },
    up_levels: &UP_LEVELS,
    norm_out: NormKeys {
        weight: "conv_norm_out.weight",
        bias: "conv_norm_out.bias",
    },
    conv_out: ConvKeys {
        weight: "conv_out.weight",
        bias: "conv_out.bias",
    },
    timestep: TimestepKeys {
        linear1: ConvKeys {
            weight: "time_embedding.linear_1.weight",
            bias: "time_embedding.linear_1.bias",
        },
        linear2: ConvKeys {
            weight: "time_embedding.linear_2.weight",
            bias: "time_embedding.linear_2.bias",
        },
    },
    vae: VaeKeys {
        conv_in: ConvKeys {
            weight: "decoder.conv_in.weight",
            bias: "decoder.conv_in.bias",
        },
        norm_out: NormKeys {
            weight: "decoder.conv_norm_out.weight",
            bias: "decoder.conv_norm_out.bias",
        },
        conv_out: ConvKeys {
            weight: "decoder.conv_out.weight",
            bias: "decoder.conv_out.bias",
        },
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Owns one unique temp file; removal happens on every exit path.
    struct TempFile(std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn checkpoint(keys: &[(String, String)]) -> (TempFile, CheckpointMap) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (_, key) in keys {
            let start = payload.len();
            payload.extend_from_slice(&0.5_f32.to_le_bytes());
            let end = payload.len();
            header.insert(
                key.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
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
            "honkbox-ash-model-spec-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        (
            TempFile(path.clone()),
            CheckpointMap::read(&path).expect("map"),
        )
    }

    #[test]
    fn sd15_spec_declares_level_topology_and_required_keys() {
        assert_eq!(SD15_SINGLE_LEVEL.down_levels.len(), 1);
        assert_eq!(SD15_SINGLE_LEVEL.down_levels[0].resnets.len(), 2);
        assert_eq!(SD15_SINGLE_LEVEL.up_levels[0].resnets.len(), 2);
        assert!(SD15_SINGLE_LEVEL.down_levels[0].downsample.is_some());
        assert!(SD15_SINGLE_LEVEL.up_levels[0].upsample.is_some());
        let keys = tensor_keys(&SD15_SINGLE_LEVEL);
        assert!(
            keys.iter()
                .any(|(_, key)| key == "mid_block.resnets.0.norm2.weight")
        );
        assert!(keys.iter().any(|(_, key)| key == "decoder.conv_out.bias"));
    }

    #[test]
    fn validate_reports_missing_tensor_role_and_key() {
        let mut keys = tensor_keys(&SD15_SINGLE_LEVEL);
        let (_file, map) = checkpoint(&keys);
        assert!(validate(&SD15_SINGLE_LEVEL, &map).is_ok());

        keys.retain(|(_, key)| key != "time_embedding.linear_2.bias");
        let (_file, map) = checkpoint(&keys);
        match validate(&SD15_SINGLE_LEVEL, &map) {
            Err(SpecError::Missing { role, key }) => {
                assert_eq!(role, "timestep linear 2");
                assert_eq!(key, "time_embedding.linear_2.bias");
            }
            Ok(()) => panic!("missing tensor accepted"),
        }
    }

    #[test]
    fn validate_accepts_single_file_prefixed_keys() {
        let keys: Vec<_> = tensor_keys(&SD15_SINGLE_LEVEL)
            .into_iter()
            .map(|(role, key)| {
                let prefix = if key.starts_with("decoder.") {
                    "first_stage_model."
                } else {
                    "model.diffusion_model."
                };
                (role, format!("{prefix}{key}"))
            })
            .collect();
        let (_file, map) = checkpoint(&keys);
        assert!(validate(&SD15_SINGLE_LEVEL, &map).is_ok());
        assert!(map.get("conv_in.weight").is_none());
    }
}
