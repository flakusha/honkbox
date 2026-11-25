//! Declarative tensor-name map for a checkpoint family and `UNet` level topology.
//!
//! ponytail: names and presence only, for the reduced graph that executes
//! today; no full-fidelity block traversal is declared here.

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

/// One `ResNet` block: two conv+norm pairs, a timestep projection and an
/// optional 1×1 shortcut when the channel count changes.
#[derive(Debug, Clone, Copy)]
pub struct ResnetKeys<'a> {
    /// First conv.
    pub conv1: ConvKeys<'a>,
    /// First group norm.
    pub norm1: NormKeys<'a>,
    /// Timestep projection (`emb_layers.1` in the loader).
    pub time: ConvKeys<'a>,
    /// Second group norm.
    pub norm2: NormKeys<'a>,
    /// Second conv.
    pub conv2: ConvKeys<'a>,
    /// Channel-changing shortcut, if the loader needs one.
    pub shortcut: Option<ConvKeys<'a>>,
}

/// One level: resnets plus an optional downsampler.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec<'a> {
    /// Human-readable level name for errors.
    pub name: &'a str,
    /// `ResNet` blocks at this level.
    pub resnets: &'a [ResnetKeys<'a>],
    /// Downsampler conv, if this level has one.
    pub downsample: Option<ConvKeys<'a>>,
}

/// Timestep MLP names.
#[derive(Debug, Clone, Copy)]
pub struct TimestepKeys<'a> {
    /// First linear layer.
    pub linear1: ConvKeys<'a>,
    /// Second linear layer.
    pub linear2: ConvKeys<'a>,
}

/// Mid-block cross-attention projection weights and biases.
#[derive(Debug, Clone, Copy)]
pub struct CrossKeys<'a> {
    /// Query projection weight.
    pub q: &'a str,
    /// Query projection bias.
    pub qb: &'a str,
    /// Key projection weight.
    pub k: &'a str,
    /// Key projection bias.
    pub kb: &'a str,
    /// Value projection weight.
    pub v: &'a str,
    /// Value projection bias.
    pub vb: &'a str,
    /// Output projection weight and bias.
    pub out: ConvKeys<'a>,
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
    /// Bottleneck resnets, in execution order.
    pub mid_level: LevelSpec<'a>,
    /// Cross-attention projections between the mid resnets.
    pub cross: CrossKeys<'a>,
    /// 1×1 channel projections after the host nearest upsample.
    pub projections: &'a [ConvKeys<'a>],
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

fn weight_key(role: &str, key: &str, out: &mut KeyList) {
    out.push((role.to_owned(), key.to_owned()));
}

/// All `(role, key)` pairs declared by a spec, in traversal order.
pub fn tensor_keys(spec: &UnetModelSpec<'_>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    conv_keys("entry", spec.conv_in, &mut out);
    for level in spec
        .down_levels
        .iter()
        .chain(std::iter::once(&spec.mid_level))
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
                &format!("{} resnet {index} time", level.name),
                resnet.time,
                &mut out,
            );
            norm_keys(
                &format!("{} resnet {index} norm2", level.name),
                resnet.norm2,
                &mut out,
            );
            conv_keys(
                &format!("{} resnet {index} conv2", level.name),
                resnet.conv2,
                &mut out,
            );
            if let Some(keys) = resnet.shortcut {
                conv_keys(
                    &format!("{} resnet {index} shortcut", level.name),
                    keys,
                    &mut out,
                );
            }
        }
        if let Some(keys) = level.downsample {
            conv_keys(&format!("{} downsample", level.name), keys, &mut out);
        }
    }
    weight_key("cross q", spec.cross.q, &mut out);
    weight_key("cross q bias", spec.cross.qb, &mut out);
    weight_key("cross k", spec.cross.k, &mut out);
    weight_key("cross k bias", spec.cross.kb, &mut out);
    weight_key("cross v", spec.cross.v, &mut out);
    weight_key("cross v bias", spec.cross.vb, &mut out);
    conv_keys("cross out", spec.cross.out, &mut out);
    for (index, keys) in spec.projections.iter().enumerate() {
        conv_keys(&format!("projection {index}"), *keys, &mut out);
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

// Where `CheckpointMap` has an alias the spec uses the diffusers-style name;
// later levels use the loader's original name without `model.diffusion_model.`.
static DOWN_LEVELS: [LevelSpec<'static>; 3] = [
    LevelSpec {
        name: "down level 0",
        resnets: &[ResnetKeys {
            conv1: ConvKeys {
                weight: "down_blocks.0.resnets.0.conv1.weight",
                bias: "down_blocks.0.resnets.0.conv1.bias",
            },
            norm1: NormKeys {
                weight: "down_blocks.0.resnets.0.norm1.weight",
                bias: "down_blocks.0.resnets.0.norm1.bias",
            },
            time: ConvKeys {
                weight: "down_blocks.0.resnets.0.emb_layers.1.weight",
                bias: "down_blocks.0.resnets.0.emb_layers.1.bias",
            },
            norm2: NormKeys {
                weight: "down_blocks.0.resnets.0.norm2.weight",
                bias: "down_blocks.0.resnets.0.norm2.bias",
            },
            conv2: ConvKeys {
                weight: "down_blocks.0.resnets.0.conv2.weight",
                bias: "down_blocks.0.resnets.0.conv2.bias",
            },
            shortcut: None,
        }],
        downsample: Some(ConvKeys {
            weight: "down_blocks.0.downsamplers.0.conv.weight",
            bias: "down_blocks.0.downsamplers.0.conv.bias",
        }),
    },
    LevelSpec {
        name: "down level 1",
        resnets: &[ResnetKeys {
            conv1: ConvKeys {
                weight: "input_blocks.4.0.conv1.weight",
                bias: "input_blocks.4.0.conv1.bias",
            },
            norm1: NormKeys {
                weight: "input_blocks.4.0.norm1.weight",
                bias: "input_blocks.4.0.norm1.bias",
            },
            time: ConvKeys {
                weight: "input_blocks.4.0.emb_layers.1.weight",
                bias: "input_blocks.4.0.emb_layers.1.bias",
            },
            norm2: NormKeys {
                weight: "input_blocks.4.0.norm2.weight",
                bias: "input_blocks.4.0.norm2.bias",
            },
            conv2: ConvKeys {
                weight: "input_blocks.4.0.conv2.weight",
                bias: "input_blocks.4.0.conv2.bias",
            },
            shortcut: Some(ConvKeys {
                weight: "input_blocks.4.0.skip_connection.weight",
                bias: "input_blocks.4.0.skip_connection.bias",
            }),
        }],
        downsample: Some(ConvKeys {
            weight: "input_blocks.6.0.op.weight",
            bias: "input_blocks.6.0.op.bias",
        }),
    },
    LevelSpec {
        name: "down level 2",
        resnets: &[ResnetKeys {
            conv1: ConvKeys {
                weight: "input_blocks.7.0.conv1.weight",
                bias: "input_blocks.7.0.conv1.bias",
            },
            norm1: NormKeys {
                weight: "input_blocks.7.0.norm1.weight",
                bias: "input_blocks.7.0.norm1.bias",
            },
            time: ConvKeys {
                weight: "input_blocks.7.0.emb_layers.1.weight",
                bias: "input_blocks.7.0.emb_layers.1.bias",
            },
            norm2: NormKeys {
                weight: "input_blocks.7.0.norm2.weight",
                bias: "input_blocks.7.0.norm2.bias",
            },
            conv2: ConvKeys {
                weight: "input_blocks.7.0.conv2.weight",
                bias: "input_blocks.7.0.conv2.bias",
            },
            shortcut: Some(ConvKeys {
                weight: "input_blocks.7.0.skip_connection.weight",
                bias: "input_blocks.7.0.skip_connection.bias",
            }),
        }],
        downsample: Some(ConvKeys {
            weight: "input_blocks.9.0.op.weight",
            bias: "input_blocks.9.0.op.bias",
        }),
    },
];

static MID_RESNETS: [ResnetKeys<'static>; 2] = [
    ResnetKeys {
        conv1: ConvKeys {
            weight: "mid_block.resnets.0.conv1.weight",
            bias: "mid_block.resnets.0.conv1.bias",
        },
        norm1: NormKeys {
            weight: "mid_block.resnets.0.norm1.weight",
            bias: "mid_block.resnets.0.norm1.bias",
        },
        time: ConvKeys {
            weight: "mid_block.resnets.0.emb_layers.1.weight",
            bias: "mid_block.resnets.0.emb_layers.1.bias",
        },
        norm2: NormKeys {
            weight: "mid_block.resnets.0.norm2.weight",
            bias: "mid_block.resnets.0.norm2.bias",
        },
        conv2: ConvKeys {
            weight: "mid_block.resnets.0.conv2.weight",
            bias: "mid_block.resnets.0.conv2.bias",
        },
        shortcut: None,
    },
    ResnetKeys {
        conv1: ConvKeys {
            weight: "middle_block.2.conv1.weight",
            bias: "middle_block.2.conv1.bias",
        },
        norm1: NormKeys {
            weight: "middle_block.2.norm1.weight",
            bias: "middle_block.2.norm1.bias",
        },
        time: ConvKeys {
            weight: "middle_block.2.emb_layers.1.weight",
            bias: "middle_block.2.emb_layers.1.bias",
        },
        norm2: NormKeys {
            weight: "middle_block.2.norm2.weight",
            bias: "middle_block.2.norm2.bias",
        },
        conv2: ConvKeys {
            weight: "middle_block.2.conv2.weight",
            bias: "middle_block.2.conv2.bias",
        },
        shortcut: None,
    },
];

static CROSS: CrossKeys<'static> = CrossKeys {
    q: "middle_block.1.transformer_blocks.0.attn2.to_q.weight",
    qb: "middle_block.1.transformer_blocks.0.attn2.to_q.bias",
    k: "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
    kb: "middle_block.1.transformer_blocks.0.attn2.to_k.bias",
    v: "middle_block.1.transformer_blocks.0.attn2.to_v.weight",
    vb: "middle_block.1.transformer_blocks.0.attn2.to_v.bias",
    out: ConvKeys {
        weight: "middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
        bias: "middle_block.1.transformer_blocks.0.attn2.to_out.0.bias",
    },
};

static PROJECTIONS: [ConvKeys<'static>; 2] = [
    ConvKeys {
        weight: "output_blocks.7.0.skip_connection.weight",
        bias: "output_blocks.7.0.skip_connection.bias",
    },
    ConvKeys {
        weight: "output_blocks.10.0.skip_connection.weight",
        bias: "output_blocks.10.0.skip_connection.bias",
    },
];

/// Keys for the reduced graph that `unet_real` actually loads and predicts;
/// VAE entry/exit keys cover the reduced decoder in `vae_real`.
///
/// ponytail: reduced topology, not SD1.5 fidelity. Add real up blocks and
/// their skip connections here when full-fidelity parity is required.
pub static SD15_REDUCED: UnetModelSpec<'static> = UnetModelSpec {
    family: "sd15-reduced",
    conv_in: ConvKeys {
        weight: "conv_in.weight",
        bias: "conv_in.bias",
    },
    down_levels: &DOWN_LEVELS,
    mid_level: LevelSpec {
        name: "mid level",
        resnets: &MID_RESNETS,
        downsample: None,
    },
    cross: CROSS,
    projections: &PROJECTIONS,
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
    fn reduced_spec_declares_executed_topology_and_required_keys() {
        assert_eq!(SD15_REDUCED.down_levels.len(), 3);
        for level in SD15_REDUCED.down_levels {
            assert_eq!(level.resnets.len(), 1);
            assert!(level.downsample.is_some());
        }
        assert_eq!(SD15_REDUCED.mid_level.resnets.len(), 2);
        assert_eq!(SD15_REDUCED.projections.len(), 2);
        let keys = tensor_keys(&SD15_REDUCED);
        for key in [
            "down_blocks.0.resnets.0.emb_layers.1.bias",
            "input_blocks.4.0.skip_connection.weight",
            "middle_block.1.transformer_blocks.0.attn2.to_q.bias",
            "middle_block.1.transformer_blocks.0.attn2.to_k.bias",
            "middle_block.1.transformer_blocks.0.attn2.to_v.bias",
            "middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
            "output_blocks.10.0.skip_connection.weight",
            "time_embedding.linear_2.bias",
            "decoder.conv_out.bias",
        ] {
            assert!(
                keys.iter().any(|(_, declared)| declared == key),
                "missing {key}"
            );
        }
    }

    #[test]
    fn validate_reports_missing_tensor_role_and_key() {
        let mut keys = tensor_keys(&SD15_REDUCED);
        let (_file, map) = checkpoint(&keys);
        assert!(validate(&SD15_REDUCED, &map).is_ok());

        keys.retain(|(_, key)| key != "time_embedding.linear_2.bias");
        let (_file, map) = checkpoint(&keys);
        match validate(&SD15_REDUCED, &map) {
            Err(SpecError::Missing { role, key }) => {
                assert_eq!(role, "timestep linear 2");
                assert_eq!(key, "time_embedding.linear_2.bias");
            }
            Ok(()) => panic!("missing tensor accepted"),
        }
    }

    #[test]
    fn validate_accepts_single_file_prefixed_keys() {
        let keys: Vec<_> = tensor_keys(&SD15_REDUCED)
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
        assert!(validate(&SD15_REDUCED, &map).is_ok());
        assert!(map.get("conv_in.weight").is_none());
    }
}
