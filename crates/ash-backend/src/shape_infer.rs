//! Infer traversal stage shapes from checkpoint tensor metadata.
//!
//! ponytail: only channels/kernel/linear widths come from metadata. Spatial
//! sizes still come from the caller's sampler; a full model loader needs real
//! checkpoint-level topology and is not attempted here.

use crate::{checkpoint::CheckpointMap, model_spec::UnetModelSpec};
use std::error::Error;
use std::fmt;

/// Channels and square kernel side from one `[out_c, in_c, k, k]` weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvInfo {
    /// Input channels.
    pub in_c: usize,
    /// Output channels.
    pub out_c: usize,
    /// Square kernel side.
    pub kernel: usize,
}

/// Shapes needed to configure one `UNet` traversal.
#[derive(Debug, Clone, Copy)]
pub struct UnetShapes {
    /// Latent channels from `conv_in`.
    pub latent_channels: usize,
    /// Entry convolution kernel side.
    pub entry_kernel: usize,
    /// Encoder downsampler, if the level defines one.
    pub down: Option<ConvInfo>,
    /// Mid-level channel width.
    pub mid_channels: usize,
    /// Decoder upsampler, if the level defines one.
    pub up: Option<ConvInfo>,
    /// Output channel width.
    pub output_channels: usize,
    /// Timestep embedding width.
    pub timestep_dim: usize,
    /// Timestep MLP hidden width.
    pub timestep_hidden: usize,
    /// Timestep output width.
    pub timestep_channels: usize,
    /// VAE decoder entry convolution.
    pub vae_entry: ConvInfo,
    /// VAE decoder exit convolution.
    pub vae_exit: ConvInfo,
}

/// Bad or inconsistent tensor metadata.
#[derive(Debug)]
pub enum ShapeError {
    /// Tensor shape has an unexpected rank.
    Rank {
        /// Checkpoint key.
        key: String,
        /// Declared shape.
        actual: Vec<usize>,
        /// Expected rank.
        expected: usize,
    },
    /// Two related tensors disagree on channel width.
    Mismatch {
        /// Checkpoint key that disagrees.
        key: String,
        /// Inferred value.
        actual: usize,
        /// Required value.
        expected: usize,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rank {
                key,
                actual,
                expected,
            } => {
                write!(f, "{key}: shape {actual:?} needs rank {expected}")
            }
            Self::Mismatch {
                key,
                actual,
                expected,
            } => {
                write!(f, "{key}: got {actual} channels, expected {expected}")
            }
        }
    }
}

impl Error for ShapeError {}

fn shape<'a>(map: &'a CheckpointMap, key: &str) -> Result<&'a [usize], ShapeError> {
    map.get_aliased(key)
        .map(|meta| meta.shape.as_slice())
        .ok_or_else(|| ShapeError::Rank {
            key: key.to_owned(),
            actual: Vec::new(),
            expected: 4,
        })
}

fn conv(map: &CheckpointMap, key: &str) -> Result<ConvInfo, ShapeError> {
    let dims = shape(map, key)?;
    if dims.len() != 4 || dims[2] != dims[3] {
        return Err(ShapeError::Rank {
            key: key.to_owned(),
            actual: dims.to_vec(),
            expected: 4,
        });
    }
    Ok(ConvInfo {
        out_c: dims[0],
        in_c: dims[1],
        kernel: dims[2],
    })
}

fn linear(map: &CheckpointMap, key: &str) -> Result<(usize, usize), ShapeError> {
    let dims = shape(map, key)?;
    if dims.len() != 2 {
        return Err(ShapeError::Rank {
            key: key.to_owned(),
            actual: dims.to_vec(),
            expected: 2,
        });
    }
    Ok((dims[0], dims[1]))
}

/// Derive channel and linear widths from a spec's tensor shapes.
pub fn infer(spec: &UnetModelSpec<'_>, map: &CheckpointMap) -> Result<UnetShapes, ShapeError> {
    let entry = conv(map, spec.conv_in.weight)?;
    let down = match spec.down_levels.first().and_then(|level| level.downsample) {
        Some(keys) => Some(conv(map, keys.weight)?),
        None => None,
    };
    let mid = conv(map, spec.mid_level.resnets[0].conv1.weight)?;
    let up = match spec.up_levels.first().and_then(|level| level.upsample) {
        Some(keys) => Some(conv(map, keys.weight)?),
        None => None,
    };
    let output = conv(map, spec.conv_out.weight)?;
    let (hidden, dim) = linear(map, spec.timestep.linear1.weight)?;
    let (channels, hidden_out) = linear(map, spec.timestep.linear2.weight)?;
    if hidden_out != hidden {
        return Err(ShapeError::Mismatch {
            key: spec.timestep.linear2.weight.to_owned(),
            actual: hidden_out,
            expected: hidden,
        });
    }
    if channels != hidden {
        return Err(ShapeError::Mismatch {
            key: spec.timestep.linear2.weight.to_owned(),
            actual: channels,
            expected: hidden,
        });
    }
    Ok(UnetShapes {
        latent_channels: entry.out_c,
        entry_kernel: entry.kernel,
        down,
        mid_channels: mid.out_c,
        up,
        output_channels: output.out_c,
        timestep_dim: dim,
        timestep_hidden: hidden,
        timestep_channels: channels,
        vae_entry: conv(map, spec.vae.conv_in.weight)?,
        vae_exit: conv(map, spec.vae.conv_out.weight)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_spec::{SD15_SINGLE_LEVEL, tensor_keys};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn checkpoint(
        keys: &[(String, String)],
        override_shape: Option<&(&str, Vec<usize>)>,
    ) -> (std::path::PathBuf, CheckpointMap) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (_, key) in keys {
            let dims = if let Some((name, shape)) = &override_shape {
                if key == name {
                    shape.clone()
                } else {
                    default_shape(key)
                }
            } else {
                default_shape(key)
            };
            let count: usize = dims.iter().product();
            let start = payload.len();
            for index in 0..count {
                let value = f32::from(u8::try_from(index % 256).expect("0..256")) * 0.1;
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let end = payload.len();
            header.insert(
                key.clone(),
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
            "honkbox-ash-shape-infer-{}-{stamp}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        (path.clone(), CheckpointMap::read(&path).expect("map"))
    }

    fn default_shape(key: &str) -> Vec<usize> {
        if key == "time_embedding.linear_1.weight" {
            vec![8, 4]
        } else if key == "time_embedding.linear_2.weight" {
            vec![8, 8]
        } else if key.ends_with(".weight") {
            vec![4, 4, 3, 3]
        } else {
            vec![4]
        }
    }

    #[test]
    fn infer_reads_channels_kernel_and_linear_widths() {
        let (path, map) = checkpoint(&tensor_keys(&SD15_SINGLE_LEVEL), None);
        let shapes = infer(&SD15_SINGLE_LEVEL, &map).expect("infer");
        assert_eq!(shapes.latent_channels, 4);
        assert_eq!(shapes.entry_kernel, 3);
        assert_eq!(shapes.mid_channels, 4);
        assert_eq!(shapes.output_channels, 4);
        assert_eq!(shapes.timestep_dim, 4);
        assert_eq!(shapes.timestep_hidden, 8);
        assert_eq!(shapes.timestep_channels, 8);
        assert_eq!(shapes.down.expect("down").kernel, 3);
        assert_eq!(shapes.up.expect("up").in_c, 4);
        assert_eq!(shapes.vae_entry.out_c, 4);
        assert_eq!(shapes.vae_exit.in_c, 4);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn infer_rejects_bad_rank_and_channel_mismatch() {
        let keys = tensor_keys(&SD15_SINGLE_LEVEL);
        let (path, map) = checkpoint(&keys, Some(&("time_embedding.linear_1.weight", vec![8])));
        assert!(matches!(
            infer(&SD15_SINGLE_LEVEL, &map),
            Err(ShapeError::Rank { .. })
        ));
        std::fs::remove_file(path).expect("cleanup");

        let (path, map) = checkpoint(&keys, Some(&("time_embedding.linear_2.weight", vec![2, 8])));
        assert!(matches!(
            infer(&SD15_SINGLE_LEVEL, &map),
            Err(ShapeError::Mismatch { .. })
        ));
        std::fs::remove_file(path).expect("cleanup");
    }
}
