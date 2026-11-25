//! Infer traversal stage shapes from checkpoint tensor metadata.
//!
//! ponytail: only channels/kernel/linear widths come from metadata, following
//! the reduced topology in [`crate::model_spec`]. Spatial sizes still come from
//! the caller's sampler; no full-fidelity loader is attempted here.

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
    /// Latent channels consumed by `conv_in`.
    pub latent_channels: usize,
    /// Entry convolution kernel side.
    pub entry_kernel: usize,
    /// Three stride-2 downsample convolutions, in execution order.
    pub downsamples: [ConvInfo; 3],
    /// Mid-level channel width.
    pub mid_channels: usize,
    /// Two 1×1 projections after the host nearest upsample.
    pub projections: [ConvInfo; 2],
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
    let downsamples = spec
        .down_levels
        .iter()
        .map(|level| {
            let keys = level.downsample.ok_or_else(|| ShapeError::Rank {
                key: level.name.to_owned(),
                actual: Vec::new(),
                expected: 4,
            })?;
            conv(map, keys.weight)
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| ShapeError::Mismatch {
            key: "down levels".to_owned(),
            actual: spec.down_levels.len(),
            expected: 3,
        })?;
    let mid = conv(map, spec.mid_level.resnets[0].conv1.weight)?;
    let projections = spec
        .projections
        .iter()
        .map(|keys| conv(map, keys.weight))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| ShapeError::Mismatch {
            key: "projections".to_owned(),
            actual: spec.projections.len(),
            expected: 2,
        })?;
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
        latent_channels: entry.in_c,
        entry_kernel: entry.kernel,
        downsamples,
        mid_channels: mid.out_c,
        projections,
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
    use crate::model_spec::{SD15_REDUCED, tensor_keys};
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

    impl std::ops::Deref for TempFile {
        type Target = std::path::PathBuf;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    fn checkpoint(
        keys: &[(String, String)],
        override_shape: Option<&(&str, Vec<usize>)>,
    ) -> (TempFile, CheckpointMap) {
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
            "honkbox-ash-shape-infer-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        let map = CheckpointMap::read(&path).expect("map");
        (TempFile(path), map)
    }

    /// Shapes of the reduced graph the real loader actually reads.
    fn default_shape(key: &str) -> Vec<usize> {
        match key {
            "time_embedding.linear_1.weight" => vec![1280, 320],
            "time_embedding.linear_2.weight" => vec![1280, 1280],
            "conv_in.weight" => vec![320, 4, 3, 3],
            "down_blocks.0.downsamplers.0.conv.weight" => vec![320, 320, 3, 3],
            "input_blocks.6.0.op.weight" => vec![640, 640, 3, 3],
            "input_blocks.9.0.op.weight" | "mid_block.resnets.0.conv1.weight" => {
                vec![1280, 1280, 3, 3]
            }
            "output_blocks.7.0.skip_connection.weight" => vec![640, 1280, 1, 1],
            "output_blocks.10.0.skip_connection.weight" => vec![320, 640, 1, 1],
            "conv_out.weight" => vec![4, 320, 3, 3],
            "decoder.conv_in.weight" => vec![128, 4, 3, 3],
            "decoder.conv_out.weight" => vec![3, 128, 3, 3],
            _ if key.ends_with(".weight") => vec![4, 4, 3, 3],
            _ => vec![4],
        }
    }

    #[test]
    fn infer_reads_executed_channels_kernels_and_linear_widths() {
        let (_file, map) = checkpoint(&tensor_keys(&SD15_REDUCED), None);
        let shapes = infer(&SD15_REDUCED, &map).expect("infer");
        assert_eq!(shapes.latent_channels, 4);
        assert_eq!(shapes.entry_kernel, 3);
        assert_eq!(
            shapes
                .downsamples
                .map(|info| (info.in_c, info.out_c, info.kernel)),
            [(320, 320, 3), (640, 640, 3), (1280, 1280, 3)]
        );
        assert_eq!(shapes.mid_channels, 1280);
        assert_eq!(
            shapes
                .projections
                .map(|info| (info.in_c, info.out_c, info.kernel)),
            [(1280, 640, 1), (640, 320, 1)]
        );
        assert_eq!(shapes.output_channels, 4);
        assert_eq!(shapes.timestep_dim, 320);
        assert_eq!(shapes.timestep_hidden, 1280);
        assert_eq!(shapes.timestep_channels, 1280);
        assert_eq!(shapes.vae_entry.out_c, 128);
        assert_eq!(shapes.vae_exit.in_c, 128);
    }

    #[test]
    fn infer_rejects_bad_rank_and_channel_mismatch() {
        let keys = tensor_keys(&SD15_REDUCED);
        let (_file, map) = checkpoint(&keys, Some(&("time_embedding.linear_1.weight", vec![8])));
        assert!(matches!(
            infer(&SD15_REDUCED, &map),
            Err(ShapeError::Rank { .. })
        ));

        let (_file, map) = checkpoint(&keys, Some(&("time_embedding.linear_2.weight", vec![2, 8])));
        assert!(matches!(
            infer(&SD15_REDUCED, &map),
            Err(ShapeError::Mismatch { .. })
        ));
    }
}
