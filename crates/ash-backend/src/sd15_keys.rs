//! SD 1.5 tensor names for the stages this backend actually implements.
//!
//! Keys follow the single-file checkpoint layout (`model.diffusion_model.…`,
//! `first_stage_model.decoder.…`). The vendored stable-diffusion.cpp model
//! headers use the same suffixes after stripping those prefixes. This is a
//! partial map, not a claim of full SD 1.5 parity: only the time embedding,
//! first convolution, middle cross-attention and VAE decoder projection that
//! the current toy stages consume are listed. New families get their own map.

use crate::{checkpoint::CheckpointMap, unet_checkpoint::load_tensor};
use std::io;
use std::path::Path;

/// `UNet` first convolution.
pub const UNET_INPUT_WEIGHT: &str = "model.diffusion_model.input_blocks.0.0.weight";
/// Bias of the `UNet` first convolution.
pub const UNET_INPUT_BIAS: &str = "model.diffusion_model.input_blocks.0.0.bias";
/// First timestep linear layer.
pub const TIME_EMBED_0_WEIGHT: &str = "model.diffusion_model.time_embed.0.weight";
/// Bias of the first timestep linear layer.
pub const TIME_EMBED_0_BIAS: &str = "model.diffusion_model.time_embed.0.bias";
/// Second timestep linear layer.
pub const TIME_EMBED_2_WEIGHT: &str = "model.diffusion_model.time_embed.2.weight";
/// Bias of the second timestep linear layer.
pub const TIME_EMBED_2_BIAS: &str = "model.diffusion_model.time_embed.2.bias";

/// Middle-block self-attention query projection.
pub const SELF_ATTN_Q: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight";
/// Middle-block cross-attention query projection.
pub const CROSS_ATTN_Q: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight";
/// Middle-block cross-attention key projection.
pub const CROSS_ATTN_K: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight";
/// Middle-block cross-attention value projection.
pub const CROSS_ATTN_V: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight";
/// Middle-block cross-attention output projection.
pub const CROSS_ATTN_OUT: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight";
/// Bias of the middle-block self-attention query projection.
pub const SELF_ATTN_Q_BIAS: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.bias";
/// Bias of the middle-block cross-attention query projection.
pub const CROSS_ATTN_Q_BIAS: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.bias";
/// Bias of the middle-block cross-attention key projection.
pub const CROSS_ATTN_K_BIAS: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.bias";
/// Bias of the middle-block cross-attention value projection.
pub const CROSS_ATTN_V_BIAS: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.bias";
/// Bias of the middle-block cross-attention output projection.
pub const CROSS_ATTN_OUT_BIAS: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias";
/// VAE decoder entry convolution.
pub const VAE_DECODER_IN: &str = "first_stage_model.decoder.conv_in.weight";
/// VAE decoder output convolution.
pub const VAE_DECODER_OUT: &str = "first_stage_model.decoder.conv_out.weight";

/// Every key this partial map requires before inference can be attempted.
pub const REQUIRED: &[&str] = &[
    UNET_INPUT_WEIGHT,
    UNET_INPUT_BIAS,
    TIME_EMBED_0_WEIGHT,
    TIME_EMBED_0_BIAS,
    TIME_EMBED_2_WEIGHT,
    TIME_EMBED_2_BIAS,
    SELF_ATTN_Q,
    CROSS_ATTN_Q,
    CROSS_ATTN_K,
    CROSS_ATTN_V,
    CROSS_ATTN_OUT,
    VAE_DECODER_IN,
    VAE_DECODER_OUT,
];

fn convertible(dtype: &str) -> bool {
    matches!(dtype, "F32" | "F16" | "BF16")
}

/// Missing or unconvertible tensors discovered in a checkpoint header.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct KeyAudit {
    /// Required names absent from the header.
    pub missing: Vec<&'static str>,
    /// Required names whose dtype cannot become f32.
    pub unconvertible: Vec<&'static str>,
}

impl KeyAudit {
    /// True when every required key is present and convertible.
    pub fn is_ready(&self) -> bool {
        self.missing.is_empty() && self.unconvertible.is_empty()
    }
}

/// Inspect a header without reading any tensor payload.
pub fn audit(map: &CheckpointMap) -> KeyAudit {
    let mut report = KeyAudit::default();
    for name in REQUIRED {
        match map.get(name) {
            None => report.missing.push(name),
            Some(meta) if !convertible(&meta.dtype) => report.unconvertible.push(name),
            Some(_) => {}
        }
    }
    report
}

/// SD 1.5 keeps `to_k` and `to_v` separate; local cross-attention wants them
/// stacked as `[2·dim, dim]`. Row order is K then V, matching `FusedQkv`.
pub fn load_fused_kv(
    map: &CheckpointMap,
    path: &Path,
    key_name: &str,
    value_name: &str,
    dim: usize,
) -> io::Result<Vec<f32>> {
    let shape = [dim, dim];
    let mut key = load_tensor(map, path, key_name, &shape)?;
    let value = load_tensor(map, path, value_name, &shape)?;
    key.extend_from_slice(&value);
    Ok(key)
}

/// SD 1.5 stores `to_k`/`to_v` biases separately; stack them K then V to match
/// [`load_fused_kv`] and [`crate::cross_attention::FusedQkv`].
pub fn load_fused_kv_bias(
    map: &CheckpointMap,
    path: &Path,
    key_name: &str,
    value_name: &str,
    dim: usize,
) -> io::Result<Vec<f32>> {
    let mut key = load_tensor(map, path, key_name, &[dim])?;
    let value = load_tensor(map, path, value_name, &[dim])?;
    key.extend_from_slice(&value);
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn checkpoint(
        entries: &[(&str, &str, Vec<usize>)],
        payload: &[u8],
    ) -> (std::path::PathBuf, CheckpointMap) {
        let mut header = serde_json::Map::new();
        let mut cursor = 0usize;
        for (name, dtype, shape) in entries {
            let elements = shape.iter().product::<usize>();
            let width = match *dtype {
                "F32" => 4,
                "F64" => 8,
                _ => panic!("test dtype"),
            };
            let end = cursor + elements * width;
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [cursor, end],
                }),
            );
            cursor = end;
        }
        let header_bytes = serde_json::to_vec(&header).expect("header");
        let mut bytes = u64::try_from(header_bytes.len())
            .expect("length")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(payload);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-sd15-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write checkpoint");
        let map = CheckpointMap::read(&path).expect("map");
        (path, map)
    }

    #[test]
    fn audit_reports_missing_and_unconvertible() {
        let entries: Vec<_> = REQUIRED
            .iter()
            .map(|name| (*name, "F32", vec![2usize, 2]))
            .collect();
        let (path, map) = checkpoint(&entries, &[0u8; 208]);
        let report = audit(&map);
        assert!(report.is_ready(), "{report:?}");
        std::fs::remove_file(path).expect("cleanup");

        let (path, map) = checkpoint(&[], &[]);
        let report = audit(&map);
        assert_eq!(report.missing.len(), REQUIRED.len());
        assert!(!report.is_ready());
        std::fs::remove_file(path).expect("cleanup");

        let mut one_f64: Vec<_> = REQUIRED
            .iter()
            .map(|name| (*name, "F32", vec![2usize, 2]))
            .collect();
        one_f64[0].1 = "F64";
        let (path, map) = checkpoint(&one_f64, &[0u8; 224]);
        let report = audit(&map);
        assert_eq!(report.unconvertible, vec![REQUIRED[0]]);
        assert!(!report.is_ready());
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn fused_kv_stacks_key_before_value() {
        let mut payload = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let entries = [
            (CROSS_ATTN_K, "F32", vec![2usize, 2]),
            (CROSS_ATTN_V, "F32", vec![2usize, 2]),
        ];
        let (path, map) = checkpoint(&entries, &payload);
        let fused = load_fused_kv(&map, &path, CROSS_ATTN_K, CROSS_ATTN_V, 2).expect("fused kv");
        assert_eq!(fused, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(load_fused_kv(&map, &path, CROSS_ATTN_K, CROSS_ATTN_V, 3).is_err());
        std::fs::remove_file(path).expect("cleanup");
    }
}
