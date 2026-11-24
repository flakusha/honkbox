//! SDXL checkpoint header inspection for the experimental ash backend.
//!
//! Reads tensor names and shapes only — no payload bytes are decoded. The
//! `UNet`/VAE dimensions are a hard contract (a wrong shape cannot be loaded);
//! the two CLIP towers are a readiness report, because an SDXL checkpoint can
//! carry the diffusion stack without usable text encoders.

use crate::checkpoint::{CheckpointMap, TensorMeta};
use honkbox_sd_core::SdError;
use std::io::Read;
use std::path::Path;

const UNET_ENTRY: &str = "model.diffusion_model.input_blocks.0.0.weight";
const TIME_EMBED_0: &str = "model.diffusion_model.time_embed.0.weight";
const TIME_EMBED_2: &str = "model.diffusion_model.time_embed.2.weight";
const LABEL_EMB: &str = "model.diffusion_model.label_emb.0.0.weight";
const CROSS_ATTN_K: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight";
const VAE_CONV_IN: &str = "first_stage_model.decoder.conv_in.weight";
const VAE_NORM_OUT: &str = "first_stage_model.decoder.norm_out.weight";
const VAE_CONV_OUT: &str = "first_stage_model.decoder.conv_out.weight";

/// Exactly-checked diffusion and VAE shapes (the keys above, in that order).
const SHAPE_KEYS: [(&str, &[usize]); 8] = [
    (UNET_ENTRY, &[320, 4, 3, 3]),
    (TIME_EMBED_0, &[1280, 320]),
    (TIME_EMBED_2, &[1280, 1280]),
    (LABEL_EMB, &[1280, 2816]),
    (CROSS_ATTN_K, &[1280, 2048]),
    (VAE_CONV_IN, &[512, 4, 3, 3]),
    (VAE_NORM_OUT, &[128]),
    (VAE_CONV_OUT, &[3, 128, 3, 3]),
];

const CLIP_L: &str = "conditioner.embedders.0.transformer.text_model.";
const CLIP_G: &str = "conditioner.embedders.1.model.";
const CLIP_L_COUNT: usize = 197;
const CLIP_G_COUNT: usize = 391;

/// Representative CLIP-L tensors (prefix `conditioner.embedders.0…`).
const CLIP_L_KEYS: [(&str, &[usize]); 5] = [
    (
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        &[49408, 768],
    ),
    (
        "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
        &[77, 768],
    ),
    (
        "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
        &[768, 768],
    ),
    (
        "conditioner.embedders.0.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight",
        &[768, 768],
    ),
    (
        "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
        &[768],
    ),
];

/// Representative `OpenCLIP` `bigG` tensors (prefix `conditioner.embedders.1…`).
const CLIP_G_KEYS: [(&str, &[usize]); 6] = [
    (
        "conditioner.embedders.1.model.token_embedding.weight",
        &[49408, 1280],
    ),
    (
        "conditioner.embedders.1.model.positional_embedding",
        &[77, 1280],
    ),
    (
        "conditioner.embedders.1.model.text_projection",
        &[1280, 1280],
    ),
    (
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
        &[3840, 1280],
    ),
    (
        "conditioner.embedders.1.model.transformer.resblocks.31.mlp.c_fc.weight",
        &[5120, 1280],
    ),
    ("conditioner.embedders.1.model.ln_final.weight", &[1280]),
];

/// Header cap, matching [`crate::checkpoint`]'s safetensors limit.
const MAX_HEADER: u64 = 16 * 1024 * 1024;

/// Channel and width geometry required by the SDXL stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdxlShapes {
    /// Latent channels in/out of the diffusion model and VAE.
    pub latent_channels: usize,
    /// Spatial kernel side of the `UNet` entry and VAE exit convolutions.
    pub entry_kernel: usize,
    /// Timestep embedding input width.
    pub timestep_dim: usize,
    /// Timestep embedding hidden width.
    pub timestep_hidden: usize,
    /// Timestep projection output width.
    pub timestep_channels: usize,
    /// ADM label embedding input width.
    pub label_emb_in: usize,
    /// Cross-attention context width.
    pub context_dim: usize,
    /// VAE decoder entry convolution input channels.
    pub vae_entry_in: usize,
    /// VAE decoder entry convolution output channels.
    pub vae_entry_out: usize,
    /// VAE decoder exit normalization width.
    pub vae_exit_in: usize,
    /// VAE decoder output channels.
    pub vae_exit_out: usize,
}

/// Header-only check of the two SDXL text towers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdxlReadiness {
    /// Diffusion and VAE shapes match exactly.
    pub unet_ready: bool,
    /// Missing or wrongly shaped diffusion/VAE tensors, named by key.
    pub unet_missing: Vec<String>,
    /// Both CLIP towers have the expected tensor count and representative keys.
    pub clip_ready: bool,
    /// Missing or wrongly shaped CLIP tensors, named by key.
    pub clip_missing: Vec<String>,
}

fn fail(path: &Path, message: impl std::fmt::Display) -> SdError {
    SdError::Backend(format!("checkpoint '{}': {message}", path.display()))
}

fn expect<'a>(
    path: &Path,
    map: &'a CheckpointMap,
    key: &str,
    shape: &[usize],
) -> Result<&'a TensorMeta, SdError> {
    let meta = map
        .get(key)
        .ok_or_else(|| fail(path, format!("missing tensor '{key}'")))?;
    if meta.shape.as_slice() != shape {
        return Err(fail(
            path,
            format!(
                "tensor '{key}' has shape {:?}, expected {shape:?}",
                meta.shape
            ),
        ));
    }
    Ok(meta)
}

/// Count header keys under `prefix`. `CheckpointMap` has no key iterator, so
/// this re-reads the header bytes only; payload bytes are still untouched.
fn count_prefix(path: &Path, prefix: &str) -> Result<usize, SdError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| fail(path, format!("cannot read safetensors header: {e}")))?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .map_err(|e| fail(path, format!("short safetensors header length: {e}")))?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > MAX_HEADER {
        return Err(fail(path, "invalid safetensors header length"));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| fail(path, "safetensors header length exceeds usize"))?;
    let mut bytes = vec![0u8; header_len];
    file.read_exact(&mut bytes)
        .map_err(|e| fail(path, format!("short safetensors header: {e}")))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)
        .map_err(|e| fail(path, format!("invalid safetensors header: {e}")))?;
    Ok(header.keys().filter(|key| key.starts_with(prefix)).count())
}

fn tower_missing(
    path: &Path,
    map: &CheckpointMap,
    prefix: &str,
    expected_count: usize,
    keys: &[(&str, &[usize])],
) -> Result<Vec<String>, SdError> {
    let count = count_prefix(path, prefix)?;
    let mut missing = Vec::new();
    if count != expected_count {
        missing.push(format!(
            "'{prefix}' has {count} tensors, expected {expected_count}"
        ));
    }
    for (key, shape) in keys {
        match map.get(key) {
            None => missing.push(format!("missing tensor '{key}'")),
            Some(meta) if meta.shape.as_slice() != *shape => missing.push(format!(
                "tensor '{key}' has shape {:?}, expected {shape:?}",
                meta.shape
            )),
            Some(_) => {}
        }
    }
    Ok(missing)
}

/// Validate the diffusion/VAE header and report text-tower readiness.
///
/// `UNet` or VAE shape errors are returned as [`SdError::Backend`]; a missing
/// text tower only sets [`SdxlReadiness::clip_ready`] to `false`.
pub fn inspect_sdxl_checkpoint(path: &Path) -> Result<(SdxlShapes, SdxlReadiness), SdError> {
    let map = CheckpointMap::read(path)
        .map_err(|e| fail(path, format!("cannot read safetensors header: {e}")))?;

    let entry = expect(path, &map, UNET_ENTRY, SHAPE_KEYS[0].1)?;
    let time_0 = expect(path, &map, TIME_EMBED_0, SHAPE_KEYS[1].1)?;
    let time_2 = expect(path, &map, TIME_EMBED_2, SHAPE_KEYS[2].1)?;
    let label_emb = expect(path, &map, LABEL_EMB, SHAPE_KEYS[3].1)?;
    let cross_k = expect(path, &map, CROSS_ATTN_K, SHAPE_KEYS[4].1)?;
    let vae_in = expect(path, &map, VAE_CONV_IN, SHAPE_KEYS[5].1)?;
    let vae_norm = expect(path, &map, VAE_NORM_OUT, SHAPE_KEYS[6].1)?;
    let vae_out = expect(path, &map, VAE_CONV_OUT, SHAPE_KEYS[7].1)?;

    let shapes = SdxlShapes {
        latent_channels: entry.shape[1],
        entry_kernel: entry.shape[2],
        timestep_dim: time_0.shape[1],
        timestep_hidden: time_0.shape[0],
        timestep_channels: time_2.shape[0],
        label_emb_in: label_emb.shape[1],
        context_dim: cross_k.shape[1],
        vae_entry_in: vae_in.shape[1],
        vae_entry_out: vae_in.shape[0],
        vae_exit_in: vae_norm.shape[0],
        vae_exit_out: vae_out.shape[0],
    };

    let mut clip_missing = tower_missing(path, &map, CLIP_L, CLIP_L_COUNT, &CLIP_L_KEYS)?;
    clip_missing.extend(tower_missing(
        path,
        &map,
        CLIP_G,
        CLIP_G_COUNT,
        &CLIP_G_KEYS,
    )?);

    Ok((
        shapes,
        SdxlReadiness {
            unet_ready: true,
            unet_missing: Vec::new(),
            clip_ready: clip_missing.is_empty(),
            clip_missing,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    struct TempFile(PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn full_entries() -> Vec<(String, Vec<usize>)> {
        let mut entries: Vec<(String, Vec<usize>)> = SHAPE_KEYS
            .iter()
            .chain(CLIP_L_KEYS.iter())
            .chain(CLIP_G_KEYS.iter())
            .map(|(key, shape)| ((*key).to_owned(), shape.to_vec()))
            .collect();
        for (prefix, expected) in [(CLIP_L, CLIP_L_COUNT), (CLIP_G, CLIP_G_COUNT)] {
            let mut pad = 0;
            while entries
                .iter()
                .filter(|(key, _)| key.starts_with(prefix))
                .count()
                < expected
            {
                entries.push((format!("{prefix}pad.{pad:04}"), vec![1]));
                pad += 1;
            }
        }
        entries
    }

    /// Sparse `BOOL` payload: shapes match the real checkpoint, but the weight
    /// bytes are zeros (only the header is inspected).
    ///
    /// `BOOL` is one byte wide, so a tensor spans exactly `product(shape)`
    /// bytes; the ranges are contiguous from offset 0, and the file length is
    /// `8 + header_len + payload_len` — the 8-byte length prefix included —
    /// as the safetensors reader requires.
    fn fixture(entries: &[(String, Vec<usize>)]) -> TempFile {
        const DTYPE: &str = "BOOL";
        const DTYPE_WIDTH: usize = 1;

        let mut header = serde_json::Map::new();
        let mut cursor = 0usize;
        for (key, shape) in entries {
            let bytes = shape
                .iter()
                .copied()
                .product::<usize>()
                .checked_mul(DTYPE_WIDTH)
                .expect("tensor byte length");
            let end = cursor + bytes;
            header.insert(
                key.clone(),
                serde_json::json!({
                    "dtype": DTYPE,
                    "shape": shape,
                    "data_offsets": [cursor, end],
                }),
            );
            cursor = end;
        }
        let payload_len = cursor;
        let header_bytes = serde_json::to_vec(&header).expect("header");
        let header_len = u64::try_from(header_bytes.len()).expect("header length");
        let mut file = header_len.to_le_bytes().to_vec();
        file.extend_from_slice(&header_bytes);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-sdxl-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, file).expect("write header");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen header");
        let file_len = 8u64
            .checked_add(header_len)
            .and_then(|len| len.checked_add(u64::try_from(payload_len).ok()?))
            .expect("file length");
        file.set_len(file_len).expect("extend sparse payload");
        TempFile(path)
    }

    #[test]
    fn full_header_is_ready_with_expected_shapes() {
        let path = fixture(&full_entries());
        let (shapes, ready) = inspect_sdxl_checkpoint(&path.0).expect("inspect");
        assert_eq!(
            shapes,
            SdxlShapes {
                latent_channels: 4,
                entry_kernel: 3,
                timestep_dim: 320,
                timestep_hidden: 1280,
                timestep_channels: 1280,
                label_emb_in: 2816,
                context_dim: 2048,
                vae_entry_in: 4,
                vae_entry_out: 512,
                vae_exit_in: 128,
                vae_exit_out: 3,
            }
        );
        assert!(ready.unet_ready, "{:?}", ready.unet_missing);
        assert!(ready.clip_ready, "{:?}", ready.clip_missing);
        assert_eq!(ready.unet_missing, Vec::<String>::new());
        assert_eq!(ready.clip_missing, Vec::<String>::new());
    }

    #[test]
    fn missing_unet_key_is_a_named_error() {
        let mut entries = full_entries();
        entries.retain(|(key, _)| key.as_str() != UNET_ENTRY);
        let path = fixture(&entries);
        match inspect_sdxl_checkpoint(&path.0) {
            Err(SdError::Backend(message)) => {
                assert!(message.contains(UNET_ENTRY), "{message}");
            }
            other => panic!("expected named backend error, got {other:?}"),
        }
    }

    #[test]
    fn missing_clip_g_key_reports_not_ready() {
        let mut entries = full_entries();
        entries.retain(|(key, _)| key.as_str() != CLIP_G_KEYS[0].0);
        let path = fixture(&entries);
        let (_, ready) = inspect_sdxl_checkpoint(&path.0).expect("inspect");
        assert!(ready.unet_ready);
        assert!(!ready.clip_ready);
        assert!(
            ready
                .clip_missing
                .iter()
                .any(|m| m.contains(CLIP_G_KEYS[0].0))
        );
    }

    #[test]
    fn wrong_label_emb_shape_fails() {
        let mut entries = full_entries();
        for (key, shape) in &mut entries {
            if key.as_str() == LABEL_EMB {
                *shape = vec![1280, 2817];
            }
        }
        let path = fixture(&entries);
        match inspect_sdxl_checkpoint(&path.0) {
            Err(SdError::Backend(message)) => {
                assert!(message.contains(LABEL_EMB), "{message}");
                assert!(message.contains("expected"), "{message}");
            }
            other => panic!("expected shape error, got {other:?}"),
        }
    }
}
