//! Header-only integration tests for SDXL inspection and the SD 1.5 regression.
//!
//! Fixtures are synthetic safetensors headers mirroring the tensor names and
//! shapes of a real single-file SDXL checkpoint (no weight bytes are written —
//! only the declared shapes and byte ranges matter to the header reader).
//!
//! The reader also requires the file to be long enough for the declared
//! payload: `8 + header_len + total_payload`. The payload bytes themselves are
//! never written; the file is extended sparsely to that length.
#![cfg(feature = "ash-sd")]

use honkbox_ash_backend::{sd, sdxl};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Unique temp file, removed on every exit path.
struct TempCheckpoint(PathBuf);

impl TempCheckpoint {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempCheckpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Clone)]
struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

fn tensor(name: impl Into<String>, shape: &[usize]) -> Tensor {
    Tensor {
        name: name.into(),
        dtype: "F32",
        shape: shape.to_vec(),
    }
}

fn f16(mut tensor: Tensor) -> Tensor {
    tensor.dtype = "F16";
    tensor
}

fn dtype_width(dtype: &str) -> usize {
    match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        other => panic!("unsupported fixture dtype '{other}'"),
    }
}

/// Write a self-consistent safetensors header over a sparse payload.
fn write_checkpoint(tensors: &[Tensor]) -> TempCheckpoint {
    let mut header = serde_json::Map::new();
    let mut cursor = 0usize;
    for entry in tensors {
        let elements: usize = entry.shape.iter().copied().product();
        let end = cursor + elements * dtype_width(entry.dtype);
        header.insert(
            entry.name.clone(),
            serde_json::json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [cursor, end],
            }),
        );
        cursor = end;
    }
    let header_bytes = serde_json::to_vec(&header).expect("header");
    let mut prefix = u64::try_from(header_bytes.len())
        .expect("header length")
        .to_le_bytes()
        .to_vec();
    prefix.extend_from_slice(&header_bytes);
    let file_len = u64::try_from(prefix.len() + cursor).expect("file length");

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "honkbox-ash-sdxl-{}-{}-{stamp}.safetensors",
        std::process::id(),
        TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create synthetic checkpoint");
    file.write_all(&prefix).expect("write header");
    file.set_len(file_len).expect("extend sparse payload");
    TempCheckpoint(path)
}

/// `UNet` + VAE keys whose shapes define [`sdxl::SdxlShapes`].
fn sdxl_unet_tensors() -> Vec<Tensor> {
    const U: &str = "model.diffusion_model.";
    const V: &str = "first_stage_model.decoder.";
    let weights: &[(&str, &[usize])] = &[
        ("input_blocks.0.0.weight", &[320, 4, 3, 3]),
        ("time_embed.0.weight", &[1280, 320]),
        ("time_embed.2.weight", &[1280, 1280]),
        ("label_emb.0.0.weight", &[1280, 2816]),
        (
            "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
            &[1280, 2048],
        ),
        (
            "middle_block.1.transformer_blocks.0.attn2.to_v.weight",
            &[1280, 2048],
        ),
        ("out.2.weight", &[4, 320, 3, 3]),
    ];
    let biases: &[(&str, usize)] = &[
        ("input_blocks.0.0.bias", 320),
        ("time_embed.0.bias", 1280),
        ("time_embed.2.bias", 1280),
        ("label_emb.0.0.bias", 1280),
        ("out.2.bias", 4),
    ];
    let vae: &[(&str, &[usize])] = &[
        ("conv_in.weight", &[512, 4, 3, 3]),
        ("norm_out.weight", &[128]),
        ("conv_out.weight", &[3, 128, 3, 3]),
    ];

    let mut out: Vec<Tensor> = weights
        .iter()
        .map(|(name, shape)| f16(tensor(format!("{U}{name}"), shape)))
        .chain(
            biases
                .iter()
                .map(|(name, n)| f16(tensor(format!("{U}{name}"), &[*n]))),
        )
        .chain(
            vae.iter()
                .map(|(name, shape)| f16(tensor(format!("{V}{name}"), shape))),
        )
        .collect();
    out.push(f16(tensor(format!("{V}conv_in.bias"), &[512usize])));
    out.push(f16(tensor(format!("{V}norm_out.bias"), &[128usize])));
    out.push(f16(tensor(format!("{V}conv_out.bias"), &[3usize])));
    out
}

/// CLIP-L tower: 3 embeddings + 12 layers x 16 tensors + final norm = 197.
fn clip_l_tensors() -> Vec<Tensor> {
    const P: &str = "conditioner.embedders.0.transformer.text_model.";
    const LAYER: &[&str] = &[
        "layer_norm1.bias",
        "layer_norm1.weight",
        "layer_norm2.bias",
        "layer_norm2.weight",
        "mlp.fc1.bias",
        "mlp.fc1.weight",
        "mlp.fc2.bias",
        "mlp.fc2.weight",
        "self_attn.k_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.out_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.v_proj.weight",
    ];
    let mut out = vec![
        tensor(format!("{P}embeddings.position_ids"), &[4]),
        tensor(format!("{P}embeddings.position_embedding.weight"), &[4]),
        tensor(format!("{P}embeddings.token_embedding.weight"), &[4]),
        tensor(format!("{P}final_layer_norm.bias"), &[4]),
        tensor(format!("{P}final_layer_norm.weight"), &[4]),
    ];
    for layer in 0..12 {
        for suffix in LAYER {
            out.push(tensor(format!("{P}encoder.layers.{layer}.{suffix}"), &[4]));
        }
    }
    assert_eq!(out.len(), 197, "CLIP-L group size");
    out.into_iter()
        .map(|entry| {
            let shape: &[usize] = if entry.name.ends_with("embeddings.token_embedding.weight") {
                &[49408, 768]
            } else if entry.name.ends_with("embeddings.position_embedding.weight") {
                &[77, 768]
            } else if entry.name.ends_with("final_layer_norm.weight") {
                &[768]
            } else if entry
                .name
                .ends_with("encoder.layers.0.self_attn.q_proj.weight")
                || entry
                    .name
                    .ends_with("encoder.layers.11.self_attn.q_proj.weight")
            {
                &[768, 768]
            } else {
                &[4]
            };
            f16(tensor(entry.name, shape))
        })
        .collect()
}

/// `OpenCLIP` bigG tower: 7 standalone + 32 blocks x 12 tensors = 391.
fn clip_g_tensors() -> Vec<Tensor> {
    const P: &str = "conditioner.embedders.1.model.";
    const BLOCK: &[&str] = &[
        "attn.in_proj_bias",
        "attn.in_proj_weight",
        "attn.out_proj.bias",
        "attn.out_proj.weight",
        "ln_1.bias",
        "ln_1.weight",
        "ln_2.bias",
        "ln_2.weight",
        "mlp.c_fc.bias",
        "mlp.c_fc.weight",
        "mlp.c_proj.bias",
        "mlp.c_proj.weight",
    ];
    let mut out = vec![
        tensor(format!("{P}ln_final.bias"), &[4]),
        tensor(format!("{P}ln_final.weight"), &[4]),
        tensor(format!("{P}logit_scale"), &[1]),
        tensor(format!("{P}positional_embedding"), &[4]),
        tensor(format!("{P}text_projection"), &[4]),
        tensor(format!("{P}token_embedding.weight"), &[4]),
        tensor(
            format!("{P}transformer.text_model.embeddings.position_ids"),
            &[4],
        ),
    ];
    for block in 0..32 {
        for suffix in BLOCK {
            out.push(tensor(
                format!("{P}transformer.resblocks.{block}.{suffix}"),
                &[4],
            ));
        }
    }
    assert_eq!(out.len(), 391, "CLIP-G group size");
    out.into_iter()
        .map(|entry| {
            let shape: &[usize] = if entry.name.ends_with("token_embedding.weight") {
                &[49408, 1280]
            } else if entry.name.ends_with("positional_embedding") {
                &[77, 1280]
            } else if entry.name.ends_with("text_projection") {
                &[1280, 1280]
            } else if entry.name.ends_with("ln_final.weight") {
                &[1280]
            } else if entry.name.ends_with("resblocks.0.attn.in_proj_weight") {
                &[3840, 1280]
            } else if entry.name.ends_with("resblocks.31.mlp.c_fc.weight") {
                &[5120, 1280]
            } else {
                &[4]
            };
            f16(tensor(entry.name, shape))
        })
        .collect()
}

fn sdxl_tensors() -> Vec<Tensor> {
    let mut out = sdxl_unet_tensors();
    out.extend(clip_l_tensors());
    out.extend(clip_g_tensors());
    out
}

#[test]
fn sdxl_header_yields_shapes_and_ready_clip() {
    let file = write_checkpoint(&sdxl_tensors());
    let (shapes, readiness) =
        sdxl::inspect_sdxl_checkpoint(file.path()).expect("complete synthetic SDXL header");

    assert_eq!(shapes.latent_channels, 4);
    assert_eq!(shapes.entry_kernel, 3);
    assert_eq!(shapes.timestep_dim, 320);
    assert_eq!(shapes.timestep_hidden, 1280);
    assert_eq!(shapes.timestep_channels, 1280);
    assert_eq!(shapes.label_emb_in, 2816);
    assert_eq!(shapes.context_dim, 2048);
    assert_eq!(shapes.vae_entry_in, 4);
    assert_eq!(shapes.vae_entry_out, 512);
    assert_eq!(shapes.vae_exit_in, 128);
    assert_eq!(shapes.vae_exit_out, 3);

    assert!(readiness.unet_ready, "{readiness:?}");
    assert!(readiness.unet_missing.is_empty(), "{readiness:?}");
    assert!(readiness.clip_ready, "{readiness:?}");
    assert!(readiness.clip_missing.is_empty(), "{readiness:?}");
}

#[test]
fn truncated_clip_g_names_a_missing_key() {
    let mut tensors = sdxl_tensors();
    tensors.retain(|entry| {
        !entry.name.starts_with("conditioner.embedders.1.model.")
            || entry.name.ends_with("ln_final.bias")
    });
    let file = write_checkpoint(&tensors);

    let (shapes, readiness) =
        sdxl::inspect_sdxl_checkpoint(file.path()).expect("truncated CLIP-G header");
    assert_eq!(shapes.context_dim, 2048);
    assert!(readiness.unet_ready, "{readiness:?}");
    assert!(!readiness.clip_ready, "{readiness:?}");
    assert!(
        readiness
            .clip_missing
            .iter()
            .any(|name| name.contains("conditioner.embedders.1.model.")),
        "missing key must be named: {:?}",
        readiness.clip_missing
    );
}

/// Original single-file SD 1.5 names consumed by `CheckpointMap::get_aliased`.
fn sd15_tensors() -> Vec<Tensor> {
    const RESNETS: &[&str] = &[
        "input_blocks.1.0",
        "input_blocks.2.0",
        "input_blocks.4.0",
        "input_blocks.7.0",
        "middle_block.0",
        "middle_block.2",
        "output_blocks.3.0",
        "output_blocks.4.0",
    ];
    const RESNET_PARTS: &[&str] = &[
        "in_layers.2.weight",
        "in_layers.2.bias",
        "in_layers.0.weight",
        "in_layers.0.bias",
        "emb_layers.1.weight",
        "emb_layers.1.bias",
        "out_layers.3.weight",
        "out_layers.3.bias",
        "out_layers.0.weight",
        "out_layers.0.bias",
        "skip_connection.weight",
        "skip_connection.bias",
    ];
    let mut keys = vec![
        "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
        "model.diffusion_model.input_blocks.0.0.bias".to_owned(),
    ];
    for base in RESNETS {
        for part in RESNET_PARTS {
            keys.push(format!("model.diffusion_model.{base}.{part}"));
        }
    }
    for name in [
        "input_blocks.3.0.op.weight",
        "input_blocks.3.0.op.bias",
        "input_blocks.6.0.op.weight",
        "input_blocks.6.0.op.bias",
        "input_blocks.9.0.op.weight",
        "input_blocks.9.0.op.bias",
        "output_blocks.5.2.conv.weight",
        "output_blocks.5.2.conv.bias",
        "middle_block.1.transformer_blocks.0.attn2.to_q.weight",
        "middle_block.1.transformer_blocks.0.attn2.to_k.weight",
        "middle_block.1.transformer_blocks.0.attn2.to_v.weight",
        "middle_block.1.transformer_blocks.0.attn2.to_q.bias",
        "middle_block.1.transformer_blocks.0.attn2.to_k.bias",
        "middle_block.1.transformer_blocks.0.attn2.to_v.bias",
        "middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
        "middle_block.1.transformer_blocks.0.attn2.to_out.0.bias",
        "output_blocks.7.0.skip_connection.weight",
        "output_blocks.7.0.skip_connection.bias",
        "output_blocks.10.0.skip_connection.weight",
        "output_blocks.10.0.skip_connection.bias",
        "out.0.weight",
        "out.0.bias",
        "out.2.weight",
        "out.2.bias",
        "time_embed.0.weight",
        "time_embed.0.bias",
        "time_embed.2.weight",
        "time_embed.2.bias",
    ] {
        keys.push(format!("model.diffusion_model.{name}"));
    }
    for name in [
        "conv_in.weight",
        "conv_in.bias",
        "norm_out.weight",
        "norm_out.bias",
        "conv_out.weight",
        "conv_out.bias",
    ] {
        keys.push(format!("first_stage_model.decoder.{name}"));
    }
    for name in [
        "embeddings.token_embedding.weight",
        "embeddings.position_embedding.weight",
        "embeddings.position_ids",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.k_proj.weight",
        "encoder.layers.11.self_attn.v_proj.weight",
        "encoder.layers.11.mlp.fc1.weight",
        "final_layer_norm.weight",
    ] {
        keys.push(format!("cond_stage_model.transformer.text_model.{name}"));
    }
    keys.push("cond_stage_model.transformer.text_projection.weight".to_owned());

    keys.into_iter()
        .map(|name| {
            let shape: &[usize] = if name.ends_with("time_embed.0.weight") {
                &[8, 4]
            } else if name.ends_with("time_embed.2.weight") {
                &[8, 8]
            } else if name.ends_with(".weight") && !name.starts_with("cond_stage_model.") {
                &[4, 4, 3, 3]
            } else {
                &[4]
            };
            tensor(name, shape)
        })
        .collect()
}

#[test]
fn sd15_original_name_layout_still_inspects() {
    let file = write_checkpoint(&sd15_tensors());
    let shapes = sd::inspect_checkpoint(file.path()).expect("SD 1.5 original-name header");

    assert_eq!(shapes.latent_channels, 4);
    assert_eq!(shapes.entry_kernel, 3);
    assert_eq!(shapes.timestep_dim, 4);
    assert_eq!(shapes.timestep_hidden, 8);
    assert_eq!(shapes.timestep_channels, 8);
    assert_eq!(shapes.vae_entry.in_c, 4);
    assert_eq!(shapes.vae_exit.out_c, 4);
}
