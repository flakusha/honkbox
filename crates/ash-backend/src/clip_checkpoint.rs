//! SD 1.5 CLIP text-tower tensors → [`ClipTextEncoderWeights`].
//!
//! Checkpoints store linear weights `[out, in]`; the local matmuls expect
//! `[in, out]`, so matrices are transposed once at load time. Linear biases
//! and the final layer norm are loaded too, feeding the bias-aware encoder and
//! EOS pooling path.

use crate::{
    checkpoint::CheckpointMap,
    text_encoder::{ClipPooling, ClipTextEncoderWeights},
    transformer::{TransformerBias, TransformerWeights},
    unet_checkpoint::load_tensor,
};
use std::io;
use std::path::Path;

const PREFIX: &str = "cond_stage_model.transformer.text_model.";

/// Geometry of the text tower to load.
#[derive(Debug, Clone, Copy)]
pub struct ClipGeometry {
    /// Token embedding vocabulary rows.
    pub vocab: usize,
    /// Position embedding rows (prompt capacity).
    pub tokens: usize,
    /// Hidden width.
    pub dim: usize,
    /// MLP expansion width.
    pub ffn_hidden: usize,
    /// Number of encoder blocks.
    pub layers: usize,
}

/// SD 1.5 text tower geometry (CLIP ViT-L/14, 77 tokens).
pub const SD15_CLIP: ClipGeometry = ClipGeometry {
    vocab: 49408,
    tokens: 77,
    dim: 768,
    ffn_hidden: 3072,
    layers: 12,
};

fn name(suffix: &str) -> String {
    // Single-file SD 1.5 nests transformer layers under `encoder.`.
    if suffix.starts_with("layers.") {
        format!("{PREFIX}encoder.{suffix}")
    } else {
        format!("{PREFIX}{suffix}")
    }
}

/// Required tensor names and shapes for one text tower.
pub fn required_tensors(geometry: ClipGeometry) -> Vec<(String, Vec<usize>)> {
    let mut out = vec![
        (
            name("embeddings.token_embedding.weight"),
            vec![geometry.vocab, geometry.dim],
        ),
        (
            name("embeddings.position_embedding.weight"),
            vec![geometry.tokens, geometry.dim],
        ),
    ];
    for index in 0..geometry.layers {
        for kind in ["q", "k", "v", "out"] {
            out.push((
                name(&format!("layers.{index}.self_attn.{kind}_proj.weight")),
                vec![geometry.dim, geometry.dim],
            ));
            out.push((
                name(&format!("layers.{index}.self_attn.{kind}_proj.bias")),
                vec![geometry.dim],
            ));
        }
        out.push((
            name(&format!("layers.{index}.mlp.fc1.weight")),
            vec![geometry.ffn_hidden, geometry.dim],
        ));
        out.push((
            name(&format!("layers.{index}.mlp.fc1.bias")),
            vec![geometry.ffn_hidden],
        ));
        out.push((
            name(&format!("layers.{index}.mlp.fc2.weight")),
            vec![geometry.dim, geometry.ffn_hidden],
        ));
        out.push((
            name(&format!("layers.{index}.mlp.fc2.bias")),
            vec![geometry.dim],
        ));
        for number in 1..=2 {
            out.push((
                name(&format!("layers.{index}.layer_norm{number}.weight")),
                vec![geometry.dim],
            ));
            out.push((
                name(&format!("layers.{index}.layer_norm{number}.bias")),
                vec![geometry.dim],
            ));
        }
    }
    out.push((name("final_layer_norm.weight"), vec![geometry.dim]));
    out.push((name("final_layer_norm.bias"), vec![geometry.dim]));
    out
}

/// Missing or wrongly shaped CLIP tensors in a checkpoint header.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ClipAudit {
    /// Required names absent from the header.
    pub missing: Vec<String>,
    /// Required names present with an unexpected shape.
    pub wrong_shape: Vec<String>,
}

impl ClipAudit {
    /// True when every required tensor is present with the expected shape.
    pub fn is_ready(&self) -> bool {
        self.missing.is_empty() && self.wrong_shape.is_empty()
    }
}

/// Inspect CLIP tensor metadata only; no payload is read.
pub fn audit(map: &CheckpointMap, geometry: ClipGeometry) -> ClipAudit {
    let mut report = ClipAudit::default();
    for (key, shape) in required_tensors(geometry) {
        match map.get(&key) {
            None => report.missing.push(key),
            Some(meta) if meta.shape != shape => report.wrong_shape.push(key),
            Some(_) => {}
        }
    }
    report
}

fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    out
}

fn load_matrix(
    map: &CheckpointMap,
    path: &Path,
    key: &str,
    out: usize,
    into: usize,
) -> io::Result<Vec<f32>> {
    Ok(transpose(
        &load_tensor(map, path, key, &[out, into])?,
        out,
        into,
    ))
}

/// Owned f32 tensors for one encoder block, already in local layout.
pub struct LayerTensors {
    /// Query projection `[dim, dim]`.
    pub query: Vec<f32>,
    /// Key projection `[dim, dim]`.
    pub key: Vec<f32>,
    /// Value projection `[dim, dim]`.
    pub value: Vec<f32>,
    /// Attention output projection `[dim, dim]`.
    pub output: Vec<f32>,
    /// MLP expansion `[dim, hidden]`.
    pub expand: Vec<f32>,
    /// MLP contraction `[hidden, dim]`.
    pub contract: Vec<f32>,
    /// Query projection bias `[dim]`.
    pub query_bias: Vec<f32>,
    /// Key projection bias `[dim]`.
    pub key_bias: Vec<f32>,
    /// Value projection bias `[dim]`.
    pub value_bias: Vec<f32>,
    /// Attention output projection bias `[dim]`.
    pub output_bias: Vec<f32>,
    /// MLP expansion bias `[hidden]`.
    pub expand_bias: Vec<f32>,
    /// MLP contraction bias `[dim]`.
    pub contract_bias: Vec<f32>,
    /// First layer-norm gain.
    pub norm1_gain: Vec<f32>,
    /// First layer-norm bias.
    pub norm1_bias: Vec<f32>,
    /// Second layer-norm gain.
    pub norm2_gain: Vec<f32>,
    /// Second layer-norm bias.
    pub norm2_bias: Vec<f32>,
}

fn load_layer(
    map: &CheckpointMap,
    path: &Path,
    index: usize,
    geometry: ClipGeometry,
) -> io::Result<LayerTensors> {
    let dim = geometry.dim;
    let hidden = geometry.ffn_hidden;
    let attn = |kind: &str| name(&format!("layers.{index}.self_attn.{kind}_proj.weight"));
    let attn_bias = |kind: &str| name(&format!("layers.{index}.self_attn.{kind}_proj.bias"));
    let norm =
        |number: usize, kind: &str| name(&format!("layers.{index}.layer_norm{number}.{kind}"));
    let first_norm_gain = norm(1, "weight");
    let first_norm_bias = norm(1, "bias");
    let second_norm_gain = norm(2, "weight");
    let second_norm_bias = norm(2, "bias");
    let q_key = attn("q");
    let k_key = attn("k");
    let v_key = attn("v");
    let o_key = attn("out");
    let fc1_key = name(&format!("layers.{index}.mlp.fc1.weight"));
    let fc2_key = name(&format!("layers.{index}.mlp.fc2.weight"));
    Ok(LayerTensors {
        query: load_matrix(map, path, &q_key, dim, dim)?,
        key: load_matrix(map, path, &k_key, dim, dim)?,
        value: load_matrix(map, path, &v_key, dim, dim)?,
        output: load_matrix(map, path, &o_key, dim, dim)?,
        expand: load_matrix(map, path, &fc1_key, hidden, dim)?,
        contract: load_matrix(map, path, &fc2_key, dim, hidden)?,
        query_bias: load_tensor(map, path, &attn_bias("q"), &[dim])?,
        key_bias: load_tensor(map, path, &attn_bias("k"), &[dim])?,
        value_bias: load_tensor(map, path, &attn_bias("v"), &[dim])?,
        output_bias: load_tensor(map, path, &attn_bias("out"), &[dim])?,
        expand_bias: load_tensor(
            map,
            path,
            &name(&format!("layers.{index}.mlp.fc1.bias")),
            &[hidden],
        )?,
        contract_bias: load_tensor(
            map,
            path,
            &name(&format!("layers.{index}.mlp.fc2.bias")),
            &[dim],
        )?,
        norm1_gain: load_tensor(map, path, &first_norm_gain, &[dim])?,
        norm1_bias: load_tensor(map, path, &first_norm_bias, &[dim])?,
        norm2_gain: load_tensor(map, path, &second_norm_gain, &[dim])?,
        norm2_bias: load_tensor(map, path, &second_norm_bias, &[dim])?,
    })
}

/// Owned f32 tensors for the whole text tower.
pub struct ClipTensors {
    token_table: Vec<f32>,
    position_table: Vec<f32>,
    layers: Vec<LayerTensors>,
    ffn_hidden: usize,
    eps: f32,
    final_norm_gain: Vec<f32>,
    final_norm_bias: Vec<f32>,
}

impl ClipTensors {
    /// Read and validate every declared tensor without touching payloads of
    /// unrelated model components.
    pub fn load(
        map: &CheckpointMap,
        path: &Path,
        geometry: ClipGeometry,
        eps: f32,
    ) -> io::Result<Self> {
        let token_key = name("embeddings.token_embedding.weight");
        let position_key = name("embeddings.position_embedding.weight");
        if geometry.vocab == 0
            || geometry.tokens == 0
            || geometry.dim == 0
            || geometry.ffn_hidden == 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "clip checkpoint: geometry must be non-zero",
            ));
        }
        let token_table = load_tensor(map, path, &token_key, &[geometry.vocab, geometry.dim])?;
        let position_table =
            load_tensor(map, path, &position_key, &[geometry.tokens, geometry.dim])?;
        let final_norm_gain =
            load_tensor(map, path, &name("final_layer_norm.weight"), &[geometry.dim])?;
        let final_norm_bias =
            load_tensor(map, path, &name("final_layer_norm.bias"), &[geometry.dim])?;
        let mut layers = Vec::with_capacity(geometry.layers);
        for index in 0..geometry.layers {
            layers.push(load_layer(map, path, index, geometry)?);
        }
        Ok(Self {
            token_table,
            position_table,
            layers,
            ffn_hidden: geometry.ffn_hidden,
            eps,
            final_norm_gain,
            final_norm_bias,
        })
    }

    /// Borrow the tensors as the encoder's weight bundle.
    pub fn weights(&self) -> ClipTextEncoderWeights<'_> {
        let blocks: Vec<TransformerWeights<'_>> = self
            .layers
            .iter()
            .map(|layer| TransformerWeights {
                wq: &layer.query,
                wk: &layer.key,
                wv: &layer.value,
                wo: &layer.output,
                w1: &layer.expand,
                w2: &layer.contract,
                norm1_gamma: &layer.norm1_gain,
                norm1_beta: &layer.norm1_bias,
                norm2_gamma: &layer.norm2_gain,
                norm2_beta: &layer.norm2_bias,
                eps: self.eps,
            })
            .collect();
        let biases: Vec<TransformerBias<'_>> = self
            .layers
            .iter()
            .map(|layer| TransformerBias {
                bq: Some(&layer.query_bias),
                bk: Some(&layer.key_bias),
                bv: Some(&layer.value_bias),
                bo: Some(&layer.output_bias),
                b1: Some(&layer.expand_bias),
                b2: Some(&layer.contract_bias),
            })
            .collect();
        ClipTextEncoderWeights {
            token_table: &self.token_table,
            position_table: &self.position_table,
            layers: blocks,
            biases,
            ffn_hidden: self.ffn_hidden,
        }
    }

    /// Final layer norm and EOS id for pooled conditioning.
    pub fn pooling(&self) -> ClipPooling<'_> {
        ClipPooling {
            gamma: &self.final_norm_gain,
            beta: &self.final_norm_bias,
            eps: self.eps,
            eos: 49407,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn layer_keys_match_single_file_layout() {
        assert_eq!(
            name("layers.0.self_attn.q_proj.weight"),
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            name("embeddings.token_embedding.weight"),
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
        );
    }

    /// Owns one unique temp file; removal happens on every exit path.
    struct TempFile(PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn fixture(omit: Option<&str>) -> (TempFile, CheckpointMap) {
        let geometry = ClipGeometry {
            vocab: 3,
            tokens: 2,
            dim: 2,
            ffn_hidden: 3,
            layers: 1,
        };
        let mut weights: Vec<(String, Vec<usize>)> = vec![
            (name("embeddings.token_embedding.weight"), vec![3, 2]),
            (name("embeddings.position_embedding.weight"), vec![2, 2]),
        ];
        for index in 0..geometry.layers {
            for kind in ["q", "k", "v", "out"] {
                weights.push((
                    name(&format!("layers.{index}.self_attn.{kind}_proj.weight")),
                    vec![2, 2],
                ));
                weights.push((
                    name(&format!("layers.{index}.self_attn.{kind}_proj.bias")),
                    vec![2],
                ));
            }
            weights.push((name(&format!("layers.{index}.mlp.fc1.weight")), vec![3, 2]));
            weights.push((name(&format!("layers.{index}.mlp.fc1.bias")), vec![3]));
            weights.push((name(&format!("layers.{index}.mlp.fc2.weight")), vec![2, 3]));
            weights.push((name(&format!("layers.{index}.mlp.fc2.bias")), vec![2]));
            for number in 1..=2 {
                for field in ["weight", "bias"] {
                    weights.push((
                        name(&format!("layers.{index}.layer_norm{number}.{field}")),
                        vec![2],
                    ));
                }
            }
        }
        weights.push((name("final_layer_norm.weight"), vec![2]));
        weights.push((name("final_layer_norm.bias"), vec![2]));
        weights.retain(|(key, _)| Some(key.as_str()) != omit);
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (key, shape) in weights {
            let count: usize = shape.iter().product();
            let start = payload.len();
            for index in 0..count {
                let value = f32::from(u8::try_from(index % 9).expect("0..9")) * 0.1;
                payload.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                key,
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, payload.len()],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header).expect("header");
        let mut bytes = u64::try_from(header_bytes.len())
            .expect("length")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-clip-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write checkpoint");
        let map = CheckpointMap::read(&path).expect("map");
        (TempFile(path), map)
    }

    fn geometry() -> ClipGeometry {
        ClipGeometry {
            vocab: 3,
            tokens: 2,
            dim: 2,
            ffn_hidden: 3,
            layers: 1,
        }
    }

    #[test]
    fn loads_and_transposes_checkpoint_matrices() {
        let (file, map) = fixture(None);
        let tensors = ClipTensors::load(&map, &file.0, geometry(), 1e-5).expect("load");
        let weights = tensors.weights();
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.ffn_hidden, 3);
        assert_eq!(weights.token_table.len(), 6);
        assert_eq!(weights.position_table.len(), 4);
        let layer = &weights.layers[0];
        assert_eq!(layer.wq.len(), 4);
        assert_eq!(layer.w1.len(), 6);
        assert_eq!(layer.w2.len(), 6);
        // fc1 checkpoint is [3,2] filled 0.0,0.1,...; local [2,3] is its
        // transpose: [0.0,0.2,0.4, 0.1,0.3,0.5].
        assert_eq!(layer.w1[1], 0.2);
        assert_eq!(layer.w1[2], 0.4);
    }

    #[test]
    fn audit_checks_required_clip_tensors() {
        let (file, map) = fixture(None);
        assert!(audit(&map, geometry()).is_ready());
        drop(file);

        let omitted = name("layers.0.self_attn.q_proj.weight");
        let (file, map) = fixture(Some(&omitted));
        let report = audit(&map, geometry());
        assert!(!report.is_ready());
        assert!(report.missing.contains(&omitted));
        drop(file);
    }

    #[test]
    fn reports_missing_tensors_and_bad_geometry() {
        let missing = name("layers.0.self_attn.q_proj.weight");
        let (file, map) = fixture(Some(&missing));
        assert!(ClipTensors::load(&map, &file.0, geometry(), 1e-5).is_err());
        let mut bad = geometry();
        bad.dim = 0;
        assert!(ClipTensors::load(&map, &file.0, bad, 1e-5).is_err());
    }
}
