//! Strict safetensors header reader: checkpoint key → tensor slot map.
//!
//! The catalog already probes headers leniently for metadata; inference needs
//! the opposite — names, shapes and byte ranges that are safe to read before
//! any GPU work is queued. Weight bytes are only copied when asked for; no
//! model data is bundled.

use serde::Deserialize;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Safetensors header cap, same bound as the catalog probe.
const MAX_HEADER: u64 = 16 * 1024 * 1024;

/// Map one diffusers-style model-spec key to its original SD 1.5 checkpoint
/// name. Mirrors the subset of vendored stable-diffusion.cpp
/// `convert_diffusers_*_to_original_sd1` used by this backend's spec.
///
/// ponytail: explicit prefixes, not a generic name-conversion engine; extend
/// only when a new spec key is added.
fn sd15_original_key(name: &str) -> Cow<'_, str> {
    let vae = name.starts_with("decoder.");
    let mut mapped = if vae {
        name.strip_prefix("decoder.")
            .expect("checked prefix")
            .to_owned()
    } else {
        name.to_owned()
    };

    if !vae {
        for (from, to) in [
            ("conv_in.", "input_blocks.0.0."),
            ("conv_norm_out.", "out.0."),
            ("conv_out.", "out.2."),
            ("time_embedding.linear_1.", "time_embed.0."),
            ("time_embedding.linear_2.", "time_embed.2."),
            ("down_blocks.0.downsamplers.0.conv.", "input_blocks.3.0.op."),
            ("up_blocks.1.upsamplers.0.conv.", "output_blocks.5.2.conv."),
            ("down_blocks.0.resnets.0.", "input_blocks.1.0."),
            ("down_blocks.0.resnets.1.", "input_blocks.2.0."),
            ("mid_block.resnets.0.", "middle_block.0."),
            ("up_blocks.1.resnets.0.", "output_blocks.3.0."),
            ("up_blocks.1.resnets.1.", "output_blocks.4.0."),
            ("norm1.", "in_layers.0."),
            ("conv1.", "in_layers.2."),
            ("norm2.", "out_layers.0."),
            ("conv2.", "out_layers.3."),
        ] {
            if let Some(pos) = mapped.find(from) {
                mapped.replace_range(pos..pos + from.len(), to);
            }
        }
    } else if let Some(rest) = mapped.strip_prefix("conv_norm_out.") {
        mapped = format!("norm_out.{rest}");
    }

    let prefix = if vae {
        "first_stage_model.decoder."
    } else {
        "model.diffusion_model."
    };
    Cow::Owned(format!("{prefix}{mapped}"))
}

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One tensor located in a safetensors file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    /// Checkpoint key (`__metadata__` is never included).
    pub name: String,
    /// Safetensors dtype name (`F32`, `F16`, …).
    pub dtype: String,
    /// Tensor dimensions, in file order.
    pub shape: Vec<usize>,
    /// Byte offset relative to the start of the data section.
    pub offset: usize,
    /// Exact byte length of the tensor payload.
    pub byte_len: usize,
}

/// Validated tensor map for one safetensors checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointMap {
    tensors: BTreeMap<String, TensorMeta>,
    data_start: u64,
    file_len: u64,
}

fn dtype_width(dtype: &str) -> Option<usize> {
    match dtype {
        "F64" | "I64" => Some(8),
        "F32" | "I32" => Some(4),
        "F16" | "BF16" | "I16" => Some(2),
        "I8" | "U8" | "BOOL" | "F8_E4M3FN" | "F8_E5M2" => Some(1),
        _ => None,
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

impl CheckpointMap {
    /// Read and validate the header of `path`. No tensor payload is read.
    pub fn read(path: &Path) -> io::Result<Self> {
        let file_len = std::fs::metadata(path)?.len();
        let mut file = std::fs::File::open(path)?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > MAX_HEADER || header_len > file_len.saturating_sub(8) {
            return Err(invalid("safetensors: invalid header length"));
        }
        let mut bytes =
            vec![0u8; usize::try_from(header_len).map_err(|_| invalid("header too large"))?];
        file.read_exact(&mut bytes)?;
        let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes)
            .map_err(|e| invalid(format!("safetensors header: {e}")))?;

        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::with_capacity(header.len());
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry: HeaderTensor = serde_json::from_value(value)
                .map_err(|e| invalid(format!("safetensors tensor '{name}': {e}")))?;
            let width = dtype_width(&entry.dtype)
                .ok_or_else(|| invalid(format!("safetensors tensor '{name}': unknown dtype")))?;
            let elements = entry.shape.iter().copied().try_fold(1usize, |acc, dim| {
                (dim > 0)
                    .then_some(acc)
                    .and_then(|acc| acc.checked_mul(dim))
            });
            let elements = elements.ok_or_else(|| {
                invalid(format!(
                    "safetensors tensor '{name}': empty or overflowing shape"
                ))
            })?;
            let byte_len = elements.checked_mul(width).ok_or_else(|| {
                invalid(format!("safetensors tensor '{name}': byte length overflow"))
            })?;
            let [start, end] = entry.data_offsets;
            if end.checked_sub(start) != Some(byte_len) {
                return Err(invalid(format!(
                    "safetensors tensor '{name}': offsets do not match shape and dtype"
                )));
            }
            ranges.push((start, end));
            tensors.insert(
                name.clone(),
                TensorMeta {
                    name,
                    dtype: entry.dtype,
                    shape: entry.shape,
                    offset: start,
                    byte_len,
                },
            );
        }

        ranges.sort_unstable();
        let mut cursor = 0usize;
        for (start, end) in ranges {
            if start != cursor {
                return Err(invalid("safetensors: tensor ranges overlap or leave gaps"));
            }
            cursor = end;
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| invalid("safetensors: data start overflow"))?;
        let data_len =
            u64::try_from(cursor).map_err(|_| invalid("safetensors: data section exceeds u64"))?;
        if data_start
            .checked_add(data_len)
            .is_none_or(|end| end > file_len)
        {
            return Err(invalid("safetensors: tensor data exceeds file length"));
        }
        Ok(Self {
            tensors,
            data_start,
            file_len,
        })
    }

    /// Look up one tensor by checkpoint key.
    pub fn get(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// Look up a model-spec key in spec-local and single-file layouts.
    ///
    /// Exact matches win. Specs use diffusers names; single-file SD 1.5
    /// checkpoints use the original names mapped by
    /// [`sd15_original_key`]. Fixtures with bare keys keep working.
    pub fn get_aliased(&self, name: &str) -> Option<&TensorMeta> {
        if let Some(meta) = self.tensors.get(name) {
            return Some(meta);
        }
        let alias = sd15_original_key(name);
        if let Some(meta) = self.tensors.get(alias.as_ref()) {
            return Some(meta);
        }
        // Spec-local fixture layout: spec name with only the family prefix.
        let prefix = if name.starts_with("decoder.") {
            "first_stage_model."
        } else {
            "model.diffusion_model."
        };
        self.tensors.get(&format!("{prefix}{name}"))
    }

    /// Number of tensors in the map.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the checkpoint declares no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Byte offset of the data section in the file.
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// Total file length observed at read time.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Read one tensor's payload bytes from the file.
    pub fn read_tensor(&self, path: &Path, name: &str) -> io::Result<Vec<u8>> {
        let meta = self
            .get(name)
            .ok_or_else(|| invalid(format!("safetensors: no tensor named '{name}'")))?;
        let mut file = std::fs::File::open(path)?;
        let offset = self
            .data_start
            .checked_add(u64::try_from(meta.offset).map_err(|_| invalid("offset exceeds u64"))?)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; meta.byte_len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

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

    impl std::ops::Deref for TempFile {
        type Target = std::path::PathBuf;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    fn checkpoint(entries: &serde_json::Value, payload: &[u8]) -> TempFile {
        let header = serde_json::to_vec(entries).expect("header");
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-map-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = u64::try_from(header.len())
            .expect("len")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(payload);
        std::fs::write(&path, bytes).expect("write checkpoint");
        TempFile(path)
    }

    fn two_tensor_header() -> serde_json::Value {
        serde_json::json!({
            "__metadata__": { "format": "pt" },
            "weight": { "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16] },
            "bias": { "dtype": "F16", "shape": [2], "data_offsets": [16, 20] }
        })
    }

    #[test]
    fn reads_tensor_map_and_payload() {
        let payload = [
            1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];
        let path = checkpoint(&two_tensor_header(), &payload);
        let map = CheckpointMap::read(&path).expect("map");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("__metadata__"), None);
        let weight = map.get("weight").expect("weight");
        assert_eq!(
            (weight.dtype.as_str(), weight.shape.as_slice()),
            ("F32", &[2usize, 2][..])
        );
        assert_eq!((weight.offset, weight.byte_len), (0, 16));
        assert_eq!(
            map.read_tensor(&path, "weight").expect("bytes"),
            payload[..16]
        );
        assert_eq!(
            map.read_tensor(&path, "bias").expect("bytes"),
            payload[16..]
        );
        assert!(map.read_tensor(&path, "missing").is_err());
    }

    #[test]
    fn maps_diffusers_spec_names_to_original_checkpoint_names() {
        assert_eq!(
            sd15_original_key("conv_in.weight").as_ref(),
            "model.diffusion_model.input_blocks.0.0.weight"
        );
        assert_eq!(
            sd15_original_key("down_blocks.0.resnets.0.norm1.weight").as_ref(),
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight"
        );
        assert_eq!(
            sd15_original_key("up_blocks.1.upsamplers.0.conv.weight").as_ref(),
            "model.diffusion_model.output_blocks.5.2.conv.weight"
        );
        assert_eq!(
            sd15_original_key("decoder.conv_norm_out.weight").as_ref(),
            "first_stage_model.decoder.norm_out.weight"
        );
    }

    #[test]
    fn rejects_bad_header_and_ranges() {
        let empty = checkpoint(&serde_json::json!({}), b"");
        let map = CheckpointMap::read(&empty).expect("empty map");
        assert!(map.is_empty());

        let overlapping = serde_json::json!({
            "a": { "dtype": "F32", "shape": [2], "data_offsets": [0, 8] },
            "b": { "dtype": "F32", "shape": [2], "data_offsets": [4, 12] }
        });
        let path = checkpoint(&overlapping, &[0u8; 12]);
        assert!(CheckpointMap::read(&path).is_err());

        let unknown = serde_json::json!({
            "a": { "dtype": "F128", "shape": [2], "data_offsets": [0, 8] }
        });
        let path = checkpoint(&unknown, &[0u8; 8]);
        assert!(CheckpointMap::read(&path).is_err());
    }
}
