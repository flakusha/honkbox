//! Cheap checkpoint-family detection from tensor names.
//!
//! The experimental ash backend currently implements a single SD 1.5-level
//! topology. SDXL shares the `UNet` prefix but adds adm conditioning and a
//! second text tower, so reporting it as "missing SD 1.5 keys" is wrong.

use crate::checkpoint::CheckpointMap;

/// Family recognised from header names only; no payload is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFamily {
    /// SD 1.x single-file layout targeted by the current backend.
    Sd15,
    /// SDXL: dual text towers, adm conditioning, 2048-wide context.
    Sdxl,
    /// No recognised marker; do not guess a family.
    Unknown,
}

impl CheckpointFamily {
    /// Short name used in CLI output.
    pub fn label(self) -> &'static str {
        match self {
            Self::Sd15 => "sd15",
            Self::Sdxl => "sdxl",
            Self::Unknown => "unknown",
        }
    }
}

const SDXL_MARKERS: [&str; 4] = [
    "model.diffusion_model.label_emb.0.0.weight",
    "conditioner.embedders.1.transformer.text_model.embeddings.token_embedding.weight",
    "cond_stage_model.1.transformer.text_model.embeddings.token_embedding.weight",
    // Real single-file SDXL (comfy/fooocus) OpenCLIP bigG tower.
    "conditioner.embedders.1.model.token_embedding.weight",
];

const SD15_MARKER: &str = "model.diffusion_model.input_blocks.0.0.weight";

/// Detect a family from distinctive tensor names.
pub fn detect(map: &CheckpointMap) -> CheckpointFamily {
    if SDXL_MARKERS.iter().any(|key| map.get(key).is_some()) {
        CheckpointFamily::Sdxl
    } else if map.get(SD15_MARKER).is_some() {
        CheckpointFamily::Sd15
    } else {
        CheckpointFamily::Unknown
    }
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

    fn header(keys: &[&str]) -> (TempFile, CheckpointMap) {
        let mut entries = serde_json::Map::new();
        let mut payload = Vec::new();
        for key in keys {
            let start = payload.len();
            payload.extend_from_slice(&0.0_f32.to_le_bytes());
            entries.insert(
                (*key).to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [start, payload.len()],
                }),
            );
        }
        let bytes = serde_json::to_vec(&entries).expect("header");
        let mut file = u64::try_from(bytes.len())
            .expect("length")
            .to_le_bytes()
            .to_vec();
        file.extend_from_slice(&bytes);
        file.extend_from_slice(&payload);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-family-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, file).expect("write checkpoint");
        (
            TempFile(path.clone()),
            CheckpointMap::read(&path).expect("map"),
        )
    }

    #[test]
    fn detects_families_from_marker_keys() {
        let (_file, map) = header(&[SD15_MARKER]);
        assert_eq!(detect(&map), CheckpointFamily::Sd15);

        let (_file, map) = header(&[SD15_MARKER, SDXL_MARKERS[0]]);
        assert_eq!(detect(&map), CheckpointFamily::Sdxl);

        let (_file, map) = header(&[SDXL_MARKERS[3]]);
        assert_eq!(detect(&map), CheckpointFamily::Sdxl);

        let (_file, map) = header(&[]);
        assert_eq!(detect(&map), CheckpointFamily::Unknown);
    }
}
