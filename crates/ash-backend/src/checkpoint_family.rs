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
    /// SD 2-base single-file layout with the 1024-wide OpenCLIP text tower.
    Sd2,
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
            Self::Sd2 => "sd2",
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
/// Shared CLIP position embedding key. SD 1.5 is 768-wide; SD 2-base uses the
/// 1024-wide OpenCLIP tower (77 positions, 1024 hidden).
const CLIP_POSITION: &str =
    "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight";
const CLIP_TOKENS: usize = 77;
const SD2_CLIP_HIDDEN: usize = 1024;

fn has_sd_unet_input(map: &CheckpointMap) -> bool {
    map.get(SD15_MARKER).is_some_and(|tensor| {
        tensor.shape.len() == 4
            && tensor.shape[1] == 4
            && tensor.shape[2] == 3
            && tensor.shape[3] == 3
    })
}

fn has_sd2_text_tower(map: &CheckpointMap) -> bool {
    map.get(CLIP_POSITION)
        .is_some_and(|tensor| tensor.shape == [CLIP_TOKENS, SD2_CLIP_HIDDEN])
}

/// Detect a family from tensor names and header shapes; no payload is read.
pub fn detect(map: &CheckpointMap) -> CheckpointFamily {
    if SDXL_MARKERS.iter().any(|key| map.get(key).is_some()) {
        CheckpointFamily::Sdxl
    } else if !has_sd_unet_input(map) {
        CheckpointFamily::Unknown
    } else if has_sd2_text_tower(map) {
        CheckpointFamily::Sd2
    } else {
        CheckpointFamily::Sd15
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
        let shaped: Vec<_> = keys
            .iter()
            .map(|key| {
                let shape: &[usize] = if *key == SD15_MARKER {
                    &[320, 4, 3, 3]
                } else if *key == CLIP_POSITION {
                    &[CLIP_TOKENS, SD2_CLIP_HIDDEN]
                } else {
                    &[1]
                };
                (*key, shape)
            })
            .collect();
        header_shaped(&shaped)
    }

    fn header_shaped(keys: &[(&str, &[usize])]) -> (TempFile, CheckpointMap) {
        let mut entries = serde_json::Map::new();
        let mut payload = Vec::new();
        for (key, shape) in keys {
            let start = payload.len();
            payload.resize(payload.len() + shape.iter().product::<usize>() * 4, 0);
            entries.insert(
                (*key).to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": *shape,
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

    #[test]
    fn detects_sd2_by_openclip_position_width() {
        let (_file, map) = header(&[SD15_MARKER, CLIP_POSITION]);
        assert_eq!(detect(&map), CheckpointFamily::Sd2);
        assert_eq!(CheckpointFamily::Sd2.label(), "sd2");

        let (_file, map) =
            header_shaped(&[(SD15_MARKER, &[320, 4, 3, 3]), (CLIP_POSITION, &[77, 768])]);
        assert_eq!(detect(&map), CheckpointFamily::Sd15);

        let (_file, map) = header_shaped(&[
            (SD15_MARKER, &[320, 4, 3, 3]),
            (CLIP_POSITION, &[77, 1024]),
            (SDXL_MARKERS[0], &[1]),
        ]);
        assert_eq!(detect(&map), CheckpointFamily::Sdxl);
    }

    #[test]
    fn refuses_wrong_unet_geometry_and_missing_keys() {
        for shape in [&[320, 5, 3, 3][..], &[320, 4, 1, 1][..], &[320, 4, 3][..]] {
            let (_file, map) = header_shaped(&[(SD15_MARKER, shape), (CLIP_POSITION, &[77, 1024])]);
            assert_eq!(detect(&map), CheckpointFamily::Unknown, "{shape:?}");
        }
        let (_file, map) = header_shaped(&[(CLIP_POSITION, &[77, 1024])]);
        assert_eq!(detect(&map), CheckpointFamily::Unknown);
    }
}
