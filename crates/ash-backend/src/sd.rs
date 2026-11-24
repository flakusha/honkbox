//! `SdPipeline` impl for the direct-Vulkan path: skeleton only.
//!
//! [`AshSdPipeline`] validates the request, device and checkpoint container
//! before sampling. Actual diffusion kernels are still missing, so
//! [`SdPipeline::generate`] stops with a named sampling error rather than
//! returning a fake image. The shortcut [`honkbox_sd_core::SdCppPipeline`]
//! stays the default backend; callers swap the constructor, never the call
//! sites.

use crate::{
    checkpoint::CheckpointMap,
    device::{CapacityRequirements, VulkanDevice},
    model::{ModelSpec, Op},
    model_spec::{SD15_SINGLE_LEVEL, validate},
    shape_infer::{ShapeError, UnetShapes, infer},
};
use honkbox_sd_core::{GeneratedImage, MAX_EDGE, SdError, SdPipeline, Txt2ImgRequest};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sidecar/backend tag for the experimental direct-Vulkan SD path.
pub const ASH_SD_BACKEND_ID: &str = "ash-sd";

/// Direct-Vulkan SD pipeline skeleton — the eventual swap-in replacement for
/// [`honkbox_sd_core::SdCppPipeline`].
///
/// Same trait seam as the other backends: the server holds
/// `Box<dyn SdPipeline>` and only the constructor branches on the feature.
#[derive(Debug)]
pub struct AshSdPipeline {
    device: Arc<VulkanDevice>,
    model_path: PathBuf,
}

/// Passes the first checkpoints will use. Grows with the kernel inventory.
const SD_SPEC: ModelSpec = ModelSpec {
    name: "sd",
    ops: &[Op::ElementwiseAdd, Op::MatMul],
};

impl AshSdPipeline {
    /// Bind the execution device and the already-resolved checkpoint path.
    pub fn new(device: Arc<VulkanDevice>, model_path: PathBuf) -> Self {
        Self { device, model_path }
    }

    /// The device this pipeline will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Admission gate: call before any model load is reserved on this
    /// pipeline, so an unsupported device is named before work is queued.
    pub fn admit(&self, req: &CapacityRequirements) -> Result<(), crate::VkError> {
        self.device.info.admit(req)
    }
}

/// `UNet` input convolution — present in SD 1.x/2.x and SDXL safetensors.
const UNET_INPUT_TENSOR: &str = "model.diffusion_model.input_blocks.0.0.weight";
/// Latent channels the diffusion backbone consumes.
const LATENT_CHANNELS: usize = 4;

fn validate_unet_input(map: &CheckpointMap) -> Result<(), String> {
    let tensor = map
        .get(UNET_INPUT_TENSOR)
        .ok_or_else(|| format!("missing UNet input tensor '{UNET_INPUT_TENSOR}'"))?;
    if tensor.shape.len() != 4
        || tensor.shape[1] != LATENT_CHANNELS
        || tensor.shape[2] != 3
        || tensor.shape[3] != 3
    {
        return Err(format!(
            "UNet input tensor '{UNET_INPUT_TENSOR}' has shape {:?}, expected [*, 4, 3, 3]",
            tensor.shape
        ));
    }
    if !matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(format!(
            "UNet input tensor '{UNET_INPUT_TENSOR}' has dtype {}",
            tensor.dtype
        ));
    }
    Ok(())
}

/// Reject checkpoints that cannot be loaded before any GPU work is queued.
///
/// Safetensors gets a real tensor-map check: the `UNet` input must exist with
/// the expected NCHW shape and a floating dtype. GGUF has no safetensors map,
/// so it is refused by name instead of passing admission with unknown weights.
fn admit_checkpoint(path: &Path) -> Result<(), SdError> {
    let fail = |msg: String| SdError::Backend(format!("checkpoint '{}': {msg}", path.display()));
    let meta = std::fs::metadata(path).map_err(|e| fail(e.to_string()))?;
    if !meta.is_file() || meta.len() == 0 {
        return Err(fail("not a non-empty file".into()));
    }
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "safetensors" => {
            let map = CheckpointMap::read(path).map_err(|e| fail(e.to_string()))?;
            validate_unet_input(&map).map_err(fail)?;
        }
        "gguf" => {
            let mut file = std::fs::File::open(path).map_err(|e| fail(e.to_string()))?;
            let mut magic = [0u8; 4];
            file.read_exact(&mut magic)
                .map_err(|e| fail(format!("truncated GGUF header: {e}")))?;
            if &magic != b"GGUF" {
                return Err(fail("bad GGUF magic".into()));
            }
            return Err(fail(
                "GGUF checkpoints have no safetensors tensor map".into(),
            ));
        }
        other => return Err(fail(format!("unsupported checkpoint format '{other}'"))),
    }
    Ok(())
}

/// Validate a safetensors checkpoint against the declared model spec and
/// return the stage widths inferred from its tensor metadata.
///
/// No device is needed: this is the CLI/capability seam for checking a model
/// before reserving the GPU slot. Sampling still needs the full loader.
pub fn inspect_checkpoint(path: &Path) -> Result<UnetShapes, SdError> {
    let fail = |msg: String| SdError::Backend(format!("checkpoint '{}': {msg}", path.display()));
    let map = CheckpointMap::read(path).map_err(|e| fail(e.to_string()))?;
    validate(&SD15_SINGLE_LEVEL, &map).map_err(|e| fail(e.to_string()))?;
    infer(&SD15_SINGLE_LEVEL, &map).map_err(|e: ShapeError| fail(e.to_string()))
}

impl SdPipeline for AshSdPipeline {
    fn backend_id(&self) -> &'static str {
        ASH_SD_BACKEND_ID
    }

    fn generate(
        &self,
        req: &Txt2ImgRequest,
        cancel: &AtomicBool,
    ) -> Result<GeneratedImage, SdError> {
        let edge = req.width.max(req.height);
        if edge > MAX_EDGE {
            return Err(SdError::TooLarge { edge });
        }
        if req.width == 0 || req.height == 0 {
            return Err(SdError::Backend("image dimensions must be positive".into()));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(SdError::Cancelled);
        }
        SD_SPEC.validate(&self.device.info).map_err(|e| {
            SdError::Backend(format!(
                "{}: device cannot host model: {e}",
                self.backend_id()
            ))
        })?;
        admit_checkpoint(&self.model_path)?;
        // Sampling needs text encoders, diffusion backbone, VAE and a
        // scheduler; only the first two compute passes exist so far. Stop
        // here rather than return a plausible-looking image.
        Err(SdError::Backend(format!(
            "{}: sampling not implemented for '{}' ({}x{}, {} steps)",
            self.backend_id(),
            req.model_slug,
            req.width,
            req.height,
            req.steps
        )))
    }
}

// The seam contract (`SdPipeline: Send + Sync`) must keep holding as fields
// grow.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AshSdPipeline>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VkError;
    use crate::model_spec::tensor_keys;
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

    fn temp_checkpoint(ext: &str, bytes: &[u8]) -> TempFile {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-{}-{}-{stamp}.{ext}",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write temp checkpoint");
        TempFile(path)
    }

    fn spec_checkpoint(omit: Option<&str>, prefixed: bool) -> TempFile {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (_, key) in tensor_keys(&SD15_SINGLE_LEVEL) {
            if omit == Some(key.as_str()) {
                continue;
            }
            let dims = if key == "time_embedding.linear_1.weight" {
                vec![8, 4]
            } else if key == "time_embedding.linear_2.weight" {
                vec![8, 8]
            } else if key.ends_with(".weight") {
                vec![4, 4, 3, 3]
            } else {
                vec![4]
            };
            let count: usize = dims.iter().product();
            let start = payload.len();
            payload.resize(start + count * 4, 0);
            let end = payload.len();
            let prefix = if key.starts_with("decoder.") {
                "first_stage_model."
            } else {
                "model.diffusion_model."
            };
            let header_key = if prefixed {
                format!("{prefix}{key}")
            } else {
                key
            };
            header.insert(
                header_key,
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
        temp_checkpoint("safetensors", &bytes)
    }

    #[test]
    fn inspect_checkpoint_validates_then_infers() {
        let valid = spec_checkpoint(None, false);
        let shapes = inspect_checkpoint(&valid).expect("inspect");
        assert_eq!(shapes.latent_channels, 4);
        assert_eq!(shapes.timestep_hidden, 8);

        let missing = spec_checkpoint(Some("time_embedding.linear_2.bias"), false);
        assert!(
            matches!(inspect_checkpoint(&missing), Err(SdError::Backend(msg)) if msg.contains("timestep linear 2")),
            "missing declared tensor must be named"
        );
    }

    #[test]
    fn inspect_accepts_single_file_prefixed_keys() {
        let path = spec_checkpoint(None, true);
        let shapes = inspect_checkpoint(&path).expect("prefixed inspect");
        assert_eq!(shapes.latent_channels, 4);
        assert_eq!(shapes.timestep_hidden, 8);
    }

    fn safetensors_bytes() -> Vec<u8> {
        safetensors_with_tensor("[8,4,3,3]", "F32", 1152)
    }

    fn safetensors_with_tensor(shape: &str, dtype: &str, byte_len: usize) -> Vec<u8> {
        let header = format!(
            "{{\"{UNET_INPUT_TENSOR}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[0,{byte_len}]}}}}"
        );
        let mut bytes = u64::try_from(header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + byte_len, 0);
        bytes
    }

    fn request(width: u32, height: u32) -> Txt2ImgRequest {
        Txt2ImgRequest {
            prompt: "test".to_owned(),
            seed: 42,
            steps: 4,
            width,
            height,
            model_slug: "comfy:checkpoints/sdxl.safetensors".to_owned(),
            clip: None,
            vae: None,
            sampler: None,
            scheduler: None,
            cfg_scale: None,
            negative_prompt: None,
            loras: None,
            init_image: None,
            mask_image: None,
            strength: None,
            hires: None,
            control_net: None,
            control_image: None,
            controlnet_strength: None,
            control_type: None,
        }
    }

    fn device() -> Option<Arc<VulkanDevice>> {
        match VulkanDevice::headless() {
            Ok(device) => Some(Arc::new(device)),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                None
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        }
    }

    #[test]
    fn admit_gates_on_the_selected_device() {
        let Some(device) = device() else { return };
        let pipeline = AshSdPipeline::new(device, PathBuf::from("unused"));
        assert_eq!(pipeline.admit(&CapacityRequirements::default()), Ok(()));
        let impossible = CapacityRequirements {
            subgroup_size: u32::MAX,
            ..CapacityRequirements::default()
        };
        assert!(matches!(
            pipeline.admit(&impossible),
            Err(VkError::InsufficientCapacity {
                capability: "subgroupSize",
                ..
            })
        ));
    }

    #[test]
    fn checkpoint_admission_accepts_sd_safetensors() {
        let safetensors = temp_checkpoint("safetensors", &safetensors_bytes());
        assert_eq!(admit_checkpoint(&safetensors), Ok(()));
        let gguf = temp_checkpoint("gguf", b"GGUF\x03\0\0\0rest");
        assert!(
            matches!(admit_checkpoint(&gguf), Err(SdError::Backend(msg)) if msg.contains("tensor map")),
            "GGUF must be refused by name"
        );
    }

    #[test]
    fn checkpoint_admission_checks_unet_tensor() {
        let empty_header = br#"{"__metadata__":{}}"#;
        let mut empty_bytes = u64::try_from(empty_header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        empty_bytes.extend_from_slice(empty_header);
        let missing = temp_checkpoint("safetensors", &empty_bytes);
        assert!(
            matches!(admit_checkpoint(&missing), Err(SdError::Backend(msg)) if msg.contains("missing UNet")),
            "empty tensor map must be refused"
        );
        let wrong = temp_checkpoint(
            "safetensors",
            &safetensors_with_tensor("[8,3,3,3]", "F32", 864),
        );
        assert!(
            matches!(admit_checkpoint(&wrong), Err(SdError::Backend(msg)) if msg.contains("shape")),
            "wrong latent channels must be refused"
        );
    }

    #[test]
    fn checkpoint_admission_rejects_bad_inputs() {
        let missing = std::env::temp_dir().join("honkbox-ash-no-such-checkpoint.safetensors");
        let empty = temp_checkpoint("safetensors", b"");
        let wrong_ext = temp_checkpoint("bin", b"data");
        let bad_magic = temp_checkpoint("gguf", b"NOTGGUF!");
        let truncated = temp_checkpoint("safetensors", b"\x02\0");
        let mut bad_len = 1000u64.to_le_bytes().to_vec();
        bad_len.extend_from_slice(b"{}");
        let huge_header = temp_checkpoint("safetensors", &bad_len);
        for path in [
            missing.as_path(),
            empty.as_path(),
            wrong_ext.as_path(),
            bad_magic.as_path(),
            truncated.as_path(),
            huge_header.as_path(),
        ] {
            assert!(
                matches!(admit_checkpoint(path), Err(SdError::Backend(_))),
                "expected rejection for {}",
                path.display()
            );
        }
    }

    #[test]
    fn generate_rejects_checkpoint_without_unet_tensor() {
        let Some(device) = device() else { return };
        let empty_header = br#"{"__metadata__":{}}"#;
        let mut empty_bytes = u64::try_from(empty_header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        empty_bytes.extend_from_slice(empty_header);
        let path = temp_checkpoint("safetensors", &empty_bytes);
        let pipeline = AshSdPipeline::new(device, path.clone());
        assert!(matches!(
            pipeline.generate(&request(512, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("UNet")
        ));
    }

    #[test]
    fn generate_checks_request_then_checkpoint_then_sampling() {
        let Some(device) = device() else { return };
        let missing = PathBuf::from("honkbox-ash-no-such-checkpoint.safetensors");
        let pipeline = AshSdPipeline::new(device.clone(), missing.clone());

        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            pipeline.generate(&request(512, 512), &cancelled),
            Err(SdError::Cancelled)
        ));

        assert!(matches!(
            pipeline.generate(&request(3000, 512), &AtomicBool::new(false)),
            Err(SdError::TooLarge { edge: 3000 })
        ));
        assert!(matches!(
            pipeline.generate(&request(0, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("dimensions")
        ));
        assert!(matches!(
            pipeline.generate(&request(512, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("checkpoint")
        ));

        let valid = temp_checkpoint("safetensors", &safetensors_bytes());
        let ready = AshSdPipeline::new(device, valid.clone());
        assert!(matches!(
            ready.generate(&request(512, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("sampling not implemented")
        ));
    }
}
