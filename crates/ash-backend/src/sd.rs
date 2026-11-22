//! `SdPipeline` impl for the direct-Vulkan path: skeleton only.
//!
//! [`AshSdPipeline`] validates the request, device and checkpoint container
//! before sampling. Actual diffusion kernels are still missing, so
//! [`SdPipeline::generate`] stops with a named sampling error rather than
//! returning a fake image. The shortcut [`honkbox_sd_core::SdCppPipeline`]
//! stays the default backend; callers swap the constructor, never the call
//! sites.

use crate::{
    device::{CapacityRequirements, VulkanDevice},
    model::{ModelSpec, Op},
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

/// Reject checkpoints that cannot be loaded before any GPU work is queued.
/// Only container-level checks: weights are read when sampling lands.
fn validate_checkpoint(path: &Path) -> Result<(), SdError> {
    let fail = |msg: String| SdError::Backend(format!("checkpoint '{}': {msg}", path.display()));
    let meta = std::fs::metadata(path).map_err(|e| fail(e.to_string()))?;
    if !meta.is_file() || meta.len() == 0 {
        return Err(fail("not a non-empty file".into()));
    }
    let mut file = std::fs::File::open(path).map_err(|e| fail(e.to_string()))?;
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "safetensors" => {
            let mut len = [0u8; 8];
            file.read_exact(&mut len)
                .map_err(|e| fail(format!("truncated safetensors header: {e}")))?;
            let header = u64::from_le_bytes(len);
            if header == 0 || header > meta.len().saturating_sub(8) {
                return Err(fail(format!(
                    "invalid safetensors header length {header} for {} bytes",
                    meta.len()
                )));
            }
        }
        "gguf" => {
            let mut magic = [0u8; 4];
            file.read_exact(&mut magic)
                .map_err(|e| fail(format!("truncated GGUF header: {e}")))?;
            if &magic != b"GGUF" {
                return Err(fail("bad GGUF magic".into()));
            }
        }
        other => return Err(fail(format!("unsupported checkpoint format '{other}'"))),
    }
    Ok(())
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
        validate_checkpoint(&self.model_path)?;
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_checkpoint(ext: &str, bytes: &[u8]) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("honkbox-ash-{}-{stamp}.{ext}", std::process::id()));
        std::fs::write(&path, bytes).expect("write temp checkpoint");
        path
    }

    fn safetensors_bytes() -> Vec<u8> {
        let header = br#"{"__metadata__":{}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
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
    fn checkpoint_validation_accepts_supported_formats() {
        let safetensors = temp_checkpoint("safetensors", &safetensors_bytes());
        let gguf = temp_checkpoint("gguf", b"GGUF\x03\0\0\0rest");
        assert_eq!(validate_checkpoint(&safetensors), Ok(()));
        assert_eq!(validate_checkpoint(&gguf), Ok(()));
        std::fs::remove_file(safetensors).expect("cleanup");
        std::fs::remove_file(gguf).expect("cleanup");
    }

    #[test]
    fn checkpoint_validation_rejects_bad_inputs() {
        let missing = std::env::temp_dir().join("honkbox-ash-no-such-checkpoint.safetensors");
        let empty = temp_checkpoint("safetensors", b"");
        let wrong_ext = temp_checkpoint("bin", b"data");
        let bad_magic = temp_checkpoint("gguf", b"NOTGGUF!");
        let truncated = temp_checkpoint("safetensors", b"\x02\0");
        let mut bad_len = 1000u64.to_le_bytes().to_vec();
        bad_len.extend_from_slice(b"{}");
        let huge_header = temp_checkpoint("safetensors", &bad_len);
        for path in [
            &missing,
            &empty,
            &wrong_ext,
            &bad_magic,
            &truncated,
            &huge_header,
        ] {
            assert!(
                matches!(validate_checkpoint(path), Err(SdError::Backend(_))),
                "expected rejection for {}",
                path.display()
            );
        }
        for path in [empty, wrong_ext, bad_magic, truncated, huge_header] {
            std::fs::remove_file(path).expect("cleanup");
        }
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
        std::fs::remove_file(valid).expect("cleanup");
    }
}
