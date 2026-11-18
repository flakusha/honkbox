//! `SdPipeline` impl for the direct-Vulkan path: skeleton only.
//!
//! [`AshSdPipeline`] is constructible and carries the future execution
//! device, but [`SdPipeline::generate`] returns a clear unimplemented error
//! until kernels land (kernel inventory: `.tmp/research/`
//! `ReviewedAsh-DirectVulkan.md` §2; effort phases §6). The shortcut
//! [`honkbox_sd_core::SdCppPipeline`] stays the default backend; callers
//! swap the constructor, never the call sites.

use crate::device::VulkanDevice;
use honkbox_sd_core::{GeneratedImage, SdError, SdPipeline, Txt2ImgRequest};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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
}

impl AshSdPipeline {
    /// Bind the (future) execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this pipeline will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}

impl SdPipeline for AshSdPipeline {
    fn backend_id(&self) -> &'static str {
        ASH_SD_BACKEND_ID
    }

    fn generate(
        &self,
        req: &Txt2ImgRequest,
        _cancel: &AtomicBool,
    ) -> Result<GeneratedImage, SdError> {
        // P0 skeleton: nothing is dispatchable yet; kernels arrive in P1+
        // (elementwise lift first, then matmul/norm, conv2d, attention, VAE
        // slice per ReviewedAsh-DirectVulkan §2). The cancel flag rides the
        // seam for the future kernel loop to poll.
        Err(SdError::Backend(format!(
            "{}: not implemented yet (experimental skeleton); requested {}x{}, \
             {} steps, model '{}'",
            self.backend_id(),
            req.width,
            req.height,
            req.steps,
            req.model_slug
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

    #[test]
    fn generate_is_a_clear_unimplemented_error() {
        // Needs a device only to satisfy the future shape; skips when the
        // machine has no Vulkan (see device tests for the policy).
        let device = match VulkanDevice::headless() {
            Ok(device) => Arc::new(device),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        };
        let pipeline = AshSdPipeline::new(device);
        assert_eq!(pipeline.backend_id(), "ash-sd");
        let req = Txt2ImgRequest {
            prompt: "test".to_owned(),
            seed: 42,
            steps: 4,
            width: 512,
            height: 512,
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
        };
        let err = pipeline.generate(&req, &AtomicBool::new(false))
            .expect_err("skeleton must not render");
        let SdError::Backend(msg) = err else {
            panic!("expected Backend variant, got {err:?}")
        };
        assert!(msg.contains("not implemented"), "unclear error: {msg}");
    }
}
