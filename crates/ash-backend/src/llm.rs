//! Placeholder direct-Vulkan LLM seam: real `LlmPipeline`, no inference yet.
//!
//! Unlike the deterministic `StubLlm` (which fakes replies for tests), this
//! backend NEVER produces output: every verb returns a named
//! [`honkbox_llm_worker::LlmError`] naming the missing capability. It exists
//! so the server can construct and route the direct-Vulkan chat path end to
//! end (device selection, residency, accounting) while the transformer
//! kernels (embedding, attention over KV cache, sampler) land slice by slice.

use crate::device::VulkanDevice;
use honkbox_llm_worker::{ChatRequest, ChatResponse, LlmError, LlmPipeline};
use std::sync::Arc;

pub mod state;

/// Why the direct-Vulkan chat path cannot serve a request yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotYet;

impl std::fmt::Display for NotYet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "direct-Vulkan chat is not implemented yet: transformer decode \
             kernels land in later slices; no output is faked",
        )
    }
}

impl std::error::Error for NotYet {}

/// Placeholder direct-Vulkan chat pipeline (backend id `ash-llm`).
///
/// Holds the selected execution device so later slices build kernels on it
/// without re-plumbing the server seam. Every [`LlmPipeline`] verb fails
/// with [`LlmError::Load`] wrapping [`NotYet`] — the pipeline has no model,
/// so load-time failure is the honest classification, and it surfaces
/// before any decode could run.
///
/// Cancellation: no generation ever starts, so [`LlmPipeline::abort`] is the
/// trait-default no-op and `complete_stream_cancellable` keeps the legacy
/// forwarding behavior — there is no in-flight run to observe a token on.
#[derive(Debug)]
pub struct AshLlmPipeline {
    device: Arc<VulkanDevice>,
}

impl AshLlmPipeline {
    /// Bind the execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this seam will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}

fn unimplemented() -> LlmError {
    LlmError::Load(NotYet.to_string())
}

impl LlmPipeline for AshLlmPipeline {
    fn complete(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(unimplemented())
    }

    fn complete_stream(
        &self,
        _req: &ChatRequest,
        _sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        // The sink is never invoked: failing before any token exists is the
        // same honest posture as `complete`.
        Err(unimplemented())
    }

    fn backend_id(&self) -> &'static str {
        "ash-llm"
    }
}

// Pin the worker-seam thread-safety contract on the stub so it cannot
// regress as fields grow (same posture as the STT/TTS seams in `stubs.rs`).
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AshLlmPipeline>();
};

#[cfg(test)]
mod tests {
    use super::AshLlmPipeline;
    use crate::VkError;
    use crate::device::VulkanDevice;
    use honkbox_llm_worker::{ChatMessage, ChatRequest, LlmError, LlmPipeline, RequestCancel};
    use std::sync::Arc;

    fn request() -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage {
                content: "hello".to_string(),
                ..ChatMessage::default()
            }],
            ..ChatRequest::default()
        }
    }

    #[test]
    fn backend_id_is_ash_llm() {
        let device = Arc::new(match VulkanDevice::headless() {
            Ok(dev) => dev,
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("device creation failed: {e}"),
        });
        assert_eq!(AshLlmPipeline::new(device).backend_id(), "ash-llm");
    }

    #[test]
    fn complete_returns_named_unimplemented_error() {
        let device = Arc::new(match VulkanDevice::headless() {
            Ok(dev) => dev,
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("device creation failed: {e}"),
        });
        let pipeline = AshLlmPipeline::new(device);
        let err = pipeline
            .complete(&request())
            .expect_err("stub must not serve");
        assert!(matches!(err, LlmError::Load(_)));
        assert!(err.to_string().contains("not implemented"));
        assert!(err.to_string().contains("no output is faked"));
    }

    #[test]
    fn complete_stream_errors_without_emitting_tokens() {
        let device = Arc::new(match VulkanDevice::headless() {
            Ok(dev) => dev,
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("device creation failed: {e}"),
        });
        let pipeline = AshLlmPipeline::new(device);
        let mut tokens = Vec::new();
        let result = pipeline.complete_stream(&request(), &mut |t| tokens.push(t.to_string()));
        let err = result.expect_err("stub must not serve");
        assert!(matches!(err, LlmError::Load(_)));
        assert_eq!(
            tokens,
            Vec::<String>::new(),
            "no token may be emitted before failure"
        );
    }

    #[test]
    fn cancellable_stream_keeps_legacy_path() {
        let device = Arc::new(match VulkanDevice::headless() {
            Ok(dev) => dev,
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                return;
            }
            Err(e) => panic!("device creation failed: {e}"),
        });
        let pipeline = AshLlmPipeline::new(device);
        let cancel = RequestCancel::new();
        let mut tokens = Vec::new();
        let err = pipeline
            .complete_stream_cancellable(&request(), &mut |t| tokens.push(t.to_string()), &cancel)
            .expect_err("stub must not serve");
        assert!(matches!(err, LlmError::Load(_)));
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn pipeline_serves_on_a_real_device_when_present() {
        // Device-in-a-box: construction + seam binding on a real (or
        // lavapipe) device; skipped on loader-less machines.
        match VulkanDevice::headless() {
            Ok(dev) => {
                let device = Arc::new(dev);
                let pipeline = AshLlmPipeline::new(Arc::clone(&device));
                assert_eq!(pipeline.device().info.name, device.info.name);
            }
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
            }
            Err(e) => panic!("device creation failed: {e}"),
        }
    }
}
