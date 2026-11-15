//! Placeholder seam types for the direct-Vulkan LLM/STT/TTS paths.
//!
//! Per `ReviewedAsh-FeaturePlan` §1 the shortcut stack stays canonical for
//! these modalities (llama.cpp / whisper.cpp are already Vulkan-native; CPU
//! Kokoro overlaps GPU work), so these stubs only reserve the seams and
//! backend ids — defined in this crate, NOT wired into `server` state (that
//! is a later ticket). The DI shape mirrors sd-core: constructible now,
//! erroring verbs until real implementations exist.

use crate::device::VulkanDevice;
use std::fmt;
use std::sync::Arc;

/// Failure of a placeholder direct-Vulkan seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StubError {
    /// The direct-Vulkan implementation for this modality does not exist yet.
    Unimplemented(&'static str),
}

impl fmt::Display for StubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unimplemented(modality) => {
                write!(f, "direct-Vulkan {modality} seam is not implemented yet")
            }
        }
    }
}

impl std::error::Error for StubError {}

/// Placeholder direct-Vulkan LLM seam (backend id `ash-llm`).
#[derive(Debug)]
pub struct AshLlmPipeline {
    device: Arc<VulkanDevice>,
}

impl AshLlmPipeline {
    /// Bind the (future) execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this seam will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Backend tag (`"ash-llm"`).
    pub fn backend_id(&self) -> &'static str {
        "ash-llm"
    }

    /// Placeholder for the future `LlmPipeline`-shaped completion call.
    pub fn complete(&self) -> Result<(), StubError> {
        Err(StubError::Unimplemented("llm"))
    }
}

/// Placeholder direct-Vulkan STT seam (backend id `ash-stt`).
#[derive(Debug)]
pub struct AshSttPipeline {
    device: Arc<VulkanDevice>,
}

impl AshSttPipeline {
    /// Bind the (future) execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this seam will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Backend tag (`"ash-stt"`).
    pub fn backend_id(&self) -> &'static str {
        "ash-stt"
    }

    /// Placeholder for the future `SttPipeline`-shaped transcribe call.
    pub fn transcribe(&self) -> Result<(), StubError> {
        Err(StubError::Unimplemented("stt"))
    }
}

/// Placeholder direct-Vulkan TTS seam (backend id `ash-tts`).
#[derive(Debug)]
pub struct AshTtsPipeline {
    device: Arc<VulkanDevice>,
}

impl AshTtsPipeline {
    /// Bind the (future) execution device.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self { device }
    }

    /// The device this seam will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Backend tag (`"ash-tts"`).
    pub fn backend_id(&self) -> &'static str {
        "ash-tts"
    }

    /// Placeholder for the future `TtsPipeline`-shaped synthesize call.
    pub fn synthesize(&self) -> Result<(), StubError> {
        Err(StubError::Unimplemented("tts"))
    }
}

// The seams will ultimately sit behind `Send + Sync` pipeline traits; pin
// that contract on the stubs now so it cannot regress as fields grow.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AshLlmPipeline>();
    assert_send_sync::<AshSttPipeline>();
    assert_send_sync::<AshTtsPipeline>();
};
