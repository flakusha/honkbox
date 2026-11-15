//! Experimental direct-Vulkan backend (`ash`) — skeleton + P0 device groundwork.
//!
//! Design docs: `.tmp/research/ReviewedAsh-FeaturePlan.md` (feature wiring,
//! experimental policy) and `.tmp/research/ReviewedAsh-DirectVulkan.md`
//! (loader choice, kernel inventory). The default build stays on the shortcut
//! stack (`honkbox-sd-core`'s [`honkbox_sd_core::SdCppPipeline`]); this crate
//! is the promoted home of the ground-up path and compiles to an empty lib
//! unless the `experimental-ash-sd` feature is on
//! (`server/experimental-ash-sd` → `honkbox-ash-backend/experimental-ash-sd`
//! → `honkbox-sd-core/ash-backend`).
//!
//! # Layout
//! - [`raw`]: the single `unsafe` boundary — thin 1:1 wrappers over ash FFI.
//! - [`device`]: [`device::VulkanDevice`] (loader + headless instance +
//!   logical device + compute queue) and physical-device selection
//!   (discrete > integrated > cpu, then VRAM), exposing properties, limits,
//!   and device-local VRAM for future capacity math.
//! - [`sd`]: [`sd::AshSdPipeline`], implementing `honkbox-sd-core`'s
//!   `SdPipeline`; `generate` returns a clear unimplemented error for now.
//! - [`stubs`]: placeholder LLM/STT/TTS seam types (backend ids `ash-llm`,
//!   `ash-stt`, `ash-tts`); not wired into server state (later ticket).
//!
//! # Unsafe policy
//! The workspace forbids `unsafe` everywhere (`unsafe_code = "forbid"`), but
//! Vulkan is a C ABI: no Vulkan operation is expressible without it. This
//! crate re-enables the lint locally (see its `Cargo.toml` for the documented
//! justification) and structurally confines every `unsafe` block to [`raw`] —
//! one audited wrapper per Vulkan entry point, each with a SAFETY comment.
//! The crate-root `#![deny(unsafe_code)]` turns unsafe anywhere else into a
//! compile error; everything above [`raw`] is safe abstraction.
#![deny(unsafe_code)]

/// Failures from the Vulkan init path (loader, instance, device selection,
/// logical device creation).
#[cfg(feature = "experimental-ash-sd")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VkError {
    /// System Vulkan loader missing or failed to load (`Entry::load`).
    Loader(String),
    /// Instance-level API failure (`vkCreateInstance`, enumeration, queries).
    Instance(String),
    /// Logical-device failure (`vkCreateDevice`, queue retrieval).
    Device(String),
    /// No physical device is present.
    NoDevices,
    /// Physical devices exist but none exposes a compute-capable queue family.
    NoComputeQueue,
}

#[cfg(feature = "experimental-ash-sd")]
impl VkError {
    /// Wrap an ash [`ash::vk::Result`] from an instance-level `op`.
    pub(crate) fn instance(op: &'static str, res: ash::vk::Result) -> Self {
        Self::Instance(format!("{op} failed: {res:?}"))
    }

    /// Wrap an ash [`ash::vk::Result`] from a device-level `op`.
    pub(crate) fn device(op: &'static str, res: ash::vk::Result) -> Self {
        Self::Device(format!("{op} failed: {res:?}"))
    }
}

#[cfg(feature = "experimental-ash-sd")]
impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loader(e) => write!(f, "vulkan loader failed to load: {e}"),
            Self::Instance(e) => write!(f, "vulkan instance error: {e}"),
            Self::Device(e) => write!(f, "vulkan device error: {e}"),
            Self::NoDevices => write!(f, "no vulkan physical devices found"),
            Self::NoComputeQueue => write!(f, "no physical device with a compute queue family"),
        }
    }
}

#[cfg(feature = "experimental-ash-sd")]
impl std::error::Error for VkError {}

#[cfg(feature = "experimental-ash-sd")]
pub mod device;
#[cfg(feature = "experimental-ash-sd")]
pub mod raw;
#[cfg(feature = "experimental-ash-sd")]
pub mod sd;
#[cfg(feature = "experimental-ash-sd")]
pub mod stubs;
