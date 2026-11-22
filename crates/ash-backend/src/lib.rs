//! Experimental direct-Vulkan backend (`ash`) — device groundwork + first
//! compute kernel.
//!
//! Design record: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md`
//! (the older `ReviewedAsh-*.md` refs in earlier comments were never present
//! in this repo). The default build stays on the shortcut stack
//! (`honkbox-sd-core`'s [`honkbox_sd_core::SdCppPipeline`]); this crate is the
//! promoted home of the ground-up path. `ash-compute` builds the kernel stack
//! alone; `experimental-ash-sd` adds the `SdPipeline` seam and forwards to
//! `honkbox-sd-core/ash-backend`.
//!
//! # Layout
//! - [`raw`]: the single `unsafe` boundary — thin wrappers over ash FFI.
//! - [`device`]: [`device::VulkanDevice`] (loader + headless instance +
//!   logical device + compute queue) and physical-device selection
//!   (discrete > integrated > cpu, then VRAM), plus capacity admission
//!   ([`device::PhysicalDeviceInfo::admit`]) and limit-bounded dispatch plans.
//! - [`compute`]: [`compute::ElementwiseAdd`] and the shared [`compute::Kernel`].
//! - [`matmul`]: [`matmul::MatMul`], row-major GEMM on the same scaffold.
//! - [`model`]: [`model::Op`] vocabulary and [`model::ModelSpec`] validation.
//! - [`softmax`]: [`softmax::Softmax`], stable row-wise normalization.
//! - [`attention`]: [`attention::Attention`], scaled dot-product attention.
//! - [`norm`]: [`norm::LayerNorm`], affine row-wise normalization.
//! - [`conv2d`]: [`conv2d::Conv2d`], direct NCHW convolution.
//! - [`sd`]: [`sd::AshSdPipeline`], implementing `honkbox-sd-core`'s
//!   `SdPipeline`; `generate` validates request, device and checkpoint, then
//!   stops with a named sampling error until diffusion kernels land.
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
#[cfg(feature = "ash-compute")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VkError {
    /// System Vulkan loader missing or failed to load (`Entry::load`).
    Loader(String),
    /// Instance-level API failure (`vkCreateInstance`, enumeration, queries).
    Instance(String),
    /// Logical-device failure (`vkCreateDevice`, queue retrieval).
    Device(String),
    /// A required device feature or limit is absent or too small.
    InsufficientCapacity {
        /// Capability name (`storageBuffer8BitAccess`, `subgroupSize`, ...).
        capability: &'static str,
        /// Requested amount (`1` for a required boolean feature).
        required: u64,
        /// Reported amount (`0` for an absent feature).
        available: u64,
    },
    /// No physical device is present.
    NoDevices,
    /// Physical devices exist but none exposes a compute-capable queue family.
    NoComputeQueue,
}

#[cfg(feature = "ash-compute")]
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

#[cfg(feature = "ash-compute")]
impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loader(e) => write!(f, "vulkan loader failed to load: {e}"),
            Self::Instance(e) => write!(f, "vulkan instance error: {e}"),
            Self::Device(e) => write!(f, "vulkan device error: {e}"),
            Self::InsufficientCapacity {
                capability,
                required,
                available,
            } if *required == 1 && *available == 0 => {
                write!(f, "vulkan device lacks required capability '{capability}'")
            }
            Self::InsufficientCapacity {
                capability,
                required,
                available,
            } => write!(
                f,
                "vulkan device capability '{capability}': need {required}, available {available}"
            ),
            Self::NoDevices => write!(f, "no vulkan physical devices found"),
            Self::NoComputeQueue => write!(f, "no physical device with a compute queue family"),
        }
    }
}

#[cfg(feature = "ash-compute")]
impl std::error::Error for VkError {}

#[cfg(feature = "ash-compute")]
pub mod attention;
#[cfg(feature = "ash-compute")]
pub mod compute;
#[cfg(feature = "ash-compute")]
pub mod conv2d;
#[cfg(feature = "ash-compute")]
pub mod device;
#[cfg(feature = "ash-compute")]
pub mod matmul;
#[cfg(feature = "ash-compute")]
pub mod model;
#[cfg(feature = "ash-compute")]
pub mod norm;
#[cfg(feature = "ash-compute")]
pub mod raw;
#[cfg(feature = "experimental-ash-sd")]
pub mod sd;
#[cfg(feature = "ash-compute")]
pub mod softmax;
#[cfg(feature = "experimental-ash-sd")]
pub mod stubs;
