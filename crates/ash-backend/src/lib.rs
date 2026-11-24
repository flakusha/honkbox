//! Experimental direct-Vulkan backend (`ash`) — device groundwork + first
//! compute kernel.
//!
//! Design record: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md`
//! (the older `ReviewedAsh-*.md` refs in earlier comments were never present
//! in this repo). The default build stays on the shortcut stack
//! (`honkbox-sd-core`'s [`honkbox_sd_core::SdCppPipeline`]); this crate is the
//! promoted home of the ground-up path. `ash-compute` builds the kernel stack
//! alone; `ash-sd` adds the `SdPipeline` seam and forwards to
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
//! - [`silu`]: [`silu::Silu`], elementwise `SiLU` activation.
//! - [`group_norm`]: [`group_norm::GroupNorm`], NCHW group normalization.
//! - [`conv_transpose`]: [`conv_transpose::ConvTranspose2d`], transposed conv.
//! - [`vae_decode`]: [`vae_decode::VaeDecode`], one decoder block composed from
//!   the kernels above (weights come from the caller, never bundled).
//! - [`sampler`]: [`sampler::DdimStep`], one DDIM eta=0 scheduler update, plus
//!   deterministic seed-derived latent noise.
//! - [`ddim_sampler`]: [`ddim_sampler::DdimSampler`], the schedule loop over
//!   caller-supplied latents and predictions (no backbone bundled).
//! - [`conditioning`]: [`conditioning::Cfg`] guidance over caller-supplied
//!   positive/negative text embeddings (no prompt→embedding faking).
//! - [`checkpoint`]: [`checkpoint::CheckpointMap`], a strict safetensors
//!   header → tensor slot map; payload bytes are read on demand.
//! - [`embedding`]: [`embedding::TextEncoder`], token + positional embedding
//!   gather and sum over caller-supplied tables (no tokenizer assets yet).
//! - [`mlp`]: [`mlp::FeedForward`], GELU activation and a two-matrix MLP block.
//! - [`tokenizer`]: [`tokenizer::ClipTokenizer`], CLIP BPE over caller-supplied
//!   `vocab.json` / `merges.txt` (assets are not bundled).
//! - [`text_encoder`]: [`text_encoder::ClipTextEncoder`], token IDs through
//!   embeddings and a caller-weighted stack of encoder blocks.
//! - [`clip_checkpoint`]: [`clip_checkpoint::ClipTensors`], SD 1.5 CLIP tensors
//!   loaded into local matmul layout.
//! - [`text_conditioning`]: [`text_conditioning::TextConditioner`], token-wise
//!   positive/negative states blended by classifier-free guidance.
//! - [`transformer`]: [`transformer::TransformerBlock`], single-head encoder
//!   block built from attention, MLP, residual adds and layer norms.
//! - [`transpose`]: [`transpose::Transpose`], matrix transpose for `Kᵀ`.
//! - [`unet_input`]: [`unet_input::UnetInputBlock`], NCHW conv → group norm →
//!   `SiLU` block over caller-supplied weights.
//! - [`unet_checkpoint`]: [`unet_checkpoint::load_unet_input`], safetensors key
//!   lookup, shape checks and f32 decode for that block.
//! - [`unet_stack`]: [`unet_stack::UnetStack`], per-block keys and chained
//!   shapes over the input-style blocks.
//! - [`unet_attention`]: [`unet_attention::UnetAttentionBlock`], group norm →
//!   spatial self-attention → residual add over NCHW tensors.
//! - [`timestep`]: [`timestep::TimestepEmbed`], sinusoidal diffusion timestep
//!   through two caller-weighted linear layers.
//! - [`unet_predict`]: [`unet_predict::UnetPredictor`], timestep + conditioning
//!   injection and the composed `UNet` stages.
//! - [`unet_downsample`]: [`unet_downsample::UnetDownsample`], strided-conv
//!   halving of the encoder feature map.
//! - [`unet_upsample`]: [`unet_upsample::UnetUpsample`], transposed-conv doubling
//!   with a skip-connection add.
//! - [`unet_traversal`]: [`unet_traversal::UnetTraversal`], one down/up level
//!   with bottleneck predict and a VAE decoder block.
//! - [`model_spec`]: [`model_spec::UnetModelSpec`], checkpoint tensor names and
//!   level topology for one model family.
//! - [`shape_infer`]: [`shape_infer::infer`], channel/kernel/linear widths read
//!   from checkpoint tensor metadata.
//! - [`cross_attention`]: [`cross_attention::CrossAttention`], fused QKV/KV
//!   projections with cross-attention and residual add.
//! - [`sd`]: [`sd::AshSdPipeline`], implementing `honkbox-sd-core`'s
//!   `SdPipeline`; `generate` validates request, device and checkpoint, then
//!   stops with a named sampling error until diffusion kernels land.
//! - [`sdxl`]: [`sdxl::inspect_sdxl_checkpoint`], header-only SDXL shape
//!   validation and dual-CLIP readiness report (no tensor payload decoded).
//! - [`llm`]: [`llm::AshLlmPipeline`], implementing the `honkbox-llm-worker`
//!   `LlmPipeline` (backend id `ash-llm`); a placeholder that returns a
//!   named unimplemented error — no fake output — until transformer decode
//!   kernels land. Gated by `ash-llm`.
//! - [`stubs`]: STT/TTS seam types `stubs::AshSttPipeline` /
//!   `stubs::AshTtsPipeline` (backend ids `ash-stt`, `ash-tts`); gated
//!   `ash-stt` / `ash-tts`, they implement the
//!   `honkbox-audio-worker` traits as placeholders returning named
//!   unimplemented errors — no fake output.
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
    /// Host or device memory exhausted (`ERROR_OUT_OF_*_MEMORY`).
    OutOfMemory {
        /// Vulkan operation that exhausted memory.
        op: &'static str,
    },
    /// The logical device was lost (`ERROR_DEVICE_LOST`).
    DeviceLost {
        /// Vulkan operation that observed the loss.
        op: &'static str,
    },
    /// A device operation exceeded its deadline (`TIMEOUT`).
    Timeout {
        /// Vulkan operation that timed out.
        op: &'static str,
    },
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
        match res {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY
            | ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Self::OutOfMemory { op },
            ash::vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost { op },
            other => Self::Device(format!("{op} failed: {other:?}")),
        }
    }
}

#[cfg(feature = "ash-compute")]
impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loader(e) => write!(f, "vulkan loader failed to load: {e}"),
            Self::Instance(e) => write!(f, "vulkan instance error: {e}"),
            Self::Device(e) => write!(f, "vulkan device error: {e}"),
            Self::OutOfMemory { op } => write!(
                f,
                "vulkan out of memory during {op}; reduce model/batch size or free device memory"
            ),
            Self::DeviceLost { op } => write!(
                f,
                "vulkan device lost during {op}; recreate the device and retry the operation"
            ),
            Self::Timeout { op } => write!(
                f,
                "vulkan {op} timed out; the device may be hung, check for a stuck submission"
            ),
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
pub mod checkpoint;
#[cfg(feature = "ash-compute")]
pub mod checkpoint_family;
#[cfg(feature = "ash-compute")]
pub mod clip_checkpoint;
#[cfg(feature = "ash-compute")]
pub mod compute;
#[cfg(feature = "ash-compute")]
pub mod conditioning;
#[cfg(feature = "ash-compute")]
pub mod conv2d;
#[cfg(feature = "ash-compute")]
pub mod conv_transpose;
#[cfg(feature = "ash-compute")]
pub mod cross_attention;
#[cfg(feature = "ash-compute")]
pub mod ddim_sampler;
#[cfg(feature = "ash-compute")]
pub mod device;
#[cfg(feature = "ash-compute")]
pub mod embedding;
#[cfg(feature = "ash-compute")]
pub mod group_norm;
#[cfg(feature = "ash-llm")]
pub mod llm;
#[cfg(feature = "ash-compute")]
pub mod matmul;
#[cfg(feature = "ash-compute")]
pub mod mlp;
#[cfg(feature = "ash-compute")]
pub mod model;
#[cfg(feature = "ash-compute")]
pub mod model_spec;
#[cfg(feature = "ash-compute")]
pub mod norm;
#[cfg(feature = "ash-sd")]
pub mod prompt;
#[cfg(feature = "ash-compute")]
pub mod raw;
#[cfg(feature = "ash-compute")]
pub mod sampler;
#[cfg(feature = "ash-sd")]
pub mod sd;
#[cfg(feature = "ash-compute")]
pub mod sd15_keys;
#[cfg(feature = "ash-sd")]
pub mod sdxl;
#[cfg(feature = "ash-compute")]
pub mod shape_infer;
#[cfg(feature = "ash-compute")]
pub mod silu;
#[cfg(feature = "ash-compute")]
pub mod softmax;
#[cfg(any(feature = "ash-sd", feature = "ash-stt", feature = "ash-tts"))]
pub mod stubs;
#[cfg(feature = "ash-compute")]
pub mod text_conditioning;
#[cfg(feature = "ash-compute")]
pub mod text_encoder;
#[cfg(feature = "ash-compute")]
pub mod timestep;
#[cfg(feature = "ash-compute")]
pub mod tokenizer;
#[cfg(feature = "ash-compute")]
pub mod transformer;
#[cfg(feature = "ash-compute")]
pub mod transpose;
#[cfg(feature = "ash-compute")]
pub mod unet_attention;
#[cfg(feature = "ash-compute")]
pub mod unet_checkpoint;
#[cfg(feature = "ash-compute")]
pub mod unet_downsample;
#[cfg(feature = "ash-compute")]
pub mod unet_input;
#[cfg(feature = "ash-compute")]
pub mod unet_predict;
#[cfg(feature = "ash-sd")]
pub mod unet_real;
#[cfg(feature = "ash-compute")]
pub mod unet_stack;
#[cfg(feature = "ash-compute")]
pub mod unet_traversal;
#[cfg(feature = "ash-compute")]
pub mod unet_upsample;
#[cfg(feature = "ash-compute")]
pub mod vae_decode;
#[cfg(feature = "ash-sd")]
pub mod vae_real;

#[cfg(all(test, feature = "ash-compute"))]
mod tests {
    use super::VkError;
    use ash::vk;

    #[test]
    fn device_errors_are_classified_and_displayed() {
        for res in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ] {
            assert_eq!(
                VkError::device("vkQueueSubmit", res),
                VkError::OutOfMemory {
                    op: "vkQueueSubmit"
                }
            );
        }
        assert_eq!(
            VkError::device("vkQueueSubmit", vk::Result::ERROR_DEVICE_LOST),
            VkError::DeviceLost {
                op: "vkQueueSubmit"
            }
        );
        assert_eq!(
            VkError::device("vkQueueSubmit", vk::Result::ERROR_UNKNOWN),
            VkError::Device("vkQueueSubmit failed: ERROR_UNKNOWN".to_owned())
        );
        assert_eq!(
            VkError::OutOfMemory {
                op: "vkQueueSubmit"
            }
            .to_string(),
            "vulkan out of memory during vkQueueSubmit; reduce model/batch size or free device memory"
        );
        assert_eq!(
            VkError::DeviceLost {
                op: "vkQueueSubmit"
            }
            .to_string(),
            "vulkan device lost during vkQueueSubmit; recreate the device and retry the operation"
        );
        assert_eq!(
            VkError::Timeout {
                op: "vkWaitForFences"
            }
            .to_string(),
            "vulkan vkWaitForFences timed out; the device may be hung, check for a stuck submission"
        );
    }
}
