//! Diffusion pipeline (samplers, schedulers, recipes); sync only, no HTTP/DB.
//!
//! Two backends behind the real [`SdPipeline`] seam: [`StubPipeline`] (deterministic
//! STUB pixels, a pure function of the seed) and [`SdCppPipeline`] (real txt2img via
//! `diffusion-rs` / stable-diffusion.cpp on a caller-provided checkpoint file).
//! Callers swap the constructor, never the call sites.

use std::fmt;
use std::io::Cursor;
use std::path::PathBuf;

use diffusion_rs::api::{
    ConfigBuilder, LoraSpec, ModelConfigBuilder, SampleMethod, Scheduler as VendorScheduler,
    parse_control_type,
};

/// Longest image edge in pixels accepted by the stub backend.
pub const MAX_EDGE: u32 = 2048;

/// Default sampler recorded in bundle sidecars when a request omits one.
pub const SAMPLER: &str = "euler";
/// Default scheduler recorded in bundle sidecars when a request omits one.
pub const SCHEDULER: &str = "normal";

/// Vendor default img2img denoise strength (stable-diffusion.cpp
/// `--strength`, 0.75); also the HTTP-surface default when a request omits
/// `strength`.
pub const IMG2IMG_STRENGTH: f32 = 0.75;

/// Vendor default `ControlNet` guidance strength (stable-diffusion.cpp
/// `--control-strength`, 0.9); also the HTTP-surface default when a request
/// omits `controlnet_strength`.
pub const CONTROLNET_STRENGTH: f32 = 0.9;

/// The engine's `ControlNet` conditioning type (`stable-diffusion.h`
/// `sd_control_type_t`: what the request's control image depicts).
/// Re-exported so callers thread the exact vendor value into
/// [`Txt2ImgRequest::control_type`] without naming the vendor crate.
pub use diffusion_rs::api::sd_control_type_t;

/// Resolve a `ControlNet` type by the engine's name or alias (`none`,
/// `openpose`, `depth`, `scribble`, `canny`, `normal`, `segment`, `tile`,
/// `repaint`, plus engine aliases like `lineart`/`hed`/`mlsd`): `None` = a
/// name the engine does not know. The name table lives in the vendor
/// (`str_to_sd_control_type`), so HTTP validation and the engine can never
/// disagree about what is accepted.
#[must_use]
pub fn control_type_from_name(name: &str) -> Option<sd_control_type_t> {
    parse_control_type(name)
}

/// Sampler selection, mirroring the vendor `SampleMethod` C enum exactly
/// (wire names are the sd.cpp CLI spellings, e.g. `dpm++2m`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// `euler` — Euler.
    Euler,
    /// `euler_a` — Euler ancestral.
    EulerA,
    /// `heun` — Heun 2nd-order.
    Heun,
    /// `dpm2` — DPM-Solver 2nd-order.
    Dpm2,
    /// `dpm++2s_a` — DPM++ 2S ancestral.
    Dpmpp2sA,
    /// `dpm++2m` — DPM++ 2M.
    Dpmpp2m,
    /// `dpm++2mv2` — DPM++ 2M v2.
    Dpmpp2mV2,
    /// `ipndm` — Improved pseudo-numerical DPM.
    Ipndm,
    /// `ipndm_v` — Improved pseudo-numerical DPM v-series.
    IpndmV,
    /// `lcm` — Latent consistency model sampling.
    Lcm,
    /// `ddim_trailing` — DDIM with trailing timesteps.
    DdimTrailing,
    /// `tcd` — Trajectory consistency distillation.
    Tcd,
    /// `res_multistep` — Residual multistep.
    ResMultistep,
    /// `res_2s` — Residual multistep 2S ancestral.
    Res2s,
    /// `er_sde` — Extended reverse-time SDE.
    ErSde,
    /// `euler_cfg_pp` — Euler with CFG++ rescaling.
    EulerCfgPp,
    /// `euler_a_cfg_pp` — Euler ancestral with CFG++ rescaling.
    EulerACfgPp,
    /// `euler_ge` — Euler with gradient estimation.
    EulerGe,
    /// `dpm++2m_sde` — DPM++ 2M SDE.
    Dpmpp2mSde,
    /// `dpm++2m_sde_bt` — DPM++ 2M SDE with Brownian tree.
    Dpmpp2mSdeBt,
    /// `lms` — Linear multistep.
    Lms,
}

/// Every accepted sampler wire name, in enum order (sd.cpp CLI spellings).
pub const SAMPLER_NAMES: &[&str] = &[
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
    "lms",
];

impl Sampler {
    /// Parse a wire name ([`SAMPLER_NAMES`]); `None` = unknown name (the
    /// HTTP surface turns this into a 400 naming the culprit and listing
    /// the accepted values).
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "euler" => Self::Euler,
            "euler_a" => Self::EulerA,
            "heun" => Self::Heun,
            "dpm2" => Self::Dpm2,
            "dpm++2s_a" => Self::Dpmpp2sA,
            "dpm++2m" => Self::Dpmpp2m,
            "dpm++2mv2" => Self::Dpmpp2mV2,
            "ipndm" => Self::Ipndm,
            "ipndm_v" => Self::IpndmV,
            "lcm" => Self::Lcm,
            "ddim_trailing" => Self::DdimTrailing,
            "tcd" => Self::Tcd,
            "res_multistep" => Self::ResMultistep,
            "res_2s" => Self::Res2s,
            "er_sde" => Self::ErSde,
            "euler_cfg_pp" => Self::EulerCfgPp,
            "euler_a_cfg_pp" => Self::EulerACfgPp,
            "euler_ge" => Self::EulerGe,
            "dpm++2m_sde" => Self::Dpmpp2mSde,
            "dpm++2m_sde_bt" => Self::Dpmpp2mSdeBt,
            "lms" => Self::Lms,
            _ => return None,
        })
    }

    /// Wire name ([`SAMPLER_NAMES`] spelling) for sidecar recording.
    pub fn name(self) -> &'static str {
        SAMPLER_NAMES[self as usize]
    }
}

/// Scheduler (denoiser sigma schedule) selection, mirroring the vendor
/// `Scheduler` C enum exactly (sd.cpp CLI spellings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheduler {
    /// `discrete` — discrete sigma schedule (the fixed recipe's `normal`).
    Discrete,
    /// `karras` — Karras sigma schedule.
    Karras,
    /// `exponential` — exponential sigma schedule.
    Exponential,
    /// `ays` — align your steps schedule.
    Ays,
    /// `gits` — gradient estimation noise schedule.
    Gits,
    /// `sgm_uniform` — SGM uniform schedule.
    SgmUniform,
    /// `simple` — simple linear schedule.
    Simple,
    /// `smoothstep` — smoothstep schedule.
    Smoothstep,
    /// `kl_optimal` — KL-optimal schedule.
    KlOptimal,
    /// `lcm` — latent-consistency-model schedule.
    Lcm,
    /// `bong_tangent` — Bong tangent schedule.
    BongTangent,
    /// `ltx2` — LTX-2 schedule.
    Ltx2,
    /// `logit_normal` — logit-normal schedule.
    LogitNormal,
    /// `flux2` — Flux 2 schedule.
    Flux2,
    /// `flux` — Flux schedule.
    Flux,
    /// `beta` — beta-distribution schedule.
    Beta,
    /// `llada_image` — LLaDA-image schedule.
    LladaImage,
}

/// Every accepted scheduler wire name, in enum order. Note: the fixed
/// recipe's recorded name `normal` ([`SCHEDULER`]) IS the vendor's
/// `DISCRETE` schedule — wire spellings follow sd.cpp, where `discrete`
/// is the default denoiser schedule for txt2img.
pub const SCHEDULER_NAMES: &[&str] = &[
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
    "logit_normal",
    "flux2",
    "flux",
    "beta",
    "llada_image",
];

impl Scheduler {
    /// Parse a wire name ([`SCHEDULER_NAMES`]); `None` = unknown name.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "discrete" => Self::Discrete,
            "karras" => Self::Karras,
            "exponential" => Self::Exponential,
            "ays" => Self::Ays,
            "gits" => Self::Gits,
            "sgm_uniform" => Self::SgmUniform,
            "simple" => Self::Simple,
            "smoothstep" => Self::Smoothstep,
            "kl_optimal" => Self::KlOptimal,
            "lcm" => Self::Lcm,
            "bong_tangent" => Self::BongTangent,
            "ltx2" => Self::Ltx2,
            "logit_normal" => Self::LogitNormal,
            "flux2" => Self::Flux2,
            "flux" => Self::Flux,
            "beta" => Self::Beta,
            "llada_image" => Self::LladaImage,
            _ => return None,
        })
    }

    /// Wire name for sidecar recording.
    pub fn name(self) -> &'static str {
        SCHEDULER_NAMES[self as usize]
    }
}

/// One resolved `LoRA` adapter for a generation: an absolute weight-file
/// path (resolved by `server` from a catalog slug) plus its multiplier.
#[derive(Debug, Clone)]
pub struct LoraRef {
    /// Absolute path to the `LoRA` weight file.
    pub path: PathBuf,
    /// Multiplier applied to the adapter delta (`1.0` = as-trained; the
    /// accepted HTTP range is documented on the DTO).
    pub strength: f32,
}

/// Hires-fix second pass: the vendor's native two-pass upscale (generate at
/// the base size, interpolate the latent up to the target, re-denoise at
/// [`HiresFix::denoising_strength`]; stable-diffusion.cpp `sd_hires_params_t`
/// with the latent upscaler). An explicit `target` wins over `scale`.
/// Only the real [`SdCppPipeline`] honors it; [`StubPipeline`] ignores it
/// like the other backend-only fields.
#[derive(Debug, Clone, PartialEq)]
pub struct HiresFix {
    /// Upscale factor over the base size when `target` is `None`.
    pub scale: f32,
    /// Explicit target `(width, height)` in pixels; `None` = base * scale.
    pub target: Option<(u32, u32)>,
    /// Second-pass denoising strength in `(0, 1]`.
    pub denoising_strength: f32,
}

/// Final hires-fix output `(width, height)`: the explicit target when set,
/// otherwise the base size scaled and rounded to nearest (minimum 1 px;
/// an absurd scale clamps to `u32::MAX` per edge — the HTTP surface caps
/// targets at [`MAX_EDGE`] long before that).
pub fn hires_target(base: (u32, u32), fix: &HiresFix) -> (u32, u32) {
    if let Some(target) = fix.target {
        target
    } else {
        let edge = |v: u32| {
            let scaled = (f64::from(v) * f64::from(fix.scale)).round().max(1.0);
            // Upstream validates the scale into [1.0, 4.0], so the rounded
            // product is a small positive integer; the >= 1 floor keeps the
            // cast sign-safe.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pixels = scaled as u32;
            pixels
        };
        (edge(base.0), edge(base.1))
    }
}

/// Minimal txt2img request crossing the `server` → `sd-core` seam. The same
/// struct carries img2img/inpaint inputs: `init_image` present switches the
/// real backend to img2img (mask optional), and `control_image` +
/// `control_net` present together switch it to a ControlNet-conditioned
/// generation; the stub honestly ignores all four image-input fields.
#[derive(Debug, Clone)]
pub struct Txt2ImgRequest {
    /// Text prompt.
    pub prompt: String,
    /// RNG seed; same seed = byte-identical output from the stub.
    pub seed: u64,
    /// Diffusion steps.
    pub steps: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Registry slug (e.g. `comfy:checkpoints/sdxl.safetensors`).
    pub model_slug: String,
    /// Optional text-encoder slug (resolved by `server`); feeds the `llm`
    /// (Qwen-VL-class) slot the new archs (krea2/moss/anima/qwen-2.1) read.
    /// `None` = single-file checkpoint with bundled encoders (e.g. SDXL).
    pub clip: Option<PathBuf>,
    /// Optional standalone VAE slug (resolved by `server`); `None` = VAE
    /// bundled in (or converted from) the diffusion checkpoint.
    pub vae: Option<PathBuf>,
    /// Sampler override; `None` = the fixed recipe ([`SAMPLER`], euler).
    pub sampler: Option<Sampler>,
    /// Scheduler override; `None` = the fixed recipe (`normal`/discrete).
    pub scheduler: Option<Scheduler>,
    /// CFG guidance override; `None` = the fixed recipe default (7.0).
    pub cfg_scale: Option<f32>,
    /// Negative prompt; `None`/empty = none (vendor default `""`).
    pub negative_prompt: Option<String>,
    /// Per-request `LoRA` adapters; `None` = none. All entries must share one
    /// parent directory (the vendor applies a single `LoRA` dir per model
    /// config) — `server` enforces that at slug-resolution time.
    pub loras: Option<Vec<LoraRef>>,
    /// img2img input image (PNG file path); `None` = txt2img.
    pub init_image: Option<PathBuf>,
    /// Inpaint mask (PNG file path); requires [`Txt2ImgRequest::init_image`].
    pub mask_image: Option<PathBuf>,
    /// img2img denoise strength in `(0, 1]`; `None` = the vendor default
    /// [`IMG2IMG_STRENGTH`]. Only meaningful with [`Txt2ImgRequest::init_image`].
    pub strength: Option<f32>,
    /// Optional hires-fix second pass (see [`HiresFix`]); `None` = single-pass
    /// generation. Only [`SdCppPipeline`] honors it; the stub ignores it.
    pub hires: Option<HiresFix>,
    /// `ControlNet` weights file (resolved by `server` from a `controlnet`
    /// catalog slug). `Some` switches the real backend to a
    /// ControlNet-conditioned generation and REQUIRES
    /// [`Txt2ImgRequest::control_image`]: weights alone condition nothing,
    /// so a lone half is a generation error, not a silent no-op.
    pub control_net: Option<PathBuf>,
    /// `ControlNet` conditioning image (PNG bytes); requires
    /// [`Txt2ImgRequest::control_net`]. The real backend writes it to a temp
    /// file (the vendor reads every image input from a path) and the engine
    /// resizes it to the generation size; the image is applied as-is, the
    /// way a preprocessed conditioning map arrives from a client.
    pub control_image: Option<Vec<u8>>,
    /// `ControlNet` guidance strength in `(0, 1]`; `None` = the vendor default
    /// [`CONTROLNET_STRENGTH`]. Only meaningful with
    /// [`Txt2ImgRequest::control_image`].
    pub controlnet_strength: Option<f32>,
    /// `ControlNet` conditioning type (the engine's `--control-type`: what
    /// the conditioning image depicts). `None` = the vendor default `none`,
    /// where the image rides along as a plain preprocessed map (classic
    /// `ControlNet`s work that way); a union `ControlNet` REQUIRES a named
    /// type and the engine refuses to run without one. Only meaningful with
    /// [`Txt2ImgRequest::control_net`] + [`Txt2ImgRequest::control_image`].
    pub control_type: Option<sd_control_type_t>,
}

/// RGBA8 image produced by [`SdPipeline::generate`].
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes, row-major RGBA.
    pub rgba: Vec<u8>,
}

/// Failures from [`SdPipeline::generate`] or PNG encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdError {
    /// Longest edge exceeds [`MAX_EDGE`].
    TooLarge {
        /// The offending `width.max(height)`.
        edge: u32,
    },
    /// Image encoding failed.
    Encode(String),
    /// Real-backend failure (config build, `gen_img`, file IO).
    Backend(String),
}

impl fmt::Display for SdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { edge } => {
                write!(f, "image edge {edge}px exceeds cap of {MAX_EDGE}px")
            }
            Self::Encode(e) => write!(f, "png encode failed: {e}"),
            Self::Backend(e) => write!(f, "sd-cpp backend failed: {e}"),
        }
    }
}

impl std::error::Error for SdError {}

/// Sync image pipeline seam; `server` codes against this trait, never a backend.
pub trait SdPipeline: Send + Sync {
    /// Sample one image; deterministic per backend contract.
    fn generate(&self, req: &Txt2ImgRequest) -> Result<GeneratedImage, SdError>;
    /// Backend tag recorded in bundle sidecars (`"stub"` here).
    fn backend_id(&self) -> &'static str;
}

/// Deterministic stub: pixels are a pure function of the seed
/// (SAME seed = byte-identical PNG); honors `width`/`height`. Recipe
/// overrides (`sampler`/`scheduler`/`cfg_scale`/`negative_prompt`), the
/// hires-fix pass, and the image inputs
/// (`loras`/`init_image`/`mask_image`/`strength`/`control_net`/
/// `control_image`/`controlnet_strength`/`control_type`) are honestly
/// ignored: the stub has no sampler, no img2img branch, and no `ControlNet`
/// — they only affect the real `SdCppPipeline` backend. Callers that need
/// control inputs applied must check the backend (the HTTP surface answers
/// `controlnet_unsupported` instead of rendering uncontrolled).
#[derive(Debug, Default, Clone, Copy)]
pub struct StubPipeline;

impl SdPipeline for StubPipeline {
    fn backend_id(&self) -> &'static str {
        "stub"
    }

    fn generate(&self, req: &Txt2ImgRequest) -> Result<GeneratedImage, SdError> {
        let edge = req.width.max(req.height);
        if edge > MAX_EDGE {
            return Err(SdError::TooLarge { edge });
        }
        let mut rgba = vec![0u8; req.width as usize * req.height as usize * 4];
        for (i, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let h = splitmix64(req.seed ^ i as u64).to_le_bytes();
            px[0] = h[7];
            px[1] = h[6];
            px[2] = h[5];
            px[3] = 255;
        }
        Ok(GeneratedImage {
            width: req.width,
            height: req.height,
            rgba,
        })
    }
}

/// Backend tag when compiled with `vulkan` (Vulkan execution on the dGPU).
#[cfg(feature = "vulkan")]
pub const SD_CPP_BACKEND_ID: &str = "sd-cpp-vulkan";
/// Backend tag for a CPU-only build (no `vulkan` feature).
#[cfg(not(feature = "vulkan"))]
pub const SD_CPP_BACKEND_ID: &str = "sd-cpp-cpu";

/// Real txt2img via `diffusion-rs` (stable-diffusion.cpp) on a caller-provided
/// single-file checkpoint (e.g. an SDXL finetune `.safetensors`).
///
/// Recipe: euler sampler, discrete (`normal`) scheduler, CFG 7.0 by default;
/// a [`Txt2ImgRequest`] may override sampler/scheduler/CFG/negative prompt
/// per request and run an optional hires-fix second pass ([`HiresFix`]). VAE tiling stays forced ON (required: the 1024² fp16 VAE decode otherwise exceeds the per-allocation limit). `gen_img` writes one PNG to a
/// temp path; this impl reads it back and decodes to RGBA so the seam stays
/// container-free. `Send + Sync` holds: all `gen_img` state is function-local.
#[derive(Debug, Clone)]
pub struct SdCppPipeline {
    /// Absolute path to the checkpoint file (resolved by `server` via `registry`).
    model_path: PathBuf,
}

impl SdCppPipeline {
    /// Point the pipeline at a checkpoint file; nothing loads until `generate`.
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }
}

/// Tensor-name prefix sd.cpp version-detect needs on the `.model()` slot.
const DIFFUSION_PREFIX: &str = "model.diffusion_model.";

/// Largest safetensors header we sniff (a GGUF magic decodes as a giant len).
const HEADER_CAP: u64 = 64 * 1024 * 1024;

/// Pure sniff over a checkpoint's leading bytes: true when the safetensors
/// header parses and NONE of its tensor names carry [`DIFFUSION_PREFIX`]
/// (split-arch diffusion-only files, e.g. qwen-2.1-convrot). Garbage,
/// truncated, prefixed, or non-safetensors (GGUF) bytes all give false.
fn safetensors_header_is_prefixless(header: &[u8]) -> bool {
    let Some(len_bytes) = header.get(..8) else {
        return false;
    };
    let Ok(len_bytes): Result<[u8; 8], _> = len_bytes.try_into() else {
        return false;
    };
    // `end` saturates, so a GGUF-magic "len" can never match a real slice.
    let len = u64::from_le_bytes(len_bytes);
    let end = usize::try_from(len).map_or(usize::MAX, |l| 8usize.saturating_add(l));
    let Some(body) = header.get(8..end) else {
        return false;
    };
    let Ok(tensors) = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(body)
    else {
        return false;
    };
    let mut any_tensor = false;
    for key in tensors.keys() {
        if key == "__metadata__" {
            continue;
        }
        any_tensor = true;
        if key.starts_with(DIFFUSION_PREFIX) {
            return false;
        }
    }
    any_tensor
}

/// True when `path` is a prefixless split safetensors file (see above);
/// reads only the 8 B length plus the header JSON. Unreadable, over-cap
/// (GGUF), or truncated files give false (`.model()` fallback).
fn checkpoint_needs_diffusion_slot(path: &std::path::Path) -> bool {
    use std::io::Read as _;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut len_bytes = [0u8; 8];
    if file.read_exact(&mut len_bytes).is_err() {
        return false;
    }
    let len = u64::from_le_bytes(len_bytes);
    if len == 0 || len > HEADER_CAP {
        return false;
    }
    let Ok(len) = usize::try_from(len) else {
        return false;
    };
    let mut header = vec![0u8; 8 + len];
    header[..8].copy_from_slice(&len_bytes);
    if file.read_exact(&mut header[8..]).is_err() {
        return false;
    }
    safetensors_header_is_prefixless(&header)
}

/// Map a [`Sampler`] onto the vendor `SampleMethod` C enum (variants and
/// order mirror [`SAMPLER_NAMES`] / the sd.cpp headers; exhaustiveness is
/// compile-checked by the match).
fn sampler_vendor(sampler: Sampler) -> diffusion_rs::api::SampleMethod {
    use diffusion_rs::api::SampleMethod as V;
    match sampler {
        Sampler::Euler => V::EULER_SAMPLE_METHOD,
        Sampler::EulerA => V::EULER_A_SAMPLE_METHOD,
        Sampler::Heun => V::HEUN_SAMPLE_METHOD,
        Sampler::Dpm2 => V::DPM2_SAMPLE_METHOD,
        Sampler::Dpmpp2sA => V::DPMPP2S_A_SAMPLE_METHOD,
        Sampler::Dpmpp2m => V::DPMPP2M_SAMPLE_METHOD,
        Sampler::Dpmpp2mV2 => V::DPMPP2Mv2_SAMPLE_METHOD,
        Sampler::Ipndm => V::IPNDM_SAMPLE_METHOD,
        Sampler::IpndmV => V::IPNDM_V_SAMPLE_METHOD,
        Sampler::Lcm => V::LCM_SAMPLE_METHOD,
        Sampler::DdimTrailing => V::DDIM_TRAILING_SAMPLE_METHOD,
        Sampler::Tcd => V::TCD_SAMPLE_METHOD,
        Sampler::ResMultistep => V::RES_MULTISTEP_SAMPLE_METHOD,
        Sampler::Res2s => V::RES_2S_SAMPLE_METHOD,
        Sampler::ErSde => V::ER_SDE_SAMPLE_METHOD,
        Sampler::EulerCfgPp => V::EULER_CFG_PP_SAMPLE_METHOD,
        Sampler::EulerACfgPp => V::EULER_A_CFG_PP_SAMPLE_METHOD,
        Sampler::EulerGe => V::EULER_GE_SAMPLE_METHOD,
        Sampler::Dpmpp2mSde => V::DPMPP2M_SDE_SAMPLE_METHOD,
        Sampler::Dpmpp2mSdeBt => V::DPMPP2M_SDE_BT_SAMPLE_METHOD,
        Sampler::Lms => V::LMS_SAMPLE_METHOD,
    }
}

/// Map a [`Scheduler`] onto the vendor `Scheduler` C enum.
fn scheduler_vendor(scheduler: Scheduler) -> diffusion_rs::api::Scheduler {
    use diffusion_rs::api::Scheduler as V;
    match scheduler {
        Scheduler::Discrete => V::DISCRETE_SCHEDULER,
        Scheduler::Karras => V::KARRAS_SCHEDULER,
        Scheduler::Exponential => V::EXPONENTIAL_SCHEDULER,
        Scheduler::Ays => V::AYS_SCHEDULER,
        Scheduler::Gits => V::GITS_SCHEDULER,
        Scheduler::SgmUniform => V::SGM_UNIFORM_SCHEDULER,
        Scheduler::Simple => V::SIMPLE_SCHEDULER,
        Scheduler::Smoothstep => V::SMOOTHSTEP_SCHEDULER,
        Scheduler::KlOptimal => V::KL_OPTIMAL_SCHEDULER,
        Scheduler::Lcm => V::LCM_SCHEDULER,
        Scheduler::BongTangent => V::BONG_TANGENT_SCHEDULER,
        Scheduler::Ltx2 => V::LTX2_SCHEDULER,
        Scheduler::LogitNormal => V::LOGIT_NORMAL_SCHEDULER,
        Scheduler::Flux2 => V::FLUX2_SCHEDULER,
        Scheduler::Flux => V::FLUX_SCHEDULER,
        Scheduler::Beta => V::BETA_SCHEDULER,
        Scheduler::LladaImage => V::LLADA_IMAGE_SCHEDULER,
    }
}

/// `ControlNet` inputs from [`Txt2ImgRequest`]: `Some((weights, image))` when
/// both halves are present, `None` = plain generation. A lone half is an
/// error: the engine ignores a control image when no `ControlNet` is loaded
/// (the weights without one condition nothing), so rendering without the
/// requested conditioning would misrepresent the output.
fn control_pair(req: &Txt2ImgRequest) -> Result<Option<(&std::path::Path, &[u8])>, SdError> {
    match (req.control_net.as_deref(), req.control_image.as_deref()) {
        (Some(weights), Some(png)) => Ok(Some((weights, png))),
        (None, None) => Ok(None),
        (Some(_), None) => Err(SdError::Backend(
            "control_net weights without control_image: nothing to condition on".to_owned(),
        )),
        (None, Some(_)) => Err(SdError::Backend(
            "control_image without control_net weights: the engine ignores a control image when no ControlNet is loaded".to_owned(),
        )),
    }
}

/// Stage `ControlNet` for a generation: resolve the pair
/// ([`control_pair`]), write the conditioning PNG to a unique temp file
/// (the vendor reads every image input from a path), point the generation
/// config at that path and the model config at the weights (they load with
/// the sd context). Returns the temp path for post-run cleanup; `None` =
/// plain generation.
fn stage_control(
    req: &Txt2ImgRequest,
    config: &mut diffusion_rs::api::ConfigBuilder,
    model: &mut diffusion_rs::api::ModelConfigBuilder,
) -> Result<Option<PathBuf>, SdError> {
    let Some((weights, png)) = control_pair(req)? else {
        return Ok(None);
    };
    // Loud, early refusal: the engine logs-and-moves-on when ControlNet
    // weights fail to load and would render the prompt unconditioned, so a
    // weights path that does not resolve to a readable file is an error
    // here, never an unconditioned render.
    if !weights.exists() {
        return Err(SdError::Backend(format!(
            "control_net weights not found: {}",
            weights.display()
        )));
    }
    let path = std::env::temp_dir().join(format!(
        "honkbox-control-{}-{}-{}.png",
        std::process::id(),
        req.seed,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    ));
    std::fs::write(&path, png).map_err(|e| SdError::Backend(e.to_string()))?;
    model.control_net(weights.to_path_buf());
    config.control_image(path.clone());
    // The named conditioning type rides into the engine (a union ControlNet
    // refuses to run without one); `None` stays the vendor default `none` —
    // the preprocessed-map behavior, byte-identical to before typed
    // execution landed.
    config.control_type(
        req.control_type
            .unwrap_or(sd_control_type_t::SD_CONTROL_TYPE_NONE),
    );
    Ok(Some(path))
}

/// Build the vendor hires-fix bundle for `req`: latent-interpolation
/// upscaler plus target size (explicit target, else base * scale rounded)
/// and second-pass denoise. `None` = single-pass generation.
fn hires_vendor_params(
    req: &Txt2ImgRequest,
) -> Result<Option<(diffusion_rs::api::Upscaler, diffusion_rs::api::HiresParams)>, SdError> {
    let Some(hires) = &req.hires else {
        return Ok(None);
    };
    let (target_width, target_height) = hires_target((req.width, req.height), hires);
    let params = diffusion_rs::api::HiresParamsBuilder::default()
        .width(target_width.cast_signed())
        .height(target_height.cast_signed())
        .denoising_strength(hires.denoising_strength)
        .build()
        .map_err(|e| SdError::Backend(e.to_string()))?;
    Ok(Some((
        diffusion_rs::api::Upscaler::SD_HIRES_UPSCALER_LATENT,
        params,
    )))
}

/// Per-request model-config wiring for [`SdCppPipeline::generate`]: the
/// scheduler recipe (vendor design keeps it on the model config), split
/// encoder/VAE files (`--clip` feeds the llm (Qwen-VL-class) slot, `--vae`
/// the standalone VAE), per-request `LoRA` adapters (the vendor maps file
/// STEMS inside one directory to (path, multiplier) pairs, so `server`
/// resolves catalog slugs to same-dir absolute paths and we re-derive the
/// parts here; high-noise split stays fixed off, not exposed on the API),
/// and the vendor-native hires-fix second pass (the latent is interpolated
/// up to the target after the base pass, then re-denoised at
/// `denoising_strength`; the interpolation-only upscaler needs no extra
/// weights — the ESRGAN-backed variant is not wired).
fn wire_model(req: &Txt2ImgRequest, model: &mut ModelConfigBuilder) -> Result<(), SdError> {
    model.scheduler(match req.scheduler {
        Some(scheduler) => scheduler_vendor(scheduler),
        None => VendorScheduler::DISCRETE_SCHEDULER,
    });
    if let Some(clip) = req.clip.clone() {
        model.llm(clip);
    }
    if let Some(vae) = req.vae.clone() {
        model.vae(vae);
    }
    if let Some(loras) = &req.loras {
        let first = loras
            .first()
            .ok_or_else(|| SdError::Backend("empty lora list".to_owned()))?;
        let dir = first
            .path
            .parent()
            .ok_or_else(|| SdError::Backend("lora path has no parent directory".to_owned()))?;
        let specs = loras
            .iter()
            .map(|lora| LoraSpec {
                file_name: lora
                    .path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or_default()
                    .to_owned(),
                is_high_noise: false,
                multiplier: lora.strength,
            })
            .collect();
        model.lora_models(dir, specs);
    }
    if let Some((upscaler, params)) = hires_vendor_params(req)? {
        model.hires_params(upscaler, params, None::<&std::path::Path>);
    }
    Ok(())
}

impl SdPipeline for SdCppPipeline {
    fn backend_id(&self) -> &'static str {
        SD_CPP_BACKEND_ID
    }

    fn generate(&self, req: &Txt2ImgRequest) -> Result<GeneratedImage, SdError> {
        #[cfg(feature = "vulkan")]
        use diffusion_rs::api::{BackendDevice, Module};

        let edge = req.width.max(req.height);
        if edge > MAX_EDGE {
            return Err(SdError::TooLarge { edge });
        }
        // Unique temp PNG: pid + nanos, so concurrent generates never collide.
        let out = std::env::temp_dir().join(format!(
            "honkbox-{}-{}-{}.png",
            std::process::id(),
            req.seed,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        ));
        let mut config = ConfigBuilder::default();
        config
            .prompt(req.prompt.clone())
            .negative_prompt(req.negative_prompt.clone().unwrap_or_default())
            // Recipe defaults (euler, CFG 7.0) match the pre-passthrough
            // fixed slice exactly; `None` fields are byte-identical behavior.
            .cfg_scale(req.cfg_scale.unwrap_or(7.0))
            .seed(req.seed.cast_signed())
            .steps(req.steps.cast_signed())
            .width(req.width.cast_signed())
            .height(req.height.cast_signed())
            .sampling_method(match req.sampler {
                Some(sampler) => sampler_vendor(sampler),
                None => SampleMethod::EULER_SAMPLE_METHOD,
            })
            // img2img/inpaint: the vendor reads both images from paths
            // (empty path = txt2img) and takes the denoise strength here;
            // the None values are byte-identical to the vendor defaults.
            .init_img(req.init_image.clone().unwrap_or_default())
            .mask_img(req.mask_image.clone().unwrap_or_default())
            .strength(req.strength.unwrap_or(IMG2IMG_STRENGTH))
            // ControlNet guidance; only consumed when a control image is
            // set (the pair check in [`stage_control`] rejects a lone
            // half), and the None branch is byte-identical to the vendor
            // default (0.9).
            .control_strength(req.controlnet_strength.unwrap_or(CONTROLNET_STRENGTH))
            .output(out.clone());
        let mut model = ModelConfigBuilder::default();
        // ControlNet conditioning: temp PNG + named control type via
        // [`stage_control`]; an unnamed type stays the vendor default (NONE,
        // preprocessed image).
        let control_path = stage_control(req, &mut config, &mut model)?;
        let config = config
            .build()
            .map_err(|e| SdError::Backend(e.to_string()))?;
        // VAE tiling: 1024² fp16 decode needs a ~4.8 GB contiguous Vulkan buffer,
        // over the per-allocation limit (OOM evidence 2026-11-15, 7900 XTX).
        // Sampling stays full-frame; only the VAE decode is tiled.
        // Prefixless split checkpoints (qwen-2.1-convrot) miss version-detect
        // through `.model()`; their loader only prepends the missing prefix on
        // the `.diffusion_model()` slot. GGUF/unreadable files take `.model()`.
        if checkpoint_needs_diffusion_slot(&self.model_path) {
            model.diffusion_model(self.model_path.clone());
        } else {
            model.model(self.model_path.clone());
        }
        model.n_threads(0).vae_tiling(true);
        // Per-request wiring: scheduler recipe, split files, LoRAs, hires.
        wire_model(req, &mut model)?;
        #[cfg(feature = "vulkan")]
        model.backend(std::collections::HashMap::from([(
            Module::Model,
            vec![BackendDevice::VULKAN0],
        )]));
        let mut model_config = model.build().map_err(|e| SdError::Backend(e.to_string()))?;
        diffusion_rs::api::gen_img(&config, &mut model_config)
            .map_err(|e| SdError::Backend(e.to_string()))?;
        let bytes = std::fs::read(&out);
        let _ = std::fs::remove_file(&out);
        if let Some(path) = &control_path {
            let _ = std::fs::remove_file(path);
        }
        let bytes = bytes.map_err(|e| SdError::Backend(e.to_string()))?;
        let img = image::load_from_memory(&bytes)
            .map_err(|e| SdError::Encode(e.to_string()))?
            .to_rgba8();
        Ok(GeneratedImage {
            width: img.width(),
            height: img.height(),
            rgba: img.into_raw(),
        })
    }
}

/// Output container for [`encode_image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Lossless PNG (default; deterministic bytes for a fixed seed).
    Png,
    /// Lossy JPEG, quality 90.
    Jpeg,
    /// Lossless WebP (`image` 0.25 has no lossy knob).
    Webp,
}

impl OutputFormat {
    /// File extension for bundle art (`jpg`, not `jpeg`).
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
            OutputFormat::Webp => "webp",
        }
    }

    /// Full lowercase name recorded in bundle sidecars (`jpeg`, not `jpg`).
    pub fn name(self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpeg",
            OutputFormat::Webp => "webp",
        }
    }

    /// IANA media type for the encoded bytes (HTTP responses).
    pub fn mime(self) -> &'static str {
        match self {
            OutputFormat::Png => "image/png",
            OutputFormat::Jpeg => "image/jpeg",
            OutputFormat::Webp => "image/webp",
        }
    }

    /// Parse a wire format name (the `name()` spelling); `None` = unknown.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "png" => Some(OutputFormat::Png),
            "jpeg" => Some(OutputFormat::Jpeg),
            "webp" => Some(OutputFormat::Webp),
            _ => None,
        }
    }

    /// Every encoder compiled into this crate (the authoritative source for
    /// capability advertisement — `GET /v1/capabilities` `formats.write`
    /// lists exactly these).
    pub const ALL: [OutputFormat; 3] = [OutputFormat::Png, OutputFormat::Jpeg, OutputFormat::Webp];

    /// `true` when `output_compression` affects this container: JPEG quality
    /// only (PNG is deterministic; the WebP encoder is lossless-only in
    /// `image` 0.25).
    #[must_use]
    pub const fn compression_applies(self) -> bool {
        matches!(self, OutputFormat::Jpeg)
    }
}

/// JPEG quality used when the caller does not override it.
pub const DEFAULT_JPEG_QUALITY: u8 = 90;

/// Encode an image in the requested container; consumes the image.
///
/// JPEG uses [`DEFAULT_JPEG_QUALITY`]; WebP is lossless-only in `image` 0.25
/// (no quality knob).
pub fn encode_image(img: GeneratedImage, format: OutputFormat) -> Result<Vec<u8>, SdError> {
    encode_image_with_quality(img, format, DEFAULT_JPEG_QUALITY)
}

/// [`encode_image`] with an explicit JPEG quality (`0-100`; ignored for PNG
/// and the lossless WebP encoder).
pub fn encode_image_with_quality(
    img: GeneratedImage,
    format: OutputFormat,
    jpeg_quality: u8,
) -> Result<Vec<u8>, SdError> {
    let buf: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(img.width, img.height, img.rgba)
            .ok_or_else(|| SdError::Encode("bad pixel buffer length".to_string()))?;
    let mut out = Vec::new();
    match format {
        // Byte settings unchanged from the original `encode_png`.
        OutputFormat::Png => {
            image::DynamicImage::ImageRgba8(buf)
                .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
                .map_err(|e| SdError::Encode(e.to_string()))?;
        }
        OutputFormat::Jpeg => {
            let rgb = image::DynamicImage::ImageRgba8(buf).to_rgb8();
            let mut enc =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, jpeg_quality);
            enc.encode(
                rgb.as_raw(),
                img.width,
                img.height,
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| SdError::Encode(e.to_string()))?;
        }
        OutputFormat::Webp => {
            image::codecs::webp::WebPEncoder::new_lossless(&mut out)
                .encode(
                    &buf.into_raw(),
                    img.width,
                    img.height,
                    image::ExtendedColorType::Rgba8,
                )
                .map_err(|e| SdError::Encode(e.to_string()))?;
        }
    }
    Ok(out)
}

/// `splitmix64`: cheap per-pixel hash, no tables, no deps.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(seed: u64, width: u32, height: u32) -> Txt2ImgRequest {
        Txt2ImgRequest {
            prompt: "a red fox, portrait".to_string(),
            seed,
            steps: 20,
            width,
            height,
            model_slug: "comfy:checkpoints/x.safetensors".to_string(),
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

    #[test]
    fn control_pair_accepts_only_complete_pairs() {
        // Plain generation: no control fields at all.
        let plain = req(1, 64, 64);
        assert!(control_pair(&plain).unwrap().is_none());

        // Both halves: the weights path and the raw PNG bytes come through.
        let png = b"\x89PNG\r\n\x1a\nfake";
        let mut both = req(1, 64, 64);
        both.control_net = Some(PathBuf::from("/models/controlnet/c.safetensors"));
        both.control_image = Some(png.to_vec());
        let (weights, image) = control_pair(&both).unwrap().expect("complete pair");
        assert_eq!(
            weights,
            std::path::Path::new("/models/controlnet/c.safetensors")
        );
        assert_eq!(image, png);

        // A lone half would render without its conditioning (the engine
        // drops an unpaired control image): both mismatches are errors.
        let mut lone_weights = req(1, 64, 64);
        lone_weights.control_net = Some(PathBuf::from("/models/controlnet/c.safetensors"));
        assert_eq!(
            control_pair(&lone_weights),
            Err(SdError::Backend(
                "control_net weights without control_image: nothing to condition on".to_owned(),
            )),
        );

        let mut lone_image = req(1, 64, 64);
        lone_image.control_image = Some(png.to_vec());
        assert_eq!(
            control_pair(&lone_image),
            Err(SdError::Backend(
                "control_image without control_net weights: the engine ignores a control image when no ControlNet is loaded".to_owned(),
            )),
        );
    }

    #[test]
    fn control_type_from_name_uses_the_engine_name_table() {
        let canny = control_type_from_name("canny").expect("canny is a vendor name");
        assert_ne!(canny, sd_control_type_t::SD_CONTROL_TYPE_NONE);
        assert_eq!(
            control_type_from_name("none"),
            Some(sd_control_type_t::SD_CONTROL_TYPE_NONE)
        );
        // Engine aliases resolve onto the same conditioning type.
        assert_eq!(control_type_from_name("lineart"), Some(canny));
        assert_eq!(
            control_type_from_name("hed"),
            control_type_from_name("scribble")
        );
        // Anything else is rejected — empty, unknown, and wrong-case alike
        // (the engine's name match is case-sensitive).
        assert_eq!(control_type_from_name("bogus"), None);
        assert_eq!(control_type_from_name(""), None);
        assert_eq!(control_type_from_name("CANNY"), None);
    }

    #[test]
    fn stage_control_threads_type_and_weights_and_refuses_missing_weights() {
        let weights = std::env::temp_dir().join(format!(
            "honkbox-test-cnet-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        ));
        std::fs::write(&weights, b"cn").expect("write weights");
        let png: &[u8] = b"\x89PNG\r\n\x1a\nfake";

        // A named type + real weights file ride onto the vendor configs.
        let mut both = req(7, 64, 64);
        both.control_net = Some(weights.clone());
        both.control_image = Some(png.to_vec());
        both.control_type = control_type_from_name("canny");
        let mut config = ConfigBuilder::default();
        config.prompt("a honk".to_owned());
        let mut model = ModelConfigBuilder::default();
        let staged = stage_control(&both, &mut config, &mut model).expect("a complete pair stages");
        let staged = staged.expect("pair present");
        // The builder validates that a checkpoint slot is set (the engine
        // path always has one); a stand-in path never reaches the engine.
        model.model(PathBuf::from("comfy:checkpoints/x.safetensors"));
        let built = config.build().expect("config builds");
        let model_built = model.build().expect("model builds");
        assert_eq!(
            built.control_type(),
            sd_control_type_t::SD_CONTROL_TYPE_CANNY
        );
        assert_eq!(built.control_image_path(), staged);
        assert_eq!(model_built.control_net_path(), weights);
        let _ = std::fs::remove_file(&staged);

        // No named type: the vendor default NONE (preprocessed map).
        let mut plain = req(7, 64, 64);
        plain.control_net = Some(weights.clone());
        plain.control_image = Some(png.to_vec());
        let mut config = ConfigBuilder::default();
        config.prompt("a honk".to_owned());
        let mut model = ModelConfigBuilder::default();
        let staged = stage_control(&plain, &mut config, &mut model)
            .expect("a complete pair stages")
            .expect("pair present");
        let _ = std::fs::remove_file(&staged);
        model.model(PathBuf::from("comfy:checkpoints/x.safetensors"));
        assert_eq!(
            config.build().expect("config builds").control_type(),
            sd_control_type_t::SD_CONTROL_TYPE_NONE
        );

        // A weights path that does not resolve is refused before any engine
        // work, naming the file.
        let mut missing = req(7, 64, 64);
        missing.control_net = Some(PathBuf::from("/nonexistent/cn.safetensors"));
        missing.control_image = Some(png.to_vec());
        let mut config = ConfigBuilder::default();
        config.prompt("a honk".to_owned());
        let mut model = ModelConfigBuilder::default();
        let err = stage_control(&missing, &mut config, &mut model).expect_err("missing weights");
        let SdError::Backend(msg) = err else {
            panic!("expected Backend variant, got {err:?}")
        };
        assert!(msg.contains("/nonexistent/cn.safetensors"), "{msg}");

        let _ = std::fs::remove_file(&weights);
    }

    #[test]
    fn sampler_names_roundtrip_and_reject_unknown() {
        for (i, name) in SAMPLER_NAMES.iter().enumerate() {
            let parsed = Sampler::parse(name).expect("parse");
            assert_eq!(parsed.name(), *name);
            assert_eq!(parsed as usize, i);
        }
        assert!(Sampler::parse("eulers").is_none());
        assert!(Sampler::parse("EULER").is_none());
        assert!(Sampler::parse("").is_none());
    }

    #[test]
    fn scheduler_names_roundtrip_and_reject_unknown() {
        for (i, name) in SCHEDULER_NAMES.iter().enumerate() {
            let parsed = Scheduler::parse(name).expect("parse");
            assert_eq!(parsed.name(), *name);
            assert_eq!(parsed as usize, i);
        }
        assert!(Scheduler::parse("normal").is_none());
        assert!(Scheduler::parse("Karras").is_none());
    }

    #[test]
    fn defaults_are_the_fixed_recipe() {
        // `None` overrides mean today's recipe: euler + `normal` (discrete)
        // + CFG 7.0, no negative prompt — recorded names stay stable.
        assert_eq!(SAMPLER, "euler");
        assert_eq!(SCHEDULER, "normal");
        assert_eq!(Sampler::parse(SAMPLER), Some(Sampler::Euler));
        assert_eq!(Scheduler::parse("discrete"), Some(Scheduler::Discrete));
        let r = req(1, 8, 8);
        assert!(r.sampler.is_none() && r.scheduler.is_none());
        assert!(r.cfg_scale.is_none() && r.negative_prompt.is_none());
    }

    #[test]
    fn stub_is_byte_deterministic() {
        let a = StubPipeline.generate(&req(42, 32, 32)).expect("gen");
        let b = StubPipeline.generate(&req(42, 32, 32)).expect("gen");
        assert_eq!(a.rgba, b.rgba);
        let pa = encode_image(a, OutputFormat::Png).expect("png");
        let pb = encode_image(b, OutputFormat::Png).expect("png");
        assert_eq!(pa, pb);
        assert_ne!(pa, [] as [u8; 0]);
    }

    #[test]
    fn stub_honors_size() {
        let img = StubPipeline.generate(&req(7, 8, 6)).expect("gen");
        assert_eq!((img.width, img.height), (8, 6));
        assert_eq!(img.rgba.len(), 8 * 6 * 4);
    }

    #[test]
    fn stub_caps_longest_edge() {
        assert!(StubPipeline.generate(&req(1, 2049, 16)).is_err());
        assert!(StubPipeline.generate(&req(1, 16, 2049)).is_err());
        assert!(StubPipeline.generate(&req(1, 2048, 2048)).is_ok());
    }

    #[test]
    fn different_seeds_differ() {
        let a = StubPipeline.generate(&req(1, 16, 16)).expect("gen");
        let b = StubPipeline.generate(&req(2, 16, 16)).expect("gen");
        assert_ne!(a.rgba, b.rgba);
    }

    /// Image-input fields never change the stub output: same seed with and
    /// without img2img/mask/`LoRA`/`ControlNet` fields is byte-identical
    /// (honest ignore; the HTTP surface refuses control inputs on non-sd-cpp
    /// backends instead of relying on this).
    #[test]
    fn stub_ignores_image_inputs() {
        let plain = req(9, 16, 16);
        let with = Txt2ImgRequest {
            loras: Some(vec![LoraRef {
                path: PathBuf::from("/models/loras/x.safetensors"),
                strength: 0.8,
            }]),
            init_image: Some(PathBuf::from("/tmp/in.png")),
            mask_image: Some(PathBuf::from("/tmp/mask.png")),
            strength: Some(0.5),
            hires: Some(HiresFix {
                scale: 2.0,
                target: None,
                denoising_strength: 0.7,
            }),
            control_net: Some(PathBuf::from("/models/controlnet/c.safetensors")),
            control_image: Some(vec![0x89, b'P', b'N', b'G']),
            controlnet_strength: Some(0.5),
            ..req(9, 16, 16)
        };
        let a = StubPipeline.generate(&plain).expect("gen");
        let b = StubPipeline.generate(&with).expect("gen");
        assert_eq!(a.rgba, b.rgba);
        // The stub ignores the hires pass, so the output stays at the base
        // size (the real backend would land at the hires target).
        assert_eq!((b.width, b.height), (16, 16));
    }

    /// Hires-fix target resolution: explicit target wins; scaling rounds to
    /// nearest and never collapses to zero.
    #[test]
    fn hires_target_resolution() {
        let scaled = HiresFix {
            scale: 2.0,
            target: None,
            denoising_strength: 0.7,
        };
        assert_eq!(hires_target((1024, 512), &scaled), (2048, 1024));
        let frac = HiresFix {
            scale: 1.25,
            target: None,
            denoising_strength: 0.7,
        };
        assert_eq!(hires_target((1024, 640), &frac), (1280, 800));
        let explicit = HiresFix {
            scale: 2.0,
            target: Some((640, 1536)),
            denoising_strength: 0.7,
        };
        assert_eq!(hires_target((1024, 1024), &explicit), (640, 1536));
        let tiny = HiresFix {
            scale: 0.001,
            target: None,
            denoising_strength: 0.7,
        };
        assert_eq!(hires_target((16, 16), &tiny), (1, 1));
    }

    /// Minimal safetensors prelude: 8 B LE header len plus a JSON header.
    fn header_bytes(keys: &[&str]) -> Vec<u8> {
        let tensors: serde_json::Map<String, serde_json::Value> = keys
            .iter()
            .map(|k| ((*k).to_string(), serde_json::json!({"dtype": "F32"})))
            .collect();
        let body = serde_json::to_vec(&tensors).expect("json");
        let mut out = (body.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn prefixed_header_stays_on_model_slot() {
        let header = header_bytes(&["model.diffusion_model.txt_in.weight"]);
        assert!(!safetensors_header_is_prefixless(&header));
    }

    #[test]
    fn prefixless_header_routes_to_diffusion_slot() {
        let header = header_bytes(&["txt_in.text_norm.weight", "decoder.conv.weight"]);
        assert!(safetensors_header_is_prefixless(&header));
    }

    #[test]
    fn garbage_header_falls_back_to_model_slot() {
        // GGUF magic decodes as a giant len: no header can match.
        assert!(!safetensors_header_is_prefixless(b"GGUF\x03\0\0\0rest"));
        assert!(!safetensors_header_is_prefixless(&[]));
        assert!(!safetensors_header_is_prefixless(&[1, 2, 3]));
        // Declared len longer than the body (truncated file).
        let mut truncated = 999u64.to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert!(!safetensors_header_is_prefixless(&truncated));
        // Valid JSON but not a tensor map.
        let mut arr = 2u64.to_le_bytes().to_vec();
        arr.extend_from_slice(b"[]");
        assert!(!safetensors_header_is_prefixless(&arr));
    }

    #[test]
    fn formats_roundtrip_dims_and_pixels() {
        for format in [OutputFormat::Png, OutputFormat::Jpeg, OutputFormat::Webp] {
            let img = StubPipeline.generate(&req(42, 16, 16)).expect("gen");
            let bytes = encode_image(img, format).expect("encode");
            assert!(!bytes.is_empty(), "{format:?}");
            let decoded = image::load_from_memory(&bytes).expect("decode").to_rgba8();
            assert_eq!((decoded.width(), decoded.height()), (16, 16), "{format:?}");
        }
        // Lossless containers preserve every pixel.
        for format in [OutputFormat::Png, OutputFormat::Webp] {
            let img = StubPipeline.generate(&req(42, 16, 16)).expect("gen");
            let want = img.rgba.clone();
            let bytes = encode_image(img, format).expect("encode");
            let decoded = image::load_from_memory(&bytes).expect("decode").to_rgba8();
            assert_eq!(decoded.as_raw(), &want, "{format:?}");
        }
        // JPEG quality 90 on noise: small mean error, not garbage.
        let img = StubPipeline.generate(&req(42, 16, 16)).expect("gen");
        let want = img.rgba.clone();
        let bytes = encode_image(img, OutputFormat::Jpeg).expect("encode");
        let decoded = image::load_from_memory(&bytes).expect("decode").to_rgba8();
        let err: u64 = decoded
            .as_raw()
            .iter()
            .zip(want.iter())
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum();
        assert!(
            err < 40 * want.len() as u64,
            "jpeg mean abs error {}",
            err / want.len() as u64
        );
    }

    #[test]
    fn output_format_naming_and_mime_agree() {
        for (format, name, extension, mime) in [
            (OutputFormat::Png, "png", "png", "image/png"),
            (OutputFormat::Jpeg, "jpeg", "jpg", "image/jpeg"),
            (OutputFormat::Webp, "webp", "webp", "image/webp"),
        ] {
            assert_eq!(format.name(), name, "{format:?}");
            assert_eq!(format.extension(), extension, "{format:?}");
            assert_eq!(format.mime(), mime, "{format:?}");
            assert_eq!(OutputFormat::from_name(name), Some(format), "{name}");
        }
        assert_eq!(OutputFormat::from_name("jpg"), None);
        assert_eq!(OutputFormat::from_name("gif"), None);
    }

    #[test]
    fn jpeg_quality_knob_changes_bytes_and_default_matches_90() {
        let img = StubPipeline.generate(&req(42, 16, 16)).expect("gen");
        let default_bytes = encode_image(img.clone(), OutputFormat::Jpeg).expect("encode");
        let q90 = encode_image_with_quality(img.clone(), OutputFormat::Jpeg, 90).expect("encode");
        assert_eq!(default_bytes, q90, "default = quality 90");
        let q10 = encode_image_with_quality(img.clone(), OutputFormat::Jpeg, 10).expect("encode");
        assert_ne!(default_bytes, q10, "quality changes JPEG bytes");
        // The knob is a JPEG-only control: PNG/WebP bytes are unaffected.
        let png_a = encode_image_with_quality(img.clone(), OutputFormat::Png, 10).expect("encode");
        let png_b = encode_image_with_quality(img.clone(), OutputFormat::Png, 90).expect("encode");
        assert_eq!(png_a, png_b);
    }
}
