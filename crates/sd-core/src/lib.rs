//! Diffusion pipeline (samplers, schedulers, recipes); sync only, no HTTP/DB.
//!
//! Two backends behind the real [`SdPipeline`] seam: [`StubPipeline`] (deterministic
//! STUB pixels, a pure function of the seed) and [`SdCppPipeline`] (real txt2img via
//! `diffusion-rs` / stable-diffusion.cpp on a caller-provided checkpoint file).
//! Callers swap the constructor, never the call sites.

use std::fmt;
use std::io::Cursor;
use std::path::PathBuf;

/// Longest image edge in pixels accepted by the stub backend.
pub const MAX_EDGE: u32 = 2048;

/// Default sampler recorded in bundle sidecars when a request omits one.
pub const SAMPLER: &str = "euler";
/// Default scheduler recorded in bundle sidecars when a request omits one.
pub const SCHEDULER: &str = "normal";

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

/// Minimal txt2img request crossing the `server` → `sd-core` seam.
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
/// overrides (`sampler`/`scheduler`/`cfg_scale`/`negative_prompt`) are
/// honestly ignored: the stub has no sampler — they only affect the
/// real `SdCppPipeline` backend.
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
/// per request. VAE tiling stays forced ON (required: the 1024² fp16 VAE decode otherwise exceeds the per-allocation limit). `gen_img` writes one PNG to a
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

impl SdPipeline for SdCppPipeline {
    fn backend_id(&self) -> &'static str {
        SD_CPP_BACKEND_ID
    }

    fn generate(&self, req: &Txt2ImgRequest) -> Result<GeneratedImage, SdError> {
        #[cfg(feature = "vulkan")]
        use diffusion_rs::api::{BackendDevice, Module};
        use diffusion_rs::api::{
            ConfigBuilder, ModelConfigBuilder, SampleMethod, Scheduler as VendorScheduler,
        };

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
        let config = ConfigBuilder::default()
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
            .output(out.clone())
            .build()
            .map_err(|e| SdError::Backend(e.to_string()))?;
        let mut model = ModelConfigBuilder::default();
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
        // Scheduler lives on the model config (vendor design); default is the
        // fixed recipe's discrete (`normal`) schedule.
        model.scheduler(match req.scheduler {
            Some(scheduler) => scheduler_vendor(scheduler),
            None => VendorScheduler::DISCRETE_SCHEDULER,
        });
        // Split encoder/VAE files (new archs ship them separately): `--clip`
        // feeds the llm (Qwen-VL-class) slot, `--vae` the standalone VAE.
        if let Some(clip) = req.clip.clone() {
            model.llm(clip);
        }
        if let Some(vae) = req.vae.clone() {
            model.vae(vae);
        }
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
}

/// Encode an image in the requested container; consumes the image.
///
/// JPEG uses quality 90; WebP is lossless-only in `image` 0.25 (no quality knob).
pub fn encode_image(img: GeneratedImage, format: OutputFormat) -> Result<Vec<u8>, SdError> {
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
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90);
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
        }
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
}
