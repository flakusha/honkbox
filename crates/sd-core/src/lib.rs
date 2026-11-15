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

/// Fixed sampler recorded in bundle sidecars for the minimal slice.
pub const SAMPLER: &str = "euler";
/// Fixed scheduler recorded in bundle sidecars for the minimal slice.
pub const SCHEDULER: &str = "normal";

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
/// (SAME seed = byte-identical PNG); honors `width`/`height`.
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
/// Fixed slice recipe: euler sampler, discrete (`normal`) scheduler, CFG 7.0, VAE tiling ON (required: the 1024² fp16 VAE decode otherwise exceeds the per-allocation limit). `gen_img` writes one PNG to a
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

impl SdPipeline for SdCppPipeline {
    fn backend_id(&self) -> &'static str {
        SD_CPP_BACKEND_ID
    }

    fn generate(&self, req: &Txt2ImgRequest) -> Result<GeneratedImage, SdError> {
        #[cfg(feature = "vulkan")]
        use diffusion_rs::api::{BackendDevice, Module};
        use diffusion_rs::api::{ConfigBuilder, ModelConfigBuilder, SampleMethod};

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
            .seed(req.seed.cast_signed())
            .steps(req.steps.cast_signed())
            .width(req.width.cast_signed())
            .height(req.height.cast_signed())
            .sampling_method(SampleMethod::EULER_SAMPLE_METHOD)
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
        }
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
