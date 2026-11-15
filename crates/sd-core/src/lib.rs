//! Diffusion pipeline (samplers, schedulers, recipes); sync only, no HTTP/DB.
//!
//! The only backend at init is [`StubPipeline`]: deterministic STUB pixels (a
//! pure function of the seed) behind the real [`SdPipeline`] seam. Real
//! `diffusion-rs` wiring arrives later; callers swap the constructor, never
//! the call sites.

use std::fmt;
use std::io::Cursor;

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
    /// PNG encoding failed.
    Encode(String),
}

impl fmt::Display for SdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { edge } => {
                write!(f, "image edge {edge}px exceeds cap of {MAX_EDGE}px")
            }
            Self::Encode(e) => write!(f, "png encode failed: {e}"),
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

/// Encode an image as (deterministic) PNG bytes; consumes the image.
pub fn encode_png(img: GeneratedImage) -> Result<Vec<u8>, SdError> {
    let buf: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(img.width, img.height, img.rgba)
            .ok_or_else(|| SdError::Encode("bad pixel buffer length".to_string()))?;
    let mut out = Vec::new();
    image::DynamicImage::ImageRgba8(buf)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| SdError::Encode(e.to_string()))?;
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
        }
    }

    #[test]
    fn stub_is_byte_deterministic() {
        let a = StubPipeline.generate(&req(42, 32, 32)).expect("gen");
        let b = StubPipeline.generate(&req(42, 32, 32)).expect("gen");
        assert_eq!(a.rgba, b.rgba);
        let pa = encode_png(a).expect("png");
        let pb = encode_png(b).expect("png");
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
}
