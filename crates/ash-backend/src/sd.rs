//! `SdPipeline` impl for the direct-Vulkan path: skeleton only.
//!
//! [`AshSdPipeline`] validates the request, device and checkpoint container
//! before sampling. Actual diffusion kernels are still missing, so
//! [`SdPipeline::generate`] stops with a named sampling error rather than
//! returning a fake image. The shortcut [`honkbox_sd_core::SdCppPipeline`]
//! stays the default backend; callers swap the constructor, never the call
//! sites.

use crate::{
    VkError,
    checkpoint::CheckpointMap,
    checkpoint_family::{self, CheckpointFamily},
    clip_checkpoint::{ClipTensors, SD15_CLIP},
    ddim_sampler::DdimSampler,
    device::{CapacityRequirements, VulkanDevice},
    model::{ModelSpec, Op},
    model_spec::{SD15_SINGLE_LEVEL, validate},
    prompt,
    sampler::seeded_uniform_noise,
    shape_infer::{ShapeError, UnetShapes, infer},
    text_conditioning::TextConditioner,
    unet_checkpoint::decode,
    unet_real::{UnetReal, UnetRealWeights},
    vae_real::{VaeReal, VaeRealWeights},
};
use honkbox_sd_core::{GeneratedImage, MAX_EDGE, SdError, SdPipeline, Txt2ImgRequest};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    model_path: PathBuf,
    vocab_dir: Option<PathBuf>,
}

/// Passes the first checkpoints will use. Grows with the kernel inventory.
pub const SD_OPS: &[Op] = &[
    Op::ElementwiseAdd,
    Op::MatMul,
    Op::Softmax,
    Op::Attention,
    Op::LayerNorm,
    Op::Conv2d,
    Op::Silu,
    Op::GroupNorm,
    Op::SamplingStep,
    Op::ClassifierFreeGuidance,
    Op::EmbeddingGather,
    Op::Gelu,
    Op::Transpose,
];

const SD_SPEC: ModelSpec = ModelSpec {
    name: "sd15-ash",
    ops: SD_OPS,
};

impl AshSdPipeline {
    /// Bind the execution device and the already-resolved checkpoint path.
    pub fn new(device: Arc<VulkanDevice>, model_path: PathBuf) -> Self {
        Self::new_with_vocab_dir(device, model_path, None)
    }

    /// Same seam with an explicit CLIP tokenizer asset directory.
    pub fn new_with_vocab_dir(
        device: Arc<VulkanDevice>,
        model_path: PathBuf,
        vocab_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            device,
            model_path,
            vocab_dir,
        }
    }

    /// The device this pipeline will execute on.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Admission gate: call before any model load is reserved on this
    /// pipeline, so an unsupported device is named before work is queued.
    pub fn admit(&self, req: &CapacityRequirements) -> Result<(), crate::VkError> {
        self.device.info.admit(req)
    }
}

/// `UNet` input convolution — present in SD 1.x/2.x and SDXL safetensors.
const UNET_INPUT_TENSOR: &str = "model.diffusion_model.input_blocks.0.0.weight";
/// Latent channels the diffusion backbone consumes.
const LATENT_CHANNELS: usize = 4;
/// CLIP sequence length used by SD 1.5.
const CLIP_TOKENS: usize = 77;
/// CLIP EOS token id used for padding and pooling.
const CLIP_EOS: u32 = 49407;
/// CLIP vocabulary size and hidden width.
const CLIP_VOCAB: usize = 49408;
const CLIP_DIM: usize = 768;
/// Discrete diffusion steps baked into the SD 1.5 alpha schedule.
const DIFFUSION_STEPS: usize = 1000;
/// Default classifier-free guidance scale.
const DEFAULT_CFG: f32 = 7.0;
/// Sentinel used to translate a mid-run cancel into `SdError::Cancelled`.
const CANCELLED: &str = "ash-sd: cancelled";

fn validate_unet_input(map: &CheckpointMap) -> Result<(), String> {
    let tensor = map
        .get(UNET_INPUT_TENSOR)
        .ok_or_else(|| format!("missing UNet input tensor '{UNET_INPUT_TENSOR}'"))?;
    if tensor.shape.len() != 4
        || tensor.shape[1] != LATENT_CHANNELS
        || tensor.shape[2] != 3
        || tensor.shape[3] != 3
    {
        return Err(format!(
            "UNet input tensor '{UNET_INPUT_TENSOR}' has shape {:?}, expected [*, 4, 3, 3]",
            tensor.shape
        ));
    }
    if !matches!(tensor.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(format!(
            "UNet input tensor '{UNET_INPUT_TENSOR}' has dtype {}",
            tensor.dtype
        ));
    }
    Ok(())
}

/// Reject checkpoints that cannot be loaded before any GPU work is queued.
///
/// Safetensors gets a real tensor-map check: the `UNet` input must exist with
/// the expected NCHW shape and a floating dtype. GGUF has no safetensors map,
/// so it is refused by name instead of passing admission with unknown weights.
fn admit_checkpoint(path: &Path) -> Result<(), SdError> {
    let fail = |msg: String| SdError::Backend(format!("checkpoint '{}': {msg}", path.display()));
    let meta = std::fs::metadata(path).map_err(|e| fail(e.to_string()))?;
    if !meta.is_file() || meta.len() == 0 {
        return Err(fail("not a non-empty file".into()));
    }
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "safetensors" => {
            let map = CheckpointMap::read(path).map_err(|e| fail(e.to_string()))?;
            validate_unet_input(&map).map_err(fail)?;
        }
        "gguf" => {
            let mut file = std::fs::File::open(path).map_err(|e| fail(e.to_string()))?;
            let mut magic = [0u8; 4];
            file.read_exact(&mut magic)
                .map_err(|e| fail(format!("truncated GGUF header: {e}")))?;
            if &magic != b"GGUF" {
                return Err(fail("bad GGUF magic".into()));
            }
            return Err(fail(
                "GGUF checkpoints have no safetensors tensor map".into(),
            ));
        }
        other => return Err(fail(format!("unsupported checkpoint format '{other}'"))),
    }
    Ok(())
}

/// Validate a safetensors checkpoint against the declared model spec and
/// return the stage widths inferred from its tensor metadata.
///
/// No device is needed: this is the CLI/capability seam for checking a model
/// before reserving the GPU slot. Sampling still needs the full loader.
pub fn inspect_checkpoint(path: &Path) -> Result<UnetShapes, SdError> {
    let fail = |msg: String| SdError::Backend(format!("checkpoint '{}': {msg}", path.display()));
    let map = CheckpointMap::read(path).map_err(|e| fail(e.to_string()))?;
    validate(&SD15_SINGLE_LEVEL, &map).map_err(|e| fail(e.to_string()))?;
    infer(&SD15_SINGLE_LEVEL, &map).map_err(|e: ShapeError| fail(e.to_string()))
}

fn backend_error(message: impl std::fmt::Display) -> SdError {
    SdError::Backend(format!("ash-sd: {message}"))
}

/// Read the checkpoint's alpha schedule, or derive the SD 1.5 scaled-linear
/// fallback. ponytail: one host-side table; add other schedulers when a model
/// family declares one.
#[allow(clippy::cast_precision_loss)] // ponytail: index <= 999, exact in f32
fn alphas_cumprod(map: &CheckpointMap, path: &Path) -> Result<Vec<f32>, SdError> {
    if let Some(meta) = map.get_aliased("alphas_cumprod")
        && meta.shape == [DIFFUSION_STEPS]
    {
        let bytes = map.read_tensor(path, &meta.name).map_err(backend_error)?;
        let values = decode(&bytes, meta).map_err(backend_error)?;
        if values
            .iter()
            .all(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
        {
            return Ok(values);
        }
        return Err(backend_error("checkpoint alphas_cumprod is out of range"));
    }
    let (start, end) = (0.00085_f32.sqrt(), 0.012_f32.sqrt());
    let mut product = 1.0_f32;
    Ok((0..DIFFUSION_STEPS)
        .map(|index| {
            let fraction = index as f32 / (DIFFUSION_STEPS - 1) as f32;
            let beta = (start + (end - start) * fraction).powi(2);
            product *= 1.0 - beta;
            product
        })
        .collect())
}

/// Discrete timesteps and their alphas, plus the clean terminal alpha.
fn schedule(steps: u32, alphas: &[f32]) -> Result<(Vec<usize>, Vec<f32>), SdError> {
    if steps == 0 || alphas.len() != DIFFUSION_STEPS {
        return Err(backend_error(
            "steps must be positive with a 1000-step schedule",
        ));
    }
    let count = usize::try_from(steps).map_err(backend_error)?;
    let timesteps = if count == 1 {
        vec![DIFFUSION_STEPS - 1]
    } else {
        (0..count)
            .map(|index| DIFFUSION_STEPS - 1 - ((DIFFUSION_STEPS - 1) * index / (count - 1)))
            .collect()
    };
    let mut cumprod: Vec<f32> = timesteps.iter().map(|&t| alphas[t]).collect();
    cumprod.push(1.0);
    Ok((timesteps, cumprod))
}

fn latent_shape(width: u32, height: u32) -> Result<(usize, usize), SdError> {
    if width == 0 || height == 0 || !width.is_multiple_of(8) || !height.is_multiple_of(8) {
        return Err(backend_error(
            "image dimensions must be positive multiples of 8",
        ));
    }
    Ok(((height / 8) as usize, (width / 8) as usize))
}

/// NCHW `[1,3,H,W]` f32 → row-major RGBA8 with alpha 255.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped to 0..=255 above
fn rgba(pixels: &[f32], width: usize, height: usize) -> Result<Vec<u8>, SdError> {
    let plane = width
        .checked_mul(height)
        .ok_or_else(|| backend_error("pixel overflow"))?;
    if pixels.len() != 3 * plane {
        return Err(backend_error(format!(
            "decoder returned {} floats, expected {}",
            pixels.len(),
            3 * plane
        )));
    }
    let mut out = vec![0_u8; plane * 4];
    for index in 0..plane {
        for channel in 0..3 {
            let value = (pixels[channel * plane + index] + 1.0) * 127.5;
            out[index * 4 + channel] = value.clamp(0.0, 255.0) as u8;
        }
        out[index * 4 + 3] = 255;
    }
    Ok(out)
}

impl SdPipeline for AshSdPipeline {
    fn backend_id(&self) -> &'static str {
        ASH_SD_BACKEND_ID
    }

    #[allow(clippy::too_many_lines)] // one linear load → encode → sample → decode pipeline
    fn generate(
        &self,
        req: &Txt2ImgRequest,
        cancel: &AtomicBool,
    ) -> Result<GeneratedImage, SdError> {
        let edge = req.width.max(req.height);
        if edge > MAX_EDGE {
            return Err(SdError::TooLarge { edge });
        }
        if req.width == 0 || req.height == 0 {
            return Err(SdError::Backend("image dimensions must be positive".into()));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(SdError::Cancelled);
        }
        SD_SPEC.validate(&self.device.info).map_err(|e| {
            SdError::Backend(format!(
                "{}: device cannot host model: {e}",
                self.backend_id()
            ))
        })?;
        admit_checkpoint(&self.model_path)?;
        let (latent_height, latent_width) = latent_shape(req.width, req.height)?;
        // ponytail: reduced UNet/VAE path assumes square latents; generalize
        // square_side() in both modules before accepting arbitrary WxH.
        if latent_height != latent_width {
            return Err(backend_error(
                "reduced ash path supports square latents only",
            ));
        }
        let map = CheckpointMap::read(&self.model_path).map_err(backend_error)?;
        match checkpoint_family::detect(&map) {
            CheckpointFamily::Sd15 => {}
            other => {
                return Err(backend_error(format!(
                    "only SD 1.5 checkpoints are supported, found {}",
                    other.label()
                )));
            }
        }
        if cancel.load(Ordering::Acquire) {
            return Err(SdError::Cancelled);
        }
        let tokenizer = prompt::tokenizer_from(self.vocab_dir.as_deref()).map_err(backend_error)?;
        let positive = prompt::pad_ids(
            tokenizer.encode(&req.prompt).map_err(backend_error)?,
            CLIP_TOKENS,
            CLIP_EOS,
        );
        let negative = prompt::pad_ids(
            tokenizer
                .encode(req.negative_prompt.as_deref().unwrap_or(""))
                .map_err(backend_error)?,
            CLIP_TOKENS,
            CLIP_EOS,
        );
        let clip =
            ClipTensors::load(&map, &self.model_path, SD15_CLIP, 1e-5).map_err(backend_error)?;
        let conditioner =
            TextConditioner::new(Arc::clone(&self.device), SD15_CLIP.layers, SD15_CLIP.heads)
                .map_err(|e| backend_error(format!("text encoder: {e}")))?;
        let scale = req.cfg_scale.unwrap_or(DEFAULT_CFG);
        let context = conditioner
            .condition(
                &positive,
                &negative,
                &clip.weights(),
                CLIP_VOCAB,
                CLIP_DIM,
                scale,
            )
            .map_err(|e| backend_error(format!("conditioning: {e}")))?;
        if cancel.load(Ordering::Acquire) {
            return Err(SdError::Cancelled);
        }
        let unet_weights = UnetRealWeights::load(&map, &self.model_path).map_err(backend_error)?;
        let vae_weights = VaeRealWeights::load(&map, &self.model_path).map_err(backend_error)?;
        let unet = UnetReal::new(Arc::clone(&self.device))
            .map_err(|e| backend_error(format!("unet: {e}")))?;
        let vae = VaeReal::new(Arc::clone(&self.device))
            .map_err(|e| backend_error(format!("vae: {e}")))?;
        let sampler = DdimSampler::new(Arc::clone(&self.device))
            .map_err(|e| backend_error(format!("sampler: {e}")))?;
        let alphas = alphas_cumprod(&map, &self.model_path)?;
        let (timesteps, cumprod) = schedule(req.steps, &alphas)?;
        let noise = seeded_uniform_noise(LATENT_CHANNELS * latent_height * latent_width, req.seed)
            .map_err(|e| backend_error(format!("latent noise: {e}")))?;
        let latent = sampler
            .run(&noise, &cumprod, |current, index| {
                if cancel.load(Ordering::Acquire) {
                    return Err(VkError::Device(CANCELLED.into()));
                }
                let timestep = timesteps[index];
                unet.predict(current, timestep, &context, &unet_weights)
            })
            .map_err(|e| {
                if matches!(&e, VkError::Device(message) if message == CANCELLED) {
                    SdError::Cancelled
                } else {
                    backend_error(format!("sampling: {e}"))
                }
            })?;
        let pixels = vae
            .decode(&latent, &vae_weights)
            .map_err(|e| backend_error(format!("vae: {e}")))?;
        Ok(GeneratedImage {
            width: req.width,
            height: req.height,
            rgba: rgba(&pixels, latent_width * 8, latent_height * 8)?,
        })
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
    use crate::model_spec::tensor_keys;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Owns one unique temp file; removal happens on every exit path.
    struct TempFile(std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl std::ops::Deref for TempFile {
        type Target = std::path::PathBuf;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    fn temp_checkpoint(ext: &str, bytes: &[u8]) -> TempFile {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-{}-{}-{stamp}.{ext}",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write temp checkpoint");
        TempFile(path)
    }

    fn spec_checkpoint(omit: Option<&str>, prefixed: bool) -> TempFile {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (_, key) in tensor_keys(&SD15_SINGLE_LEVEL) {
            if omit == Some(key.as_str()) {
                continue;
            }
            let dims = if key == "time_embedding.linear_1.weight" {
                vec![8, 4]
            } else if key == "time_embedding.linear_2.weight" {
                vec![8, 8]
            } else if key.ends_with(".weight") {
                vec![4, 4, 3, 3]
            } else {
                vec![4]
            };
            let count: usize = dims.iter().product();
            let start = payload.len();
            payload.resize(start + count * 4, 0);
            let end = payload.len();
            let prefix = if key.starts_with("decoder.") {
                "first_stage_model."
            } else {
                "model.diffusion_model."
            };
            let header_key = if prefixed {
                format!("{prefix}{key}")
            } else {
                key
            };
            header.insert(
                header_key,
                serde_json::json!({
                    "dtype": "F32",
                    "shape": dims,
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header).expect("header");
        let mut bytes = u64::try_from(header_bytes.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload);
        temp_checkpoint("safetensors", &bytes)
    }

    #[test]
    fn inspect_checkpoint_validates_then_infers() {
        let valid = spec_checkpoint(None, false);
        let shapes = inspect_checkpoint(&valid).expect("inspect");
        assert_eq!(shapes.latent_channels, 4);
        assert_eq!(shapes.timestep_hidden, 8);

        let missing = spec_checkpoint(Some("time_embedding.linear_2.bias"), false);
        assert!(
            matches!(inspect_checkpoint(&missing), Err(SdError::Backend(msg)) if msg.contains("timestep linear 2")),
            "missing declared tensor must be named"
        );
    }

    #[test]
    fn inspect_accepts_single_file_prefixed_keys() {
        let path = spec_checkpoint(None, true);
        let shapes = inspect_checkpoint(&path).expect("prefixed inspect");
        assert_eq!(shapes.latent_channels, 4);
        assert_eq!(shapes.timestep_hidden, 8);
    }

    fn safetensors_bytes() -> Vec<u8> {
        safetensors_with_tensor("[8,4,3,3]", "F32", 1152)
    }

    fn safetensors_with_tensor(shape: &str, dtype: &str, byte_len: usize) -> Vec<u8> {
        let header = format!(
            "{{\"{UNET_INPUT_TENSOR}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[0,{byte_len}]}}}}"
        );
        let mut bytes = u64::try_from(header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + byte_len, 0);
        bytes
    }

    fn request(width: u32, height: u32) -> Txt2ImgRequest {
        Txt2ImgRequest {
            prompt: "test".to_owned(),
            seed: 42,
            steps: 4,
            width,
            height,
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
        }
    }

    fn device() -> Option<Arc<VulkanDevice>> {
        match VulkanDevice::headless() {
            Ok(device) => Some(Arc::new(device)),
            Err(VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue) => {
                eprintln!("skipping: no usable Vulkan environment");
                None
            }
            Err(e) => panic!("vulkan api failed: {e:?}"),
        }
    }

    #[test]
    fn admit_gates_on_the_selected_device() {
        let Some(device) = device() else { return };
        let pipeline = AshSdPipeline::new(device, PathBuf::from("unused"));
        assert_eq!(pipeline.admit(&CapacityRequirements::default()), Ok(()));
        let impossible = CapacityRequirements {
            subgroup_size: u32::MAX,
            ..CapacityRequirements::default()
        };
        assert!(matches!(
            pipeline.admit(&impossible),
            Err(VkError::InsufficientCapacity {
                capability: "subgroupSize",
                ..
            })
        ));
    }

    #[test]
    fn checkpoint_admission_accepts_sd_safetensors() {
        let safetensors = temp_checkpoint("safetensors", &safetensors_bytes());
        assert_eq!(admit_checkpoint(&safetensors), Ok(()));
        let gguf = temp_checkpoint("gguf", b"GGUF\x03\0\0\0rest");
        assert!(
            matches!(admit_checkpoint(&gguf), Err(SdError::Backend(msg)) if msg.contains("tensor map")),
            "GGUF must be refused by name"
        );
    }

    #[test]
    fn checkpoint_admission_checks_unet_tensor() {
        let empty_header = br#"{"__metadata__":{}}"#;
        let mut empty_bytes = u64::try_from(empty_header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        empty_bytes.extend_from_slice(empty_header);
        let missing = temp_checkpoint("safetensors", &empty_bytes);
        assert!(
            matches!(admit_checkpoint(&missing), Err(SdError::Backend(msg)) if msg.contains("missing UNet")),
            "empty tensor map must be refused"
        );
        let wrong = temp_checkpoint(
            "safetensors",
            &safetensors_with_tensor("[8,3,3,3]", "F32", 864),
        );
        assert!(
            matches!(admit_checkpoint(&wrong), Err(SdError::Backend(msg)) if msg.contains("shape")),
            "wrong latent channels must be refused"
        );
    }

    #[test]
    fn checkpoint_admission_rejects_bad_inputs() {
        let missing = std::env::temp_dir().join("honkbox-ash-no-such-checkpoint.safetensors");
        let empty = temp_checkpoint("safetensors", b"");
        let wrong_ext = temp_checkpoint("bin", b"data");
        let bad_magic = temp_checkpoint("gguf", b"NOTGGUF!");
        let truncated = temp_checkpoint("safetensors", b"\x02\0");
        let mut bad_len = 1000u64.to_le_bytes().to_vec();
        bad_len.extend_from_slice(b"{}");
        let huge_header = temp_checkpoint("safetensors", &bad_len);
        for path in [
            missing.as_path(),
            empty.as_path(),
            wrong_ext.as_path(),
            bad_magic.as_path(),
            truncated.as_path(),
            huge_header.as_path(),
        ] {
            assert!(
                matches!(admit_checkpoint(path), Err(SdError::Backend(_))),
                "expected rejection for {}",
                path.display()
            );
        }
    }

    #[test]
    fn generate_rejects_checkpoint_without_unet_tensor() {
        let Some(device) = device() else { return };
        let empty_header = br#"{"__metadata__":{}}"#;
        let mut empty_bytes = u64::try_from(empty_header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        empty_bytes.extend_from_slice(empty_header);
        let path = temp_checkpoint("safetensors", &empty_bytes);
        let pipeline = AshSdPipeline::new(device, path.clone());
        assert!(matches!(
            pipeline.generate(&request(512, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("UNet")
        ));
    }

    #[test]
    fn generate_checks_request_then_checkpoint_then_sampling() {
        let Some(device) = device() else { return };
        let missing = PathBuf::from("honkbox-ash-no-such-checkpoint.safetensors");
        let pipeline = AshSdPipeline::new(device.clone(), missing.clone());

        let cancelled = AtomicBool::new(true);
        assert!(matches!(
            pipeline.generate(&request(512, 512), &cancelled),
            Err(SdError::Cancelled)
        ));

        assert!(matches!(
            pipeline.generate(&request(3000, 512), &AtomicBool::new(false)),
            Err(SdError::TooLarge { edge: 3000 })
        ));
        assert!(matches!(
            pipeline.generate(&request(0, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("dimensions")
        ));
        assert!(matches!(
            pipeline.generate(&request(512, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("checkpoint")
        ));

        let valid = temp_checkpoint("safetensors", &safetensors_bytes());
        let ready = AshSdPipeline::new(device, valid.clone());
        assert!(matches!(
            ready.generate(&request(512, 512), &AtomicBool::new(false)),
            Err(SdError::Backend(msg)) if msg.contains("CLIP_VOCAB_DIR")
        ));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // synthetic index <= 999
    fn fallback_alphas_and_schedule_are_denoising_ordered() {
        let alphas: Vec<f32> = (0..DIFFUSION_STEPS)
            .map(|i| 0.999 - i as f32 * 0.0009)
            .collect();
        assert!(alphas.windows(2).all(|w| w[0] > w[1]));
        let (timesteps, cumprod) = schedule(4, &alphas).expect("schedule");
        assert_eq!(timesteps, vec![999, 666, 333, 0]);
        assert!(cumprod.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(cumprod.len(), 5);
        assert_eq!(cumprod[4], 1.0);
    }

    #[test]
    fn latent_shape_and_rgba_validate_geometry() {
        assert_eq!(latent_shape(64, 128).expect("shape"), (16, 8));
        assert!(latent_shape(63, 128).is_err());
        let pixels = vec![1.0_f32; 3 * 64];
        let image = rgba(&pixels, 8, 8).expect("rgba");
        assert_eq!(image.len(), 8 * 8 * 4);
        assert!(
            image
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px == &[255, 255, 255, 255])
        );
        assert!(rgba(&pixels, 4, 4).is_err());
    }
}
