//! Structural parity for the direct-Vulkan SD 1.5 backend.
//!
//! Opt-in via `HONKBOX_ASH_PARITY_MODEL` (local single-file SD 1.5 checkpoint)
//! and `HONKBOX_ASH_CLIP_VOCAB_DIR` (directory with CLIP `vocab.json` +
//! `merges.txt`). Byte equality against sd.cpp is impossible without GPU
//! determinism; this checks the contract and same-seed repeatability instead.

#![cfg(feature = "ash-sd")]

use honkbox_ash_backend::device::VulkanDevice;
use honkbox_ash_backend::sd::AshSdPipeline;
use honkbox_sd_core::{GeneratedImage, OutputFormat, SdPipeline, Txt2ImgRequest, encode_image};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn request(width: u32, height: u32) -> Txt2ImgRequest {
    Txt2ImgRequest {
        prompt: "a red cube on a blue table".to_owned(),
        seed: 42,
        steps: 1,
        width,
        height,
        model_slug: "parity:ash.safetensors".to_owned(),
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

fn structural_checks(backend: &AshSdPipeline, image: &GeneratedImage, req: &Txt2ImgRequest) {
    assert_eq!((image.width, image.height), (req.width, req.height));
    assert_eq!(
        image.rgba.len(),
        req.width as usize * req.height as usize * 4
    );
    assert!(image.rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255));
    let bytes = encode_image(image.clone(), OutputFormat::Png).expect("png encode");
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let width = u32::from_be_bytes(bytes[16..20].try_into().expect("IHDR width"));
    let height = u32::from_be_bytes(bytes[20..24].try_into().expect("IHDR height"));
    assert_eq!((width, height), (req.width, req.height));
    assert_ne!(backend.backend_id(), "");
}

#[test]
#[ignore = "opt-in: needs a local SD 1.5 checkpoint and CLIP vocab assets"]
fn ash_sd_structural_and_repeatability_parity() {
    let Some(model) = std::env::var("HONKBOX_ASH_PARITY_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!("skipping ash parity: set HONKBOX_ASH_PARITY_MODEL=<checkpoint>");
        return;
    };
    let vocab = std::env::var_os("HONKBOX_ASH_CLIP_VOCAB_DIR").map(PathBuf::from);
    assert!(model.is_file(), "parity model is not a file");
    let device =
        Arc::new(VulkanDevice::headless().expect("ash parity needs a headless Vulkan device"));
    let pipeline = AshSdPipeline::new_with_vocab_dir(device, model, vocab);
    let req = request(64, 64);
    let first = pipeline
        .generate(&req, &AtomicBool::new(false))
        .expect("ash generate failed");
    structural_checks(&pipeline, &first, &req);
    let second = pipeline
        .generate(&req, &AtomicBool::new(false))
        .expect("ash repeat generate failed");
    assert_eq!(first.rgba, second.rgba, "same seed must repeat");
    assert!(matches!(
        pipeline.generate(&request(63, 64), &AtomicBool::new(false)),
        Err(honkbox_sd_core::SdError::Backend(_))
    ));
}
