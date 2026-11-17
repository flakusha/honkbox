//! Parity harness: structural agreement between backends on identical
//! [`Txt2ImgRequest`] inputs.
//!
//! Without GPU determinism, byte-level cross-backend parity is impossible: the
//! real `sd.cpp` backend runs a full diffusion model while [`StubPipeline`]
//! hashes the seed, so their pixels never match. What CAN be checked:
//!
//! - Both backends honor the [`MAX_EDGE`] cap (error, not panic) and requested
//!   dimensions.
//! - Encoded PNGs carry the magic bytes and decode via the `image` crate.
//! - Encoded files are non-empty and PNG decode round-trips to the exact
//!   requested dimensions.
//! - [`StubPipeline`] is byte-identical for the same fixed seed (the only
//!   byte-level determinism guarantee available, and what dev consumers see).
//!
//! The real-backend case is opt-in: `#[ignore]` + `HONKBOX_PARITY_MODEL=path`
//! (a local single-file checkpoint, e.g. an SDXL `.safetensors`). Default
//! `cargo test --workspace` stays green on CPU-only CI with no model download;
//! `-- --include-ignored` with the env unset skips the real backend with a
//! clear message. Run everything with `bun run test:parity`.

use std::path::PathBuf;

use honkbox_sd_core::{
    GeneratedImage, MAX_EDGE, OutputFormat, SdCppPipeline, SdError, SdPipeline, StubPipeline,
    Txt2ImgRequest, encode_image,
};

const PARITY_MODEL_ENV: &str = "HONKBOX_PARITY_MODEL";

fn png(png_bytes: &[u8]) -> image::DynamicImage {
    assert_eq!(
        &png_bytes[..8],
        b"\x89PNG\r\n\x1a\n",
        "output is not a PNG (bad magic)"
    );
    image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)
        .expect("PNG magic present but image crate cannot decode it")
}

fn structural_checks(backend: &dyn SdPipeline, img: &GeneratedImage, req: &Txt2ImgRequest) {
    assert_eq!(img.width, req.width, "backend ignored requested width");
    assert_eq!(img.height, req.height, "backend ignored requested height");
    assert_eq!(
        img.rgba.len(),
        req.width as usize * req.height as usize * 4,
        "RGBA buffer is not width*height*4"
    );
    assert!(
        img.rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255),
        "alpha channel is not fully opaque"
    );
    let bytes = encode_image(img.clone(), OutputFormat::Png).expect("PNG encode failed");
    assert!(!bytes.is_empty(), "encoded PNG is empty");
    let decoded = png(&bytes);
    assert_eq!(
        (decoded.width(), decoded.height()),
        (req.width, req.height),
        "decoded PNG dimensions differ from the request"
    );
    // Backend tag is non-empty and recorded consistently on repeat calls.
    let id = backend.backend_id();
    assert!(!id.is_empty(), "backend_id is empty");
    assert_eq!(backend.backend_id(), id, "backend_id changed between calls");
}

fn request(width: u32, height: u32, seed: u64) -> Txt2ImgRequest {
    Txt2ImgRequest {
        prompt: "a red cube on a blue table".to_owned(),
        seed,
        steps: 4,
        width,
        height,
        model_slug: "parity:test/model.safetensors".to_owned(),
        clip: None,
        vae: None,
        sampler: None,
        scheduler: None,
        cfg_scale: None,
        negative_prompt: None,
    }
}

/// Stub output passes all structural checks across geometries.
#[test]
fn stub_is_structurally_valid() {
    let stub = StubPipeline;
    for (w, h) in [(64, 64), (256, 128), (1, 1), (7, 13)] {
        let req = request(w, h, 42);
        let img = stub.generate(&req).expect("stub generate failed");
        structural_checks(&stub, &img, &req);
    }
}

/// Over-`MAX_EDGE` requests fail with [`SdError::TooLarge`] (stub path; the
/// real backend's identical cap check is covered in the ignored parity test).
#[test]
fn max_edge_cap_is_enforced() {
    let err = StubPipeline
        .generate(&request(MAX_EDGE + 1, 64, 42))
        .expect_err("edge above MAX_EDGE must fail");
    assert!(
        matches!(err, SdError::TooLarge { edge } if edge == MAX_EDGE + 1),
        "expected TooLarge, got {err:?}"
    );
    let err = StubPipeline
        .generate(&request(64, MAX_EDGE + 1, 42))
        .expect_err("edge above MAX_EDGE must fail");
    assert!(matches!(err, SdError::TooLarge { .. }), "got {err:?}");
    // Exactly MAX_EDGE passes.
    StubPipeline
        .generate(&request(MAX_EDGE, MAX_EDGE, 42))
        .expect("MAX_EDGE exactly must succeed");
}

/// The only byte-level determinism guarantee available: the stub is a pure
/// function of the seed. Guards accidental nondeterminism in what dev
/// consumers see; PNG encoding is lossless so bytes are stable too.
#[test]
fn stub_fixed_seed_is_byte_identical() {
    let req = request(128, 96, 0xDEAD_BEEF);
    let a = encode_image(StubPipeline.generate(&req).unwrap(), OutputFormat::Png).unwrap();
    let b = encode_image(StubPipeline.generate(&req).unwrap(), OutputFormat::Png).unwrap();
    assert_eq!(a, b, "same seed + same input must give byte-identical PNG");
    // Different seed gives different pixels (seed actually feeds the output).
    let other = encode_image(
        StubPipeline
            .generate(&request(128, 96, 0xDEAD_BEEF + 1))
            .unwrap(),
        OutputFormat::Png,
    )
    .unwrap();
    assert_ne!(a, other, "different seeds produced identical PNG bytes");
}

/// Stub determinism is per-seed, not per-input-object: same seed via different
/// request text/steps still yields identical bytes.
#[test]
fn stub_depends_only_on_seed_and_geometry() {
    let mut req = request(96, 96, 7);
    let a = encode_image(StubPipeline.generate(&req).unwrap(), OutputFormat::Png).unwrap();
    req.prompt = "something else entirely".to_owned();
    req.steps = 50;
    let b = encode_image(StubPipeline.generate(&req).unwrap(), OutputFormat::Png).unwrap();
    assert_eq!(a, b, "stub output changed when only prompt/steps changed");
}

/// Real-backend parity: opt-in via `HONKBOX_PARITY_MODEL=path` (local
/// single-file checkpoint). Runs `SdCppPipeline` on the same inputs as the
/// stub and applies the structural checks. Skipped (with a message) when the
/// env is unset so default CPU-only CI and `--include-ignored` runs stay green.
#[test]
#[ignore = "opt-in: needs HONKBOX_PARITY_MODEL=path to a local checkpoint"]
fn sd_cpp_parity_with_stub_structure() {
    let Some(model) = std::env::var(PARITY_MODEL_ENV)
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
    else {
        eprintln!("skipping sd-cpp parity: set {PARITY_MODEL_ENV}=<checkpoint path> to run");
        return;
    };
    assert!(
        model.is_file(),
        "{PARITY_MODEL_ENV}={} is not a readable file",
        model.display()
    );
    let real = SdCppPipeline::new(model);
    let stub = StubPipeline;
    // Small-but-sane generation: keeps the real run to seconds, still exercises
    // the full pipeline (sampling + VAE decode + PNG round-trip).
    for (w, h) in [(64, 64), (256, 128)] {
        let req = request(w, h, 42);
        let real_img = real.generate(&req).expect("sd-cpp generate failed");
        structural_checks(&real, &real_img, &req);
        // Structural agreement: the stub produces the same shape for the same
        // request (never pixel-equal — that's the documented GPU-nondeterminism
        // boundary this harness is honest about).
        let stub_img = stub.generate(&req).expect("stub generate failed");
        assert_eq!(
            (stub_img.width, stub_img.height),
            (real_img.width, real_img.height)
        );
    }
    // Cap enforcement on the real path too.
    let err = real
        .generate(&request(MAX_EDGE + 1, 64, 42))
        .expect_err("real backend must enforce MAX_EDGE");
    assert!(
        matches!(err, SdError::TooLarge { .. }),
        "expected TooLarge, got {err:?}"
    );
}
