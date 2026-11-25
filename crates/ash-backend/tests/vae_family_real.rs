//! Opt-in real-artifact header evidence for explicit VAE families.
//!
//! Set `HONKBOX_ASH_VAE_FAMILY` (`flux2-small` or `qwen-image21`),
//! `HONKBOX_ASH_VAE_MODEL`, and `HONKBOX_ASH_VAE_MODEL_SHA256`.
//! `HONKBOX_ASH_VAE_LICENSE` records operator-supplied terms when present.
//! No checkpoint is downloaded or bundled; this test performs no generation.

#![cfg(feature = "ash-sd")]

use honkbox_ash_backend::{
    checkpoint::CheckpointMap,
    device::VulkanDevice,
    vae_family::{spec_by_name, validate_required_keys},
    vae_family_adapters::{DecodeShapes, FamilyAdapter, OwnedAdapterWeights, encode_latent},
};
use honkbox_residency::{ResourceKind, global};
use sha2::{Digest, Sha256};
use std::{fs::File, io, io::Read, path::Path};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("string write");
    }
    out
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut bytes = vec![0u8; 64 * 1024];
    loop {
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        hasher.update(&bytes[..count]);
    }
    Ok(hex(&hasher.finalize()))
}

#[test]
#[ignore = "opt-in: needs a user-supplied VAE checkpoint and SHA-256"]
fn vae_family_real_header_evidence() {
    let family = std::env::var("HONKBOX_ASH_VAE_FAMILY")
        .unwrap_or_else(|_| panic!("real VAE test requires HONKBOX_ASH_VAE_FAMILY"));
    let spec = spec_by_name(&family).expect("declared VAE family");
    let model = std::env::var_os("HONKBOX_ASH_VAE_MODEL")
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
        .expect("real VAE test requires HONKBOX_ASH_VAE_MODEL");
    let expected_sha = std::env::var("HONKBOX_ASH_VAE_MODEL_SHA256")
        .unwrap_or_else(|_| panic!("real VAE test requires HONKBOX_ASH_VAE_MODEL_SHA256"));
    let license = std::env::var("HONKBOX_ASH_VAE_LICENSE")
        .unwrap_or_else(|_| "operator-supplied (not independently verified)".into());
    assert!(
        model.is_file(),
        "VAE model is not a file: {}",
        model.display()
    );
    let actual_sha = sha256_file(&model).expect("checkpoint SHA-256");
    assert!(
        actual_sha.eq_ignore_ascii_case(&expected_sha),
        "checkpoint SHA-256 mismatch: got {actual_sha}, expected {expected_sha}"
    );

    let map = CheckpointMap::read(&model).expect("checkpoint header");
    validate_required_keys(spec, &map).expect("required family keys");

    let shapes = DecodeShapes::new(spec, 1, 2, 3).expect("shapes");
    let values = vec![0.0_f32; shapes.latent_c * shapes.latent_h * shapes.latent_w];
    let encoded = encode_latent(
        spec,
        &values,
        shapes.latent_c,
        shapes.latent_h,
        shapes.latent_w,
    )
    .expect("synthetic encode shape");

    let _lease = global()
        .try_acquire(ResourceKind::Sd)
        .expect("real VAE test requires the exclusive SD lease");
    let device = VulkanDevice::headless().expect("real VAE test needs a headless Vulkan device");
    let adapter = FamilyAdapter::new(spec, device.into()).expect("adapter pipelines");
    let owned = OwnedAdapterWeights::synthetic(spec, &shapes);
    let latent = vec![0.0_f32; 2 * 3 * spec.diffusion_channels];
    let _ = adapter
        .decode(&latent, &shapes, &owned.borrowed())
        .expect("synthetic decode shape on selected device");

    println!(
        "vae family={} model={} sha256={actual_sha} license={license} encoded={}x{} channels={}",
        spec.name,
        model.display(),
        encoded.height,
        encoded.width,
        encoded.channels
    );
}
