//! Opt-in Real-ESRGAN RRDB + PixelShuffle spike.
//!
//! Set `HONKBOX_ASH_UPSCALER_MODEL` and
//! `HONKBOX_ASH_UPSCALER_MODEL_SHA256`. `HONKBOX_ASH_UPSCALER_LICENSE` may
//! record the operator-supplied license. No checkpoint is downloaded or bundled.

#![cfg(feature = "ash-sd")]

use honkbox_ash_backend::{
    checkpoint::CheckpointMap,
    device::VulkanDevice,
    rrdb::{RrdbBlock, RrdbShape, RrdbWeights},
};
use honkbox_residency::{ResourceKind, global};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
};

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
    let mut bytes = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        hasher.update(&bytes[..count]);
    }
    Ok(hex(&hasher.finalize()))
}

fn sha256_f32s(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("upscaler spike requires {name}=<local safetensors>"))
}

#[test]
#[ignore = "opt-in: needs a local Real-ESRGAN checkpoint and SHA-256"]
fn upscaler_spike() {
    let model = required_path("HONKBOX_ASH_UPSCALER_MODEL");
    let expected_sha = std::env::var("HONKBOX_ASH_UPSCALER_MODEL_SHA256")
        .unwrap_or_else(|_| panic!("upscaler spike requires HONKBOX_ASH_UPSCALER_MODEL_SHA256"));
    let license = std::env::var("HONKBOX_ASH_UPSCALER_LICENSE")
        .unwrap_or_else(|_| "operator-supplied (not independently verified)".into());
    assert!(
        model.is_file(),
        "upscaler model is not a file: {}",
        model.display()
    );
    let actual_sha = sha256_file(&model).expect("checkpoint SHA-256");
    assert!(
        actual_sha.eq_ignore_ascii_case(&expected_sha),
        "checkpoint SHA-256 mismatch: got {actual_sha}, expected {expected_sha}"
    );

    let map = CheckpointMap::read(&model).expect("checkpoint header");
    let weights = RrdbWeights::from_checkpoint(&map, &model).expect("RRDB weights");
    let shape = RrdbShape {
        batch: 1,
        height: 4,
        width: 4,
    };
    let input = (0..64 * shape.height * shape.width)
        .map(|i| f32::from(u16::try_from(i % 257).expect("0..257")) / 128.0 - 1.0)
        .collect::<Vec<_>>();

    let _lease = global()
        .try_acquire(ResourceKind::Sd)
        .expect("upscaler spike requires the exclusive SD lease");
    let device = VulkanDevice::headless().expect("upscaler spike needs a headless Vulkan device");
    let block = RrdbBlock::new(device.into()).expect("RRDB pipelines");
    let first = block.run(&input, shape, &weights).expect("RRDB first run");
    let second = block.run(&input, shape, &weights).expect("RRDB repeat run");
    let first_sha = sha256_f32s(&first);
    let second_sha = sha256_f32s(&second);
    assert_eq!(first_sha, second_sha, "same input must hash identically");
    println!(
        "upscaler spike: model={} license={license} output_sha256={first_sha} elements={}",
        model.display(),
        first.len()
    );
}
