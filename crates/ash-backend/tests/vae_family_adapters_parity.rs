//! Required-Vulkan parity for the two explicit VAE family adapters.

#![cfg(feature = "ash-sd")]

use honkbox_ash_backend::{
    VkError,
    device::VulkanDevice,
    vae_family::{FLUX2_SMALL, QWEN_IMAGE21, VaeFamilySpec},
    vae_family_adapters::{DecodeShapes, FamilyAdapter, OwnedAdapterWeights, decode_cpu},
};
use honkbox_residency::{ResourceKind, global};
use std::sync::Arc;

fn device() -> Option<Arc<VulkanDevice>> {
    let required = std::env::var("HONKBOX_ASH_REQUIRE_VULKAN").as_deref() == Ok("1");
    match VulkanDevice::headless() {
        Ok(device) => Some(Arc::new(device)),
        Err(error @ (VkError::Loader(_) | VkError::NoDevices | VkError::NoComputeQueue)) => {
            assert!(!required, "required Vulkan device unavailable: {error:?}");
            eprintln!("skipping: no usable Vulkan environment: {error:?}");
            None
        }
        Err(error) => panic!("vulkan api failed: {error:?}"),
    }
}

fn check(spec: &'static VaeFamilySpec, packed_h: usize, packed_w: usize) {
    let Some(device) = device() else {
        return;
    };
    let _lease = global()
        .try_acquire(ResourceKind::Sd)
        .expect("VAE parity requires the exclusive SD lease");
    let shapes = DecodeShapes::new(spec, 1, packed_h, packed_w).expect("shapes");
    let owned = OwnedAdapterWeights::synthetic(spec, &shapes);
    let len = packed_h * packed_w * spec.diffusion_channels;
    let latent: Vec<f32> = (0..len)
        .map(|i| f32::from(u16::try_from(i % 11).unwrap_or(0)) * 0.01 - 0.05)
        .collect();

    let actual = FamilyAdapter::new(spec, device)
        .expect("adapter pipelines")
        .decode(&latent, &shapes, &owned.borrowed())
        .expect("gpu decode");
    let expected = decode_cpu(spec, &latent, &shapes, &owned.borrowed()).expect("cpu decode");

    assert_eq!(
        actual.len(),
        shapes.image_c * shapes.image_h * shapes.image_w,
        "{} image volume",
        spec.name
    );
    for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "{} output[{index}]: got {got}, want {want}",
            spec.name
        );
    }
}

#[test]
fn flux2_small_rectangular_parity() {
    check(&FLUX2_SMALL, 2, 3);
}

#[test]
fn qwen_image21_rectangular_parity() {
    check(&QWEN_IMAGE21, 2, 3);
}
