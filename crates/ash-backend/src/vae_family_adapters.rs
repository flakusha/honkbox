//! Explicit synthetic VAE adapters for the declared Ash families.
//!
//! This module consumes `vae_family` specs and the reduced `VaeDecode` block.
//! It proves geometry, normalization, pack/unpack and one small decode
//! topology without loading a checkpoint or claiming family availability.

use crate::{
    VkError,
    conv_transpose::ConvTransposeShape,
    conv2d::ConvShape,
    device::VulkanDevice,
    vae_decode::{VaeDecode, VaeWeights, cpu_decode_reference},
    vae_family::{
        self, PackedLatent, VaeFamilyError, VaeFamilySpec, check_diffusion_channels,
        diffusion_to_vae, unpack_latents, vae_to_diffusion,
    },
};
use std::{fmt, sync::Arc};

/// Failure from adapter geometry, weights or Vulkan dispatch.
#[derive(Debug)]
pub enum AdapterError {
    /// A family contract guard rejected the request.
    Family(VaeFamilyError),
    /// A Vulkan operation failed.
    Vulkan(VkError),
}

impl From<VaeFamilyError> for AdapterError {
    fn from(error: VaeFamilyError) -> Self {
        Self::Family(error)
    }
}

impl From<VkError> for AdapterError {
    fn from(error: VkError) -> Self {
        Self::Vulkan(error)
    }
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Family(error) => fmt::Display::fmt(error, formatter),
            Self::Vulkan(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Geometry of the two reduced decode blocks for one family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeShapes {
    /// Batch size preserved end to end.
    pub batch: usize,
    /// Packed diffusion height.
    pub packed_h: usize,
    /// Packed diffusion width.
    pub packed_w: usize,
    /// Unpacked VAE channels.
    pub latent_c: usize,
    /// Unpacked VAE height.
    pub latent_h: usize,
    /// Unpacked VAE width.
    pub latent_w: usize,
    /// Hidden channels after the first block.
    pub hidden_c: usize,
    /// Hidden height after the first block.
    pub hidden_h: usize,
    /// Hidden width after the first block.
    pub hidden_w: usize,
    /// Decoded image channels.
    pub image_c: usize,
    /// Decoded image height.
    pub image_h: usize,
    /// Decoded image width.
    pub image_w: usize,
}

fn product(
    spec: &VaeFamilySpec,
    what: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, AdapterError> {
    left.checked_mul(right).ok_or_else(|| {
        VaeFamilyError::InvalidShape {
            family: spec.name,
            what,
            detail: "dimension overflow".to_owned(),
        }
        .into()
    })
}

impl DecodeShapes {
    /// Derive the reduced topology from a declared family spec.
    pub fn new(
        spec: &'static VaeFamilySpec,
        batch: usize,
        packed_h: usize,
        packed_w: usize,
    ) -> Result<Self, AdapterError> {
        check_diffusion_channels(spec, spec.diffusion_channels)?;
        if batch == 0 || packed_h == 0 || packed_w == 0 {
            return Err(VaeFamilyError::InvalidShape {
                family: spec.name,
                what: "decode batch or plane",
                detail: "zero dimension".to_owned(),
            }
            .into());
        }
        let latent_h = product(spec, "latent height", packed_h, spec.pack_factor)?;
        let latent_w = product(spec, "latent width", packed_w, spec.pack_factor)?;
        let hidden_h = product(spec, "hidden height", latent_h, 4)?;
        let hidden_w = product(spec, "hidden width", latent_w, 4)?;
        let image_h = product(spec, "image height", hidden_h, 4)?;
        let image_w = product(spec, "image width", hidden_w, 4)?;
        Ok(Self {
            batch,
            packed_h,
            packed_w,
            latent_c: spec.latent_channels,
            latent_h,
            latent_w,
            hidden_c: spec.latent_channels,
            hidden_h,
            hidden_w,
            image_c: spec.image_channels,
            image_h,
            image_w,
        })
    }

    fn up_shape(
        &self,
        in_c: usize,
        out_c: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) -> ConvTransposeShape {
        ConvTransposeShape {
            batch: self.batch,
            in_c,
            out_c,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel: 4,
            stride: 4,
            padding: 0,
        }
    }

    fn proj_shape(&self, in_c: usize, out_c: usize, height: usize, width: usize) -> ConvShape {
        ConvShape {
            batch: self.batch,
            in_c,
            out_c,
            in_h: height,
            in_w: width,
            out_h: height,
            out_w: width,
            kernel: 1,
            stride: 1,
            padding: 0,
        }
    }

    /// Shape of the first upsample block.
    pub fn first_up(&self) -> ConvTransposeShape {
        self.up_shape(
            self.latent_c,
            self.hidden_c,
            self.latent_h,
            self.latent_w,
            self.hidden_h,
            self.hidden_w,
        )
    }

    /// Shape of the first projection.
    pub fn first_proj(&self) -> ConvShape {
        self.proj_shape(self.hidden_c, self.hidden_c, self.hidden_h, self.hidden_w)
    }

    /// Shape of the second upsample block.
    pub fn second_up(&self) -> ConvTransposeShape {
        self.up_shape(
            self.hidden_c,
            self.image_c,
            self.hidden_h,
            self.hidden_w,
            self.image_h,
            self.image_w,
        )
    }

    /// Shape of the second projection.
    pub fn second_proj(&self) -> ConvShape {
        self.proj_shape(self.image_c, self.image_c, self.image_h, self.image_w)
    }
}

/// Weights for the two reduced blocks, in execution order.
pub struct AdapterWeights<'a> {
    /// First block weights.
    pub first: VaeWeights<'a>,
    /// Second block weights.
    pub second: VaeWeights<'a>,
}

/// Owned deterministic weights for synthetic fixtures.
pub struct OwnedAdapterWeights {
    /// First block upsample weights.
    pub first_up: Vec<f32>,
    /// First block upsample bias.
    pub first_up_bias: Vec<f32>,
    /// First block norm scale.
    pub first_gamma: Vec<f32>,
    /// First block norm shift.
    pub first_beta: Vec<f32>,
    /// First block projection weights.
    pub first_proj: Vec<f32>,
    /// First block projection bias.
    pub first_proj_bias: Vec<f32>,
    /// Second block upsample weights.
    pub second_up: Vec<f32>,
    /// Second block upsample bias.
    pub second_up_bias: Vec<f32>,
    /// Second block norm scale.
    pub second_gamma: Vec<f32>,
    /// Second block norm shift.
    pub second_beta: Vec<f32>,
    /// Second block projection weights.
    pub second_proj: Vec<f32>,
    /// Second block projection bias.
    pub second_proj_bias: Vec<f32>,
}

impl OwnedAdapterWeights {
    /// Build deterministic small weights for a family/geometry pair.
    #[allow(clippy::cast_precision_loss)]
    pub fn synthetic(_spec: &'static VaeFamilySpec, shapes: &DecodeShapes) -> Self {
        let fill = |len: usize, seed: u32| -> Vec<f32> {
            (0..len)
                .map(|i| (u32::try_from(i).unwrap_or(0) + seed) % 17)
                .map(|v| f32::from(u16::try_from(v).unwrap_or(0)) * 0.05 - 0.4)
                .collect()
        };
        Self {
            first_up: fill(shapes.latent_c * shapes.hidden_c * 16, 1),
            first_up_bias: fill(shapes.hidden_c, 2),
            first_gamma: fill(shapes.hidden_c, 3),
            first_beta: fill(shapes.hidden_c, 4),
            first_proj: fill(shapes.hidden_c * shapes.hidden_c, 5),
            first_proj_bias: fill(shapes.hidden_c, 6),
            second_up: fill(shapes.hidden_c * shapes.image_c * 16, 7),
            second_up_bias: fill(shapes.image_c, 8),
            second_gamma: fill(shapes.image_c, 9),
            second_beta: fill(shapes.image_c, 10),
            second_proj: fill(shapes.image_c * shapes.image_c, 11),
            second_proj_bias: fill(shapes.image_c, 12),
        }
    }

    /// Borrow these buffers as kernel weights.
    pub fn borrowed(&self) -> AdapterWeights<'_> {
        AdapterWeights {
            first: VaeWeights {
                up_weights: &self.first_up,
                up_bias: &self.first_up_bias,
                norm_gamma: &self.first_gamma,
                norm_beta: &self.first_beta,
                proj_weights: &self.first_proj,
                proj_bias: &self.first_proj_bias,
            },
            second: VaeWeights {
                up_weights: &self.second_up,
                up_bias: &self.second_up_bias,
                norm_gamma: &self.second_gamma,
                norm_beta: &self.second_beta,
                proj_weights: &self.second_proj,
                proj_bias: &self.second_proj_bias,
            },
        }
    }
}

fn validate_weights(
    spec: &VaeFamilySpec,
    shapes: &DecodeShapes,
    weights: &AdapterWeights<'_>,
) -> Result<(), AdapterError> {
    let checks = [
        (
            "first.up_weights",
            weights.first.up_weights,
            shapes.latent_c * shapes.hidden_c * 16,
        ),
        ("first.up_bias", weights.first.up_bias, shapes.hidden_c),
        (
            "first.norm_gamma",
            weights.first.norm_gamma,
            shapes.hidden_c,
        ),
        ("first.norm_beta", weights.first.norm_beta, shapes.hidden_c),
        (
            "first.proj_weights",
            weights.first.proj_weights,
            shapes.hidden_c * shapes.hidden_c,
        ),
        ("first.proj_bias", weights.first.proj_bias, shapes.hidden_c),
        (
            "second.up_weights",
            weights.second.up_weights,
            shapes.hidden_c * shapes.image_c * 16,
        ),
        ("second.up_bias", weights.second.up_bias, shapes.image_c),
        (
            "second.norm_gamma",
            weights.second.norm_gamma,
            shapes.image_c,
        ),
        ("second.norm_beta", weights.second.norm_beta, shapes.image_c),
        (
            "second.proj_weights",
            weights.second.proj_weights,
            shapes.image_c * shapes.image_c,
        ),
        ("second.proj_bias", weights.second.proj_bias, shapes.image_c),
    ];
    for (name, slice, expected) in checks {
        if slice.is_empty() {
            return Err(VaeFamilyError::MissingKey {
                family: spec.name,
                key: format!("synthetic.{name}"),
            }
            .into());
        }
        if slice.len() != expected {
            return Err(VaeFamilyError::InvalidShape {
                family: spec.name,
                what: "synthetic weight length",
                detail: format!("synthetic.{name} expected {expected}, got {}", slice.len()),
            }
            .into());
        }
    }
    Ok(())
}

fn prepare_latent(
    spec: &'static VaeFamilySpec,
    shapes: &DecodeShapes,
    diffusion_latent: &[f32],
) -> Result<Vec<f32>, AdapterError> {
    let expected = product(
        spec,
        "diffusion latent volume",
        product(spec, "diffusion plane", shapes.packed_h, shapes.packed_w)?,
        spec.diffusion_channels,
    )?;
    if diffusion_latent.len() != expected {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "diffusion latent",
            detail: format!("expected {expected} values, got {}", diffusion_latent.len()),
        }
        .into());
    }
    let normalized = diffusion_to_vae(spec, diffusion_latent)?;
    let unpacked = unpack_latents(
        spec,
        &normalized,
        spec.diffusion_channels,
        shapes.packed_h,
        shapes.packed_w,
    )?;
    let expected = product(
        spec,
        "unpacked latent volume",
        product(spec, "latent plane", shapes.latent_h, shapes.latent_w)?,
        shapes.latent_c,
    )?;
    if unpacked.len() != expected {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "unpacked latent",
            detail: format!("expected {expected} values, got {}", unpacked.len()),
        }
        .into());
    }
    Ok(unpacked)
}

/// Encode caller-supplied VAE latents into the family diffusion layout.
pub fn encode_latent(
    spec: &'static VaeFamilySpec,
    latents: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<PackedLatent, VaeFamilyError> {
    let packed = vae_family::pack_latents(spec, latents, channels, height, width)?;
    let data = vae_to_diffusion(spec, &packed.data)?;
    Ok(PackedLatent {
        data,
        channels: packed.channels,
        height: packed.height,
        width: packed.width,
    })
}

/// CPU reference for the full reduced decode topology.
pub fn decode_cpu(
    spec: &'static VaeFamilySpec,
    diffusion_latent: &[f32],
    shapes: &DecodeShapes,
    weights: &AdapterWeights<'_>,
) -> Result<Vec<f32>, AdapterError> {
    validate_weights(spec, shapes, weights)?;
    let latent = prepare_latent(spec, shapes, diffusion_latent)?;
    let first = cpu_decode_reference(
        &latent,
        shapes.first_up(),
        2,
        1e-5,
        shapes.first_proj(),
        &weights.first,
    );
    Ok(cpu_decode_reference(
        &first,
        shapes.second_up(),
        1,
        1e-5,
        shapes.second_proj(),
        &weights.second,
    ))
}

/// Vulkan-backed adapter over the shared decoder block kernels.
pub struct FamilyAdapter {
    spec: &'static VaeFamilySpec,
    first: VaeDecode,
    second: VaeDecode,
}

impl FamilyAdapter {
    /// Build both pipelines on the selected device.
    pub fn new(spec: &'static VaeFamilySpec, device: Arc<VulkanDevice>) -> Result<Self, VkError> {
        Ok(Self {
            spec,
            first: VaeDecode::new(Arc::clone(&device))?,
            second: VaeDecode::new(device)?,
        })
    }

    /// The pinned family contract used by this adapter.
    pub fn spec(&self) -> &'static VaeFamilySpec {
        self.spec
    }

    /// Decode one diffusion latent through the reduced two-block topology.
    pub fn decode(
        &self,
        diffusion_latent: &[f32],
        shapes: &DecodeShapes,
        weights: &AdapterWeights<'_>,
    ) -> Result<Vec<f32>, AdapterError> {
        validate_weights(self.spec, shapes, weights)?;
        let latent = prepare_latent(self.spec, shapes, diffusion_latent)?;
        let first = self.first.run(
            &latent,
            shapes.first_up(),
            2,
            1e-5,
            shapes.first_proj(),
            &weights.first,
        )?;
        Ok(self.second.run(
            &first,
            shapes.second_up(),
            1,
            1e-5,
            shapes.second_proj(),
            &weights.second,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vae_family::{FLUX2_SMALL, QWEN_IMAGE21};

    fn latent(spec: &VaeFamilySpec, shapes: &DecodeShapes) -> Vec<f32> {
        let len = shapes.packed_h * shapes.packed_w * spec.diffusion_channels;
        (0..len)
            .map(|i| f32::from(u16::try_from(i % 13).unwrap_or(0)) * 0.01 - 0.06)
            .collect()
    }

    #[test]
    fn adapter_shapes_are_family_specific() {
        let flux = DecodeShapes::new(&FLUX2_SMALL, 1, 2, 3).expect("flux shapes");
        assert_eq!((flux.latent_c, flux.latent_h, flux.latent_w), (32, 4, 6));
        assert_eq!((flux.packed_h, flux.packed_w), (2, 3));
        assert_eq!((flux.image_c, flux.image_h, flux.image_w), (3, 64, 96));
        let qwen = DecodeShapes::new(&QWEN_IMAGE21, 1, 2, 3).expect("qwen shapes");
        assert_eq!((qwen.latent_c, qwen.latent_h, qwen.latent_w), (64, 2, 3));
        assert_eq!((qwen.image_c, qwen.image_h, qwen.image_w), (4, 32, 48));
    }

    #[test]
    fn encode_then_decode_plane_returns_family_latents() {
        for spec in [&FLUX2_SMALL, &QWEN_IMAGE21] {
            let shapes = DecodeShapes::new(spec, 1, 2, 3).expect("shapes");
            let values: Vec<f32> = (0..shapes.latent_c * shapes.latent_h * shapes.latent_w)
                .map(|i| f32::from(u16::try_from(i % 11).unwrap_or(0)) * 0.02 - 0.1)
                .collect();
            let encoded = encode_latent(
                spec,
                &values,
                shapes.latent_c,
                shapes.latent_h,
                shapes.latent_w,
            )
            .expect("encode");
            let unpacked = unpack_latents(
                spec,
                &diffusion_to_vae(spec, &encoded.data).expect("denormalize"),
                spec.diffusion_channels,
                encoded.height,
                encoded.width,
            )
            .expect("decode plane");
            assert_eq!(unpacked.len(), values.len());
            for (got, want) in unpacked.iter().zip(&values) {
                assert!((got - want).abs() < 1e-4, "{}: {got} != {want}", spec.name);
            }
        }
    }

    #[test]
    fn missing_synthetic_block_is_named() {
        let spec = &FLUX2_SMALL;
        let shapes = DecodeShapes::new(spec, 1, 2, 2).expect("shapes");
        let mut owned = OwnedAdapterWeights::synthetic(spec, &shapes);
        owned.second_up_bias.clear();
        let error = decode_cpu(spec, &latent(spec, &shapes), &shapes, &owned.borrowed())
            .expect_err("missing block must fail");
        assert!(
            error.to_string().contains("synthetic.second.up_bias"),
            "{error}"
        );
    }

    #[test]
    fn wrong_latent_volume_is_named() {
        let spec = &QWEN_IMAGE21;
        let shapes = DecodeShapes::new(spec, 1, 2, 3).expect("shapes");
        let owned = OwnedAdapterWeights::synthetic(spec, &shapes);
        let error = decode_cpu(spec, &[0.0], &shapes, &owned.borrowed())
            .expect_err("short latent must fail");
        assert!(error.to_string().contains("diffusion latent"), "{error}");
    }
}
