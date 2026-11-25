//! Explicit VAE family contracts for FLUX.2-small and Qwen-Image-2.1.
//!
//! Every channel count, scale, mean/std vector and tensor name here is pinned
//! to `vendor/diffusion-rs/sys/stable-diffusion.cpp` at revision
//! `5dcfa79801213f23e27b3e83617350ebb8ea93aa` (MIT):
//!
//! - FLUX.2-small: `src/model/vae/auto_encoder_kl.hpp:539-660,664-693,773-836`
//!   and `src/name_conversion.cpp:1157-1250,1347-1360`;
//! - Qwen-Image-2.1: `src/model/vae/wan_vae.hpp:1077-1095,1355-1428` and
//!   `src/name_conversion.cpp:1253-1345`.
//!
//! No SD1.5 channel count, `1/0.18215` scale or generic alias is inherited.
//! Production names that the pinned conversion does not resolve to a unique
//! Wan block path are refused by name instead of guessed. This module is a
//! data/shape contract only: it does not run a VAE, own a device or decode a
//! real checkpoint.

use crate::checkpoint::{CheckpointMap, TensorMeta};
use std::fmt;

/// The two VAE families with an explicit contract here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaeFamily {
    /// FLUX.2-small: 32 base channels, 2×2 packed to 128 diffusion channels.
    Flux2Small,
    /// Qwen-Image-2.1: 64-channel latent, no 2×2 packing, RGBA output.
    QwenImage21,
}

/// One pinned Diffusers → original checkpoint tensor mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredKey {
    /// Diffusers spelling from the pinned conversion; empty when the pinned
    /// conversion does not disambiguate this tensor (original name only).
    pub diffusers: &'static str,
    /// Canonical stable-diffusion.cpp tensor name.
    pub original: &'static str,
    /// Rank implied by the pinned block; `0` when conv-vs-linear is resolved
    /// from checkpoint metadata at load time (never guessed here).
    pub rank: usize,
}

/// One tensor family whose production name cannot be established from the
/// pinned sources. Supplying such a key is refused, never aliased.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefusedKey {
    /// Exact representative tensor name.
    pub key: &'static str,
    /// Short family label reported in errors.
    pub family: &'static str,
    /// Why this family is not admitted.
    pub reason: &'static str,
}

/// All geometry, constants and key rules for one family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VaeFamilySpec {
    /// Family identity.
    pub family: VaeFamily,
    /// Stable name used in errors and logs.
    pub name: &'static str,
    /// VAE (unpacked) latent channels.
    pub latent_channels: usize,
    /// Diffusion latent channels.
    pub diffusion_channels: usize,
    /// Spatial packing factor (`2` for FLUX.2, `1` for Qwen-Image-2.1).
    pub pack_factor: usize,
    /// Image pixels per latent pixel.
    pub image_scale: usize,
    /// Decoded image channels.
    pub image_channels: usize,
    /// Decoded image channel order.
    pub output_order: &'static str,
    /// Normalization scale (pinned: `1.0` for both declared families).
    pub scale_factor: f32,
    /// Normalization shift (pinned: `0.0` for both declared families).
    pub shift_factor: f32,
    /// Per diffusion channel means, in pinned order.
    pub mean: &'static [f32],
    /// Per diffusion channel standard deviations, in pinned order.
    pub std: &'static [f32],
    /// Admitted exact tensor names.
    pub keys: &'static [RequiredKey],
    /// SD1.5-style aliases that must never be substituted.
    pub alias_traps: &'static [&'static str],
    /// Unresolved production key families, refused by name.
    pub refused_keys: &'static [RefusedKey],
}

/// Failures from family selection, shape guards and key validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaeFamilyError {
    /// Family name is not one of the two declared contracts.
    RefusedFamily {
        /// Rejected family name.
        name: String,
    },
    /// A channel count belongs to another family or to no family.
    ChannelMismatch {
        /// Selected family name.
        family: &'static str,
        /// Channel role: latent or diffusion.
        role: &'static str,
        /// Channels required by the spec.
        expected: usize,
        /// Channels supplied by the caller.
        actual: usize,
    },
    /// Tensor shape or layout is invalid.
    InvalidShape {
        /// Selected family name.
        family: &'static str,
        /// Invalid item.
        what: &'static str,
        /// Human-readable shape detail.
        detail: String,
    },
    /// A required tensor is absent.
    MissingKey {
        /// Selected family name.
        family: &'static str,
        /// Missing canonical tensor name.
        key: String,
    },
    /// Both spellings of one required tensor are present.
    DuplicateKey {
        /// Selected family name.
        family: &'static str,
        /// Canonical tensor name.
        key: String,
        /// Conflicting alias tensor name.
        alias: String,
    },
    /// A forbidden alias is present; no fallback is performed.
    UnexpectedKey {
        /// Selected family name.
        family: &'static str,
        /// Unexpected tensor name.
        key: String,
    },
    /// A required tensor has the wrong rank.
    ShapeMismatch {
        /// Selected family name.
        family: &'static str,
        /// Tensor name.
        key: String,
        /// Rank required by the pinned block.
        expected: usize,
        /// Rank found in the checkpoint.
        actual: usize,
    },
    /// An unresolved production key family was supplied.
    RefusedKeyFamily {
        /// Selected family name.
        family: &'static str,
        /// Refused key family label.
        keys: &'static str,
        /// Why the family is unresolved.
        reason: &'static str,
    },
}

impl fmt::Display for VaeFamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RefusedFamily { name } => {
                write!(f, "vae_family: refused family '{name}' (no explicit spec)")
            }
            Self::ChannelMismatch {
                family,
                role,
                expected,
                actual,
            } => write!(
                f,
                "vae_family: {family} {role} channels expected {expected}, got {actual}"
            ),
            Self::InvalidShape {
                family,
                what,
                detail,
            } => write!(f, "vae_family: {family} invalid {what}: {detail}"),
            Self::MissingKey { family, key } => {
                write!(f, "vae_family: {family} missing required tensor '{key}'")
            }
            Self::DuplicateKey { family, key, alias } => write!(
                f,
                "vae_family: {family} duplicate tensor '{key}' and '{alias}'"
            ),
            Self::UnexpectedKey { family, key } => {
                write!(f, "vae_family: {family} unexpected alias tensor '{key}'")
            }
            Self::ShapeMismatch {
                family,
                key,
                expected,
                actual,
            } => write!(
                f,
                "vae_family: {family} tensor '{key}' rank expected {expected}, got {actual}"
            ),
            Self::RefusedKeyFamily {
                family,
                keys,
                reason,
            } => write!(
                f,
                "vae_family: {family} refused key family '{keys}': {reason}"
            ),
        }
    }
}

impl std::error::Error for VaeFamilyError {}

// Pinned `auto_encoder_kl.hpp:780-795` — 128 channels, scale 1, shift 0.
const FLUX2_MEAN: [f32; 128] = [
    -0.0676, -0.0715, -0.0753, -0.0745, 0.0223, 0.0180, 0.0142, 0.0184, -0.0001, -0.0063, -0.0002,
    -0.0031, -0.0272, -0.0281, -0.0276, -0.0290, -0.0769, -0.0672, -0.0902, -0.0892, 0.0168,
    0.0152, 0.0079, 0.0086, 0.0083, 0.0015, 0.0003, -0.0043, -0.0439, -0.0419, -0.0438, -0.0431,
    -0.0102, -0.0132, -0.0066, -0.0048, -0.0311, -0.0306, -0.0279, -0.0180, 0.0030, 0.0015, 0.0126,
    0.0145, 0.0347, 0.0338, 0.0337, 0.0283, 0.0020, 0.0047, 0.0047, 0.0050, 0.0123, 0.0081, 0.0081,
    0.0146, 0.0681, 0.0679, 0.0767, 0.0732, -0.0462, -0.0474, -0.0392, -0.0511, -0.0528, -0.0477,
    -0.0470, -0.0517, -0.0317, -0.0316, -0.0345, -0.0283, 0.0510, 0.0445, 0.0578, 0.0458, -0.0412,
    -0.0458, -0.0487, -0.0467, -0.0088, -0.0106, -0.0088, -0.0046, -0.0376, -0.0432, -0.0436,
    -0.0499, 0.0118, 0.0166, 0.0203, 0.0279, 0.0113, 0.0129, 0.0016, 0.0072, -0.0118, -0.0018,
    -0.0141, -0.0054, -0.0091, -0.0138, -0.0145, -0.0187, 0.0323, 0.0305, 0.0259, 0.0300, 0.0540,
    0.0614, 0.0495, 0.0590, -0.0511, -0.0603, -0.0478, -0.0524, -0.0227, -0.0274, -0.0154, -0.0255,
    -0.0572, -0.0565, -0.0518, -0.0496, 0.0116, 0.0054, 0.0163, 0.0104,
];

// Pinned `auto_encoder_kl.hpp:797-812`.
const FLUX2_STD: [f32; 128] = [
    1.8029, 1.7786, 1.7868, 1.7837, 1.7717, 1.7590, 1.7610, 1.7479, 1.7336, 1.7373, 1.7340, 1.7343,
    1.8626, 1.8527, 1.8629, 1.8589, 1.7593, 1.7526, 1.7556, 1.7583, 1.7363, 1.7400, 1.7355, 1.7394,
    1.7342, 1.7246, 1.7392, 1.7304, 1.7551, 1.7513, 1.7559, 1.7488, 1.8449, 1.8454, 1.8550, 1.8535,
    1.8240, 1.7813, 1.7854, 1.7945, 1.8047, 1.7876, 1.7695, 1.7676, 1.7782, 1.7667, 1.7925, 1.7848,
    1.7579, 1.7407, 1.7483, 1.7368, 1.7961, 1.7998, 1.7920, 1.7925, 1.7780, 1.7747, 1.7727, 1.7749,
    1.7526, 1.7447, 1.7657, 1.7495, 1.7775, 1.7720, 1.7813, 1.7813, 1.8162, 1.8013, 1.8023, 1.8033,
    1.7527, 1.7331, 1.7563, 1.7482, 1.7610, 1.7507, 1.7681, 1.7613, 1.7665, 1.7545, 1.7828, 1.7726,
    1.7896, 1.7999, 1.7864, 1.7760, 1.7613, 1.7625, 1.7560, 1.7577, 1.7783, 1.7671, 1.7810, 1.7799,
    1.7201, 1.7068, 1.7265, 1.7091, 1.7793, 1.7578, 1.7502, 1.7455, 1.7587, 1.7500, 1.7525, 1.7362,
    1.7616, 1.7572, 1.7444, 1.7430, 1.7509, 1.7610, 1.7634, 1.7612, 1.7254, 1.7135, 1.7321, 1.7226,
    1.7664, 1.7624, 1.7718, 1.7664, 1.7457, 1.7441, 1.7569, 1.7530,
];

// Pinned `wan_vae.hpp:1390-1397` — 64 channels, scale 1, shift 0.
// ponytail: keep the pinned -0.5236 literal; it is not a stand-in for FRAC_PI_6.
#[allow(clippy::approx_constant)]
const QWEN21_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];

// Pinned `wan_vae.hpp:1398-1405`.
const QWEN21_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904,
    2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075,
    3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
    3.7662, 3.6746, 3.4563, 3.8161,
];

// One representative per pinned category: encoder/decoder, quant, resnet,
// attention, sampler (down/up-sample), norm and output. Rank 0 means the
// attention projection's conv-vs-linear rank is read from checkpoint metadata
// rather than assumed here (`auto_encoder_kl.hpp:70-85,694-704`).
const FLUX2_KEYS: &[RequiredKey] = &[
    RequiredKey {
        diffusers: "encoder.conv_in.weight",
        original: "encoder.conv_in.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "encoder.conv_in.bias",
        original: "encoder.conv_in.bias",
        rank: 1,
    },
    RequiredKey {
        diffusers: "encoder.down_blocks.0.resnets.0.norm1.weight",
        original: "encoder.down.0.block.0.norm1.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "encoder.down_blocks.0.resnets.0.conv1.weight",
        original: "encoder.down.0.block.0.conv1.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
        original: "encoder.down.1.block.0.nin_shortcut.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "encoder.down_blocks.0.downsamplers.0.conv.weight",
        original: "encoder.down.0.downsample.conv.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "mid_block.attentions.0.group_norm.weight",
        original: "encoder.mid.attn_1.norm.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "mid_block.attentions.0.to_q.weight",
        original: "encoder.mid.attn_1.q.weight",
        rank: 0,
    },
    RequiredKey {
        diffusers: "mid_block.resnets.0.norm1.weight",
        original: "encoder.mid.block_1.norm1.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "encoder.conv_norm_out.weight",
        original: "encoder.norm_out.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "encoder.conv_out.weight",
        original: "encoder.conv_out.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "encoder.quant_conv.weight",
        original: "quant_conv.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "decoder.post_quant_conv.weight",
        original: "post_quant_conv.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "decoder.conv_in.weight",
        original: "decoder.conv_in.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "",
        original: "decoder.mid.block_1.norm1.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "",
        original: "decoder.mid.attn_1.norm.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "decoder.up_blocks.3.resnets.0.norm1.weight",
        original: "decoder.up.0.block.0.norm1.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "up_blocks.0.upsamplers.0.conv.weight",
        original: "decoder.up.3.upsample.conv.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "decoder.conv_norm_out.weight",
        original: "decoder.norm_out.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "decoder.conv_out.weight",
        original: "decoder.conv_out.weight",
        rank: 4,
    },
];

const FLUX2_ALIAS_TRAPS: &[&str] = &[
    "first_stage_model.encoder.conv_in.weight",
    "first_stage_model.decoder.norm_out.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
];

// Pinned `name_conversion.cpp:1277-1322` maps quant_conv/post_quant_conv to
// conv1/conv2, conv_in to encoder.conv1, and mid attention norm to middle.1.
// Down/up/resnet/shortcut/sampler spellings are refused below: the pinned
// conversion does not produce the Wan block paths in `wan_vae.hpp:378-590`
// unambiguously, so those families are not guessed.
const QWEN21_KEYS: &[RequiredKey] = &[
    RequiredKey {
        diffusers: "encoder.conv_in.weight",
        original: "encoder.conv1.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "quant_conv.weight",
        original: "conv1.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "post_quant_conv.weight",
        original: "conv2.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "mid_block.attentions.0.norm.weight",
        original: "middle.1.norm.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "",
        original: "middle.1.to_qkv.weight",
        rank: 0,
    },
    RequiredKey {
        diffusers: "",
        original: "middle.1.proj.weight",
        rank: 0,
    },
    RequiredKey {
        diffusers: "",
        original: "middle.0.residual.0.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "",
        original: "middle.2.residual.6.weight",
        rank: 4,
    },
    RequiredKey {
        diffusers: "",
        original: "head.0.weight",
        rank: 1,
    },
    RequiredKey {
        diffusers: "",
        original: "head.2.weight",
        rank: 4,
    },
];

const QWEN21_REFUSED: &[RefusedKey] = &[
    RefusedKey {
        key: "encoder.downsamples.0.downsamples.0.residual.0.weight",
        family: "qwen21.down/resnet",
        reason: "pinned conversion does not resolve to the Wan residual block path",
    },
    RefusedKey {
        key: "encoder.downsamples.0.downsamples.0.shortcut.weight",
        family: "qwen21.shortcut",
        reason: "pinned conv_shortcut conversion leaves the block index ambiguous",
    },
    RefusedKey {
        key: "encoder.downsamples.0.downsamples.2.weight",
        family: "qwen21.sampler",
        reason: "pinned down/up-sampler index does not match the Wan block path",
    },
    RefusedKey {
        key: "decoder.upsamples.0.upsamples.0.residual.0.weight",
        family: "qwen21.up/resnet",
        reason: "pinned conversion does not resolve to the Wan upsample block path",
    },
];

const QWEN21_ALIAS_TRAPS: &[&str] = &[
    "first_stage_model.conv1.weight",
    "first_stage_model.head.2.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
];

/// FLUX.2-small VAE contract: 32 base channels, 2×2 pack to 128, RGB at 16×.
pub const FLUX2_SMALL: VaeFamilySpec = VaeFamilySpec {
    family: VaeFamily::Flux2Small,
    name: "flux2-small",
    latent_channels: 32,
    diffusion_channels: 128,
    pack_factor: 2,
    image_scale: 16,
    image_channels: 3,
    output_order: "RGB",
    scale_factor: 1.0,
    shift_factor: 0.0,
    mean: &FLUX2_MEAN,
    std: &FLUX2_STD,
    keys: FLUX2_KEYS,
    alias_traps: FLUX2_ALIAS_TRAPS,
    refused_keys: &[],
};

/// Qwen-Image-2.1 VAE contract: 64 channels, no packing, RGBA at 16×.
pub const QWEN_IMAGE21: VaeFamilySpec = VaeFamilySpec {
    family: VaeFamily::QwenImage21,
    name: "qwen-image21",
    latent_channels: 64,
    diffusion_channels: 64,
    pack_factor: 1,
    image_scale: 16,
    image_channels: 4,
    output_order: "RGBA",
    scale_factor: 1.0,
    shift_factor: 0.0,
    mean: &QWEN21_MEAN,
    std: &QWEN21_STD,
    keys: QWEN21_KEYS,
    alias_traps: QWEN21_ALIAS_TRAPS,
    refused_keys: QWEN21_REFUSED,
};

/// Resolve one of the two declared family names. All other names, including
/// SD1.5, FLUX.1 and the older 16-channel Qwen VAE, are refused.
pub fn spec_by_name(name: &str) -> Result<&'static VaeFamilySpec, VaeFamilyError> {
    match name {
        "flux2-small" | "flux2-small-vae" => Ok(&FLUX2_SMALL),
        "qwen-image21" | "qwen-image-2.1" => Ok(&QWEN_IMAGE21),
        _ => Err(VaeFamilyError::RefusedFamily {
            name: name.to_owned(),
        }),
    }
}

/// Resolve a family enum to its pinned spec.
pub fn spec(family: VaeFamily) -> &'static VaeFamilySpec {
    match family {
        VaeFamily::Flux2Small => &FLUX2_SMALL,
        VaeFamily::QwenImage21 => &QWEN_IMAGE21,
    }
}

fn channel_count(
    spec: &VaeFamilySpec,
    role: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VaeFamilyError> {
    if actual == expected {
        Ok(())
    } else {
        Err(VaeFamilyError::ChannelMismatch {
            family: spec.name,
            role,
            expected,
            actual,
        })
    }
}

/// Guard for unpacked VAE latent channels; rejects 4/16/32/64/128 misuse.
pub fn check_latent_channels(spec: &VaeFamilySpec, channels: usize) -> Result<(), VaeFamilyError> {
    channel_count(spec, "latent", spec.latent_channels, channels)
}

/// Guard for diffusion latent channels; rejects 4/16/32/64/128 misuse.
pub fn check_diffusion_channels(
    spec: &VaeFamilySpec,
    channels: usize,
) -> Result<(), VaeFamilyError> {
    channel_count(spec, "diffusion", spec.diffusion_channels, channels)
}

fn check_constants(spec: &VaeFamilySpec) -> Result<(), VaeFamilyError> {
    if spec.mean.len() != spec.diffusion_channels || spec.std.len() != spec.diffusion_channels {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "normalization constants",
            detail: format!(
                "expected {} mean/std entries, got {}/{}",
                spec.diffusion_channels,
                spec.mean.len(),
                spec.std.len()
            ),
        });
    }
    Ok(())
}

/// Diffusion → VAE: `z * std / scale + mean`, per pinned channel vector.
/// `values` is channel-major: every `diffusion_channels` entries form one site.
pub fn diffusion_to_vae(spec: &VaeFamilySpec, values: &[f32]) -> Result<Vec<f32>, VaeFamilyError> {
    check_constants(spec)?;
    if values.is_empty() || !values.len().is_multiple_of(spec.diffusion_channels) {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "diffusion latent vector",
            detail: format!(
                "length {} is not a positive multiple of {}",
                values.len(),
                spec.diffusion_channels
            ),
        });
    }
    Ok(values
        .chunks_exact(spec.diffusion_channels)
        .flat_map(|site| {
            site.iter()
                .zip(spec.std)
                .zip(spec.mean)
                .map(|((&z, &std), &mean)| z * std / spec.scale_factor + mean)
        })
        .collect())
}

/// VAE → diffusion: `(z - mean) * scale / std`, the exact inverse of
/// [`diffusion_to_vae`].
pub fn vae_to_diffusion(spec: &VaeFamilySpec, values: &[f32]) -> Result<Vec<f32>, VaeFamilyError> {
    check_constants(spec)?;
    if values.is_empty() || !values.len().is_multiple_of(spec.diffusion_channels) {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "vae latent vector",
            detail: format!(
                "length {} is not a positive multiple of {}",
                values.len(),
                spec.diffusion_channels
            ),
        });
    }
    Ok(values
        .chunks_exact(spec.diffusion_channels)
        .flat_map(|site| {
            site.iter()
                .zip(spec.mean)
                .zip(spec.std)
                .map(|((&z, &mean), &std)| (z - mean) * spec.scale_factor / std)
        })
        .collect())
}

/// Packed diffusion latent in NCHW order.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedLatent {
    /// Flat NCHW data.
    pub data: Vec<f32>,
    /// Packed channel count.
    pub channels: usize,
    /// Packed height.
    pub height: usize,
    /// Packed width.
    pub width: usize,
}

fn checked_volume(spec: &VaeFamilySpec, dims: &[usize]) -> Result<usize, VaeFamilyError> {
    dims.iter().copied().try_fold(1usize, |acc, dim| {
        if dim == 0 {
            Err(VaeFamilyError::InvalidShape {
                family: spec.name,
                what: "latent shape",
                detail: "zero dimension".to_owned(),
            })
        } else {
            acc.checked_mul(dim)
                .ok_or_else(|| VaeFamilyError::InvalidShape {
                    family: spec.name,
                    what: "latent shape",
                    detail: "dimension overflow".to_owned(),
                })
        }
    })
}

/// Pack `[1, latent_channels, H, W]` into the family diffusion layout.
/// FLUX.2 uses the pinned `p*p*C + 2*h%p + w%p` channel order; Qwen is an
/// explicit 64-channel pass-through and refuses accidental 128-channel packing.
pub fn pack_latents(
    spec: &VaeFamilySpec,
    latents: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<PackedLatent, VaeFamilyError> {
    check_latent_channels(spec, channels)?;
    let volume = checked_volume(spec, &[channels, height, width])?;
    if latents.len() != volume {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "latent plane",
            detail: format!("expected {volume} values, got {}", latents.len()),
        });
    }
    if spec.pack_factor == 1 {
        return Ok(PackedLatent {
            data: latents.to_vec(),
            channels,
            height,
            width,
        });
    }
    let p = spec.pack_factor;
    if !height.is_multiple_of(p) || !width.is_multiple_of(p) {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "latent plane",
            detail: format!("{height}x{width} is not divisible by {p}"),
        });
    }
    let shape = [
        channels
            .checked_mul(p * p)
            .ok_or_else(|| VaeFamilyError::InvalidShape {
                family: spec.name,
                what: "packed channels",
                detail: "channel overflow".to_owned(),
            })?,
        height / p,
        width / p,
    ];
    let mut packed = vec![0.0f32; checked_volume(spec, &shape)?];
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let sub = (y % p) * p + (x % p);
                let dst = (c * p * p + sub) * shape[1] * shape[2] + (y / p) * shape[2] + (x / p);
                packed[dst] = latents[(c * height + y) * width + x];
            }
        }
    }
    Ok(PackedLatent {
        data: packed,
        channels: shape[0],
        height: shape[1],
        width: shape[2],
    })
}

/// Inverse of [`pack_latents`]. FLUX.2 rejects non-divisible dimensions and
/// wrong channel counts; Qwen refuses 128-channel input instead of unpacking.
pub fn unpack_latents(
    spec: &VaeFamilySpec,
    packed: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, VaeFamilyError> {
    check_diffusion_channels(spec, channels)?;
    let volume = checked_volume(spec, &[channels, height, width])?;
    if packed.len() != volume {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "packed plane",
            detail: format!("expected {volume} values, got {}", packed.len()),
        });
    }
    if spec.pack_factor == 1 {
        return Ok(packed.to_vec());
    }
    let p = spec.pack_factor;
    if !channels.is_multiple_of(p * p) {
        return Err(VaeFamilyError::InvalidShape {
            family: spec.name,
            what: "packed plane",
            detail: format!("{channels} packed channels cannot be unpacked by {p}x{p}"),
        });
    }
    let shape = [channels / (p * p), height * p, width * p];
    let mut latents = vec![0.0f32; checked_volume(spec, &shape)?];
    for c in 0..shape[0] {
        for sub in 0..p * p {
            let y0 = sub / p;
            let x0 = sub % p;
            for y in 0..height {
                for x in 0..width {
                    let src = (c * p * p + sub) * height * width + y * width + x;
                    latents[(c * shape[1] + y * p + y0) * shape[2] + x * p + x0] = packed[src];
                }
            }
        }
    }
    Ok(latents)
}

fn check_rank(
    spec: &VaeFamilySpec,
    rule: &RequiredKey,
    meta: &TensorMeta,
) -> Result<(), VaeFamilyError> {
    if rule.rank == 0 || meta.shape.len() == rule.rank {
        Ok(())
    } else {
        Err(VaeFamilyError::ShapeMismatch {
            family: spec.name,
            key: rule.original.to_owned(),
            expected: rule.rank,
            actual: meta.shape.len(),
        })
    }
}

/// Strict validation against one family's pinned key map.
///
/// Only exact names are accepted: `get_aliased` is never used. Both spellings
/// of one tensor are a duplicate, a supplied SD1.5 alias is unexpected, and
/// unresolved Wan key families are refused by name rather than guessed.
pub fn validate_required_keys(
    spec: &VaeFamilySpec,
    map: &CheckpointMap,
) -> Result<(), VaeFamilyError> {
    for alias in spec.alias_traps {
        if map.get(alias).is_some() {
            return Err(VaeFamilyError::UnexpectedKey {
                family: spec.name,
                key: (*alias).to_owned(),
            });
        }
    }
    for refused in spec.refused_keys {
        if map.get(refused.key).is_some() {
            return Err(VaeFamilyError::RefusedKeyFamily {
                family: spec.name,
                keys: refused.family,
                reason: refused.reason,
            });
        }
    }
    for rule in spec.keys {
        let original = map.get(rule.original);
        let alias = if rule.diffusers.is_empty() || rule.diffusers == rule.original {
            None
        } else {
            map.get(rule.diffusers)
        };
        match (original, alias) {
            (None, None) => {
                return Err(VaeFamilyError::MissingKey {
                    family: spec.name,
                    key: rule.original.to_owned(),
                });
            }
            (Some(_), Some(b)) => {
                return Err(VaeFamilyError::DuplicateKey {
                    family: spec.name,
                    key: rule.original.to_owned(),
                    alias: b.name.clone(),
                });
            }
            (Some(a), None) | (None, Some(a)) => check_rank(spec, rule, a)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    const RANK1: &[usize] = &[1];
    const RANK3: &[usize] = &[1, 1, 1];
    const RANK4: &[usize] = &[1, 1, 1, 1];

    struct TempFile(std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Sparse safetensors fixture; each entry uses one f32 payload slot.
    fn checkpoint(entries: &[(&str, &[usize])]) -> (TempFile, CheckpointMap) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape) in entries {
            let start = payload.len();
            let elements: usize = shape.iter().product();
            payload.resize(start + elements * 4, 0u8);
            let end = payload.len();
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
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
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-vae-family-{}-{}-{stamp}.safetensors",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).expect("write checkpoint");
        (
            TempFile(path.clone()),
            CheckpointMap::read(&path).expect("map"),
        )
    }

    fn keys_of(spec: &VaeFamilySpec) -> Vec<(&'static str, Vec<usize>)> {
        spec.keys
            .iter()
            .map(|rule| {
                let rank = if rule.rank == 0 { 4 } else { rule.rank };
                (rule.original, vec![1; rank])
            })
            .collect()
    }

    #[test]
    fn vae_family_specs_use_complete_pinned_constants() {
        assert_eq!(FLUX2_MEAN.len(), 128);
        assert_eq!(FLUX2_STD.len(), 128);
        assert_eq!(QWEN21_MEAN.len(), 64);
        assert_eq!(QWEN21_STD.len(), 64);
        assert_eq!(FLUX2_SMALL.mean[0], -0.0676);
        assert_eq!(FLUX2_SMALL.mean[127], 0.0104);
        assert_eq!(FLUX2_SMALL.std[0], 1.8029);
        assert_eq!(FLUX2_SMALL.std[127], 1.7530);
        assert_eq!(QWEN_IMAGE21.mean[0], 0.5126);
        assert_eq!(QWEN_IMAGE21.mean[63], 0.0304);
        assert_eq!(QWEN_IMAGE21.std[0], 3.2001);
        assert_eq!(QWEN_IMAGE21.std[63], 3.8161);
        assert_eq!(FLUX2_SMALL.latent_channels, 32);
        assert_eq!(FLUX2_SMALL.diffusion_channels, 128);
        assert_eq!(FLUX2_SMALL.pack_factor, 2);
        assert_eq!(FLUX2_SMALL.image_scale, 16);
        assert_eq!(FLUX2_SMALL.output_order, "RGB");
        assert_eq!(QWEN_IMAGE21.latent_channels, 64);
        assert_eq!(QWEN_IMAGE21.diffusion_channels, 64);
        assert_eq!(QWEN_IMAGE21.pack_factor, 1);
        assert_eq!(QWEN_IMAGE21.image_scale, 16);
        assert_eq!(QWEN_IMAGE21.image_channels, 4);
        assert_eq!(QWEN_IMAGE21.output_order, "RGBA");
        for spec in [&FLUX2_SMALL, &QWEN_IMAGE21] {
            assert_eq!(spec.scale_factor, 1.0);
            assert_eq!(spec.shift_factor, 0.0);
            assert!(!spec.mean.contains(&(1.0 / 0.18215)));
        }
    }

    #[test]
    fn vae_family_normalization_is_inverse_within_1e_4() {
        for spec in [&FLUX2_SMALL, &QWEN_IMAGE21] {
            let n = spec.diffusion_channels;
            let values: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32) / 17.0 - 3.0) * spec.std[i % n])
                .collect();
            let vae = diffusion_to_vae(spec, &values).expect("encode");
            let back = vae_to_diffusion(spec, &vae).expect("decode");
            assert_eq!(back.len(), values.len());
            for (a, b) in values.iter().zip(&back) {
                assert!((a - b).abs() < 1e-4, "{}: {a} vs {b}", spec.name);
            }
            assert!((vae[0] - (values[0] * spec.std[0] + spec.mean[0])).abs() < 1e-4);
        }
    }

    #[test]
    fn vae_family_sd15_scale_and_cross_family_constants_fail() {
        let sd15 = VaeFamilySpec {
            scale_factor: 1.0 / 0.18215,
            ..FLUX2_SMALL
        };
        let values = vec![0.5f32; 128];
        assert_ne!(
            diffusion_to_vae(&sd15, &values).expect("sd15")[0],
            diffusion_to_vae(&FLUX2_SMALL, &values).expect("flux")[0]
        );
        let wrong = VaeFamilySpec {
            mean: &QWEN21_MEAN,
            std: &QWEN21_STD,
            ..FLUX2_SMALL
        };
        assert!(matches!(
            diffusion_to_vae(&wrong, &values),
            Err(VaeFamilyError::InvalidShape { .. })
        ));
        assert!(matches!(
            diffusion_to_vae(&QWEN_IMAGE21, &values[..65]),
            Err(VaeFamilyError::InvalidShape { .. })
        ));
        assert!(matches!(
            spec_by_name("sd15"),
            Err(VaeFamilyError::RefusedFamily { .. })
        ));
        assert!(matches!(
            spec_by_name("qwen-image"),
            Err(VaeFamilyError::RefusedFamily { .. })
        ));
    }

    #[test]
    fn vae_family_flux2_pack_unpack_golden_order() {
        let latents = [
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            26.0, 27.0,
        ];
        assert!(matches!(
            pack_latents(&FLUX2_SMALL, &latents, 128, 2, 2),
            Err(VaeFamilyError::ChannelMismatch { .. })
        ));
        let two = VaeFamilySpec {
            latent_channels: 2,
            diffusion_channels: 8,
            ..FLUX2_SMALL
        };
        let packed = pack_latents(&two, &latents, 2, 2, 4).expect("pack");
        assert_eq!((packed.channels, packed.height, packed.width), (8, 1, 2));
        assert_eq!(
            packed.data,
            [
                10.0, 12.0, 11.0, 13.0, 14.0, 16.0, 15.0, 17.0, 20.0, 22.0, 21.0, 23.0, 24.0, 26.0,
                25.0, 27.0
            ]
        );
        let back = unpack_latents(&two, &packed.data, 8, 1, 2).expect("unpack");
        assert_eq!(back, latents);
        assert!(matches!(
            pack_latents(&two, &latents[..12], 2, 3, 2),
            Err(VaeFamilyError::InvalidShape { .. })
        ));
        assert!(matches!(
            pack_latents(&two, &latents, 2, 1, 5),
            Err(VaeFamilyError::InvalidShape { .. })
        ));
        assert!(matches!(
            unpack_latents(&FLUX2_SMALL, &[0.0; 8], 128, 1, 2),
            Err(VaeFamilyError::InvalidShape { .. })
        ));
    }

    #[test]
    fn vae_family_qwen_passes_64_channels_and_refuses_128() {
        let values: Vec<f32> = (0..64 * 15).map(|i| i as f32 * 0.25).collect();
        let packed = pack_latents(&QWEN_IMAGE21, &values, 64, 3, 5).expect("pass-through");
        assert_eq!((packed.channels, packed.height, packed.width), (64, 3, 5));
        assert_eq!(packed.data, values);
        let back = unpack_latents(&QWEN_IMAGE21, &packed.data, 64, 3, 5).expect("unpack");
        assert_eq!(back, values);
        assert!(matches!(
            pack_latents(&QWEN_IMAGE21, &values, 128, 3, 5),
            Err(VaeFamilyError::ChannelMismatch { .. })
        ));
        assert!(matches!(
            unpack_latents(&QWEN_IMAGE21, &[0.0; 128], 128, 1, 1),
            Err(VaeFamilyError::ChannelMismatch { .. })
        ));
    }

    #[test]
    fn vae_family_channel_guards_reject_all_foreign_counts() {
        for spec in [&FLUX2_SMALL, &QWEN_IMAGE21] {
            for count in [4, 16, 32, 64, 128] {
                assert_eq!(
                    check_latent_channels(spec, count).is_ok(),
                    count == spec.latent_channels
                );
                assert_eq!(
                    check_diffusion_channels(spec, count).is_ok(),
                    count == spec.diffusion_channels
                );
            }
        }
    }

    #[test]
    fn vae_family_key_validation_names_errors_and_refuses_aliases() {
        let entries = keys_of(&FLUX2_SMALL);
        let refs: Vec<(&str, &[usize])> = entries
            .iter()
            .map(|(name, shape)| (*name, shape.as_slice()))
            .collect();
        let (_temp, map) = checkpoint(&refs);
        validate_required_keys(&FLUX2_SMALL, &map).expect("complete keys");

        let mut missing = refs.clone();
        missing.retain(|(name, _)| *name != "decoder.conv_out.weight");
        let (_temp, map) = checkpoint(&missing);
        assert_eq!(
            validate_required_keys(&FLUX2_SMALL, &map),
            Err(VaeFamilyError::MissingKey {
                family: "flux2-small",
                key: "decoder.conv_out.weight".to_owned(),
            })
        );

        let mut duplicate = refs.clone();
        duplicate.push(("encoder.quant_conv.weight", RANK4));
        let (_temp, map) = checkpoint(&duplicate);
        assert_eq!(
            validate_required_keys(&FLUX2_SMALL, &map),
            Err(VaeFamilyError::DuplicateKey {
                family: "flux2-small",
                key: "quant_conv.weight".to_owned(),
                alias: "encoder.quant_conv.weight".to_owned(),
            })
        );

        let mut alias = refs.clone();
        alias.push(("first_stage_model.decoder.norm_out.weight", RANK1));
        let (_temp, map) = checkpoint(&alias);
        assert_eq!(
            validate_required_keys(&FLUX2_SMALL, &map),
            Err(VaeFamilyError::UnexpectedKey {
                family: "flux2-small",
                key: "first_stage_model.decoder.norm_out.weight".to_owned(),
            })
        );

        let mut wrong_shape: Vec<(&str, &[usize])> = entries
            .iter()
            .map(|(name, shape)| (*name, shape.as_slice()))
            .collect();
        for (name, shape) in &mut wrong_shape {
            if *name == "encoder.conv_in.weight" {
                *shape = RANK3;
            }
        }
        let (_temp, map) = checkpoint(&wrong_shape);
        assert_eq!(
            validate_required_keys(&FLUX2_SMALL, &map),
            Err(VaeFamilyError::ShapeMismatch {
                family: "flux2-small",
                key: "encoder.conv_in.weight".to_owned(),
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn vae_family_qwen_key_map_accepts_exact_and_refuses_unresolved() {
        let entries = keys_of(&QWEN_IMAGE21);
        let refs: Vec<(&str, &[usize])> = entries
            .iter()
            .map(|(name, shape)| (*name, shape.as_slice()))
            .collect();
        let (_temp, map) = checkpoint(&refs);
        validate_required_keys(&QWEN_IMAGE21, &map).expect("exact qwen keys");

        let mut unresolved = refs.clone();
        unresolved.push((
            "encoder.downsamples.0.downsamples.0.residual.0.weight",
            RANK1,
        ));
        let (_temp, map) = checkpoint(&unresolved);
        assert_eq!(
            validate_required_keys(&QWEN_IMAGE21, &map),
            Err(VaeFamilyError::RefusedKeyFamily {
                family: "qwen-image21",
                keys: "qwen21.down/resnet",
                reason: "pinned conversion does not resolve to the Wan residual block path",
            })
        );

        let mut alias_only: Vec<(&str, &[usize])> = refs
            .iter()
            .copied()
            .filter(|(name, _)| *name != "conv2.weight")
            .collect();
        alias_only.push(("first_stage_model.conv2.weight", RANK1));
        let (_temp, map) = checkpoint(&alias_only);
        assert!(matches!(
            validate_required_keys(&QWEN_IMAGE21, &map),
            Err(VaeFamilyError::MissingKey { .. }) | Err(VaeFamilyError::UnexpectedKey { .. })
        ));
    }
}
