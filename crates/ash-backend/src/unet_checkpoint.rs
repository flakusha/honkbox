//! Safetensors → `UnetInputWeights`: key lookup, shape checks, f32 decode.
//!
//! Callers name the four checkpoint keys explicitly. No model family is
//! assumed: SD 1.x/2.x and SDXL place the first convolution and its norms
//! under different paths, and guessing would bind the wrong tensors silently.

use crate::{
    checkpoint::{CheckpointMap, TensorMeta},
    unet_input::UnetInputShape,
};
use std::io::{self, ErrorKind};
use std::path::Path;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

/// Checkpoint keys for one convolution + norm block.
#[derive(Debug, Clone, Copy)]
pub struct UnetInputKeys<'a> {
    /// `[out_c, in_c, kernel, kernel]` tensor.
    pub kernel: &'a str,
    /// `[out_c]` tensor.
    pub bias: &'a str,
    /// `[out_c]` group-norm gain tensor.
    pub gamma: &'a str,
    /// `[out_c]` group-norm bias tensor.
    pub beta: &'a str,
}

/// Decoded f32 tensors ready to borrow into `UnetInputWeights`.
pub struct UnetInputTensors {
    /// Convolution kernel in row-major `[out_c, in_c, k, k]` order.
    pub kernel: Vec<f32>,
    /// Convolution bias.
    pub bias: Vec<f32>,
    /// Group-norm gain.
    pub gamma: Vec<f32>,
    /// Group-norm bias.
    pub beta: Vec<f32>,
}

/// IEEE binary16 → f32 by bit expansion (std `f16` is still unstable here).
fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let result = if exponent == 0 {
        if fraction == 0 {
            sign << 31
        } else {
            let mut mantissa = fraction;
            let mut biased = 127 - 14;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                biased -= 1;
            }
            (sign << 31) | (biased << 23) | (mantissa & 0x03ff)
        }
    } else if exponent == 0x1f {
        (sign << 31) | 0x7f80_0000 | (fraction << 13)
    } else {
        let biased = u32::try_from(i32::from(exponent) - 15 + 127).expect("half exponent range");
        (sign << 31) | (biased << 23) | (fraction << 13)
    };
    f32::from_bits(result)
}

/// bf16 is the top 16 bits of an f32; no rounding is needed.
fn bfloat16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(crate) fn decode(bytes: &[u8], meta: &TensorMeta) -> io::Result<Vec<f32>> {
    let expected = meta.byte_len;
    if bytes.len() != expected {
        return Err(invalid(format!(
            "tensor '{}': read {} bytes, expected {expected}",
            meta.name,
            bytes.len()
        )));
    }
    match meta.dtype.as_str() {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| Ok(f32::from_le_bytes(*chunk)))
            .collect(),
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| Ok(half_bits_to_f32(u16::from_le_bytes(*chunk))))
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| Ok(bfloat16_bits_to_f32(u16::from_le_bytes(*chunk))))
            .collect(),
        other => Err(invalid(format!(
            "tensor '{}': dtype '{other}' cannot be converted to f32",
            meta.name
        ))),
    }
}

/// Read one declared tensor as f32. Callers validate the shape first.
pub fn load_tensor(
    map: &CheckpointMap,
    path: &Path,
    name: &str,
    expected_shape: &[usize],
) -> io::Result<Vec<f32>> {
    let meta = map
        .get_aliased(name)
        .ok_or_else(|| invalid(format!("checkpoint missing tensor '{name}'")))?;
    if meta.shape != expected_shape {
        return Err(invalid(format!(
            "tensor '{name}' has shape {:?}, expected {expected_shape:?}",
            meta.shape
        )));
    }
    // Read by the canonical checkpoint name: `get_aliased` may have matched
    // an original single-file key that differs from the spec-style `name`.
    decode(&map.read_tensor(path, &meta.name)?, meta)
}

/// Load and validate the four tensors described by `keys` for `shape`.
///
/// Every tensor is checked against its expected shape before any bytes are
/// read, so a mismatched checkpoint fails without loading the whole payload.
pub fn load_unet_input(
    map: &CheckpointMap,
    path: &Path,
    keys: &UnetInputKeys<'_>,
    shape: UnetInputShape,
) -> io::Result<UnetInputTensors> {
    let kernel_shape = [shape.out_c, shape.in_c, shape.kernel, shape.kernel];
    let channel_shape = [shape.out_c];
    for (name, expected) in [
        (keys.kernel, &kernel_shape[..]),
        (keys.bias, &channel_shape[..]),
        (keys.gamma, &channel_shape[..]),
        (keys.beta, &channel_shape[..]),
    ] {
        let meta = map
            .get_aliased(name)
            .ok_or_else(|| invalid(format!("checkpoint missing tensor '{name}'")))?;
        if meta.shape != expected {
            return Err(invalid(format!(
                "tensor '{name}' has shape {:?}, expected {expected:?}",
                meta.shape
            )));
        }
    }
    Ok(UnetInputTensors {
        kernel: load_tensor(map, path, keys.kernel, &kernel_shape)?,
        bias: load_tensor(map, path, keys.bias, &channel_shape)?,
        gamma: load_tensor(map, path, keys.gamma, &channel_shape)?,
        beta: load_tensor(map, path, keys.beta, &channel_shape)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    const KERNEL: &str = "unet.input.kernel";
    const BIAS: &str = "unet.input.bias";
    const GAMMA: &str = "unet.norm.gamma";
    const BETA: &str = "unet.norm.beta";

    fn shape() -> UnetInputShape {
        UnetInputShape {
            batch: 1,
            in_c: 2,
            out_c: 3,
            in_h: 3,
            in_w: 3,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
        }
    }

    fn keys() -> UnetInputKeys<'static> {
        UnetInputKeys {
            kernel: KERNEL,
            bias: BIAS,
            gamma: GAMMA,
            beta: BETA,
        }
    }

    fn checkpoint(
        dtype: &str,
        kernel_shape: &str,
        bias_shape: &str,
        payload: &[u8],
    ) -> (std::path::PathBuf, CheckpointMap) {
        let width = if dtype == "F32" || dtype == "I32" {
            4
        } else {
            2
        };
        let kernel_bytes = 3 * 2 * 3 * 3 * width;
        let header = format!(
            concat!(
                "{{\"{k}\":{{\"dtype\":\"{d}\",\"shape\":{ks},\"data_offsets\":[0,{kb}]}},",
                "\"{b}\":{{\"dtype\":\"{d}\",\"shape\":{bs},\"data_offsets\":[{kb},{end}]}},",
                "\"{g}\":{{\"dtype\":\"{d}\",\"shape\":{bs},\"data_offsets\":[{end},{end2}]}},",
                "\"{e}\":{{\"dtype\":\"{d}\",\"shape\":{bs},\"data_offsets\":[{end2},{end3}]}}}}"
            ),
            k = KERNEL,
            b = BIAS,
            g = GAMMA,
            e = BETA,
            d = dtype,
            ks = kernel_shape,
            bs = bias_shape,
            kb = kernel_bytes,
            end = kernel_bytes + 3 * width,
            end2 = kernel_bytes + 6 * width,
            end3 = kernel_bytes + 9 * width,
        );
        let mut bytes = u64::try_from(header.len())
            .expect("header length")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "honkbox-ash-unet-{}-{stamp}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        let map = CheckpointMap::read(&path).expect("map");
        (path, map)
    }

    #[test]
    fn loads_f32_tensors_with_declared_shapes() {
        let kernel: Vec<f32> = (0..54)
            .map(|i| f32::from(u8::try_from(i).expect("0..54")) * 0.01)
            .collect();
        let channels = [0.5_f32, -0.25, 0.75];
        let mut payload = Vec::new();
        for value in kernel
            .iter()
            .chain(channels.iter())
            .chain(channels.iter())
            .chain(channels.iter())
        {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let (path, map) = checkpoint("F32", "[3,2,3,3]", "[3]", &payload);
        let tensors = load_unet_input(&map, &path, &keys(), shape()).expect("load");
        assert_eq!(tensors.kernel, kernel);
        assert_eq!(tensors.bias, channels);
        assert_eq!(tensors.gamma, channels);
        assert_eq!(tensors.beta, channels);
        std::fs::remove_file(path).expect("cleanup");
    }

    // Every test value is an exact multiple of 0.25, so both 16-bit formats
    // encode it losslessly and the f32 reference needs no rounding tolerance.
    fn exact_values() -> (Vec<f32>, Vec<f32>) {
        let kernel = (0..54)
            .map(|i| f32::from(u8::try_from(i % 8).expect("0..8")) * 0.25 - 0.5)
            .collect();
        (kernel, vec![0.5_f32, -0.25, 0.75])
    }

    fn half_bits(value: f32) -> u16 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let sign = u16::try_from(bits >> 31).expect("sign bit");
        let biased = i32::try_from((bits >> 23) & 0xff).expect("f32 exponent") - 127 + 15;
        let exponent = u16::try_from(biased).expect("test exponent");
        (sign << 15) | (exponent << 10) | u16::try_from((bits >> 13) & 0x3ff).expect("mantissa")
    }

    #[test]
    fn decodes_half_precision_tensors() {
        let (kernel, channels) = exact_values();
        let mut payload = Vec::new();
        for value in kernel
            .iter()
            .chain(channels.iter())
            .chain(channels.iter())
            .chain(channels.iter())
        {
            payload.extend_from_slice(&half_bits(*value).to_le_bytes());
        }
        let (path, map) = checkpoint("F16", "[3,2,3,3]", "[3]", &payload);
        let tensors = load_unet_input(&map, &path, &keys(), shape()).expect("load");
        assert_eq!(tensors.kernel, kernel);
        assert_eq!(tensors.bias, channels);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn decodes_bf16_tensors() {
        let (kernel, channels) = exact_values();
        let mut payload = Vec::new();
        for value in kernel
            .iter()
            .chain(channels.iter())
            .chain(channels.iter())
            .chain(channels.iter())
        {
            let bits = u16::try_from(value.to_bits() >> 16).expect("bf16 bits");
            payload.extend_from_slice(&bits.to_le_bytes());
        }
        let (path, map) = checkpoint("BF16", "[3,2,3,3]", "[3]", &payload);
        let tensors = load_unet_input(&map, &path, &keys(), shape()).expect("load");
        assert_eq!(tensors.kernel, kernel);
        assert_eq!(tensors.bias, channels);
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn rejects_missing_keys_wrong_shapes_and_bad_dtypes() {
        let payload = vec![0u8; 54 * 4 + 3 * 4 * 3];
        let (path, map) = checkpoint("F32", "[3,2,3,3]", "[3]", &payload);
        let missing = UnetInputKeys {
            kernel: "absent",
            ..keys()
        };
        assert!(load_unet_input(&map, &path, &missing, shape()).is_err());
        let wrong_shape = UnetInputShape {
            out_c: 4,
            ..shape()
        };
        assert!(load_unet_input(&map, &path, &keys(), wrong_shape).is_err());
        std::fs::remove_file(path).expect("cleanup");

        let (path, map) = checkpoint("I32", "[3,2,3,3]", "[3]", &payload);
        assert!(load_unet_input(&map, &path, &keys(), shape()).is_err());
        std::fs::remove_file(path).expect("cleanup");
    }
}
