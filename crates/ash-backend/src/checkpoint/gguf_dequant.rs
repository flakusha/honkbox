//! Reference CPU decode for GGUF scalar and Q4_K tensors.
//!
//! Provenance: Q4_K layout and scale/min unpacking translated from pinned
//! `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/ggml/src/ggml-quants.c`
//! (`get_scale_min_k4`, `dequantize_row_q4_K`), llama-cpp-sys-4 0.11.0 crate
//! SHA-256 `814738cddce0c910aa0f5192fd9d3e2ff7e447453e66326dd2f5df7f1375aea3`.
//! Modified to checked Rust slices, shared IEEE conversion and named errors.

use super::{GgufTensor, bfloat16_bits_to_f32, half_bits_to_f32, invalid};
use std::io;

const Q4_K_VALUES: usize = 256;
const Q4_K_BYTES: usize = 144;

/// Decode one validated GGUF tensor payload to flat f32 in file `ne` order.
///
/// Shape permutation belongs to the consumer; this function never infers a
/// model family or rewrites dimensions. Supported ggml types are F32 (0),
/// F16 (1), Q4_K (12) and BF16 (30).
pub fn decode_gguf_tensor(bytes: &[u8], tensor: &GgufTensor) -> io::Result<Vec<f32>> {
    let elements = checked_elements(tensor)?;
    if bytes.len() != tensor.byte_len {
        return Err(invalid(format!(
            "GGUF tensor '{}': read {} bytes, expected {}",
            tensor.name,
            bytes.len(),
            tensor.byte_len
        )));
    }
    let (values_per_block, bytes_per_block) = match tensor.dtype {
        0 => (1, 4),
        1 | 30 => (1, 2),
        12 => (Q4_K_VALUES, Q4_K_BYTES),
        other => {
            return Err(invalid(format!(
                "GGUF tensor '{}': unsupported dtype {other}",
                tensor.name
            )));
        }
    };
    if elements % values_per_block != 0 {
        return Err(invalid(format!(
            "GGUF tensor '{}': element count {elements} is not a multiple of {values_per_block}",
            tensor.name
        )));
    }
    let blocks = elements / values_per_block;
    let expected = blocks.checked_mul(bytes_per_block).ok_or_else(|| {
        invalid(format!(
            "GGUF tensor '{}': byte length overflow",
            tensor.name
        ))
    })?;
    if bytes.len() != expected {
        return Err(invalid(format!(
            "GGUF tensor '{}': payload has {} bytes, expected {expected}",
            tensor.name,
            bytes.len()
        )));
    }
    if tensor.dtype == 12 {
        return decode_q4_k(bytes, elements, &tensor.name);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(elements)
        .map_err(|_| invalid(format!("GGUF tensor '{}': allocation failed", tensor.name)))?;
    match tensor.dtype {
        0 => out.extend(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        ),
        1 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|chunk| half_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))),
        ),
        30 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|chunk| bfloat16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))),
        ),
        _ => unreachable!("dtype validated above"),
    }
    Ok(out)
}

fn checked_elements(tensor: &GgufTensor) -> io::Result<usize> {
    if tensor.shape.is_empty() {
        return Err(invalid(format!(
            "GGUF tensor '{}': empty shape",
            tensor.name
        )));
    }
    tensor.shape.iter().try_fold(1usize, |elements, &dim| {
        if dim == 0 {
            return Err(invalid(format!(
                "GGUF tensor '{}': zero dimension",
                tensor.name
            )));
        }
        elements.checked_mul(dim).ok_or_else(|| {
            invalid(format!(
                "GGUF tensor '{}': element count overflow",
                tensor.name
            ))
        })
    })
}

fn decode_q4_k(bytes: &[u8], elements: usize, name: &str) -> io::Result<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(elements)
        .map_err(|_| invalid(format!("GGUF tensor '{name}': allocation failed")))?;
    for (block_index, block) in bytes.chunks_exact(Q4_K_BYTES).enumerate() {
        let d = half_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = half_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
        if !d.is_finite() || !dmin.is_finite() {
            return Err(invalid(format!(
                "GGUF tensor '{name}': non-finite Q4_K scale in block {block_index}"
            )));
        }
        let scales = &block[4..16];
        for half in 0..4 {
            let quant = &block[16 + half * 32..48 + half * 32];
            let (scale, min) = scale_min(scales, half * 2);
            let scale = d * f32::from(scale);
            let min = dmin * f32::from(min);
            out.extend(
                quant
                    .iter()
                    .map(|byte| scale * f32::from(byte & 0x0f) - min),
            );
            let (scale, min) = scale_min(scales, half * 2 + 1);
            let scale = d * f32::from(scale);
            let min = dmin * f32::from(min);
            out.extend(quant.iter().map(|byte| scale * f32::from(byte >> 4) - min));
        }
    }
    Ok(out)
}

fn scale_min(scales: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        (
            (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(dtype: u32, shape: &[usize], byte_len: usize) -> GgufTensor {
        GgufTensor {
            name: "w".into(),
            dtype,
            shape: shape.to_vec(),
            offset: 0,
            byte_len,
        }
    }

    fn block(d: u16, dmin: u16, scales: [u8; 12], quant: u8) -> [u8; Q4_K_BYTES] {
        let mut bytes = [0; Q4_K_BYTES];
        bytes[..2].copy_from_slice(&d.to_le_bytes());
        bytes[2..4].copy_from_slice(&dmin.to_le_bytes());
        bytes[4..16].copy_from_slice(&scales);
        bytes[16..].fill(quant);
        bytes
    }

    #[test]
    fn gguf_dequant_q4_k_scale_boundaries_and_nibble_order() {
        let scales = [
            0x40, 0x80, 0xc0, 0x41, 0x80, 0xc0, 0x40, 0x80, 0xa7, 0xb6, 0xc5, 0xd4,
        ];
        let bytes = block(0x3c00, 0x3c00, scales, 0x11);
        assert_eq!(half_bits_to_f32(0x3c00), 1.0);
        assert_eq!(scale_min(&scales, 4), (23, 42));
        let values = decode_gguf_tensor(&bytes, &tensor(12, &[256], bytes.len())).expect("decode");
        let expected = [0.0, 0.0, 0.0, 1.0, -19.0, -21.0, 25.0, -25.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 32))
            .collect::<Vec<_>>();
        assert_eq!(values, expected);

        let unit_scales = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
        let bytes = block(0x3c00, 0, unit_scales, 0x21);
        let values =
            decode_gguf_tensor(&bytes, &tensor(12, &[8, 32], bytes.len())).expect("decode");
        let expected = [1.0, 2.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 32))
            .cycle()
            .take(256)
            .collect::<Vec<_>>();
        assert_eq!(values, expected);
    }

    #[test]
    fn gguf_dequant_q4_k_signed_scales_and_multiple_blocks() {
        let scales = [2, 2, 2, 2, 3, 3, 3, 3, 0x32, 0x32, 0x32, 0x32];
        let bytes = block(0xb800, 0x3400, scales, 0xf0);
        let values = decode_gguf_tensor(&bytes, &tensor(12, &[256], bytes.len())).expect("decode");
        let expected = [-0.75, -15.75]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 32))
            .cycle()
            .take(256)
            .collect::<Vec<_>>();
        assert_eq!(values, expected);

        let ones = block(0x3c00, 0, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], 0xff);
        let minus_ones = block(
            0,
            0x3c00,
            [0, 0, 0, 0, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10],
            0,
        );
        let mut bytes = Vec::from(ones);
        bytes.extend_from_slice(&minus_ones);
        let values = decode_gguf_tensor(&bytes, &tensor(12, &[512], bytes.len())).expect("decode");
        assert!(values[..256].iter().all(|&value| value == 15.0));
        assert!(values[256..].iter().all(|&value| value == -1.0));
    }

    #[test]
    fn gguf_dequant_scalars_preserve_ieee_bits() {
        let f16 = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e00,
        ];
        let bytes = f16
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let values =
            decode_gguf_tensor(&bytes, &tensor(1, &[f16.len()], bytes.len())).expect("f16");
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                0x0000_0000,
                0x8000_0000,
                0x3380_0000,
                0x387f_c000,
                0x3880_0000,
                0x3f80_0000,
                0x477f_e000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
            ]
        );

        let bf16 = [
            0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x3f80, 0x7f7f, 0x7f80, 0xff80, 0x7fc0,
        ];
        let bytes = bf16
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let values =
            decode_gguf_tensor(&bytes, &tensor(30, &[bf16.len()], bytes.len())).expect("bf16");
        for (value, bits) in values.iter().zip(bf16) {
            assert_eq!(value.to_bits(), u32::from(bits) << 16);
        }

        let f32 = [1.0_f32, -0.0, f32::INFINITY, f32::NAN];
        let bytes = f32
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let values =
            decode_gguf_tensor(&bytes, &tensor(0, &[f32.len()], bytes.len())).expect("f32");
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            f32.into_iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn gguf_dequant_rejects_malformed_metadata_and_payloads() {
        let valid = block(0x3c00, 0, [0; 12], 0);
        let cases = [
            (tensor(12, &[], 0), Vec::new(), "empty shape"),
            (tensor(12, &[0], 0), Vec::new(), "zero dimension"),
            (tensor(12, &[1], 0), Vec::new(), "not a multiple of 256"),
            (
                tensor(12, &[usize::MAX, 2], 0),
                Vec::new(),
                "element count overflow",
            ),
            (
                tensor(13, &[256], 176),
                vec![0; 176],
                "unsupported dtype 13",
            ),
            (
                tensor(12, &[257], 288),
                vec![0; 288],
                "not a multiple of 256",
            ),
            (
                tensor(12, &[256], 144),
                valid[..143].to_vec(),
                "read 143 bytes",
            ),
            (
                tensor(12, &[256], 145),
                [valid.as_slice(), &[0]].concat(),
                "payload has 145 bytes",
            ),
            (
                tensor(12, &[256], 144),
                block(0x7c00, 0, [0; 12], 0).to_vec(),
                "non-finite Q4_K scale",
            ),
            (
                tensor(12, &[256], 144),
                block(0, 0xfc00, [0; 12], 0).to_vec(),
                "non-finite Q4_K scale",
            ),
        ];
        for (meta, bytes, needle) in cases {
            let error = decode_gguf_tensor(&bytes, &meta).expect_err(needle);
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains(needle), "{error}");
        }
    }

    #[test]
    fn gguf_dequant_values_feed_cpu_dot_reference() {
        let bytes = block(0x3c00, 0, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], 0x21);
        let weights = decode_gguf_tensor(&bytes, &tensor(12, &[256], bytes.len())).expect("decode");
        let rhs = (0..256)
            .map(|index| f32::from(u8::try_from(index % 7).expect("0..7")) * 0.25)
            .collect::<Vec<_>>();
        let actual = weights
            .iter()
            .zip(&rhs)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let expected = (0..256)
            .map(|index| {
                let quant = if (index / 32) % 2 == 0 { 1.0 } else { 2.0 };
                quant * f32::from(u8::try_from(index % 7).expect("0..7")) * 0.25
            })
            .sum::<f32>();
        assert_eq!(actual, expected);
    }
}
