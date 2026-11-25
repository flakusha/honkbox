//! Binary-only generation provenance for `PNG`, `JPEG`, and `WebP` outputs.
// size-allow: 940
//!
//! The codec copies container bytes verbatim: it never decodes or re-encodes
//! pixels. The canonical generation record is the determinism bundle record
//! plus its type, server version, and canonical input hash. The accompanying
//! `C2PA`-shaped manifest is deliberately unsigned and timestamp-free.

use serde_json::{Value, json};

use crate::jobs::BundleInputs;
use crate::jobs::determinism::{bundle_record, bundle_record_hash, canonical_json};

/// Sidecar name for the unsigned `C2PA`-shaped manifest.
pub(crate) const MANIFEST_NAME: &str = "c2pa_manifest.json";

const MAX_METADATA_BYTES: usize = 1024 * 1024;
const XMP_NAMESPACE: &str = "https://honkbox.org/ns/provenance/1/";
const RECORD_KEYWORD: &[u8] = b"honkbox_generation_record";
const XMP_KEYWORD: &[u8] = b"XML:com.adobe.xmp";
const XMP_EXTENDED_NAMESPACE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const EXIF_DESCRIPTION: &[u8] = b"Honkbox generation provenance\0";
const EXIF_PREFIX: &[u8] = b"Exif\0\0";
const EXIF_COMMENT_PREFIX: &[u8] = b"UNICODE\0";

type CodecResult<T> = std::result::Result<T, &'static str>;

struct RecordMetadata {
    json: Vec<u8>,
    xmp: Vec<u8>,
}

/// Build the canonical generation record embedded in image metadata.
#[must_use]
pub(crate) fn generation_record(inputs: &BundleInputs) -> Value {
    let mut record = bundle_record(inputs);
    let object = record
        .as_object_mut()
        .expect("bundle_record always returns an object");
    object.insert("record_type".to_owned(), json!("honkbox.generation/1"));
    object.insert(
        "honkbox_version".to_owned(),
        json!(env!("CARGO_PKG_VERSION")),
    );
    object.insert(
        "inputs_sha256".to_owned(),
        json!(bundle_record_hash(inputs)),
    );
    record
}

/// Build a deterministic, unsigned `C2PA`-shaped manifest without a timestamp.
#[must_use]
pub(crate) fn unsigned_manifest(inputs: &BundleInputs) -> Value {
    let record = generation_record(inputs);
    let generator = format!("honkbox/{}", env!("CARGO_PKG_VERSION"));
    json!({
        "manifest_version": 1,
        "claim_generator": generator,
        "claim": {
            "created_assertions": [],
            "gathered_assertions": [],
            "redactions": []
        },
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [{
                        "action": "c2pa.created",
                        "softwareAgent": generator
                    }]
                }
            },
            {
                "label": "honkbox.generation",
                "data": record
            }
        ]
    })
}

/// Inject canonical generation metadata without changing encoded image pixels.
pub(crate) fn embed_record(
    bytes: &[u8],
    format: honkbox_sd_core::OutputFormat,
    record: &Value,
) -> Result<Vec<u8>, String> {
    let metadata = record_metadata(record).map_err(str::to_owned)?;
    match format {
        honkbox_sd_core::OutputFormat::Png => {
            embed_png(bytes, &metadata.json, &metadata.xmp).map_err(str::to_owned)
        }
        honkbox_sd_core::OutputFormat::Jpeg => embed_jpeg(bytes, &metadata).map_err(str::to_owned),
        honkbox_sd_core::OutputFormat::Webp => embed_webp(bytes, &metadata).map_err(str::to_owned),
    }
}

fn record_metadata(record: &Value) -> CodecResult<RecordMetadata> {
    let json = canonical_json(record);
    if json.len() > MAX_METADATA_BYTES {
        return Err("record exceeds 1 MiB limit");
    }

    let mut xmp = String::with_capacity(json.len() + 512);
    xmp.push_str("<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n");
    xmp.push_str("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"Honkbox\">\n");
    xmp.push_str("  <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n");
    xmp.push_str("    <rdf:Description rdf:about=\"\" xmlns:honkbox=\"");
    xmp.push_str(XMP_NAMESPACE);
    xmp.push_str("\" honkbox:generationRecord=\"");
    push_xml_escaped(&mut xmp, &json)?;
    xmp.push_str("\"/>\n  </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end=\"w\"?>\n");
    if xmp.len() > MAX_METADATA_BYTES {
        return Err("XMP packet exceeds 1 MiB limit");
    }

    Ok(RecordMetadata {
        json: json.into_bytes(),
        xmp: xmp.into_bytes(),
    })
}

fn push_xml_escaped(out: &mut String, text: &str) -> CodecResult<()> {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' | '\n' | '\r' => out.push(ch),
            '\u{fffe}' | '\u{ffff}' => return Err("record contains characters invalid in XML"),
            ch if ch < '\u{20}' => return Err("record contains characters invalid in XML"),
            ch => out.push(ch),
        }
    }
    Ok(())
}

fn slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    length: usize,
    error: &'static str,
) -> CodecResult<&'a [u8]> {
    let end = offset.checked_add(length).ok_or(error)?;
    bytes.get(offset..end).ok_or(error)
}

fn read_u16_be(bytes: &[u8], offset: usize, error: &'static str) -> CodecResult<u16> {
    Ok(u16::from_be_bytes(
        slice(bytes, offset, 2, error)?
            .try_into()
            .map_err(|_| error)?,
    ))
}

fn read_u32_be(bytes: &[u8], offset: usize, error: &'static str) -> CodecResult<u32> {
    Ok(u32::from_be_bytes(
        slice(bytes, offset, 4, error)?
            .try_into()
            .map_err(|_| error)?,
    ))
}

fn read_u32_le(bytes: &[u8], offset: usize, error: &'static str) -> CodecResult<u32> {
    Ok(u32::from_le_bytes(
        slice(bytes, offset, 4, error)?
            .try_into()
            .map_err(|_| error)?,
    ))
}

fn embed_png(bytes: &[u8], record: &[u8], xmp: &[u8]) -> CodecResult<Vec<u8>> {
    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    if bytes.get(..8) != Some(SIGNATURE) {
        return Err("PNG: invalid signature");
    }

    let mut offset = 8;
    let mut ihdr_end = None;
    let mut saw_iend = false;
    let mut saw_idat = false;
    while offset < bytes.len() {
        let length = usize::try_from(read_u32_be(bytes, offset, "PNG: truncated chunk length")?)
            .map_err(|_| "PNG: chunk too large")?;
        let kind = slice(bytes, offset + 4, 4, "PNG: truncated chunk type")?;
        if !kind.iter().all(u8::is_ascii_alphabetic) || !kind[2].is_ascii_uppercase() {
            return Err("PNG: invalid chunk type");
        }
        let data_start = offset + 8;
        let data = slice(bytes, data_start, length, "PNG: truncated chunk data")?;
        let crc_offset = data_start + length;
        let stored_crc = read_u32_be(bytes, crc_offset, "PNG: truncated chunk CRC")?;
        if crc32(kind, data) != stored_crc {
            return Err("PNG: chunk CRC mismatch");
        }
        let end = crc_offset + 4;

        if kind == b"IHDR" {
            if offset != 8 || length != 13 || ihdr_end.is_some() {
                return Err("PNG: invalid IHDR");
            }
            let width = read_u32_be(data, 0, "PNG: truncated IHDR")?;
            let height = read_u32_be(data, 4, "PNG: truncated IHDR")?;
            if width == 0 || height == 0 || width > 0x7fff_ffff || height > 0x7fff_ffff {
                return Err("PNG: invalid image dimensions");
            }
            ihdr_end = Some(end);
        } else if kind == b"IDAT" {
            saw_idat = true;
        } else if kind == b"IEND" {
            if length != 0 || end != bytes.len() {
                return Err("PNG: invalid IEND");
            }
            saw_iend = true;
            break;
        }
        offset = end;
    }

    if !saw_iend {
        return Err("PNG: missing IEND");
    }
    if !saw_idat {
        return Err("PNG: missing IDAT");
    }
    let insert_at = ihdr_end.ok_or("PNG: missing IHDR")?;
    let record_chunk = png_chunk(*b"iTXt", &png_text(RECORD_KEYWORD, record));
    let xmp_chunk = png_chunk(*b"iTXt", &png_text(XMP_KEYWORD, xmp));
    let added = record_chunk
        .len()
        .checked_add(xmp_chunk.len())
        .ok_or("PNG: output too large")?;
    let capacity = bytes
        .len()
        .checked_add(added)
        .ok_or("PNG: output too large")?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&bytes[..insert_at]);
    out.extend_from_slice(&record_chunk);
    out.extend_from_slice(&xmp_chunk);
    out.extend_from_slice(&bytes[insert_at..]);
    Ok(out)
}

fn png_text(keyword: &[u8], text: &[u8]) -> Vec<u8> {
    let mut data = Vec::with_capacity(keyword.len() + text.len() + 5);
    data.extend_from_slice(keyword);
    data.extend_from_slice(&[0, 0, 0, 0, 0]);
    data.extend_from_slice(text);
    data
}

fn png_chunk(kind: [u8; 4], data: &[u8]) -> Vec<u8> {
    let length = u32::try_from(data.len()).expect("metadata is bounded below u32::MAX");
    let mut chunk = Vec::with_capacity(data.len() + 12);
    chunk.extend_from_slice(&length.to_be_bytes());
    chunk.extend_from_slice(&kind);
    chunk.extend_from_slice(data);
    chunk.extend_from_slice(&crc32(&kind, data).to_be_bytes());
    chunk
}

static CRC32_TABLE: std::sync::LazyLock<[u32; 256]> = std::sync::LazyLock::new(|| {
    let mut table = [0; 256];
    for (index, entry) in table.iter_mut().enumerate() {
        let mut crc = u32::try_from(index).expect("CRC table index fits u32");
        for _ in 0..8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                (crc >> 1) ^ 0xedb8_8320
            };
        }
        *entry = crc;
    }
    table
});

fn crc32(kind: &[u8], data: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in kind.iter().chain(data) {
        let index = usize::try_from((crc ^ u32::from(*byte)) & 0xff).expect("CRC index fits usize");
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    !crc
}

fn embed_jpeg(bytes: &[u8], metadata: &RecordMetadata) -> CodecResult<Vec<u8>> {
    validate_jpeg(bytes)?;
    let exif = jpeg_app1(&exif_payload(&metadata.json)?)?;
    let xmp = jpeg_app1(&xmp_payload(&metadata.xmp))?;
    let added = exif
        .len()
        .checked_add(xmp.len())
        .ok_or("JPEG: output too large")?;
    let capacity = bytes
        .len()
        .checked_add(added)
        .ok_or("JPEG: output too large")?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&bytes[..2]);
    out.extend_from_slice(&exif);
    out.extend_from_slice(&xmp);
    out.extend_from_slice(&bytes[2..]);
    Ok(out)
}

fn validate_jpeg(bytes: &[u8]) -> CodecResult<()> {
    if bytes.get(..2) != Some(b"\xff\xd8") {
        return Err("JPEG: invalid SOI");
    }
    let mut offset = 2;
    let mut saw_scan = false;
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            return Err("JPEG: expected marker");
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes.get(offset).ok_or("JPEG: truncated marker")?;
        offset += 1;
        match marker {
            0x00 => return Err("JPEG: invalid marker"),
            0xd8 => return Err("JPEG: unexpected SOI"),
            0xd9 if saw_scan => return Ok(()),
            0xd9 => return Err("JPEG: missing scan"),
            0xd0..=0xd7 => return Err("JPEG: restart marker outside scan"),
            0x01 => continue,
            _ => {}
        }

        let length = usize::from(read_u16_be(bytes, offset, "JPEG: truncated segment")?);
        if length < 2 {
            return Err("JPEG: invalid segment length");
        }
        let end = offset
            .checked_add(length)
            .ok_or("JPEG: segment too large")?;
        if end > bytes.len() {
            return Err("JPEG: truncated segment");
        }
        if marker == 0xda {
            saw_scan = true;
            offset = skip_jpeg_entropy(bytes, end)?;
        } else {
            offset = end;
        }
    }
    Err("JPEG: missing EOI")
}

fn skip_jpeg_entropy(bytes: &[u8], mut offset: usize) -> CodecResult<usize> {
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            offset += 1;
            continue;
        }
        let marker_start = offset;
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes.get(offset).ok_or("JPEG: truncated scan")?;
        match marker {
            0x00 | 0xd0..=0xd7 => offset += 1,
            _ => return Ok(marker_start),
        }
    }
    Err("JPEG: missing EOI")
}

fn jpeg_app1(payload: &[u8]) -> CodecResult<Vec<u8>> {
    let length = payload
        .len()
        .checked_add(2)
        .ok_or("JPEG: APP1 segment too large")?;
    let length = u16::try_from(length).map_err(|_| "JPEG: APP1 segment exceeds 65533 bytes")?;
    let mut segment = Vec::with_capacity(usize::from(length) + 2);
    segment.extend_from_slice(&[0xff, 0xe1]);
    segment.extend_from_slice(&length.to_be_bytes());
    segment.extend_from_slice(payload);
    Ok(segment)
}

fn xmp_payload(xmp: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(XMP_EXTENDED_NAMESPACE.len() + xmp.len());
    payload.extend_from_slice(XMP_EXTENDED_NAMESPACE);
    payload.extend_from_slice(xmp);
    payload
}

fn exif_payload(json: &[u8]) -> CodecResult<Vec<u8>> {
    let tiff = exif_tiff(json)?;
    let mut payload = Vec::with_capacity(EXIF_PREFIX.len() + tiff.len());
    payload.extend_from_slice(EXIF_PREFIX);
    payload.extend_from_slice(&tiff);
    Ok(payload)
}

fn exif_tiff(json: &[u8]) -> CodecResult<Vec<u8>> {
    let comment_length = EXIF_COMMENT_PREFIX
        .len()
        .checked_add(json.len())
        .ok_or("EXIF: metadata too large")?;
    let description_offset: usize = 38;
    let comment_offset = description_offset
        .checked_add(EXIF_DESCRIPTION.len())
        .ok_or("EXIF: metadata too large")?;
    let end = comment_offset
        .checked_add(comment_length)
        .ok_or("EXIF: metadata too large")?;
    u32::try_from(end).map_err(|_| "EXIF: metadata too large")?;
    let description_count =
        u32::try_from(EXIF_DESCRIPTION.len()).map_err(|_| "EXIF: metadata too large")?;
    let comment_count = u32::try_from(comment_length).map_err(|_| "EXIF: metadata too large")?;
    let description_offset =
        u32::try_from(description_offset).map_err(|_| "EXIF: metadata too large")?;
    let comment_offset = u32::try_from(comment_offset).map_err(|_| "EXIF: metadata too large")?;

    let mut tiff = Vec::with_capacity(end);
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42_u16.to_le_bytes());
    tiff.extend_from_slice(&8_u32.to_le_bytes());
    tiff.extend_from_slice(&2_u16.to_le_bytes());
    push_ifd_entry(&mut tiff, 0x010e, 2, description_count, description_offset);
    push_ifd_entry(&mut tiff, 0x9286, 7, comment_count, comment_offset);
    tiff.extend_from_slice(&0_u32.to_le_bytes());
    tiff.extend_from_slice(EXIF_DESCRIPTION);
    tiff.extend_from_slice(EXIF_COMMENT_PREFIX);
    tiff.extend_from_slice(json);
    Ok(tiff)
}

fn push_ifd_entry(out: &mut Vec<u8>, tag: u16, field_type: u16, count: u32, value: u32) {
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&field_type.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn embed_webp(bytes: &[u8], metadata: &RecordMetadata) -> CodecResult<Vec<u8>> {
    if bytes.len() < 12 {
        return Err("WebP: truncated RIFF header");
    }
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WEBP") {
        return Err("WebP: invalid RIFF/WEBP signature");
    }
    let riff_size = usize::try_from(read_u32_le(bytes, 4, "WebP: truncated RIFF header")?)
        .map_err(|_| "WebP: RIFF size mismatch")?;
    if riff_size != bytes.len() - 8 {
        return Err("WebP: RIFF size mismatch");
    }

    let mut offset = 12;
    let mut vp8l = None;
    while offset < bytes.len() {
        let kind = slice(bytes, offset, 4, "WebP: truncated chunk header")?;
        let length = usize::try_from(read_u32_le(
            bytes,
            offset + 4,
            "WebP: truncated chunk header",
        )?)
        .map_err(|_| "WebP: chunk too large")?;
        let data_start = offset + 8;
        let data = slice(bytes, data_start, length, "WebP: truncated chunk data")?;
        let padded_end = data_start
            .checked_add(length)
            .and_then(|end| end.checked_add(length & 1))
            .ok_or("WebP: chunk too large")?;
        if padded_end > bytes.len() {
            return Err("WebP: truncated chunk padding");
        }
        match kind {
            b"VP8X" | b"VP8 " | b"ANIM" | b"ANMF" => {
                return Err("WebP: expected a simple VP8L container");
            }
            b"VP8L" => {
                if vp8l.is_some() {
                    return Err("WebP: multiple VP8L chunks");
                }
                vp8l = Some(data);
            }
            _ => {}
        }
        offset = padded_end;
    }
    let vp8l = vp8l.ok_or("WebP: missing VP8L chunk")?;
    let (width, height, alpha) = parse_vp8l(vp8l)?;

    let mut vp8x_payload = Vec::with_capacity(10);
    vp8x_payload.push(0x0c | (u8::from(alpha) << 4));
    vp8x_payload.extend_from_slice(&[0, 0, 0]);
    push_u24_le(&mut vp8x_payload, width - 1);
    push_u24_le(&mut vp8x_payload, height - 1);
    let vp8x_chunk = webp_chunk(*b"VP8X", &vp8x_payload);
    let exif = webp_chunk(*b"EXIF", &exif_payload(&metadata.json)?);
    let xmp = webp_chunk(*b"XMP ", &metadata.xmp);
    let added = vp8x_chunk
        .len()
        .checked_add(exif.len())
        .and_then(|length| length.checked_add(xmp.len()))
        .ok_or("WebP: output too large")?;
    let capacity = bytes
        .len()
        .checked_add(added)
        .ok_or("WebP: output too large")?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&vp8x_chunk);
    out.extend_from_slice(&bytes[12..]);
    out.extend_from_slice(&exif);
    out.extend_from_slice(&xmp);
    let size = u32::try_from(out.len() - 8).map_err(|_| "WebP: output too large")?;
    out[4..8].copy_from_slice(&size.to_le_bytes());
    Ok(out)
}

fn parse_vp8l(data: &[u8]) -> CodecResult<(u32, u32, bool)> {
    if data.first() != Some(&0x2f) {
        return Err("WebP: invalid VP8L signature");
    }
    let bits = read_u32_le(data, 1, "WebP: truncated VP8L header")?;
    if (bits >> 29) & 7 != 0 {
        return Err("WebP: unsupported VP8L version");
    }
    let width = (bits & 0x3fff) + 1;
    let height = ((bits >> 14) & 0x3fff) + 1;
    let alpha = (bits >> 28) & 1 != 0;
    Ok((width, height, alpha))
}

fn push_u24_le(out: &mut Vec<u8>, value: u32) {
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..3]);
}

fn webp_chunk(kind: [u8; 4], data: &[u8]) -> Vec<u8> {
    let length = u32::try_from(data.len()).expect("metadata is bounded below u32::MAX");
    let mut chunk = Vec::with_capacity(data.len() + 8 + (data.len() & 1));
    chunk.extend_from_slice(&kind);
    chunk.extend_from_slice(&length.to_le_bytes());
    chunk.extend_from_slice(data);
    if data.len() & 1 != 0 {
        chunk.push(0);
    }
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::{BundleAnnotation, BundleInputs, HiresRecord, LoraRecord};
    use honkbox_sd_core::OutputFormat;
    use image::{ExtendedColorType, ImageEncoder, ImageFormat, RgbaImage};
    use quick_xml::Reader;
    use quick_xml::events::Event;
    use serde_json::Map;

    fn inputs() -> BundleInputs {
        BundleInputs {
            model_slug: "test:checkpoint".to_owned(),
            prompt: "a tiny test image".to_owned(),
            seed: 42,
            steps: 4,
            width: 3,
            height: 2,
            format: OutputFormat::Png,
            backend_id: "test",
            model_sha256: None,
            engine_version: None,
            sampler: "euler".to_owned(),
            scheduler: "normal".to_owned(),
            cfg_scale: 7.0,
            negative_prompt: None,
            loras: vec![LoraRecord {
                slug: "test:adapter".to_owned(),
                strength: 0.5,
                base_model: None,
            }],
            strength: None,
            mask_mode: None,
            mask_sha256: None,
            checkpoint_base_model: None,
            recipe: None,
            hires: Some(HiresRecord {
                upscaler: "latent",
                target: "6x4".to_owned(),
                scale: 2.0,
                denoising_strength: 0.4,
            }),
            annotations: vec![BundleAnnotation {
                name: "test".to_owned(),
                value: json!({"enabled": true}),
            }],
        }
    }

    fn pixels() -> RgbaImage {
        RgbaImage::from_raw(
            3,
            2,
            vec![
                10, 20, 30, 255, 40, 50, 60, 128, 70, 80, 90, 0, 100, 110, 120, 64, 130, 140, 150,
                200, 160, 170, 180, 255,
            ],
        )
        .expect("tiny RGBA buffer")
    }

    fn png_source() -> Vec<u8> {
        let image = pixels();
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                ExtendedColorType::Rgba8,
            )
            .expect("encode PNG");
        bytes
    }

    fn jpeg_source() -> Vec<u8> {
        let image = image::DynamicImage::ImageRgba8(pixels()).to_rgb8();
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90)
            .encode(
                image.as_raw(),
                image.width(),
                image.height(),
                ExtendedColorType::Rgb8,
            )
            .expect("encode JPEG");
        bytes
    }

    fn webp_source() -> Vec<u8> {
        let image = pixels();
        let mut bytes = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut bytes)
            .encode(
                image.as_raw(),
                image.width(),
                image.height(),
                ExtendedColorType::Rgba8,
            )
            .expect("encode WebP");
        bytes
    }

    fn decode(bytes: &[u8], format: ImageFormat) -> RgbaImage {
        image::load_from_memory_with_format(bytes, format)
            .expect("decode image")
            .to_rgba8()
    }

    fn canonical_record() -> String {
        canonical_json(&generation_record(&inputs()))
    }

    fn png_text(bytes: &[u8], keyword: &[u8]) -> Option<Vec<u8>> {
        let mut offset = 8;
        while offset + 12 <= bytes.len() {
            let length = usize::try_from(read_u32_be(bytes, offset, "truncated").expect("length"))
                .expect("usize");
            let kind = &bytes[offset + 4..offset + 8];
            let data = &bytes[offset + 8..offset + 8 + length];
            if kind == b"iTXt" {
                let keyword_end = data.iter().position(|byte| *byte == 0)?;
                if &data[..keyword_end] == keyword {
                    let language_start = keyword_end + 3;
                    let language_end = language_start
                        + data[language_start..].iter().position(|byte| *byte == 0)?;
                    let translated_start = language_end + 1;
                    let translated_end = translated_start
                        + data[translated_start..]
                            .iter()
                            .position(|byte| *byte == 0)?;
                    return Some(data[translated_end + 1..].to_vec());
                }
            }
            offset += length + 12;
        }
        None
    }

    fn jpeg_app1_segments(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut segments = Vec::new();
        let mut offset = 2;
        while bytes.get(offset) == Some(&0xff) {
            let marker = bytes[offset + 1];
            if marker == 0xd9 || marker == 0xda {
                break;
            }
            let length = usize::from(u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]));
            if marker == 0xe1 {
                segments.push(bytes[offset + 4..offset + 2 + length].to_vec());
            }
            offset += length + 2;
        }
        segments
    }

    fn webp_chunks(bytes: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
        let mut chunks = Vec::new();
        let mut offset = 12;
        while offset + 8 <= bytes.len() {
            let kind: [u8; 4] = bytes[offset..offset + 4].try_into().expect("chunk kind");
            let length = usize::try_from(u32::from_le_bytes(
                bytes[offset + 4..offset + 8]
                    .try_into()
                    .expect("chunk length"),
            ))
            .expect("chunk length fits usize");
            chunks.push((kind, bytes[offset + 8..offset + 8 + length].to_vec()));
            offset += 8 + length + (length & 1);
        }
        chunks
    }

    fn user_comment(tiff: &[u8]) -> Vec<u8> {
        assert_eq!(&tiff[..2], b"II");
        let ifd_offset = usize::try_from(u32::from_le_bytes(
            tiff[4..8].try_into().expect("IFD offset"),
        ))
        .expect("IFD offset fits usize");
        let count = usize::from(u16::from_le_bytes(
            tiff[ifd_offset..ifd_offset + 2]
                .try_into()
                .expect("IFD count"),
        ));
        for index in 0..count {
            let entry = ifd_offset + 2 + index * 12;
            let tag = u16::from_le_bytes(tiff[entry..entry + 2].try_into().expect("tag"));
            if tag == 0x9286 {
                let length = usize::try_from(u32::from_le_bytes(
                    tiff[entry + 4..entry + 8]
                        .try_into()
                        .expect("comment count"),
                ))
                .expect("comment count fits usize");
                let offset = usize::try_from(u32::from_le_bytes(
                    tiff[entry + 8..entry + 12]
                        .try_into()
                        .expect("comment offset"),
                ))
                .expect("comment offset fits usize");
                return tiff[offset..offset + length].to_vec();
            }
        }
        panic!("missing UserComment");
    }

    fn assert_exif_record(exif: &[u8], canonical: &str) {
        let tiff = exif.strip_prefix(EXIF_PREFIX).expect("EXIF prefix");
        let comment = user_comment(tiff);
        assert!(comment.starts_with(EXIF_COMMENT_PREFIX));
        assert_eq!(&comment[EXIF_COMMENT_PREFIX.len()..], canonical.as_bytes());
    }

    fn assert_xmp_record(xmp: &[u8], canonical: &str) {
        let packet = std::str::from_utf8(xmp).expect("UTF-8 XMP");
        let mut reader = Reader::from_str(packet);
        loop {
            match reader.read_event().expect("valid XMP") {
                Event::Start(element) | Event::Empty(element) => {
                    for attribute in element.attributes() {
                        let attribute = attribute.expect("valid XMP attribute");
                        if attribute.key.local_name().as_ref() == b"generationRecord" {
                            assert_eq!(
                                attribute
                                    .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                                    .expect("XML value")
                                    .as_ref(),
                                canonical
                            );
                            return;
                        }
                    }
                }
                Event::Eof => panic!("missing generation record in XMP"),
                _ => {}
            }
        }
    }

    #[test]
    fn generation_record_preserves_unknown_hash_and_manifest_is_unsigned() {
        let record = generation_record(&inputs());
        assert_eq!(record["record_type"], "honkbox.generation/1");
        assert_eq!(record["honkbox_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(record["inputs_sha256"], bundle_record_hash(&inputs()));
        assert!(record["model_sha256"].is_null());
        assert!(record["engine_version"].is_null());

        let mut known = inputs();
        known.model_sha256 = Some("ab".repeat(32));
        known.engine_version = Some("test-engine");
        let known_record = generation_record(&known);
        assert_eq!(known_record["model_sha256"], "ab".repeat(32));
        assert_eq!(known_record["engine_version"], "test-engine");

        let manifest = unsigned_manifest(&inputs());
        assert_eq!(manifest["manifest_version"], 1);
        assert_eq!(manifest["assertions"][1]["label"], "honkbox.generation");
        assert_eq!(manifest["assertions"][1]["data"], record);
        assert_eq!(
            manifest["assertions"][0]["data"]["actions"][0]["action"],
            "c2pa.created"
        );
        let serialized = canonical_json(&manifest);
        for forbidden in ["timestamp", "created_at", "signature", "credential"] {
            assert!(!serialized.contains(forbidden));
        }
        assert_eq!(canonical_json(&unsigned_manifest(&inputs())), serialized);
    }

    #[test]
    fn png_round_trip_preserves_pixels_and_embeds_canonical_record() {
        let source = png_source();
        let embedded = embed_record(&source, OutputFormat::Png, &generation_record(&inputs()))
            .expect("embed PNG");
        assert_eq!(
            decode(&source, ImageFormat::Png),
            decode(&embedded, ImageFormat::Png)
        );
        let canonical = canonical_record();
        assert_eq!(
            png_text(&embedded, RECORD_KEYWORD).expect("record iTXt"),
            canonical.as_bytes()
        );
        assert_xmp_record(
            &png_text(&embedded, XMP_KEYWORD).expect("XMP iTXt"),
            &canonical,
        );
        let mut offset = 8;
        let mut saw_idat = false;
        let mut metadata_chunks = 0usize;
        while offset < embedded.len() {
            let kind = &embedded[offset + 4..offset + 8];
            let length = read_u32_be(&embedded, offset, "PNG chunk").expect("PNG chunk") as usize;
            if kind == b"IDAT" {
                saw_idat = true;
            }
            if kind == b"iTXt" {
                assert!(!saw_idat, "metadata must precede IDAT");
                metadata_chunks += 1;
            }
            offset += 12 + length;
        }
        assert!(saw_idat && metadata_chunks >= 2);
    }

    #[test]
    fn jpeg_round_trip_preserves_pixels_and_embeds_canonical_record() {
        let source = jpeg_source();
        let embedded = embed_record(&source, OutputFormat::Jpeg, &generation_record(&inputs()))
            .expect("embed JPEG");
        assert_eq!(
            decode(&source, ImageFormat::Jpeg),
            decode(&embedded, ImageFormat::Jpeg)
        );
        let canonical = canonical_record();
        let segments = jpeg_app1_segments(&embedded);
        assert_eq!(segments.len(), 2);
        assert_exif_record(&segments[0], &canonical);
        assert_xmp_record(
            segments[1]
                .strip_prefix(XMP_EXTENDED_NAMESPACE)
                .expect("XMP namespace"),
            &canonical,
        );
    }

    #[test]
    fn webp_round_trip_preserves_pixels_alpha_and_canonical_record() {
        let source = webp_source();
        let embedded = embed_record(&source, OutputFormat::Webp, &generation_record(&inputs()))
            .expect("embed WebP");
        assert_eq!(
            decode(&source, ImageFormat::WebP),
            decode(&embedded, ImageFormat::WebP)
        );
        let canonical = canonical_record();
        let source_chunks = webp_chunks(&source);
        let chunks = webp_chunks(&embedded);
        assert_eq!(&chunks[0].0, b"VP8X");
        assert_eq!(chunks[0].1[0] & 0x10, 0x10);
        assert_eq!(&chunks[0].1[4..7], &[2, 0, 0]);
        assert_eq!(&chunks[0].1[7..10], &[1, 0, 0]);
        assert_eq!(&chunks[1].0, b"VP8L");
        assert_eq!(chunks[1].1, source_chunks[0].1);
        assert_eq!(&chunks[2].0, b"EXIF");
        assert_exif_record(&chunks[2].1, &canonical);
        assert_eq!(&chunks[3].0, b"XMP ");
        assert_xmp_record(&chunks[3].1, &canonical);
    }

    #[test]
    fn repeated_embedding_is_byte_identical() {
        let record = generation_record(&inputs());
        for (source, format) in [
            (png_source(), OutputFormat::Png),
            (jpeg_source(), OutputFormat::Jpeg),
            (webp_source(), OutputFormat::Webp),
        ] {
            let first = embed_record(&source, format, &record).expect("first embed");
            let second = embed_record(&source, format, &record).expect("second embed");
            assert_eq!(first, second);
        }
    }

    #[test]
    fn malformed_png_fails() {
        let record = generation_record(&inputs());
        let mut bad_crc = png_source();
        let last = bad_crc.len() - 5;
        bad_crc[last] ^= 1;
        let mut no_idat = png_source()[..33].to_vec();
        no_idat.extend(png_chunk(*b"IEND", &[]));
        for bytes in [
            b"not a png".to_vec(),
            png_source()[..20].to_vec(),
            bad_crc,
            no_idat,
        ] {
            let error = embed_record(&bytes, OutputFormat::Png, &record).expect_err("bad PNG");
            assert!(error.starts_with("PNG:"), "{error}");
        }
    }

    #[test]
    fn malformed_jpeg_fails() {
        let record = generation_record(&inputs());
        let source = jpeg_source();
        for bytes in [
            b"not a jpeg".to_vec(),
            vec![0xff, 0xd8, 0xff, 0xe0, 0, 16],
            source[..source.len() - 2].to_vec(),
        ] {
            let error = embed_record(&bytes, OutputFormat::Jpeg, &record).expect_err("bad JPEG");
            assert!(error.starts_with("JPEG:"), "{error}");
        }
    }

    #[test]
    fn malformed_webp_fails() {
        let record = generation_record(&inputs());
        let mut bad_size = webp_source();
        bad_size[4] ^= 1;
        for bytes in [
            b"not a webp".to_vec(),
            webp_source()[..20].to_vec(),
            bad_size,
        ] {
            let error = embed_record(&bytes, OutputFormat::Webp, &record).expect_err("bad WebP");
            assert!(error.starts_with("WebP:"), "{error}");
        }
    }

    #[test]
    fn metadata_limits_are_enforced() {
        let mut oversized = Map::new();
        oversized.insert(
            "payload".to_owned(),
            Value::String("x".repeat(MAX_METADATA_BYTES + 1)),
        );
        assert_eq!(
            embed_record(&png_source(), OutputFormat::Png, &Value::Object(oversized))
                .expect_err("oversized record"),
            "record exceeds 1 MiB limit"
        );

        let expanded_xmp = json!({"payload": "<".repeat(MAX_METADATA_BYTES / 4)});
        assert_eq!(
            embed_record(&png_source(), OutputFormat::Png, &expanded_xmp)
                .expect_err("oversized XMP"),
            "XMP packet exceeds 1 MiB limit"
        );

        let large_jpeg_record = json!({"payload": "x".repeat(100_000)});
        assert_eq!(
            embed_record(&jpeg_source(), OutputFormat::Jpeg, &large_jpeg_record)
                .expect_err("oversized APP1"),
            "JPEG: APP1 segment exceeds 65533 bytes"
        );
    }

    #[test]
    fn crc32_matches_known_vector() {
        assert_eq!(crc32(b"123456789", &[]), 0xcbf4_3926);
    }
}
