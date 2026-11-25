//! EXIF-orientation regression tests for the image-ops input decode path.
//!
//! Fixtures are built in-tree: an asymmetric JPEG (3×2, one distinct color
//! per pixel) is encoded with the `image` crate, then an APP1 EXIF segment
//! carrying a single IFD0 orientation entry is spliced right after SOI.
//! Every expectation is hand-written from the EXIF transform table — the
//! tests never derive expectations by calling `apply_orientation` on a
//! reference image.

use image::{GenericImageView, ImageEncoder};

use super::image_ops_exec::{ImageOpWorkError, decode_flat_raster_upright};

/// `ImageOpWorkError` has no `Debug` (hand-rolled errors): decode through a
/// Display-panicking shim so tests can use `expect`-style assertions.
fn decode_upright(raw: &[u8]) -> image::DynamicImage {
    match decode_flat_raster_upright(raw) {
        Ok(decoded) => decoded,
        Err(ImageOpWorkError::Bad { message, code }) => {
            panic!("decode failed [{code}]: {message}")
        }
        Err(_) => panic!("decode failed: unexpected error variant"),
    }
}

/// Source pixel at column `x`, row `y`: a red ramp along the top row, a
/// dimmer bottom row. Asymmetric in both axes, so a 90/270 rotation is
/// detectable from dimensions alone and a flip from pixels.
fn source_pixel(x: u32, y: u32) -> image::Rgba<u8> {
    image::Rgba([30 + (x * 40) as u8, if y == 0 { 200 } else { 20 }, 90, 255])
}

/// Little-endian TIFF EXIF payload carrying one IFD0 entry: tag `0x0112`,
/// type SHORT, count 1, value `orientation`.
fn exif_payload(orientation: u16) -> Vec<u8> {
    let mut exif = Vec::new();
    exif.extend_from_slice(b"Exif\0\0"); // APP1 payload prefix (JPEG only)
    exif.extend_from_slice(b"II");
    exif.extend_from_slice(&42u16.to_le_bytes());
    exif.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    exif.extend_from_slice(&1u16.to_le_bytes()); // one IFD entry
    exif.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation tag
    exif.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
    exif.extend_from_slice(&1u32.to_le_bytes()); // count
    exif.extend_from_slice(&orientation.to_le_bytes()); // value
    exif.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    exif
}

/// Encode a 3×2 JPEG of distinct per-pixel colors with the in-tree encoder.
fn encode_source_jpeg() -> Vec<u8> {
    let mut buffer = image::RgbaImage::new(3, 2);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = source_pixel(x, y);
    }
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(buffer)
        .write_to(&mut out, image::ImageFormat::Jpeg)
        .expect("encode fixture JPEG");
    out.into_inner()
}

/// Splice an APP1 segment (`Exif\0\0` + TIFF payload) carrying `orientation`
/// right after the JPEG SOI marker.
fn jpeg_with_exif_orientation(mut jpeg: Vec<u8>, orientation: u16) -> Vec<u8> {
    let exif = exif_payload(orientation);
    let mut segment = Vec::new();
    segment.push(0xFF);
    segment.push(0xE1);
    segment.extend_from_slice(&((exif.len() as u16 + 2).to_be_bytes()));
    segment.extend_from_slice(&exif);
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    jpeg.splice(2..2, segment);
    jpeg
}

/// The source's top-left pixel must appear at (`x`, `y`) of the decoded
/// image (JPEG is lossy, so channels are compared with a small tolerance).
fn assert_top_left_at(decoded: &image::DynamicImage, x: u32, y: u32) {
    let expected = source_pixel(0, 0);
    let got = decoded.get_pixel(x, y);
    assert!(
        (0..3).all(|c| i32::abs(i32::from(got.0[c]) - i32::from(expected.0[c])) <= 8),
        "source top-left expected at ({x},{y}), got {got:?}"
    );
}

/// Orientation 1 (and absent EXIF): no transform, source pixels upright.
#[test]
fn orientation_1_decodes_upright() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 1);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (3, 2));
    assert_top_left_at(&decoded, 0, 0);
}

/// No EXIF segment at all: byte-for-byte the previous decode behavior.
#[test]
fn no_exif_decodes_unchanged() {
    let decoded = decode_upright(&encode_source_jpeg());
    assert_eq!((decoded.width(), decoded.height()), (3, 2));
    assert_top_left_at(&decoded, 0, 0);
}

/// Orientation 3: 180° rotation, dimensions unchanged.
#[test]
fn orientation_3_rotates_180() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 3);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (3, 2));
    assert_top_left_at(&decoded, 2, 1);
    // Hand-written full-image expectation: pixel (x,y) holds source (2-x,1-y).
    for y in 0..2u32 {
        for x in 0..3u32 {
            let expected = source_pixel(2 - x, 1 - y);
            let got = decoded.get_pixel(x, y);
            assert!(
                (0..3).all(|c| i32::abs(i32::from(got.0[c]) - i32::from(expected.0[c])) <= 8),
                "180° pixel ({x},{y}) expected {expected:?}, got {got:?}"
            );
        }
    }
}

/// Orientation 5: transpose (flip about the main diagonal): D(x,y)=S(y,x);
/// dimensions swap to 2×3 and the source's top-left pixel stays at (0,0).
#[test]
fn orientation_5_transposes() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 5);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (2, 3));
    assert_top_left_at(&decoded, 0, 0);
    // The off-diagonal pixels distinguish a transpose from a no-transform:
    // D(1,0)=S(0,1) and D(0,1)=S(1,0).
    for (x, y, sx, sy) in [(1u32, 0u32, 0u32, 1u32), (0u32, 1u32, 1u32, 0u32)] {
        let expected = source_pixel(sx, sy);
        let got = decoded.get_pixel(x, y);
        assert!(
            (0..3).all(|c| i32::abs(i32::from(got.0[c]) - i32::from(expected.0[c])) <= 8),
            "transpose D({x},{y}) expected S({sx},{sy}) {expected:?}, got {got:?}"
        );
    }
}

/// Orientation 7: transverse — D(x,y)=S(W-1-y,H-1-x); dimensions swap to 2×3
/// and the source's top-left pixel lands at the bottom-right corner (1,2).
#[test]
fn orientation_7_is_transverse() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 7);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (2, 3));
    assert_top_left_at(&decoded, 1, 2);
    // D(0,0)=S(W-1-0,H-1-0)=S(2,1): the source's bottom-right pixel.
    let expected = source_pixel(2, 1);
    let got = decoded.get_pixel(0, 0);
    assert!(
        (0..3).all(|c| i32::abs(i32::from(got.0[c]) - i32::from(expected.0[c])) <= 8),
        "transverse D(0,0) expected S(2,1) {expected:?}, got {got:?}"
    );
}

/// Orientation 6: rotate 90° clockwise; 3×2 stored becomes 2×3 upright with
/// the source's top-left pixel at the top-right corner.
#[test]
fn orientation_6_rotates_90_cw() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 6);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (2, 3));
    assert_top_left_at(&decoded, 1, 0);
}

/// Orientation 8: rotate 90° counter-clockwise (270° CW); dims swap to 2×3
/// with the source's top-left pixel at the bottom-left corner.
#[test]
fn orientation_8_rotates_90_ccw() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 8);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (2, 3));
    assert_top_left_at(&decoded, 0, 2);
}

/// Orientation 2: horizontal flip; dimensions unchanged, top-left pixel at
/// the top-right corner.
#[test]
fn orientation_2_flips_horizontally() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 2);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (3, 2));
    assert_top_left_at(&decoded, 2, 0);
}

/// Orientation 4: vertical flip; dimensions unchanged, top-left pixel at
/// the bottom-left corner.
#[test]
fn orientation_4_flips_vertically() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 4);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (3, 2));
    assert_top_left_at(&decoded, 0, 1);
}

/// Corrupt EXIF metadata must not fail an otherwise decodable input: the
/// orientation read is best-effort and the pixel decode still succeeds.
#[test]
fn corrupt_exif_still_decodes() {
    let raw = jpeg_with_exif_orientation(encode_source_jpeg(), 99);
    let decoded = decode_upright(&raw);
    assert_eq!((decoded.width(), decoded.height()), (3, 2));
    assert_top_left_at(&decoded, 0, 0);
}

/// The locked WebP stack honors EXIF orientation too: a lossless WebP
/// carrying the same EXIF chunk decodes upright with swapped dimensions.
#[test]
fn webp_orientation_6_rotates_90_cw() {
    let mut buffer = image::RgbaImage::new(3, 2);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = source_pixel(x, y);
    }
    let mut out = std::io::Cursor::new(Vec::new());
    let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut out);
    encoder
        // The WebP EXIF RIFF chunk carries RAW TIFF bytes — no APP1
        // `Exif\0\0` prefix (unlike the JPEG APP1 payload above).
        .set_exif_metadata(exif_payload(6)[6..].to_vec())
        .expect("set EXIF on WebP encoder");
    encoder
        .write_image(buffer.as_raw(), 3, 2, image::ExtendedColorType::Rgba8)
        .expect("encode fixture WebP");
    let decoded = decode_upright(&out.into_inner());
    assert_eq!((decoded.width(), decoded.height()), (2, 3));
    assert_top_left_at(&decoded, 1, 0);
}
