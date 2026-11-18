//! The crafted-ORA corpus for the safe ORA read path (TASK-svg-ora-support).
//!
//! Every fixture is built in memory (no files on disk) and every
//! rejection asserts the NAMED error variant -- an ORA that is not an ORA,
//! a stack that lies, a layer that escapes the archive, a decompression
//! bomb: none of them may panic and none of them may be reported as a
//! generic failure.

// No lib target on this binary crate, so include the modules directly.
#[allow(dead_code)]
#[path = "../src/container.rs"]
mod container;
#[allow(dead_code)]
#[path = "../src/ora.rs"]
mod ora;
#[allow(dead_code)]
#[path = "../src/raster.rs"]
mod raster;

use std::fmt::Write as _;
use std::io::{Cursor, Write as _};

use crate::container::ContainerError;
use crate::ora::{OraError, OraLimits};

/// The exact `mimetype` entry content of an ORA file.
const MIMETYPE: &[u8] = b"image/openraster";

/// The limits of the corpus (small on purpose: every fixture must fit).
fn limits() -> OraLimits {
    OraLimits {
        max_entries: 16,
        max_entry_bytes: 1 << 20,
        max_total_bytes: 4 << 20,
        max_layers: 4,
        max_pixels: 1 << 20,
    }
}

/// Build a zip archive in memory.
fn zip_bytes(entries: &[(&str, zip::CompressionMethod, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, method, data) in entries {
        let options = zip::write::SimpleFileOptions::default().compression_method(*method);
        writer
            .start_file(*name, options)
            .expect("the fixture entry name is accepted");
        writer
            .write_all(data)
            .expect("the fixture entry is written");
    }
    writer
        .finish()
        .expect("the fixture archive is finished")
        .into_inner()
}

/// A PNG of one solid colour.
fn png_solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(width, height, image::Rgba(rgba));
    let mut out = Vec::new();
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("the fixture PNG is encoded");
    out
}

/// A PNG that only claims to be huge (the header, no pixel data).
fn png_claiming(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    // 8-bit RGBA, deflate, adaptive filtering, no interlacing.
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    write_chunk(&mut out, *b"IHDR", &header);
    write_chunk(&mut out, *b"IEND", &[]);
    out
}

/// One PNG chunk (length, kind, data, CRC).
fn write_chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    let length = u32::try_from(data.len()).expect("the chunk data is short");
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let mut crc_input = kind.to_vec();
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// The CRC-32 (IEEE) of `bytes`.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = if crc & 1 == 1 { 0xedb8_8320 } else { 0 };
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

/// A `stack.xml` with the given canvas and layers (name, src).
fn stack_xml(width: u32, height: u32, layers: &[(&str, &str)]) -> String {
    let mut xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<image version=\"0.0.3\" x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\">\n<stack>\n"
    );
    for (name, src) in layers {
        let _ = writeln!(
            xml,
            "<layer name=\"{name}\" src=\"{src}\" x=\"0\" y=\"0\" opacity=\"1.0\"/>"
        );
    }
    xml.push_str("</stack>\n</image>\n");
    xml
}

/// A valid ORA: a canvas, one layer PNG and the two ORA conventions.
fn valid_ora(canvas: (u32, u32), layer: (u32, u32)) -> Vec<u8> {
    let png = png_solid(layer.0, layer.1, [255, 0, 0, 255]);
    let stack = stack_xml(canvas.0, canvas.1, &[("Layer 1", "data/layer1.png")]);
    zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("data/layer1.png", zip::CompressionMethod::Deflated, &png),
    ])
}

/// A valid ORA read: the canvas, its layers and the source digest.
#[test]
fn valid_ora_reads_layers_and_flattens() {
    let bytes = valid_ora((2, 2), (2, 2));

    let document = crate::ora::read_ora(&bytes, &limits()).expect("a valid ORA reads");
    assert_eq!(document.width, 2);
    assert_eq!(document.height, 2);
    assert_eq!(document.layers.len(), 1);
    let layer = &document.layers[0];
    assert_eq!(layer.name, "Layer 1");
    assert_eq!(layer.src, "data/layer1.png");
    assert_eq!(layer.width, 2);
    assert_eq!(layer.height, 2);
    assert_eq!(layer.x, 0);
    assert_eq!(layer.y, 0);
    assert_eq!(layer.opacity, 1.0);
    assert_eq!(layer.sha256.len(), 64);
    assert!(layer.sha256.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(document.sha256.len(), 64);

    let raster = crate::ora::flatten_ora(&bytes, &limits()).expect("a valid ORA flattens");
    assert_eq!(raster.width, 2);
    assert_eq!(raster.height, 2);
    assert_eq!(
        raster.pixels,
        vec![
            255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255,
        ]
    );
}

/// A one-pixel canvas keeps its own size (the layer is clipped to it).
#[test]
fn small_canvas_keeps_its_own_size() {
    let bytes = valid_ora((1, 1), (2, 2));

    let document = crate::ora::read_ora(&bytes, &limits()).expect("a valid ORA reads");
    assert_eq!(document.width, 1);
    assert_eq!(document.height, 1);
    assert_eq!(document.layers[0].width, 2);
    assert_eq!(document.layers[0].height, 2);

    let raster = crate::ora::flatten_ora(&bytes, &limits()).expect("a valid ORA flattens");
    assert_eq!(raster.width, 1);
    assert_eq!(raster.height, 1);
    assert_eq!(raster.pixels, vec![255, 0, 0, 255]);
}

/// The first entry must be `mimetype`.
#[test]
fn first_entry_must_be_mimetype() {
    let stack = stack_xml(2, 2, &[("Layer 1", "data/layer1.png")]);
    let bytes = zip_bytes(&[
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::NotOra { detail }) => assert!(
            detail.contains("mimetype"),
            "the rejection names the offending entry: {detail}"
        ),
        other => panic!("expected NotOra, got {other:?}"),
    }
}

/// The `mimetype` entry must be stored, not deflated.
#[test]
fn mimetype_must_be_stored() {
    let stack = stack_xml(2, 2, &[("Layer 1", "data/layer1.png")]);
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Deflated, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::NotOra { detail }) => assert!(
            detail.contains("compress"),
            "the rejection names the broken convention: {detail}"
        ),
        other => panic!("expected NotOra, got {other:?}"),
    }
}

/// The `mimetype` entry must be exactly `image/openraster`.
#[test]
fn mimetype_content_must_be_image_openraster() {
    let stack = stack_xml(2, 2, &[("Layer 1", "data/layer1.png")]);
    let bytes = zip_bytes(&[
        (
            "mimetype",
            zip::CompressionMethod::Stored,
            b"image/openraster\n",
        ),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::NotOra { detail }) => assert!(
            detail.contains("image/openraster"),
            "the rejection names the expected media type: {detail}"
        ),
        other => panic!("expected NotOra, got {other:?}"),
    }
}

/// An ORA without a stack is not an ORA.
#[test]
fn missing_stack_is_rejected() {
    let bytes = zip_bytes(&[("mimetype", zip::CompressionMethod::Stored, MIMETYPE)]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::MissingStack) => {}
        other => panic!("expected MissingStack, got {other:?}"),
    }
}

/// An unsafe entry name is rejected by the container reader.
#[test]
fn unsafe_entry_path_is_rejected_by_the_container() {
    let png = png_solid(2, 2, [255, 0, 0, 255]);
    let stack = stack_xml(2, 2, &[("Layer 1", "data/layer1.png")]);
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("data/layer1.png", zip::CompressionMethod::Deflated, &png),
        ("../evil.txt", zip::CompressionMethod::Stored, b"pwned"),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::Container(ContainerError::UnsafePath { reason, .. })) => {
            assert_eq!(reason, "parent_traversal");
        }
        other => panic!("expected Container(UnsafePath), got {other:?}"),
    }
}

/// Too many layers are rejected before any of them is read.
#[test]
fn too_many_layers_are_rejected() {
    let png = png_solid(2, 2, [255, 0, 0, 255]);
    let stack = stack_xml(
        2,
        2,
        &[
            ("Layer 1", "data/layer1.png"),
            ("Layer 2", "data/layer2.png"),
            ("Layer 3", "data/layer3.png"),
            ("Layer 4", "data/layer4.png"),
            ("Layer 5", "data/layer5.png"),
        ],
    );
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("data/layer1.png", zip::CompressionMethod::Deflated, &png),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::TooManyLayers { limit, found }) => {
            assert_eq!(limit, 4);
            assert_eq!(found, 5);
        }
        other => panic!("expected TooManyLayers, got {other:?}"),
    }
}

/// A layer that references a missing file is rejected.
#[test]
fn missing_layer_file_is_rejected() {
    let png = png_solid(2, 2, [255, 0, 0, 255]);
    let stack = stack_xml(2, 2, &[("Layer 1", "data/missing.png")]);
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("data/layer1.png", zip::CompressionMethod::Deflated, &png),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::LayerMissing { name }) => assert_eq!(name, "data/missing.png"),
        other => panic!("expected LayerMissing, got {other:?}"),
    }
}

/// A layer that escapes the archive root is rejected (the stack's `src`
/// needs its own path check).
#[test]
fn escaping_layer_src_is_rejected() {
    let png = png_solid(2, 2, [255, 0, 0, 255]);
    let stack = stack_xml(2, 2, &[("Layer 1", "../evil.png")]);
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("data/layer1.png", zip::CompressionMethod::Deflated, &png),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::LayerInvalid { name, detail }) => {
            assert_eq!(name, "Layer 1");
            assert!(
                detail.contains("../evil.png"),
                "the rejection names the offending path: {detail}"
            );
        }
        other => panic!("expected LayerInvalid, got {other:?}"),
    }
}

/// A layer that claims to be huge is rejected by the pixel limit.
#[test]
fn huge_layer_is_rejected() {
    let stack = stack_xml(2, 2, &[("Layer 1", "data/layer1.png")]);
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        (
            "data/layer1.png",
            zip::CompressionMethod::Deflated,
            &png_claiming(60_000, 60_000),
        ),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::LayerTooLarge { name, limit, found }) => {
            assert_eq!(name, "Layer 1");
            assert_eq!(limit, 1 << 20);
            assert_eq!(found, 60_000 * 60_000);
        }
        other => panic!("expected LayerTooLarge, got {other:?}"),
    }
}

/// A canvas bigger than the pixel limit is rejected.
#[test]
fn huge_canvas_is_rejected() {
    let png = png_solid(2, 2, [255, 0, 0, 255]);
    let stack = stack_xml(4096, 4096, &[("Layer 1", "data/layer1.png")]);
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            stack.as_bytes(),
        ),
        ("data/layer1.png", zip::CompressionMethod::Deflated, &png),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::CanvasTooLarge { limit, found }) => {
            assert_eq!(limit, 1 << 20);
            assert_eq!(found, 4096 * 4096);
        }
        other => panic!("expected CanvasTooLarge, got {other:?}"),
    }
}

/// A malformed stack is rejected (and never panics).
#[test]
fn malformed_stack_is_rejected() {
    let bytes = zip_bytes(&[
        ("mimetype", zip::CompressionMethod::Stored, MIMETYPE),
        (
            "stack.xml",
            zip::CompressionMethod::Deflated,
            b"<image width=\"2\" height=\"2\"><layer src=\"data/layer1.png\"",
        ),
    ]);

    match crate::ora::read_ora(&bytes, &limits()) {
        Err(OraError::StackInvalid { detail }) => assert!(
            !detail.is_empty(),
            "the rejection says what was wrong with the stack"
        ),
        other => panic!("expected StackInvalid, got {other:?}"),
    }
}
