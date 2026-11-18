//! The safe `OpenRaster` (ORA) read path (TASK-svg-ora-support).
//!
//! An ORA file is a zip container with two conventions on top: the FIRST
//! entry is a STORED `mimetype` entry holding exactly `image/openraster`,
//! and `stack.xml` describes the canvas plus the layers (each layer is a
//! PNG inside the archive).
//!
//! Every archive read goes through [`crate::container`], so the
//! entry-count, per-entry, total-size and path rules apply to ORA for
//! free; this module only adds the ORA-specific shape checks, the layer
//! decode under strict limits, and the deterministic flattening.

use std::fmt;
use std::io::Cursor;

use quick_xml::Reader;
use quick_xml::events::Event;
use sha2::{Digest, Sha256};

use crate::container::{Container, ContainerError, ContainerLimits};
use crate::raster::RasterImage;

/// The `mimetype` entry name (it must be the FIRST entry of the archive).
const MIMETYPE_ENTRY: &str = "mimetype";

/// The exact `mimetype` entry content of an ORA file.
const ORA_MIMETYPE: &[u8] = b"image/openraster";

/// The stack description entry name.
const STACK_ENTRY: &str = "stack.xml";

/// The longest edge (in pixels) a decoded layer may have.
const MAX_LAYER_EDGE: u32 = 8192;

/// The limits of the ORA read path.
#[allow(clippy::struct_field_names)] // every cap is a maximum; the prefix says so
pub(crate) struct OraLimits {
    /// The maximum number of entries in the archive.
    pub max_entries: u64,
    /// The maximum size of a single archive entry.
    pub max_entry_bytes: u64,
    /// The maximum total uncompressed size of the archive.
    pub max_total_bytes: u64,
    /// The maximum number of layers in the stack.
    pub max_layers: u64,
    /// The maximum pixel count of the canvas and of a single layer.
    pub max_pixels: u64,
}

/// Everything that can reject an ORA file.
#[derive(Debug)]
pub(crate) enum OraError {
    /// The archive itself was rejected by the bounded container reader.
    Container(ContainerError),
    /// The archive is not an ORA file (wrong `mimetype` conventions).
    NotOra {
        /// What was wrong with the ORA conventions.
        detail: String,
    },
    /// The archive has no `stack.xml`.
    MissingStack,
    /// The `stack.xml` is malformed or does not describe a canvas.
    StackInvalid {
        /// What was wrong with the `stack.xml`.
        detail: String,
    },
    /// The stack declares more layers than allowed.
    TooManyLayers {
        /// The layer limit that was broken.
        limit: u64,
        /// The number of layers that were found.
        found: u64,
    },
    /// A layer references a file that is not in the archive.
    LayerMissing {
        /// The referenced file name.
        name: String,
    },
    /// A layer is malformed (bad `src`, undecodable PNG, ...).
    LayerInvalid {
        /// The layer name.
        name: String,
        /// What was wrong with the layer.
        detail: String,
    },
    /// A layer decodes to more pixels than allowed.
    LayerTooLarge {
        /// The layer name.
        name: String,
        /// The pixel limit that was broken.
        limit: u64,
        /// The pixel count that was found.
        found: u64,
    },
    /// The canvas is bigger than allowed.
    CanvasTooLarge {
        /// The pixel limit that was broken.
        limit: u64,
        /// The canvas pixel count that was found.
        found: u64,
    },
}

impl fmt::Display for OraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OraError::Container(err) => write!(f, "ORA container rejected: {err}"),
            OraError::NotOra { detail } => write!(f, "not an ORA archive: {detail}"),
            OraError::MissingStack => {
                write!(
                    f,
                    "not an ORA archive: the `{STACK_ENTRY}` entry is missing"
                )
            }
            OraError::StackInvalid { detail } => {
                write!(f, "invalid ORA `{STACK_ENTRY}`: {detail}")
            }
            OraError::TooManyLayers { limit, found } => write!(
                f,
                "ORA stack declares {found} layers, exceeding the limit of {limit} layers"
            ),
            OraError::LayerMissing { name } => {
                write!(f, "ORA layer file `{name}` is missing from the archive")
            }
            OraError::LayerInvalid { name, detail } => {
                write!(f, "invalid ORA layer `{name}`: {detail}")
            }
            OraError::LayerTooLarge { name, limit, found } => write!(
                f,
                "ORA layer `{name}` has {found} pixels, exceeding the limit of {limit} pixels"
            ),
            OraError::CanvasTooLarge { limit, found } => write!(
                f,
                "ORA canvas has {found} pixels, exceeding the limit of {limit} pixels"
            ),
        }
    }
}

impl std::error::Error for OraError {}

/// One layer of an ORA stack.
#[derive(Debug, Clone)]
#[allow(dead_code)] // the corpus asserts the metadata; the route reads what it needs
pub(crate) struct OraLayer {
    /// The layer name (the `name` attribute of the `<layer>` element).
    pub name: String,
    /// The archive-relative path of the layer PNG (the `src` attribute).
    pub src: String,
    /// The horizontal offset of the layer on the canvas.
    pub x: i64,
    /// The vertical offset of the layer on the canvas.
    pub y: i64,
    /// The layer opacity, in `[0, 1]`.
    pub opacity: f32,
    /// The decoded width of the layer PNG.
    pub width: u32,
    /// The decoded height of the layer PNG.
    pub height: u32,
    /// The SHA-256 of the layer PNG bytes.
    pub sha256: String,
}

/// A read ORA file: the canvas plus its layers.
#[derive(Debug, Clone)]
#[allow(dead_code)] // the corpus asserts the metadata; the route reads what it needs
pub(crate) struct OraDocument {
    /// The canvas width in pixels.
    pub width: u32,
    /// The canvas height in pixels.
    pub height: u32,
    /// The layers, in stack order (the first layer is the bottom one).
    pub layers: Vec<OraLayer>,
    /// The SHA-256 of the whole ORA source bytes.
    pub sha256: String,
}

/// Read an ORA file: the canvas, its layers and the source digest.
///
/// # Errors
///
/// Returns [`OraError`] when the archive is not an ORA file, when the
/// container, the stack or a layer breaks one of the limits.
pub(crate) fn read_ora(bytes: &[u8], limits: &OraLimits) -> Result<OraDocument, OraError> {
    Ok(read_inner(bytes, limits)?.0)
}

/// Flatten an ORA file into the canvas raster (layers composited in stack
/// order, the first layer at the bottom).
///
/// # Errors
///
/// Returns [`OraError`] when the archive is not an ORA file, when the
/// container, the stack or a layer breaks one of the limits.
pub(crate) fn flatten_ora(bytes: &[u8], limits: &OraLimits) -> Result<RasterImage, OraError> {
    let (document, layers) = read_inner(bytes, limits)?;

    let mut canvas = image::RgbaImage::new(document.width, document.height);
    for (layer, png) in document.layers.iter().zip(layers.iter()) {
        // The decode happens HERE, once per layer: the read path only
        // validated the header claims.
        let mut pixels = decode_layer(png, limits).map_err(|detail| OraError::LayerInvalid {
            name: layer.name.clone(),
            detail,
        })?;
        apply_opacity(&mut pixels, layer.opacity);
        image::imageops::overlay(&mut canvas, &pixels, layer.x, layer.y);
    }

    Ok(RasterImage {
        width: document.width,
        height: document.height,
        pixels: canvas.into_raw(),
    })
}

/// Read the archive, the stack and every layer header; returns the document
/// plus the layer PNG bytes (NOT decoded — the decode belongs to
/// [`flatten_ora`], so the layers are decoded exactly once).
#[allow(clippy::too_many_lines)] // one linear read: container, stack, canvas, layers
fn read_inner(bytes: &[u8], limits: &OraLimits) -> Result<(OraDocument, Vec<Vec<u8>>), OraError> {
    let container =
        Container::open_zip(bytes, container_limits(limits)).map_err(OraError::Container)?;

    // The ORA conventions: `mimetype` first, stored, exactly
    // `image/openraster`.
    let Some(first) = container.entries().first() else {
        return Err(OraError::NotOra {
            detail: "the archive has no entries".to_owned(),
        });
    };
    if first.name != MIMETYPE_ENTRY {
        return Err(OraError::NotOra {
            detail: format!(
                "the first entry is `{}`, expected `{MIMETYPE_ENTRY}`",
                first.name
            ),
        });
    }
    if !first.stored {
        return Err(OraError::NotOra {
            detail: format!("the `{MIMETYPE_ENTRY}` entry is compressed; ORA stores it verbatim"),
        });
    }
    let mimetype = container
        .read_entry(MIMETYPE_ENTRY, container_limits(limits))
        .map_err(OraError::Container)?;
    if mimetype.bytes != ORA_MIMETYPE {
        return Err(OraError::NotOra {
            detail: format!(
                "the `{MIMETYPE_ENTRY}` entry is `{}`, expected `image/openraster`",
                String::from_utf8_lossy(&mimetype.bytes)
            ),
        });
    }

    // The stack description.
    if container.entry(STACK_ENTRY).is_none() {
        return Err(OraError::MissingStack);
    }
    let stack = container
        .read_entry(STACK_ENTRY, container_limits(limits))
        .map_err(OraError::Container)?;
    let stack_xml = std::str::from_utf8(&stack.bytes).map_err(|err| OraError::StackInvalid {
        detail: format!("`{STACK_ENTRY}` is not UTF-8: {err}"),
    })?;
    let stack = parse_stack(stack_xml)?;

    // The canvas.
    let (width, height) = stack.canvas;
    let canvas_pixels = u64::from(width) * u64::from(height);
    if canvas_pixels > limits.max_pixels {
        return Err(OraError::CanvasTooLarge {
            limit: limits.max_pixels,
            found: canvas_pixels,
        });
    }

    // The layer count.
    let found = stack.layers.len() as u64;
    if found > limits.max_layers {
        return Err(OraError::TooManyLayers {
            limit: limits.max_layers,
            found,
        });
    }

    // The layers themselves.
    let mut layers = Vec::with_capacity(stack.layers.len());
    let mut images = Vec::with_capacity(stack.layers.len());
    for spec in stack.layers {
        if let Err(detail) = check_src_path(&spec.src) {
            return Err(OraError::LayerInvalid {
                name: spec.name,
                detail,
            });
        }
        if container.entry(&spec.src).is_none() {
            return Err(OraError::LayerMissing {
                name: spec.src.clone(),
            });
        }
        let file = container
            .read_entry(&spec.src, container_limits(limits))
            .map_err(OraError::Container)?;

        let (decoded_width, decoded_height) =
            png_dimensions(&file.bytes).map_err(|detail| OraError::LayerInvalid {
                name: spec.name.clone(),
                detail,
            })?;
        let pixels = u64::from(decoded_width) * u64::from(decoded_height);
        if pixels > limits.max_pixels {
            return Err(OraError::LayerTooLarge {
                name: spec.name.clone(),
                limit: limits.max_pixels,
                found: pixels,
            });
        }
        // The layer's PIXELS are not decoded here: the header dimensions
        // are the claim that the cap is checked against, and the decode
        // happens once, in `flatten_ora`.
        layers.push(OraLayer {
            name: spec.name,
            src: spec.src,
            x: spec.x,
            y: spec.y,
            opacity: spec.opacity,
            width: decoded_width,
            height: decoded_height,
            sha256: file.sha256,
        });
        images.push(file.bytes);
    }

    Ok((
        OraDocument {
            width,
            height,
            layers,
            sha256: sha256_hex(bytes),
        },
        images,
    ))
}

/// The container limits derived from the ORA limits.
fn container_limits(limits: &OraLimits) -> ContainerLimits {
    ContainerLimits {
        max_entries: limits.max_entries,
        max_entry_bytes: limits.max_entry_bytes,
        max_total_bytes: limits.max_total_bytes,
    }
}

/// The SHA-256 of `bytes`, lowercase hex.
fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// The dimensions a layer PNG DECLARES (header only: no decode, no
/// allocation). The declaration is a claim -- the decode below is what
/// actually produces pixels -- so the size cap is checked against it
/// BEFORE anything is decoded.
fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    /// The PNG signature.
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

    if bytes.len() < 8 || bytes[..8] != SIGNATURE {
        return Err("the layer file is not a PNG".to_owned());
    }
    // The IHDR chunk: length (4 bytes), type (4 bytes), width (4 bytes),
    // height (4 bytes).
    if bytes.len() < 24 || &bytes[12..16] != b"IHDR" {
        return Err("the layer PNG has no IHDR chunk".to_owned());
    }
    let width = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let height = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Ok((width, height))
}

/// Decode a layer PNG under strict limits.
fn decode_layer(bytes: &[u8], limits: &OraLimits) -> Result<image::RgbaImage, String> {
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), image::ImageFormat::Png);
    let mut image_limits = image::Limits::default();
    image_limits.max_image_width = Some(MAX_LAYER_EDGE);
    image_limits.max_image_height = Some(MAX_LAYER_EDGE);
    image_limits.max_alloc = Some(max_alloc(limits));
    reader.limits(image_limits);
    reader
        .decode()
        .map(|image| image.to_rgba8())
        .map_err(|err| format!("the layer PNG cannot be decoded: {err}"))
}

/// The decoder allocation budget: the canvas budget plus room for the
/// decoder's own buffers, capped so a huge canvas cannot buy a huge
/// allocation.
fn max_alloc(limits: &OraLimits) -> u64 {
    limits
        .max_pixels
        .saturating_mul(4)
        .saturating_add(16 * 1024 * 1024)
        .min(512 * 1024 * 1024)
}

/// Scale the alpha channel of `pixels` by `opacity` (in place).
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // the clamp bounds the cast
fn apply_opacity(pixels: &mut image::RgbaImage, opacity: f32) {
    if opacity >= 1.0 {
        return;
    }
    for (_, _, pixel) in pixels.enumerate_pixels_mut() {
        let alpha = f32::from(pixel.0[3]) * opacity;
        // The value is clamped into `[0, 255]`, so the truncation is exact
        // and the cast cannot lose the sign.
        pixel.0[3] = alpha.round().clamp(0.0, 255.0) as u8;
    }
}

/// Reject a layer `src` that is not a plain archive-relative path.
fn check_src_path(src: &str) -> Result<(), String> {
    if src.is_empty() {
        return Err("the layer `src` is empty".to_owned());
    }
    if src.starts_with('/') || src.starts_with('\\') {
        return Err(format!("the layer `src` `{src}` is an absolute path"));
    }
    if src.contains('\\') {
        return Err(format!("the layer `src` `{src}` contains a backslash"));
    }
    if src.split('/').next().is_some_and(|head| head.contains(':')) {
        return Err(format!("the layer `src` `{src}` is an absolute path"));
    }
    if src.split('/').any(|part| part == "..") {
        return Err(format!("the layer `src` `{src}` escapes the archive root"));
    }
    Ok(())
}

/// A parsed `stack.xml`.
struct ParsedStack {
    /// The canvas size.
    canvas: (u32, u32),
    /// The layers, in stack order.
    layers: Vec<LayerSpec>,
}

/// A parsed `<layer>` element.
struct LayerSpec {
    /// The layer name.
    name: String,
    /// The archive-relative path of the layer PNG.
    src: String,
    /// The horizontal offset.
    x: i64,
    /// The vertical offset.
    y: i64,
    /// The opacity, in `[0, 1]`.
    opacity: f32,
}

/// Parse `stack.xml` (streaming, no entity expansion: attribute values are
/// read raw).
fn parse_stack(xml: &str) -> Result<ParsedStack, OraError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut canvas: Option<(u32, u32)> = None;
    let mut layers: Vec<LayerSpec> = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(element) | Event::Empty(element)) => {
                match element.local_name().as_ref() {
                    b"image" => canvas = Some(parse_canvas(&element)?),
                    b"layer" => layers.push(parse_layer(&element)?),
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(err) => {
                return Err(OraError::StackInvalid {
                    detail: err.to_string(),
                });
            }
        }
    }

    let canvas = canvas.ok_or_else(|| OraError::StackInvalid {
        detail: format!("the `{STACK_ENTRY}` has no `<image>` element"),
    })?;
    Ok(ParsedStack { canvas, layers })
}

/// The attributes of an element, as raw (unexpanded) values.
fn element_attributes(
    element: &quick_xml::events::BytesStart,
) -> Result<Vec<(String, String)>, OraError> {
    let mut out = Vec::new();
    for attribute in element.attributes() {
        let attribute = attribute.map_err(|err| OraError::StackInvalid {
            detail: format!("malformed attribute: {err}"),
        })?;
        out.push((
            String::from_utf8_lossy(attribute.key.as_ref()).into_owned(),
            String::from_utf8_lossy(&attribute.value).into_owned(),
        ));
    }
    Ok(out)
}

/// One attribute of an element.
fn attribute<'a>(attributes: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attributes
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

/// The canvas of the `<image>` element.
fn parse_canvas(element: &quick_xml::events::BytesStart) -> Result<(u32, u32), OraError> {
    let attributes = element_attributes(element)?;
    let width = attribute(&attributes, "width").or_else(|| attribute(&attributes, "w"));
    let height = attribute(&attributes, "height").or_else(|| attribute(&attributes, "h"));
    let width = width.ok_or_else(|| OraError::StackInvalid {
        detail: "the `<image>` element has no `width`".to_owned(),
    })?;
    let height = height.ok_or_else(|| OraError::StackInvalid {
        detail: "the `<image>` element has no `height`".to_owned(),
    })?;

    let width: u32 = width.parse().map_err(|_| OraError::StackInvalid {
        detail: format!("the `<image>` width `{width}` is not a number"),
    })?;
    let height: u32 = height.parse().map_err(|_| OraError::StackInvalid {
        detail: format!("the `<image>` height `{height}` is not a number"),
    })?;
    if width == 0 || height == 0 {
        return Err(OraError::StackInvalid {
            detail: format!(
                "the `<image>` canvas is {width}x{height}; both edges must be positive"
            ),
        });
    }
    Ok((width, height))
}

/// One `<layer>` element.
fn parse_layer(element: &quick_xml::events::BytesStart) -> Result<LayerSpec, OraError> {
    let attributes = element_attributes(element)?;
    let src = attribute(&attributes, "src").ok_or_else(|| OraError::StackInvalid {
        detail: "a `<layer>` element has no `src`".to_owned(),
    })?;
    let name = attribute(&attributes, "name").unwrap_or(src).to_owned();

    let x = parse_offset(attribute(&attributes, "x"), "x")?;
    let y = parse_offset(attribute(&attributes, "y"), "y")?;
    let opacity = match attribute(&attributes, "opacity") {
        Some(raw) => {
            let opacity: f32 = raw.parse().map_err(|_| OraError::StackInvalid {
                detail: format!("the `<layer>` opacity `{raw}` is not a number"),
            })?;
            if !opacity.is_finite() {
                return Err(OraError::StackInvalid {
                    detail: format!("the `<layer>` opacity `{raw}` is not finite"),
                });
            }
            opacity.clamp(0.0, 1.0)
        }
        None => 1.0,
    };

    Ok(LayerSpec {
        name,
        src: src.to_owned(),
        x,
        y,
        opacity,
    })
}

/// One integer attribute of a `<layer>` element.
fn parse_offset(raw: Option<&str>, key: &str) -> Result<i64, OraError> {
    match raw {
        None => Ok(0),
        Some(raw) => raw.parse().map_err(|_| OraError::StackInvalid {
            detail: format!("the `<layer>` `{key}` `{raw}` is not a number"),
        }),
    }
}
