//! The safe SVG read path (TASK-svg-ora-support).
//!
//! An SVG is untrusted XML: it can carry scripts, point at external
//! resources or hide a billion-laughs entity bomb. This module validates
//! the document BEFORE anything is rasterized, then renders it with
//! `resvg` into a plain RGBA8 raster.
//!
//! The validation is one streaming pass over the source with `quick-xml`
//! (no DTD processing, no entity expansion) and it rejects, by name:
//! scripts, external references, DTD/entity declarations, oversized
//! sources, too many elements and too many pixels.
//!
//! `resvg` is compiled WITHOUT its `text` feature, so no system font is
//! ever loaded and `<text>` is not rasterized: the same bytes always
//! produce the same pixels on every machine.

use std::fmt;

use quick_xml::Reader;
use quick_xml::events::Event;
use sha2::{Digest, Sha256};

use crate::raster::RasterImage;

/// The default viewport width (SVG user units) of a document that has
/// neither a `width` attribute nor a `viewBox`.
const DEFAULT_WIDTH: u32 = 300;

/// The default viewport height (SVG user units) of a document that has
/// neither a `height` attribute nor a `viewBox`.
const DEFAULT_HEIGHT: u32 = 150;

/// The byte patterns that mark a DTD or an entity declaration. XML
/// keywords are case sensitive, so the uppercase spellings are the only
/// ones a conforming document can use.
const DTD_MARKERS: [&[u8]; 2] = [b"<!DOCTYPE", b"<!ENTITY"];

/// The limits of the SVG read path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_field_names)] // every cap is a maximum; the prefix says so
pub(crate) struct SvgLimits {
    /// The largest accepted source size, in bytes.
    pub max_source_bytes: u64,
    /// The largest accepted number of elements.
    pub max_elements: u64,
    /// The largest accepted pixel count (`width * height`).
    pub max_pixels: u64,
}

/// Everything that can reject an SVG file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SvgError {
    /// The source is bigger than the byte limit.
    SourceTooLarge {
        /// The byte limit that was broken.
        limit: u64,
        /// The size that was found.
        found: u64,
    },
    /// The document has more elements than the limit.
    TooManyElements {
        /// The element limit that was broken.
        limit: u64,
        /// The element count that was found.
        found: u64,
    },
    /// The document carries a script.
    Script,
    /// The document references a resource outside the document.
    ExternalReference {
        /// The attribute that carries the reference.
        attr: String,
        /// The offending value.
        value: String,
    },
    /// The document declares a DTD or an entity.
    Doctype,
    /// The document is not well-formed XML.
    Invalid {
        /// What was wrong.
        detail: String,
    },
    /// The viewport is bigger than the pixel limit.
    RasterTooLarge {
        /// The pixel limit that was broken.
        limit: u64,
        /// The pixel count that was found.
        found: u64,
    },
    /// The renderer could not produce a raster.
    RasterFailed {
        /// What the renderer reported.
        detail: String,
    },
}

impl fmt::Display for SvgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SvgError::SourceTooLarge { limit, found } => write!(
                f,
                "the SVG source is {found} bytes, over the {limit} byte limit"
            ),
            SvgError::TooManyElements { limit, found } => write!(
                f,
                "the SVG has {found} elements, over the {limit} element limit"
            ),
            SvgError::Script => write!(
                f,
                "the SVG carries a script (`<script>` or an `on*` handler)"
            ),
            SvgError::ExternalReference { attr, value } => write!(
                f,
                "the SVG references the external resource `{value}` in `{attr}`"
            ),
            SvgError::Doctype => write!(f, "the SVG declares a DOCTYPE or an entity"),
            SvgError::Invalid { detail } => write!(f, "the SVG is not valid: {detail}"),
            SvgError::RasterTooLarge { limit, found } => write!(
                f,
                "the SVG viewport is {found} pixels, over the {limit} pixel limit"
            ),
            SvgError::RasterFailed { detail } => {
                write!(f, "the SVG could not be rasterized: {detail}")
            }
        }
    }
}

impl std::error::Error for SvgError {}

/// A read SVG file: its viewport plus the digest of the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SvgDocument {
    /// The viewport width, in pixels.
    pub width: u32,
    /// The viewport height, in pixels.
    pub height: u32,
    /// The SHA-256 of the source bytes, lowercase hex.
    pub sha256: String,
}

/// Validate an SVG file against the limits.
///
/// # Errors
///
/// Returns [`SvgError`] when the source is too big, carries a script or an
/// external reference, declares a DTD or an entity, is not well-formed XML
/// or describes a viewport that is over the pixel limit.
pub(crate) fn validate_svg(bytes: &[u8], limits: &SvgLimits) -> Result<SvgDocument, SvgError> {
    inspect(bytes, limits).map(|(_, document)| document)
}

/// Rasterize an SVG file into an RGBA8 raster.
///
/// The document is validated first, so nothing is rasterized before the
/// unsafe constructs are rejected.
///
/// # Errors
///
/// Returns [`SvgError`] when the document is rejected by the validation or
/// when the renderer cannot produce the raster.
pub(crate) fn rasterize_svg(bytes: &[u8], limits: &SvgLimits) -> Result<RasterImage, SvgError> {
    let (text, document) = inspect(bytes, limits)?;

    let tree =
        resvg::usvg::Tree::from_str(text, &resvg::usvg::Options::default()).map_err(|err| {
            SvgError::Invalid {
                detail: format!("the SVG could not be parsed: {err}"),
            }
        })?;

    let mut pixmap = tiny_skia::Pixmap::new(document.width, document.height).ok_or_else(|| {
        SvgError::RasterFailed {
            detail: format!(
                "a {}x{} raster could not be allocated",
                document.width, document.height
            ),
        }
    })?;
    resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());

    Ok(RasterImage {
        width: document.width,
        height: document.height,
        pixels: straight_alpha(pixmap.data()),
    })
}

/// Validate the source and return the decoded text plus the document.
fn inspect<'a>(bytes: &'a [u8], limits: &SvgLimits) -> Result<(&'a str, SvgDocument), SvgError> {
    let found = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if found > limits.max_source_bytes {
        return Err(SvgError::SourceTooLarge {
            limit: limits.max_source_bytes,
            found,
        });
    }
    // The DTD / entity gate: the document is rejected outright, so the
    // billion-laughs and XXE shapes never reach the parser at all.
    if declares_dtd(bytes) {
        return Err(SvgError::Doctype);
    }

    let text = decode_source(bytes)?;
    let shape = scan(text, limits)?;

    let width = shape
        .width
        .or_else(|| shape.view_box.map(|view_box| view_box.0))
        .unwrap_or(DEFAULT_WIDTH);
    let height = shape
        .height
        .or_else(|| shape.view_box.map(|view_box| view_box.1))
        .unwrap_or(DEFAULT_HEIGHT);
    let pixels = u64::from(width) * u64::from(height);
    if pixels > limits.max_pixels {
        return Err(SvgError::RasterTooLarge {
            limit: limits.max_pixels,
            found: pixels,
        });
    }

    let document = SvgDocument {
        width,
        height,
        sha256: sha256_hex(bytes),
    };
    Ok((text, document))
}

/// The shape of the root `<svg>` element.
#[derive(Debug, Default)]
struct RootShape {
    /// The `width` attribute, in pixels.
    width: Option<u32>,
    /// The `height` attribute, in pixels.
    height: Option<u32>,
    /// The `width` and `height` of the `viewBox`, in user units.
    view_box: Option<(u32, u32)>,
}

impl RootShape {
    /// Whether the shape was filled yet.
    fn is_empty(&self) -> bool {
        self.width.is_none() && self.height.is_none() && self.view_box.is_none()
    }
}

/// Scan the document for the unsafe constructs and read the root shape.
fn scan(text: &str, limits: &SvgLimits) -> Result<RootShape, SvgError> {
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(true);

    let mut shape = RootShape::default();
    let mut elements: u64 = 0;
    // The open-element depth: a document that leaves an element open (or
    // closes one that was never opened) is not well-formed, and the
    // validator must say so instead of passing a broken document on to
    // the rasterizer.
    let mut depth: usize = 0;
    let mut in_style = false;
    let mut style_text = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => {
                count_element(&mut elements, limits)?;
                depth = depth.saturating_add(1);
                let local_name = element.local_name();
                let name = local_name.as_ref();
                if name.eq_ignore_ascii_case(b"script") {
                    return Err(SvgError::Script);
                }
                if name.eq_ignore_ascii_case(b"style") {
                    in_style = true;
                    style_text.clear();
                }
                if name.eq_ignore_ascii_case(b"svg") && shape.is_empty() {
                    shape = read_root_shape(&element)?;
                }
                check_attributes(&element)?;
            }
            Ok(Event::Empty(element)) => {
                count_element(&mut elements, limits)?;
                let local_name = element.local_name();
                let name = local_name.as_ref();
                if name.eq_ignore_ascii_case(b"script") {
                    return Err(SvgError::Script);
                }
                if name.eq_ignore_ascii_case(b"svg") && shape.is_empty() {
                    shape = read_root_shape(&element)?;
                }
                check_attributes(&element)?;
            }
            Ok(Event::Text(text)) => {
                if in_style {
                    style_text.push_str(&decode_text(&text)?);
                }
            }
            Ok(Event::CData(text)) => {
                if in_style {
                    style_text.push_str(&text.decode().map_err(|err| SvgError::Invalid {
                        detail: format!("the SVG has text that is not valid UTF-8: {err}"),
                    })?);
                }
            }
            Ok(Event::End(element)) => {
                if depth == 0 {
                    return Err(SvgError::Invalid {
                        detail: "the SVG closes an element that was never opened".to_owned(),
                    });
                }
                depth -= 1;
                if in_style && element.local_name().as_ref().eq_ignore_ascii_case(b"style") {
                    in_style = false;
                    check_style_urls(&style_text, "style")?;
                    style_text.clear();
                }
            }
            Ok(Event::DocType(_)) => return Err(SvgError::Doctype),
            Ok(Event::Eof) => {
                if depth != 0 {
                    return Err(SvgError::Invalid {
                        detail: format!("the SVG ends with {depth} unclosed element(s)"),
                    });
                }
                break;
            }
            Ok(_) => {}
            Err(err) => {
                return Err(SvgError::Invalid {
                    detail: format!("the SVG is not well-formed: {err}"),
                });
            }
        }
    }

    Ok(shape)
}

/// Count one element and enforce the element limit.
fn count_element(elements: &mut u64, limits: &SvgLimits) -> Result<(), SvgError> {
    *elements += 1;
    if *elements > limits.max_elements {
        return Err(SvgError::TooManyElements {
            limit: limits.max_elements,
            found: *elements,
        });
    }
    Ok(())
}

/// Read the `width`, `height` and `viewBox` of the root `<svg>` element.
fn read_root_shape(element: &quick_xml::events::BytesStart<'_>) -> Result<RootShape, SvgError> {
    let mut shape = RootShape::default();
    for attribute in element.attributes() {
        let attribute = attribute.map_err(|err| SvgError::Invalid {
            detail: format!("the SVG has a malformed attribute: {err}"),
        })?;
        // The local name is bound first: `local_name()` returns a temporary,
        // and the borrow must not outlive it.
        let local_name = attribute.key.local_name();
        let name = local_name.as_ref();
        let value = String::from_utf8_lossy(&attribute.value);
        if name == b"width" {
            shape.width = Some(parse_dimension(&value)?);
        } else if name == b"height" {
            shape.height = Some(parse_dimension(&value)?);
        } else if name == b"viewBox" {
            shape.view_box = Some(parse_view_box(&value)?);
        }
    }
    Ok(shape)
}

/// Check every attribute of an element for the unsafe constructs.
fn check_attributes(element: &quick_xml::events::BytesStart<'_>) -> Result<(), SvgError> {
    for attribute in element.attributes() {
        let attribute = attribute.map_err(|err| SvgError::Invalid {
            detail: format!("the SVG has a malformed attribute: {err}"),
        })?;
        let local_name = attribute.key.local_name();
        let local = local_name.as_ref();
        // An event handler (`onload`, `onclick`, ...) is a script.
        if local.len() >= 2 && local[..2].eq_ignore_ascii_case(b"on") {
            return Err(SvgError::Script);
        }
        let value = String::from_utf8_lossy(&attribute.value);
        let trimmed = value.trim();
        if local == b"href" || local == b"src" || local == b"data" {
            if !is_local_reference(trimmed) {
                return Err(SvgError::ExternalReference {
                    attr: String::from_utf8_lossy(attribute.key.as_ref()).into_owned(),
                    value: trimmed.to_owned(),
                });
            }
        } else if local == b"style" {
            let name = String::from_utf8_lossy(attribute.key.as_ref()).into_owned();
            check_style_urls(trimmed, &name)?;
        }
    }
    Ok(())
}

/// Whether a reference stays inside the document: a same-document
/// fragment (`#id`) or a `data:` URI.
fn is_local_reference(value: &str) -> bool {
    if value.is_empty() {
        return true;
    }
    value.starts_with('#') || value.to_ascii_lowercase().starts_with("data:")
}

/// Check the `url(...)` references of a style value.
fn check_style_urls(value: &str, attr: &str) -> Result<(), SvgError> {
    let lowered = value.to_ascii_lowercase();
    let mut cursor = 0usize;
    while let Some(found) = lowered[cursor..].find("url(") {
        let start = cursor + found + "url(".len();
        let end = lowered[start..]
            .find(')')
            .map_or(value.len(), |offset| start + offset);
        let url = value[start..end]
            .trim()
            .trim_matches(|c| c == '\'' || c == '"');
        if !url.is_empty() && !is_local_reference(url) {
            return Err(SvgError::ExternalReference {
                attr: attr.to_owned(),
                value: url.to_owned(),
            });
        }
        cursor = end;
    }
    Ok(())
}

/// Parse a length in pixels (a trailing `px` is allowed).
fn parse_dimension(value: &str) -> Result<u32, SvgError> {
    let trimmed = value.trim();
    let number = trimmed.strip_suffix("px").unwrap_or(trimmed).trim();
    number.parse::<u32>().map_err(|_| SvgError::Invalid {
        detail: format!("the SVG dimension `{value}` is not a pixel length"),
    })
}

/// Parse the `width` and `height` of a `viewBox` (`min-x min-y width
/// height`).
fn parse_view_box(value: &str) -> Result<(u32, u32), SvgError> {
    let mut parts = value
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|part| !part.is_empty());
    let _min_x = parts.next();
    let _min_y = parts.next();
    let width = parts.next();
    let height = parts.next();
    match (width, height) {
        (Some(width), Some(height)) => Ok((parse_dimension(width)?, parse_dimension(height)?)),
        _ => Err(SvgError::Invalid {
            detail: format!("the SVG viewBox `{value}` is malformed"),
        }),
    }
}

/// Decode the source as UTF-8 (a leading byte-order mark is dropped).
fn decode_source(bytes: &[u8]) -> Result<&str, SvgError> {
    let body = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    std::str::from_utf8(body).map_err(|err| SvgError::Invalid {
        detail: format!("the SVG is not valid UTF-8: {err}"),
    })
}

/// Decode the text of an element.
fn decode_text(text: &quick_xml::events::BytesText<'_>) -> Result<String, SvgError> {
    text.decode()
        .map(std::borrow::Cow::into_owned)
        .map_err(|err| SvgError::Invalid {
            detail: format!("the SVG has text that is not valid UTF-8: {err}"),
        })
}

/// Whether the source declares a DTD or an entity.
fn declares_dtd(bytes: &[u8]) -> bool {
    DTD_MARKERS
        .iter()
        .any(|marker| contains_bytes(bytes, marker))
}

/// A byte-substring search without an allocation.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Copy the PREMULTIPLIED RGBA bytes of a pixmap into STRAIGHT RGBA bytes.
fn straight_alpha(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for pixel in data.as_chunks::<4>().0 {
        let alpha = pixel[3];
        if alpha == 0 {
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        let scale = u32::from(alpha);
        for channel in &pixel[..3] {
            let value = (u32::from(*channel) * 255 + scale / 2) / scale;
            out.push(u8::try_from(value).unwrap_or(u8::MAX));
        }
        out.push(alpha);
    }
    out
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
