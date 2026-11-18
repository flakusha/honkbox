//! The crafted-SVG corpus for the safe SVG read path
//! (TASK-svg-ora-support).
//!
//! Every fixture is a string in this file (no fixture files on disk) and
//! every rejection asserts the NAMED error variant: a script, an external
//! reference, a DTD, an oversized source, an oversized element tree or an
//! oversized viewport must each be reported by name — not as a generic
//! "invalid input".
//!
//! # Resource contract (parallel safety)
//!
//! Each test owns ONLY its own bytes: the fixtures are literals, the
//! rasters are allocated per test and dropped with it. There is no file,
//! no temp dir, no fixed path, no port, no environment variable and no
//! shared mutable state — so the corpus needs no lock, has no ordering
//! dependence and passes in any order.

// No lib target on this binary crate, so include the modules directly.
#[allow(dead_code)]
#[path = "../src/raster.rs"]
mod raster;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;

use svg::{SvgError, SvgLimits};

/// The limits the corpus runs with: small on purpose, so every fixture
/// that breaks a limit breaks it clearly.
fn limits() -> SvgLimits {
    SvgLimits {
        max_source_bytes: 4096,
        max_elements: 16,
        max_pixels: 4096,
    }
}

/// A minimal valid SVG with the given viewport.
fn svg_source(width: u32, height: u32) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><rect width=\"{width}\" height=\"{height}\" fill=\"#ff0000\"/></svg>"
    )
}

/// A valid SVG is read: its viewport and the digest of the source.
#[test]
fn valid_svg_reports_its_viewport() {
    let source = svg_source(40, 20);
    let document = svg::validate_svg(source.as_bytes(), &limits()).expect("the SVG is valid");
    assert_eq!(document.width, 40);
    assert_eq!(document.height, 20);
    assert_eq!(document.sha256.len(), 64, "the source digest is recorded");
}

/// A valid SVG rasterizes deterministically: the same bytes always
/// produce the same pixels.
#[test]
fn rasterizes_a_valid_svg_deterministically() {
    let source = svg_source(40, 20);
    let first = svg::rasterize_svg(source.as_bytes(), &limits()).expect("the SVG rasterizes");
    let second = svg::rasterize_svg(source.as_bytes(), &limits()).expect("the SVG rasterizes");
    assert_eq!(first.width, 40);
    assert_eq!(first.height, 20);
    assert_eq!(first.pixels.len(), 40 * 20 * 4);
    assert_eq!(first, second, "the same bytes produce the same pixels");
}

/// A `<script>` element is a script, whatever it contains.
#[test]
fn script_element_is_rejected() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><script>alert(1)</script></svg>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::Script) => {}
        other => panic!("expected Script, got {other:?}"),
    }
}

/// An event handler attribute is a script.
#[test]
fn event_handler_attribute_is_rejected() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\" onload=\"alert(1)\"/>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::Script) => {}
        other => panic!("expected Script, got {other:?}"),
    }
}

/// An external reference is rejected and the rejection names the
/// attribute and the offending value.
#[test]
fn external_href_is_rejected() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><image href=\"http://example.invalid/x.png\"/></svg>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::ExternalReference { attr, value }) => {
            assert_eq!(attr, "href");
            assert!(
                value.contains("example.invalid"),
                "the rejection names the offending value: {value}"
            );
        }
        other => panic!("expected ExternalReference, got {other:?}"),
    }
}

/// An external reference inside a `style` attribute is rejected too.
#[test]
fn external_style_url_is_rejected() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><rect style=\"fill:url(http://example.invalid/x.svg)\"/></svg>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::ExternalReference { attr, value }) => {
            assert_eq!(attr, "style");
            assert!(
                value.contains("example.invalid"),
                "the rejection names the offending value: {value}"
            );
        }
        other => panic!("expected ExternalReference, got {other:?}"),
    }
}

/// An external reference inside a `<style>` element is rejected too.
#[test]
fn external_style_element_url_is_rejected() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><style>@import url(http://example.invalid/x.css);</style></svg>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::ExternalReference { .. }) => {}
        other => panic!("expected ExternalReference, got {other:?}"),
    }
}

/// A same-document reference stays inside the document and is accepted.
#[test]
fn same_document_reference_is_accepted() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><defs><rect id=\"box\" width=\"4\" height=\"4\"/></defs><use href=\"#box\"/></svg>";
    svg::validate_svg(source.as_bytes(), &limits()).expect("a same-document reference is local");
}

/// A DTD (and with it an entity declaration) is rejected outright, so
/// neither a billion-laughs bomb nor an XXE reaches the parser.
#[test]
fn doctype_is_rejected() {
    let source = "<!DOCTYPE svg [<!ENTITY lol \"lol\">]><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">&lol;</svg>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::Doctype) => {}
        other => panic!("expected Doctype, got {other:?}"),
    }
}

/// A source over the byte limit is rejected before it is parsed.
#[test]
fn source_over_the_limit_is_rejected() {
    let mut source = svg_source(10, 10);
    source.push_str("<!-- ");
    source.push_str(&"x".repeat(4096));
    source.push_str(" -->");
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::SourceTooLarge { limit, found }) => {
            assert_eq!(limit, 4096);
            assert_eq!(found, u64::try_from(source.len()).unwrap_or(u64::MAX));
        }
        other => panic!("expected SourceTooLarge, got {other:?}"),
    }
}

/// A document with too many elements is rejected while it is scanned.
#[test]
fn too_many_elements_are_rejected() {
    let mut source =
        String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">");
    for _ in 0..32 {
        source.push_str("<rect/>");
    }
    source.push_str("</svg>");
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::TooManyElements { limit, found }) => {
            assert_eq!(limit, 16);
            assert!(found > 16, "the found count is reported: {found}");
        }
        other => panic!("expected TooManyElements, got {other:?}"),
    }
}

/// A viewport over the pixel limit is rejected — never downscaled.
#[test]
fn viewport_over_the_pixel_limit_is_rejected() {
    let source = svg_source(10_000, 10_000);
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::RasterTooLarge { limit, found }) => {
            assert_eq!(limit, 4096);
            assert_eq!(found, 100_000_000);
        }
        other => panic!("expected RasterTooLarge, got {other:?}"),
    }
}

/// Malformed XML is rejected by name, not by a panic.
#[test]
fn malformed_xml_is_rejected() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><rect>";
    match svg::validate_svg(source.as_bytes(), &limits()) {
        Err(SvgError::Invalid { .. }) => {}
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A source that is not UTF-8 is rejected by name.
#[test]
fn non_utf8_source_is_rejected() {
    let source = b"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"><text>\xff\xfe</text></svg>";
    match svg::validate_svg(source, &limits()) {
        Err(SvgError::Invalid { .. }) => {}
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A document with neither a viewport nor a `viewBox` gets the default
/// viewport, and a `viewBox` alone is enough to size the raster.
#[test]
fn viewbox_sizes_the_raster_when_no_viewport_is_given() {
    let source = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 12\"><rect width=\"24\" height=\"12\"/></svg>";
    let document = svg::validate_svg(source.as_bytes(), &limits()).expect("the SVG is valid");
    assert_eq!(document.width, 24);
    assert_eq!(document.height, 12);
}

/// A byte-order mark is tolerated (the source is still UTF-8 text).
#[test]
fn byte_order_mark_is_tolerated() {
    let mut source = b"\xEF\xBB\xBF".to_vec();
    source.extend_from_slice(svg_source(10, 10).as_bytes());
    let document = svg::validate_svg(&source, &limits()).expect("the SVG is valid");
    assert_eq!(document.width, 10);
    assert_eq!(document.height, 10);
}
