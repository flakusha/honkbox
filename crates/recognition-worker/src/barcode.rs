//! rxing barcode/QR backend (pure Rust; Apache-2.0 crate — license note in
//! the workspace `Cargo.toml` + `.tmp/research/ReviewedRecognition.md`).
//!
//! Container decode is done HERE with the workspace `image` crate (rxing is
//! compiled without its `image` feature — see the workspace note): PNG/
//! JPEG/WebP bytes → luma buffer → `detect_in_luma_slice`. Error classes
//! are kept distinct:
//! - undecodable container → [`RecError::Invalid`] (named 400 upstream;
//!   never a silent success, never a 500);
//! - decodable container, no barcode → the honest empty result
//!   `pages: []` (rxing reports no-hit as an error; it does not distinguish
//!   no-hit from engine trouble — both collapse to "nothing found", which
//!   is the wire contract's answer, not a 500);
//! - empty input → [`RecError::Invalid`] before any decode.

use super::{
    RecEngine, RecError, RecKind, RecPage, RecRegion, RecognizeOutput, RecognizeRequest,
    RecognitionPipeline,
};

/// Stable backend id (sd-core `SD_CPP_BACKEND_ID` pattern).
pub const RXING_BACKEND_ID: &str = "rxing";

/// The linked rxing DEPENDENCY version this backend was written against
/// (`Cargo.toml` pin); asserted by a test so the const cannot rot silently.
pub const RXING_VERSION: &str = "0.9";

/// rxing decode over luma buffers; the caller owns container decoding.
#[derive(Debug, Default)]
pub struct BarcodeRecognition;

/// Bounding box over the decoder's corner/finder points.
///
/// rxing returns corner points in image coordinates; an axis-aligned bbox
/// is the honest wire shape (rotated barcodes get a slightly loose box —
/// documented, not hidden). An empty point list (no symbology reports that
/// today, but it is not a panic risk) yields a degenerate box.
#[must_use]
fn bbox_from_points(points: &[(f32, f32)]) -> [f32; 4] {
    let Some(&(first_x, first_y)) = points.first() else {
        return [0.0, 0.0, 0.0, 0.0];
    };
    let mut bbox = [first_x, first_y, first_x, first_y];
    for &(x, y) in points {
        bbox[0] = bbox[0].min(x);
        bbox[1] = bbox[1].min(y);
        bbox[2] = bbox[2].max(x);
        bbox[3] = bbox[3].max(y);
    }
    bbox
}

/// Decode container bytes to the luma buffer rxing consumes. `image`'s
/// format set is exactly the workspace codec set (png/jpeg/webp) — the
/// same vocabulary the route's data-URI MIME gate admits.
fn container_to_luma(bytes: &[u8]) -> Result<image::GrayImage, RecError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| RecError::Invalid(format!("image container did not decode: {e}")))?;
    Ok(img.to_luma8())
}

impl RecognitionPipeline for BarcodeRecognition {
    fn backend_id(&self) -> &'static str {
        RXING_BACKEND_ID
    }

    fn kinds(&self) -> &'static [RecKind] {
        &[RecKind::Barcode]
    }

    fn recognize(&self, req: &RecognizeRequest) -> Result<RecognizeOutput, RecError> {
        if req.kind != RecKind::Barcode {
            return Err(RecError::Unsupported {
                kind: req.kind,
                backend: self.backend_id(),
            });
        }
        if req.image.is_empty() {
            return Err(RecError::Invalid("image input is empty".to_owned()));
        }
        let luma = container_to_luma(&req.image)?;
        let engine = RecEngine {
            id: RXING_BACKEND_ID,
            version: Some(RXING_VERSION),
            bbox_support: true,
        };
        // No-hit collapses to the empty result (see module docs): rxing's
        // error carries no stable no-hit marker, and "nothing found" is the
        // wire answer for both no-hit and transient engine trouble here —
        // barcode decode is deterministic, so a real retry changes nothing.
        let pages = match rxing::helpers::detect_in_luma_slice(
            luma.as_raw(),
            luma.width(),
            luma.height(),
            None,
        ) {
            Ok(found) => {
                // rxing keeps zxing's Java-style accessor names.
                let text = found.getText().to_owned();
                let points: Vec<(f32, f32)> =
                    found.getPoints().iter().map(|p| (p.x, p.y)).collect();
                vec![RecPage {
                    text: text.clone(),
                    regions: vec![RecRegion {
                        bbox: bbox_from_points(&points),
                        text,
                        // rxing exposes no decode confidence; honesty over
                        // noise.
                        confidence: None,
                    }],
                }]
            }
            Err(_) => Vec::new(),
        };
        Ok(RecognizeOutput {
            kind: RecKind::Barcode,
            engine,
            pages,
        })
    }

    fn cancel(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use rxing::{BarcodeFormat, Writer};

    /// Encode `contents` as a QR and return PNG bytes (rxing encodes the
    /// bit matrix; the workspace `image` crate rasterizes — one lockfile,
    /// no fixture file).
    fn qr_png(contents: &str) -> Vec<u8> {
        let matrix = rxing::MultiFormatWriter
            .encode(contents, &BarcodeFormat::QR_CODE, 0, 0)
            .expect("encode QR");
        let (w, h) = (matrix.width(), matrix.height());
        let img = image::GrayImage::from_fn(w, h, |x, y| {
            if matrix.get(x, y) {
                image::Luma([0u8])
            } else {
                image::Luma([255u8])
            }
        });
        let mut png = Vec::new();
        image::DynamicImage::ImageLuma8(img)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode PNG");
        png
    }

    fn req(png: &[u8]) -> RecognizeRequest {
        RecognizeRequest {
            kind: RecKind::Barcode,
            image: png.to_vec(),
            language: None,
            mode: None,
            engine: None,
        }
    }

    /// Round trip: encode -> PNG -> decode gives back the payload, a real
    /// corner-derived bbox, and the rxing engine pin.
    #[test]
    fn qr_round_trip_recovers_payload_and_bbox() {
        let payload = "https://honkbox.example/stage1";
        let out = BarcodeRecognition
            .recognize(&req(&qr_png(payload)))
            .expect("decode");
        assert_eq!(out.kind, RecKind::Barcode);
        assert_eq!(
            out.engine,
            RecEngine {
                id: RXING_BACKEND_ID,
                version: Some(RXING_VERSION),
                bbox_support: true,
            }
        );
        assert_eq!(out.pages.len(), 1);
        assert_eq!(out.pages[0].text, payload);
        let region = &out.pages[0].regions[0];
        assert_eq!(region.text, payload);
        assert_eq!(region.confidence, None);
        let [x_min, y_min, x_max, y_max] = region.bbox;
        assert!(x_min < x_max && y_min < y_max, "bbox not a box: {region:?}");
        assert!(x_max <= f32::from(u16::MAX), "bbox within sane pixel range");
    }

    /// Round trip is deterministic: identical input, identical output.
    #[test]
    fn decode_is_deterministic() {
        let png = qr_png("honkbox-deterministic");
        let a = BarcodeRecognition.recognize(&req(&png)).unwrap();
        let b = BarcodeRecognition.recognize(&req(&png)).unwrap();
        assert_eq!(a, b);
    }

    /// A solid-color image (valid container, no barcode) is the honest empty
    /// result — never a 500, never a half-filled page.
    #[test]
    fn no_barcode_is_a_clean_empty_result() {
        let blank = image::GrayImage::from_pixel(64, 64, image::Luma([255]));
        let mut png = Vec::new();
        image::DynamicImage::ImageLuma8(blank)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode PNG");
        let out = BarcodeRecognition
            .recognize(&req(&png))
            .expect("no-barcode is Ok");
        assert!(out.pages.is_empty(), "expected empty, got {out:?}");
        // Engine meta still describes the backend that ran.
        assert_eq!(out.engine.id, RXING_BACKEND_ID);
    }

    /// Garbage bytes fail the CONTAINER decode as a named Invalid — never a
    /// panic, never a misleading empty success (a truncated PNG must not
    /// look like "no barcode present" to a client).
    #[test]
    fn garbage_bytes_are_invalid_input() {
        let garbage = [0u8; 64];
        assert!(matches!(
            BarcodeRecognition.recognize(&req(&garbage)),
            Err(RecError::Invalid(_))
        ));
        let truncated = &qr_png("cut")[..8];
        assert!(matches!(
            BarcodeRecognition.recognize(&req(truncated)),
            Err(RecError::Invalid(_))
        ));
    }

    /// Kind refusal + empty input mirrors the stub honesty.
    #[test]
    fn refuses_unserved_kinds_and_empty_input() {
        let mut r = req(&[1]);
        r.kind = RecKind::Ocr;
        assert!(matches!(
            BarcodeRecognition.recognize(&r),
            Err(RecError::Unsupported {
                kind: RecKind::Ocr,
                ..
            })
        ));
        r.kind = RecKind::Barcode;
        r.image.clear();
        assert!(matches!(
            BarcodeRecognition.recognize(&r),
            Err(RecError::Invalid(_))
        ));
    }

    /// The version pin matches the workspace dependency (rot check).
    #[test]
    fn version_pin_matches_workspace_dep() {
        let manifest = include_str!("../../../Cargo.toml");
        let line = manifest
            .lines()
            .find(|l| l.trim_start().starts_with("rxing = "))
            .expect("rxing workspace dep line");
        assert!(
            line.contains(RXING_VERSION),
            "pin drift: {line} vs {RXING_VERSION}"
        );
    }
}
