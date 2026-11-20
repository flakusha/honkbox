//! `ocrs`/`rten` OCR backend (pure Rust, CPU only, `build = false`).
//!
//! User-downloaded `.onnx`/`.rten` text-detection and text-recognition
//! weights are loaded once at construction; honkbox bundles none (the
//! upstream weights are CC-BY-SA-4.0; attribution/share-alike applies to
//! redistributing them, which honkbox never does). Missing or unloadable
//! weights leave the engine absent, so the backend's `serves(Ocr)`
//! and the capability block answer `false` and the route answers the named
//! 501 — never a fake result. Input is the decoded image-container bytes the
//! route supplies; no path or URL is ever resolved here.
//!
//! `ocrs` 0.14 exposes word boxes (`TextWord::bounding_rect`) but no
//! per-word confidence in its public API, so `RecRegion::confidence` is
//! `None` (the honest shape, as with rxing) rather than an invented score.
//! The default alphabet is Latin; no multilingual claim is made.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use ocrs::{ImageSource, OcrEngine, OcrEngineParams, TextItem};
use rten::Model;

use super::{
    RecEngine, RecError, RecKind, RecPage, RecRegion, RecognitionPipeline, RecognizeOutput,
    RecognizeRequest,
};

/// Stable backend id (sd-core `SD_CPP_BACKEND_ID` pattern).
pub const OCRS_BACKEND_ID: &str = "ocrs-rten";

/// The linked `ocrs` DEPENDENCY version this build compiles (Cargo.toml pin).
pub const OCRS_VERSION: &str = "0.14.0";

/// The linked `rten` DEPENDENCY version this build compiles (Cargo.toml pin).
pub const RTEN_VERSION: &str = "0.28.0";

/// Latin-script OCR via `ocrs`; the engine exists only when both weights load.
pub struct OcrRecognition {
    engine: Option<OcrEngine>,
    cancelled: AtomicBool,
}

impl std::fmt::Debug for OcrRecognition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OcrRecognition")
            .field("available", &self.engine.is_some())
            .finish_non_exhaustive()
    }
}

impl OcrRecognition {
    /// Build from already-located model files. Either missing weight yields
    /// an absent engine (honest `serves(Ocr) == false`), never a stub.
    #[must_use]
    pub fn new(detection_model: Option<&Path>, recognition_model: Option<&Path>) -> Self {
        let engine = detection_model
            .zip(recognition_model)
            .and_then(|(detection, recognition)| {
                let detection = Model::load_file(detection).ok()?;
                let recognition = Model::load_file(recognition).ok()?;
                OcrEngine::new(OcrEngineParams {
                    detection_model: Some(detection),
                    recognition_model: Some(recognition),
                    ..OcrEngineParams::default()
                })
                .ok()
            });
        Self {
            engine,
            cancelled: AtomicBool::new(false),
        }
    }

    /// `true` when both models loaded and OCR can actually be served.
    #[must_use]
    pub fn available(&self) -> bool {
        self.engine.is_some()
    }

    /// Best-effort stop: checked between OCR stages.
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

impl RecognitionPipeline for OcrRecognition {
    fn backend_id(&self) -> &'static str {
        OCRS_BACKEND_ID
    }

    fn kinds(&self) -> &'static [RecKind] {
        &[RecKind::Ocr]
    }

    /// Runtime availability, not just the wire vocabulary: no weights ⇒ no
    /// OCR capability, and the route's first gate names this engine in its 501.
    fn serves(&self, kind: RecKind) -> bool {
        kind == RecKind::Ocr && self.available()
    }

    fn recognize(&self, req: &RecognizeRequest) -> Result<RecognizeOutput, RecError> {
        if req.kind != RecKind::Ocr {
            return Err(RecError::Unsupported {
                kind: req.kind,
                backend: self.backend_id(),
            });
        }
        let Some(engine) = self.engine.as_ref() else {
            return Err(RecError::Unsupported {
                kind: RecKind::Ocr,
                backend: self.backend_id(),
            });
        };
        if req.image.is_empty() {
            return Err(RecError::Invalid("image input is empty".to_owned()));
        }
        self.cancelled.store(false, Ordering::Relaxed);

        let img = image::load_from_memory(&req.image)
            .map_err(|e| RecError::Invalid(format!("image container did not decode: {e}")))?
            .into_rgb8();
        let source = ImageSource::from_bytes(img.as_raw(), img.dimensions())
            .map_err(|e| RecError::Invalid(format!("image pixels unsupported: {e}")))?;
        let input = engine
            .prepare_input(source)
            .map_err(|e| RecError::Engine(format!("prepare input: {e}")))?;
        let words = engine
            .detect_words(&input)
            .map_err(|e| RecError::Engine(format!("detect words: {e}")))?;
        if self.cancelled() {
            return Err(RecError::Engine("cancelled".to_owned()));
        }
        let lines = engine.find_text_lines(&input, &words);
        let recognized = engine
            .recognize_text(&input, &lines)
            .map_err(|e| RecError::Engine(format!("recognize text: {e}")))?;
        if self.cancelled() {
            return Err(RecError::Engine("cancelled".to_owned()));
        }

        let mut regions = Vec::new();
        let mut text_lines = Vec::new();
        for line in recognized.into_iter().flatten() {
            text_lines.push(line.to_string());
            for word in line.words() {
                let rect = word.bounding_rect();
                regions.push(RecRegion {
                    bbox: [
                        rect.left() as f32,
                        rect.top() as f32,
                        rect.right() as f32,
                        rect.bottom() as f32,
                    ],
                    text: word.to_string(),
                    // ocrs 0.14 does not expose a recognition score.
                    confidence: None,
                });
            }
        }
        let pages = if regions.is_empty() {
            Vec::new()
        } else {
            vec![RecPage {
                text: text_lines.join("\n"),
                regions,
            }]
        };
        Ok(RecognizeOutput {
            kind: RecKind::Ocr,
            engine: RecEngine {
                id: OCRS_BACKEND_ID,
                version: Some(OCRS_VERSION),
                bbox_support: true,
            },
            pages,
        })
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(kind: RecKind) -> RecognizeRequest {
        RecognizeRequest {
            kind,
            // No weights in tests: this exercises the absent-engine path
            // without shipping model files (honest capability/501 seam).
            image: vec![1, 2, 3],
            language: None,
            mode: None,
            engine: None,
        }
    }

    #[test]
    fn missing_weights_are_honestly_unserved() {
        let backend = OcrRecognition::new(None, None);
        assert!(!backend.available());
        assert!(!backend.serves(RecKind::Ocr));
        assert!(!backend.serves(RecKind::Barcode));
        assert_eq!(backend.backend_id(), "ocrs-rten");
        assert_eq!(backend.kinds(), &[RecKind::Ocr]);
        assert_eq!(OCRS_VERSION, "0.14.0");
        assert_eq!(RTEN_VERSION, "0.28.0");
    }

    #[test]
    fn missing_weights_name_the_engine_in_the_unsupported_error() {
        let err = OcrRecognition::new(None, None)
            .recognize(&req(RecKind::Ocr))
            .unwrap_err();
        match err {
            RecError::Unsupported { kind, backend } => {
                assert_eq!(kind, RecKind::Ocr);
                assert_eq!(backend, "ocrs-rten");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn unloaded_backend_refuses_barcode_by_name() {
        let err = OcrRecognition::new(None, None)
            .recognize(&req(RecKind::Barcode))
            .unwrap_err();
        assert!(matches!(
            err,
            RecError::Unsupported {
                kind: RecKind::Barcode,
                ..
            }
        ));
    }

    /// A tiny in-test PNG proves the container-decode path accepts a valid
    /// image even when no weights are present (the engine gate still fires
    /// first, so this never fabricates a result).
    #[test]
    fn tiny_generated_image_decodes_without_panicking() {
        use std::io::Cursor;
        let img =
            image::RgbImage::from_fn(8, 8, |x, y| image::Rgb([x as u8 * 31, y as u8 * 31, 0]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode PNG");
        let rgb = image::load_from_memory(&png).expect("decode PNG").to_rgb8();
        let source = ImageSource::from_bytes(rgb.as_raw(), rgb.dimensions());
        assert!(source.is_ok());
    }
}
