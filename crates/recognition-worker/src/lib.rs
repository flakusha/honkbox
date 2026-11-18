//! Sync recognition pipelines (barcode/QR decode now; OCR/VLM backends land
//! via `.plan/tickets/TASK-ocr-*.md`).
//!
//! CPU-only, tokio-free, std (+ `serde` for the wire DTOs). `barcode` (default)
//! pulls in `rxing` for QR/1D/2D decode; a stub keeps builds without it
//! honest.

use std::fmt;

#[cfg(feature = "barcode")]
mod barcode;

#[cfg(feature = "barcode")]
pub use barcode::{BarcodeRecognition, RXING_BACKEND_ID, RXING_VERSION};

/// `true` when THIS build compiles the rxing barcode backend (`barcode`
/// feature) — the same compile probe the capability advertisement reports
/// as `features.recognition.barcode`, so the document cannot drift from
/// the binary (`VAD_COMPILED` pattern).
pub const BARCODE_COMPILED: bool = cfg!(feature = "barcode");

/// Which recognition work a request asks for. The vocabulary is closed:
/// kinds without a compiled backend answer the named 501
/// (`recognition_kind_unsupported`) rather than pretending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecKind {
    /// Barcode/QR decode (symbology + payload + corner bbox).
    Barcode,
    /// Document/scene OCR (text, regions, confidence). No compiled backend
    /// yet — see `.plan/tickets/TASK-ocr-ocrs-backend.md` and siblings.
    Ocr,
}

impl RecKind {
    /// Every kind in the wire vocabulary, in stable order.
    pub const ALL: &'static [RecKind] = &[RecKind::Barcode, RecKind::Ocr];

    /// Stable wire name (matches the `api-types` `RecKind` serde spelling).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RecKind::Barcode => "barcode",
            RecKind::Ocr => "ocr",
        }
    }
}

impl fmt::Display for RecKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Worker-internal recognition request (not an API type; the server maps
/// its DTO onto this). `image` is DECODED image-container bytes (PNG/JPEG),
/// never a path or URL — the caller resolves input before this seam.
#[derive(Debug, Clone)]
pub struct RecognizeRequest {
    /// Which recognition work to run.
    pub kind: RecKind,
    /// Image-container bytes (PNG/JPEG/WebP).
    pub image: Vec<u8>,
    /// Language hint (BCP-47); OCR engines use it, barcode decode ignores it.
    pub language: Option<String>,
    /// Free-form mode hint (`text_layer`/`layout_regions`/`full_page`);
    /// barcode decode ignores it.
    pub mode: Option<String>,
    /// Optional engine override (`None` = the configured default backend).
    pub engine: Option<String>,
}

/// One located recognition hit: axis-aligned pixel bbox plus its text.
#[derive(Debug, Clone, PartialEq)]
pub struct RecRegion {
    /// Pixel bbox `[x_min, y_min, x_max, y_max]` in source-image coordinates.
    pub bbox: [f32; 4],
    /// Recognized text for this region (barcode: the full payload).
    pub text: String,
    /// Engine confidence in `0.0..=1.0`; `None` when the engine does not
    /// produce one (rxing does not).
    pub confidence: Option<f32>,
}

/// One page of recognition output. Barcode decode yields exactly one page.
#[derive(Debug, Clone, PartialEq)]
pub struct RecPage {
    /// Full recognized text (regions concatenated in engine order for OCR;
    /// the barcode payload for barcode decode).
    pub text: String,
    /// Located regions; empty when the engine cannot localize (VLM) or
    /// nothing was found.
    pub regions: Vec<RecRegion>,
}

/// Which engine produced a result, and what its output honestly contains.
#[derive(Debug, Clone, PartialEq)]
pub struct RecEngine {
    /// Stable backend id (`"stub-recognition"` | `"rxing"` | ...).
    pub id: &'static str,
    /// Engine/version pin (the DEPENDENCY version this build linked, not a
    /// runtime-reported value); `None` when the backend has no meaningful
    /// version (stub).
    pub version: Option<&'static str>,
    /// `true` only when this engine fills `RecRegion::bbox` with real
    /// coordinates — a client must not treat regions as located otherwise.
    pub bbox_support: bool,
}

/// Recognition result (worker-internal; the server DTO mirrors this shape).
#[derive(Debug, Clone, PartialEq)]
pub struct RecognizeOutput {
    /// The kind that was served.
    pub kind: RecKind,
    /// Producing engine metadata.
    pub engine: RecEngine,
    /// One entry per processed page; `pages: []` is the honest
    /// "nothing recognized" result (never an error) for no-hit inputs.
    pub pages: Vec<RecPage>,
}

/// Errors from [`RecognitionPipeline`] backends (hand-rolled, no thiserror,
/// matching the other workers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecError {
    /// Valid request for a kind this backend does not serve — the server
    /// maps this to the named 501 `recognition_kind_unsupported`.
    Unsupported {
        /// The refused kind.
        kind: RecKind,
        /// The refusing backend id.
        backend: &'static str,
    },
    /// Malformed request (empty image, undecodable container for a backend
    /// that must decode to run) — the server maps this to a named 400.
    Invalid(String),
    /// The engine ran but failed — the server maps this to a named 500.
    Engine(String),
}

impl fmt::Display for RecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { kind, backend } => {
                write!(f, "backend {backend:?} does not serve kind {:?}", kind.as_str())
            }
            Self::Invalid(e) => write!(f, "invalid recognition input: {e}"),
            Self::Engine(e) => write!(f, "recognition engine failure: {e}"),
        }
    }
}

impl std::error::Error for RecError {}

/// Sync recognition pipeline.
///
/// Backend ids: `"stub-recognition"` | `"rxing"` (with the `barcode`
/// feature) | future OCR backends.
pub trait RecognitionPipeline: Send + Sync {
    /// Stable backend id.
    fn backend_id(&self) -> &'static str;
    /// The kinds THIS backend can serve — the honesty source for the
    /// capabilities `recognition` block and the route's named 501.
    fn kinds(&self) -> &'static [RecKind];
    /// Run one recognition request over one image.
    ///
    /// A successful return with `pages: []` means the engine processed the
    /// input and found nothing; failures come back as [`RecError`].
    fn recognize(&self, req: &RecognizeRequest) -> Result<RecognizeOutput, RecError>;
    /// Best-effort cancel of any in-flight recognition on this pipeline.
    ///
    /// Contract (mirrors `LlmPipeline::cancel`):
    /// - returns promptly (atomics only; no locks the worker holds, no join);
    /// - the in-flight `recognize` call returns its normal error path soon
    ///   after (best effort — stage-1 backends are sub-millisecond and may
    ///   simply finish);
    /// - safe to call from any thread, any number of times.
    fn cancel(&self);
}

/// Always-compiling stub: deterministic canned barcode output for tests and
/// stub builds; every other kind is an honest [`RecError::Unsupported`] so
/// capability advertisement can key on [`RecognitionPipeline::kinds`].
#[derive(Debug, Default)]
pub struct StubRecognition;

/// Deterministic stub payload — stable across versions on purpose (test
/// assertions pin it).
pub const STUB_BARCODE_TEXT: &str = "HONKBOX-STUB-BARCODE";

impl RecognitionPipeline for StubRecognition {
    fn backend_id(&self) -> &'static str {
        "stub-recognition"
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
        Ok(RecognizeOutput {
            kind: RecKind::Barcode,
            engine: RecEngine {
                id: self.backend_id(),
                version: None,
                bbox_support: true,
            },
            pages: vec![RecPage {
                text: STUB_BARCODE_TEXT.to_owned(),
                regions: vec![RecRegion {
                    bbox: [0.0, 0.0, 1.0, 1.0],
                    text: STUB_BARCODE_TEXT.to_owned(),
                    confidence: Some(1.0),
                }],
            }],
        })
    }

    fn cancel(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(kind: RecKind) -> RecognizeRequest {
        RecognizeRequest {
            kind,
            image: vec![1, 2, 3],
            language: None,
            mode: None,
            engine: None,
        }
    }

    /// The stub is deterministic: two identical requests produce byte-equal
    /// outputs (client-visible replay shape for the future idempotency key).
    #[test]
    fn stub_output_is_deterministic() {
        let a = StubRecognition.recognize(&req(RecKind::Barcode)).unwrap();
        let b = StubRecognition.recognize(&req(RecKind::Barcode)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.pages.len(), 1);
        assert_eq!(a.pages[0].text, STUB_BARCODE_TEXT);
        assert_eq!(a.engine.id, "stub-recognition");
    }

    /// The stub honestly refuses kinds it does not serve, naming both the
    /// kind and the backend — the shape the route's 501 maps from.
    #[test]
    fn stub_refuses_unserved_kinds_by_name() {
        let err = StubRecognition.recognize(&req(RecKind::Ocr)).unwrap_err();
        match err {
            RecError::Unsupported { kind, backend } => {
                assert_eq!(kind, RecKind::Ocr);
                assert_eq!(backend, "stub-recognition");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// Empty image is a named invalid input, never a success or a panic.
    #[test]
    fn stub_rejects_empty_image() {
        let mut r = req(RecKind::Barcode);
        r.image.clear();
        assert!(matches!(
            StubRecognition.recognize(&r),
            Err(RecError::Invalid(_))
        ));
    }

    /// The probe matches the compiled feature set and the stub's honesty:
    /// without the `barcode` feature the probe is false, yet the vocabulary
    /// still lists Barcode (a kind is wire surface, not availability).
    #[test]
    fn vocabulary_and_probe_stay_stable() {
        assert_eq!(RecKind::ALL, &[RecKind::Barcode, RecKind::Ocr]);
        assert_eq!(RecKind::Barcode.as_str(), "barcode");
        assert_eq!(RecKind::Ocr.as_str(), "ocr");
        assert_eq!(BARCODE_COMPILED, cfg!(feature = "barcode"));
        assert_eq!(StubRecognition.kinds(), &[RecKind::Barcode]);
    }
}
