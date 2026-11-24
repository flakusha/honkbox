//! Backend constructors: stub zero-cost vs file backends via registry slug.
//!
//! Every worker touch-point lives here so the frozen-contract wiring stays in
//! one place. Worker crates keep their own DTOs (no `api-types` reuse), so
//! this module converts at the boundary in both directions.

use std::sync::Arc;

use honkbox_audio_worker::{KokoroTts, SttPipeline, StubStt, StubTts, TtsPipeline, WhisperStt};
use honkbox_llm_worker::{LlamaCppLlm, LlmPipeline, StubLlm};

/// `--llm-backend` values: instant stub or the real llama.cpp pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum LlmBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real inference via llama.cpp on a registry GGUF.
    LlamaCpp,
    /// Consume an already-running, health-proved `llama-server` over loopback.
    /// The endpoint comes from `[tools] llama_server_url`; no process is started.
    Attached,
    /// Experimental direct-Vulkan placeholder (slice 8 of the ash lane).
    /// Requires the `ash-llm` cargo feature. Construction
    /// selects a Vulkan device (loud named exit-1 on failure — never a
    /// silent stub fallback); every completion then returns the named
    /// unimplemented error. No model slug needed.
    #[cfg(feature = "ash-llm")]
    AshLlm,
}

/// `--stt-backend` values: instant stub or the real pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum SttBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real transcription via whisper on a registry model.
    Whisper,
    /// Real transcription via the audio.cpp engine (TASK-audiocpp-stt-provider).
    /// Requires the server's `audio-cpp` cargo feature; without it the
    /// constructor arm is a loud named error (honest absence, same posture
    /// as `controlnet_available` on the SD side).
    AudioCpp,
    /// Experimental direct-Vulkan placeholder (slice 15 of the ash lane).
    /// Requires the `ash-stt` cargo feature. Construction
    /// selects a Vulkan device (loud named exit-1 on failure — never a
    /// silent stub fallback); every transcription then returns the named
    /// unimplemented error. No model slug needed.
    #[cfg(feature = "ash-stt")]
    AshStt,
}

/// `--tts-backend` values: instant stub or the real kokoro pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum TtsBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real synthesis via kokoro on a registry voice/model file.
    Kokoro,
    /// Experimental direct-Vulkan placeholder (slice 15 of the ash lane).
    /// Requires the `ash-tts` cargo feature. Construction
    /// selects a Vulkan device (loud named exit-1 on failure — never a
    /// silent stub fallback); every synthesis then returns the named
    /// unimplemented error. No model slug needed.
    #[cfg(feature = "ash-tts")]
    AshTts,
}

/// Chat pipeline: stub, file backend via `resolve` slug→path, or an attached
/// `llama-server` proved over loopback HTTP.
///
/// `resolve` is `main::resolve_model` (unknown slug is a loud exit-1); a
/// missing slug with a file backend is a loud exit-1 too. `attach_url` is
/// config-only and required by the `Attached` arm.
#[must_use]
pub fn make_llm(
    backend: LlmBackend,
    model: Option<&str>,
    attach_url: Option<&str>,
    resolve: &dyn Fn(&str) -> std::path::PathBuf,
    cache: crate::config::CacheConfig,
    vram_budget_bytes: Option<u64>,
) -> Arc<dyn LlmPipeline> {
    match backend {
        LlmBackend::Stub => Arc::new(StubLlm::default()),
        LlmBackend::LlamaCpp => {
            let slug = model.unwrap_or_else(|| {
                eprintln!("error: --llm-model <slug> is required with --llm-backend llama-cpp");
                std::process::exit(1);
            });
            let path = resolve(slug);
            // The cache knobs: the declared ones, else the backend defaults;
            // the VRAM ceiling clamps the KV cache (see `swap::llm_options`).
            let mut opts = crate::swap::llm_options(cache, vram_budget_bytes);
            // Vision chat (REQ-8): a sibling mmproj projector (HF
            // `mmproj*.gguf` convention) upgrades the static pipeline to
            // multimodal — when the build compiles the mtmd path at all
            // (`llm-vision` feature; the catalog mirrors the same flag so
            // `/v1/models` never advertises vision this build cannot
            // serve). None means text-only.
            opts.mmproj_path = honkbox_llm_worker::VISION_COMPILED
                .then(|| honkbox_registry::sibling_mmproj(&path))
                .flatten();
            Arc::new(crate::or_exit(LlamaCppLlm::load(&path, &opts)))
        }
        LlmBackend::Attached => {
            let url = attach_url.unwrap_or_else(|| {
                eprintln!(
                    "error: [tools] llama_server_url is required with --llm-backend attached"
                );
                std::process::exit(1);
            });
            Arc::new(crate::or_exit(crate::llm_attached::AttachedLlm::new(
                url, model,
            )))
        }
        #[cfg(feature = "ash-llm")]
        LlmBackend::AshLlm => {
            // Device selection happens now: a Vulkan-less environment is a
            // loud named exit-1 (see `or_exit`), not a degraded stub. The
            // pipeline itself is the ash-lane placeholder — every verb
            // returns the named unimplemented error, no fake output.
            let device = Arc::new(crate::or_exit(
                honkbox_ash_backend::device::VulkanDevice::headless(),
            ));
            Arc::new(honkbox_ash_backend::llm::AshLlmPipeline::new(device))
        }
    }
}

/// `--sd-backend` values: instant stub or the real diffusion-rs pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum SdBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real txt2img via stable-diffusion.cpp on a registry checkpoint.
    SdCpp,
    /// Experimental direct-Vulkan pipeline (sampling not implemented yet).
    #[cfg(feature = "ash-sd")]
    AshSd,
}
/// Image pipeline: stub unit struct, or file backend via `resolve` slug→path.
#[must_use]
pub fn make_sd(
    backend: SdBackend,
    model: Option<&str>,
    resolve: &dyn Fn(&str) -> std::path::PathBuf,
) -> Arc<dyn honkbox_sd_core::SdPipeline> {
    match backend {
        SdBackend::Stub => Arc::new(honkbox_sd_core::StubPipeline),
        SdBackend::SdCpp => {
            let slug = model.unwrap_or_else(|| {
                eprintln!("error: --sd-model <slug> is required with --sd-backend sd-cpp");
                std::process::exit(1);
            });
            let path = resolve(slug);
            Arc::new(honkbox_sd_core::SdCppPipeline::new(path))
        }
        #[cfg(feature = "ash-sd")]
        SdBackend::AshSd => {
            let slug = model.unwrap_or_else(|| {
                eprintln!("error: --sd-model <slug> is required with --sd-backend ash-sd");
                std::process::exit(1);
            });
            let device = Arc::new(crate::or_exit(
                honkbox_ash_backend::device::VulkanDevice::headless(),
            ));
            Arc::new(honkbox_ash_backend::sd::AshSdPipeline::new(
                device,
                resolve(slug),
            ))
        }
    }
}

/// Transcription pipeline: stub unit struct, or file backend via slug→path.
/// `family` selects the audio.cpp family for `SttBackend::AudioCpp` (ignored
/// by the other arms).
#[must_use]
pub fn make_stt(
    backend: SttBackend,
    model: Option<&str>,
    family: Option<&str>,
    resolve: &dyn Fn(&str) -> std::path::PathBuf,
) -> Arc<dyn SttPipeline> {
    match backend {
        SttBackend::Stub => Arc::new(StubStt),
        SttBackend::Whisper => {
            let slug = model.unwrap_or_else(|| {
                eprintln!("error: --stt-model <slug> is required with --stt-backend whisper");
                std::process::exit(1);
            });
            let path = resolve(slug);
            Arc::new(crate::or_exit(WhisperStt::load(&path)))
        }
        SttBackend::AudioCpp => {
            let family = family.unwrap_or(honkbox_audio_worker::AUDIOCPP_DEFAULT_FAMILY);
            #[cfg(not(feature = "audio-cpp"))]
            {
                // Name the families THIS build actually has (none without
                // the engine) instead of leaving the request hanging.
                let compiled = if honkbox_audio_worker::AUDIOCPP_STT_FAMILIES.is_empty() {
                    "none".to_owned()
                } else {
                    honkbox_audio_worker::AUDIOCPP_STT_FAMILIES.join(", ")
                };
                eprintln!(
                    "error: --stt-backend audio-cpp requires the `audio-cpp` cargo feature, \
                     which is not compiled into this build (family `{family}` requested; \
                     compiled families: {compiled} — enable `audio-cpp` + \
                     `audio-cpp-qwen3-asr` for `qwen3_asr`); rebuild with \
                     `cargo build -p honkbox-server --features audio-cpp` \
                     after `bun scripts/vendor-patches.ts sync`"
                );
                std::process::exit(1);
            }
            #[cfg(feature = "audio-cpp")]
            {
                let slug = model.unwrap_or_else(|| {
                    eprintln!("error: --stt-model <slug> is required with --stt-backend audio-cpp");
                    std::process::exit(1);
                });
                let path = resolve(slug);
                Arc::new(crate::or_exit(honkbox_audio_worker::AudioCppStt::load(
                    &path, family,
                )))
            }
        }
        #[cfg(feature = "ash-stt")]
        SttBackend::AshStt => {
            // Device selection happens now: a Vulkan-less environment is a
            // loud named exit-1 (see `or_exit`), not a degraded stub. The
            // pipeline itself is the ash-lane placeholder — every verb
            // returns the named unimplemented error, no fake transcript.
            let device = Arc::new(crate::or_exit(
                honkbox_ash_backend::device::VulkanDevice::headless(),
            ));
            Arc::new(honkbox_ash_backend::stubs::AshSttPipeline::new(device))
        }
    }
}

/// Synthesis pipeline: stub unit struct, or file backend via slug→path.
#[must_use]
pub fn make_tts(
    backend: TtsBackend,
    model: Option<&str>,
    resolve: &dyn Fn(&str) -> std::path::PathBuf,
) -> Arc<dyn TtsPipeline> {
    match backend {
        TtsBackend::Stub => Arc::new(StubTts),
        TtsBackend::Kokoro => {
            let slug = model.unwrap_or_else(|| {
                eprintln!("error: --tts-model <slug> is required with --tts-backend kokoro");
                std::process::exit(1);
            });
            let path = resolve(slug);
            Arc::new(crate::or_exit(KokoroTts::load(&path)))
        }
        #[cfg(feature = "ash-tts")]
        TtsBackend::AshTts => {
            // Device selection happens now: a Vulkan-less environment is a
            // loud named exit-1 (see `or_exit`), not a degraded stub. The
            // pipeline itself is the ash-lane placeholder — every verb
            // returns the named unimplemented error, no fake audio.
            let device = Arc::new(crate::or_exit(
                honkbox_ash_backend::device::VulkanDevice::headless(),
            ));
            Arc::new(honkbox_ash_backend::stubs::AshTtsPipeline::new(device))
        }
    }
}

/// Recognition pipeline, feature-off build: the rxing barcode backend when
/// compiled, otherwise the stub. OCR is not in the wire vocabulary of this
/// build (no `ocrs-rten` engine exists here).
#[cfg(not(feature = "ocr-ocrs"))]
#[must_use]
pub fn make_recognition(
    _found: &[honkbox_registry::FoundModel],
) -> Arc<dyn honkbox_recognition_worker::RecognitionPipeline> {
    if honkbox_recognition_worker::BARCODE_COMPILED {
        Arc::new(honkbox_recognition_worker::BarcodeRecognition)
    } else {
        Arc::new(honkbox_recognition_worker::StubRecognition)
    }
}

/// Dispatch by kind so enabling OCR never takes barcode away. `backend_id`
/// names `ocrs-rten` whenever that engine is compiled: the route's first-gate
/// 501 for absent OCR weights must name the missing engine, not rxing.
#[cfg(feature = "ocr-ocrs")]
struct RecognitionRouter {
    ocr: Arc<dyn honkbox_recognition_worker::RecognitionPipeline>,
    barcode: Option<Arc<dyn honkbox_recognition_worker::RecognitionPipeline>>,
    id: &'static str,
}

#[cfg(feature = "ocr-ocrs")]
impl honkbox_recognition_worker::RecognitionPipeline for RecognitionRouter {
    fn backend_id(&self) -> &'static str {
        self.id
    }

    fn backend_id_for(&self, kind: honkbox_recognition_worker::RecKind) -> &'static str {
        use honkbox_recognition_worker::RecKind;
        match kind {
            RecKind::Barcode => self
                .barcode
                .as_ref()
                .map_or(self.id, |backend| backend.backend_id_for(kind)),
            RecKind::Ocr => self.ocr.backend_id_for(kind),
        }
    }

    fn kinds(&self) -> &'static [honkbox_recognition_worker::RecKind] {
        &[
            honkbox_recognition_worker::RecKind::Barcode,
            honkbox_recognition_worker::RecKind::Ocr,
        ]
    }

    fn serves(&self, kind: honkbox_recognition_worker::RecKind) -> bool {
        match kind {
            honkbox_recognition_worker::RecKind::Barcode => {
                self.barcode.as_ref().is_some_and(|b| b.serves(kind))
            }
            honkbox_recognition_worker::RecKind::Ocr => self.ocr.serves(kind),
        }
    }

    fn recognize(
        &self,
        req: &honkbox_recognition_worker::RecognizeRequest,
    ) -> Result<honkbox_recognition_worker::RecognizeOutput, honkbox_recognition_worker::RecError>
    {
        use honkbox_recognition_worker::{RecError, RecKind};
        match req.kind {
            RecKind::Barcode => self.barcode.as_ref().map_or_else(
                || {
                    Err(RecError::Unsupported {
                        kind: req.kind,
                        backend: self.id,
                    })
                },
                |b| b.recognize(req),
            ),
            RecKind::Ocr => self.ocr.recognize(req),
        }
    }

    fn cancel(&self) {
        self.ocr.cancel();
        if let Some(barcode) = &self.barcode {
            barcode.cancel();
        }
    }
}

/// Recognition pipeline, `ocr-ocrs` build: locate user-downloaded OCR
/// weights in the scan result and construct the engine. Missing weights
/// leave it absent — `serves(Ocr)` is false and the named 501 follows from
/// the same predicate. Barcode dispatch is preserved when compiled.
#[cfg(feature = "ocr-ocrs")]
#[must_use]
pub fn make_recognition(
    found: &[honkbox_registry::FoundModel],
) -> Arc<dyn honkbox_recognition_worker::RecognitionPipeline> {
    use honkbox_recognition_worker::{
        BarcodeRecognition, OCRS_BACKEND_ID, OcrRecognition, RecognitionPipeline,
    };
    use honkbox_registry::RootKind;

    let mut detection = None;
    let mut recognition = None;
    // `found` is sorted by slug; `or` picks the first match deterministically.
    for entry in found.iter().filter(|e| e.root == RootKind::Ocr) {
        let name = entry
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if name.starts_with("text-detection") {
            detection = detection.or(Some(entry.path.as_path()));
        } else if name.starts_with("text-rec") {
            recognition = recognition.or(Some(entry.path.as_path()));
        }
    }
    let ocr = Arc::new(OcrRecognition::new(detection, recognition));
    let barcode: Option<Arc<dyn RecognitionPipeline>> =
        if honkbox_recognition_worker::BARCODE_COMPILED {
            Some(Arc::new(BarcodeRecognition))
        } else {
            None
        };
    Arc::new(RecognitionRouter {
        ocr,
        barcode,
        id: OCRS_BACKEND_ID,
    })
}

#[cfg(all(test, feature = "ocr-ocrs"))]
mod ocr_tests {
    use super::*;
    use honkbox_recognition_worker::{BARCODE_COMPILED, RecError, RecKind, RecognizeRequest};

    #[test]
    fn absent_weights_are_unserved_and_name_the_engine() {
        let pipeline = make_recognition(&[]);
        assert_eq!(pipeline.backend_id(), "ocrs-rten");
        assert!(!pipeline.serves(RecKind::Ocr));
        // Feature-on keeps barcode when the barcode backend is compiled.
        assert_eq!(pipeline.serves(RecKind::Barcode), BARCODE_COMPILED);
        let req = RecognizeRequest {
            kind: RecKind::Ocr,
            image: vec![1, 2, 3],
            language: None,
            mode: None,
            engine: None,
        };
        match pipeline.recognize(&req).unwrap_err() {
            RecError::Unsupported { kind, backend } => {
                assert_eq!(kind, RecKind::Ocr);
                assert_eq!(backend, "ocrs-rten");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn kind_specific_backend_ids_stay_distinct() {
        let pipeline = make_recognition(&[]);
        assert_eq!(pipeline.backend_id_for(RecKind::Ocr), "ocrs-rten");
        if BARCODE_COMPILED {
            assert_eq!(pipeline.backend_id_for(RecKind::Barcode), "rxing");
        }
    }
}
