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
}

/// `--tts-backend` values: instant stub or the real kokoro pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum TtsBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real synthesis via kokoro on a registry voice/model file.
    Kokoro,
}

/// Chat pipeline: stub unit struct, or file backend via `resolve` slug→path.
///
/// `resolve` is `main::resolve_model` (unknown slug is a loud exit-1); a
/// missing slug with a file backend is a loud exit-1 too.
#[must_use]
pub fn make_llm(
    backend: LlmBackend,
    model: Option<&str>,
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
                eprintln!(
                    "error: --stt-backend audio-cpp requires the `audio-cpp` cargo feature, \
                     which is not compiled into this build (family `{family}` requested); \
                     rebuild with `cargo build -p honkbox-server --features audio-cpp` \
                     after `bun scripts/vendor-patches.ts sync`"
                );
                std::process::exit(1);
            }
            #[cfg(feature = "audio-cpp")]
            {
                let slug = model.unwrap_or_else(|| {
                    eprintln!(
                        "error: --stt-model <slug> is required with --stt-backend audio-cpp"
                    );
                    std::process::exit(1);
                });
                let path = resolve(slug);
                Arc::new(crate::or_exit(honkbox_audio_worker::AudioCppStt::load(&path, family)))
            }
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
    }
}

/// Recognition pipeline (recognition tickets): the rxing barcode backend
/// when this build compiles it (`barcode` feature — default-on, pure-Rust
/// CPU decode), otherwise the stub. No CLI selection in stage 1: barcode
/// decode needs no model, and OCR engines land with their own tickets.
#[must_use]
pub fn make_recognition() -> Arc<dyn honkbox_recognition_worker::RecognitionPipeline> {
    if honkbox_recognition_worker::BARCODE_COMPILED {
        Arc::new(honkbox_recognition_worker::BarcodeRecognition)
    } else {
        Arc::new(honkbox_recognition_worker::StubRecognition)
    }
}
