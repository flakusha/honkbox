//! Backend constructors: stub zero-cost vs file backends via registry slug.
//!
//! Every worker touch-point lives here so the frozen-contract wiring stays in
//! one place. Worker crates keep their own DTOs (no `api-types` reuse), so
//! this module converts at the boundary in both directions.

use std::sync::Arc;

use honkbox_audio_worker::{KokoroTts, SttPipeline, StubStt, StubTts, TtsPipeline, WhisperStt};
use honkbox_llm_worker::{LlamaCppLlm, LlmOptions, LlmPipeline, StubLlm};

/// `--llm-backend` values: instant stub or the real llama.cpp pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum LlmBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real inference via llama.cpp on a registry GGUF.
    LlamaCpp,
}

/// `--stt-backend` values: instant stub or the real whisper pipeline.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum SttBackend {
    /// Instant deterministic stub (no model, no GPU).
    #[default]
    Stub,
    /// Real transcription via whisper on a registry model.
    Whisper,
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
) -> Arc<dyn LlmPipeline> {
    match backend {
        LlmBackend::Stub => Arc::new(StubLlm::default()),
        LlmBackend::LlamaCpp => {
            let slug = model.unwrap_or_else(|| {
                eprintln!("error: --llm-model <slug> is required with --llm-backend llama-cpp");
                std::process::exit(1);
            });
            let path = resolve(slug);
            Arc::new(crate::or_exit(LlamaCppLlm::load(
                &path,
                &LlmOptions::default(),
            )))
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
#[must_use]
pub fn make_stt(
    backend: SttBackend,
    model: Option<&str>,
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
