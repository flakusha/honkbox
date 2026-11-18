//! audio.cpp STT backend (`audio-cpp` feature): [`AudioCppStt`] over the
//! vendored engine.
//!
//! The `audio-cpp` crate (Apache-2.0, `launcher-rs/audio-cpp-rs`) wraps the
//! vendored audio.cpp engine (`audio-cpp-sys` → `CMake` `engine_runtime` +
//! ggml 0.12.0; pinned in `vendor.manifest.toml`). Family set follows the
//! cargo features: `audio-cpp` compiles the engine core + bundled
//! silero/marblenet VAD; `audio-cpp-qwen3-asr` adds the default ASR loader
//! (`qwen3_asr`, eSpeak-free). Unknown families are a loud named error at
//! load — honest absence, never a silent fallback (see
//! `Registry::supports_family`).
//!
//! `translate` is NOT supported here: whisper.cpp owns `/v1/audio/
//! translations` in this build (audio.cpp translate-capable families are a
//! per-family decision for a later ticket).

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use audio_cpp::options::request as opts;
use audio_cpp::{Backend, Model, ModelFamily, Registry, Request, RunMode, TaskKind, WavAudio};

use super::{
    resample_linear_f32, SttError, SttPipeline, Transcript, TranscriptSegment, TranscribeRequest,
};

/// Whisper.cpp owns 16 kHz mono; audio.cpp ASR families expect the same.
const TARGET_RATE: u32 = 16_000;

/// Stable base backend id (the transcript carries `audiocpp-stt:<family>`).
pub const BACKEND_ID: &str = "audiocpp-stt";

/// audio.cpp STT pipeline. `Send + Sync` via an internal mutex — the binding
/// types are deliberately `Send + !Sync` (C handles are not thread-safe;
/// separate handles may run concurrently), so one mutex-guarded model is the
/// same shape as [`super::WhisperStt`]'s context.
pub struct AudioCppStt {
    model: Mutex<Model>,
    family: String,
}

impl std::fmt::Debug for AudioCppStt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCppStt").field("family", &self.family).finish_non_exhaustive()
    }
}

impl AudioCppStt {
    /// Load audio.cpp weights from `model_path` for `family` (e.g.
    /// `"qwen3_asr"`). The family is validated against the engine registry
    /// BEFORE load: an uncompiled family is a named [`SttError::Model`]
    /// listing what this build actually has.
    pub fn load(model_path: &Path, family: &str) -> Result<Self, SttError> {
        if !model_path.is_file() {
            return Err(SttError::Io(format!(
                "model not found: {}",
                model_path.display()
            )));
        }
        let registry =
            Registry::new().map_err(|e| SttError::Model(format!("registry init failed: {e}")))?;
        let compiled = registry
            .families()
            .map_err(|e| SttError::Model(format!("registry family probe failed: {e}")))?;
        if !registry.supports_family(family) {
            return Err(SttError::Model(format!(
                "audio.cpp family `{family}` is not compiled into this build \
                 (compiled families: {}); enable the matching `audio-cpp-*` \
                 cargo feature or pick another family",
                compiled.join(", ")
            )));
        }
        let model = registry
            .load(&model_path.to_string_lossy(), Some(ModelFamily::from(family)), None)
            .map_err(|e| SttError::Model(format!("audio.cpp load failed ({family}): {e}")))?;
        Ok(Self {
            model: Mutex::new(model),
            family: family.to_owned(),
        })
    }

    /// The family this pipeline was loaded with.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Mono PCM16 at `sample_rate` → engine buffer at 16 kHz mono f32.
    fn engine_audio(pcm16: &[i16], sample_rate: u32) -> WavAudio {
        let f32_pcm: Vec<f32> = pcm16.iter().map(|&s| f32::from(s) / 32_768.0).collect();
        let samples = resample_linear_f32(&f32_pcm, sample_rate, TARGET_RATE);
        WavAudio {
            sample_rate: i32::try_from(TARGET_RATE).unwrap_or(i32::MAX),
            channels: 1,
            samples,
        }
    }
}

impl SttPipeline for AudioCppStt {
    fn transcribe(
        &self,
        pcm16: &[i16],
        sample_rate: u32,
        req: &TranscribeRequest,
    ) -> Result<Transcript, SttError> {
        if pcm16.is_empty() {
            return Err(SttError::Empty);
        }
        if sample_rate == 0 {
            return Err(SttError::Format("sample rate must be non-zero".into()));
        }
        if req.translate {
            return Err(SttError::Format(format!(
                "translate-to-English is not supported by the audio.cpp family `{}`; \
                 `/v1/audio/translations` is whisper.cpp-only in this build",
                self.family
            )));
        }
        let started = Instant::now();
        let wav = Self::engine_audio(pcm16, sample_rate);
        // Whole-input span in ms (StubStt shape): audio.cpp word timestamps
        // are family-gated (qwen3_asr needs a forced-aligner session option),
        // so the honest MVP contract is one full-audio segment; per-word
        // segmentation upgrades later behind `RETURN_TIMESTAMPS`.
        let duration_ms = u32::try_from(
            u64::try_from(wav.samples.len()).unwrap_or(u64::MAX).saturating_mul(1_000)
                / u64::from(TARGET_RATE),
        )
        .unwrap_or(u32::MAX);
        let mut request = Request::asr(wav);
        if let Some(language) = &req.language {
            request = request.option(opts::LANGUAGE, language.clone());
        }
        let model = self
            .model
            .lock()
            .map_err(|e| SttError::Inference(format!("model mutex poisoned: {e}")))?;
        let session = model
            .create_task_session(TaskKind::Asr, RunMode::Offline, Backend::Cpu, 0, 4, None)
            .map_err(|e| SttError::Inference(format!("session create failed: {e}")))?;
        let result = session
            .run_offline(request)
            .map_err(|e| SttError::Inference(e.to_string()))?;
        let text = result
            .text_output
            .ok_or_else(|| {
                SttError::Inference(format!(
                    "family `{}` produced no text output",
                    self.family
                ))
            })?
            .text;
        Ok(Transcript {
            backend_id: format!("{}:{}", BACKEND_ID, self.family),
            text: text.trim().to_owned(),
            segments: vec![TranscriptSegment {
                start_ms: 0,
                end_ms: duration_ms,
                text: text.trim().to_owned(),
            }],
            millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Resource contract (parallel-safe by construction): every test owns
    // nothing shared — fixtures under the vendored tree are READ-ONLY, each
    // test builds its own `Registry`/`Model` handles (the engine tolerates
    // concurrent separate handles; the binding types are `Send`, the
    // default cargo harness runs these in parallel), no ports, no fixed
    // write paths, no ordering dependence. `transcribes_local_asr_model`
    // is `#[ignore]` + env-gated so CI and parallel runs never touch
    // network or operator-supplied weights.

    /// Repo-root relative default for the vendored tree (sync lands it);
    /// resolved from this crate's manifest dir.
    fn vendored_tree() -> std::path::PathBuf {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .map_or_else(|| Path::new("../..").to_path_buf(), Path::to_path_buf);
        manifest.join("vendor/audio-cpp-sys-0.9.0/audio.cpp")
    }

    #[test]
    fn unknown_family_is_a_named_error() {
        let tree = vendored_tree();
        let weights = tree.join("assets/framework/models/silero_vad/silero_vad_16k.safetensors");
        if !weights.is_file() {
            // Feature-on compile probe without the vendored tree (sync not
            // run on this machine): the engine itself built, nothing to
            // validate — skip instead of failing the gate.
            eprintln!("skip: vendored audio.cpp tree not present at {}", tree.display());
            return;
        }
        let err = AudioCppStt::load(&weights, "definitely_not_a_family")
            .expect_err("unknown family must fail");
        let msg = err.to_string();
        assert!(msg.contains("definitely_not_a_family"), "names the family: {msg}");
        assert!(msg.contains("not compiled into this build"), "honest absence: {msg}");
        assert!(
            msg.contains("silero_vad") && msg.contains("marblenet_vad"),
            "lists the compiled core families: {msg}"
        );
    }

    #[test]
    fn core_build_compiles_the_bundled_vad_families() {
        let registry = Registry::new().expect("registry");
        let compiled = registry.families().expect("families");
        for family in ["silero_vad", "marblenet_vad"] {
            assert!(
                compiled.iter().any(|f| f == family),
                "bundled `{family}` missing from {compiled:?}"
            );
        }
    }

    /// Full-ASR round trip needs a LOCAL audio.cpp ASR GGUF (weights are
    /// never downloaded): set `AUDIOCPP_ASR_MODEL` (path) and optionally
    /// `AUDIOCPP_ASR_FAMILY` (default `qwen3_asr`).
    #[test]
    #[ignore = "needs a local audio.cpp ASR GGUF (AUDIOCPP_ASR_MODEL); no CI downloads"]
    fn transcribes_local_asr_model() {
        let Some(model) = std::env::var_os("AUDIOCPP_ASR_MODEL").map(std::path::PathBuf::from)
        else {
            eprintln!("skip: AUDIOCPP_ASR_MODEL not set");
            return;
        };
        let family = std::env::var("AUDIOCPP_ASR_FAMILY")
            .unwrap_or_else(|_| crate::AUDIOCPP_DEFAULT_FAMILY.into());
        let stt = AudioCppStt::load(&model, &family).expect("load");
        let tree = vendored_tree();
        let bytes =
            std::fs::read(tree.join("assets/resources/sample_16k.wav")).expect("bundled speech");
        let (rate, pcm) = super::super::decode_wav(&bytes).expect("decode wav");
        let transcript = stt
            .transcribe(
                &pcm,
                rate,
                &TranscribeRequest {
                    model: model.display().to_string(),
                    language: None,
                    translate: false,
                },
            )
            .expect("transcribe");
        assert_eq!(transcript.backend_id, format!("{BACKEND_ID}:{family}"));
        assert!(!transcript.text.is_empty(), "non-empty transcript");
    }
}
