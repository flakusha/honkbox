//! STT backends: deterministic [`StubStt`] + whisper.cpp [`WhisperStt`].

use super::{
    SttError, SttPipeline, TranscribeRequest, Transcript, TranscriptSegment, resample_f32,
};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    convert_integer_to_float_audio,
};

/// Fixed-text hermetic STT: one segment spanning the whole input.
#[derive(Debug, Default)]
pub struct StubStt;

impl SttPipeline for StubStt {
    fn transcribe(
        &self,
        pcm16: &[i16],
        sample_rate: u32,
        _req: &TranscribeRequest,
    ) -> Result<Transcript, SttError> {
        if pcm16.is_empty() {
            return Err(SttError::Empty);
        }
        let started = Instant::now();
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )] // len/rate are small; `as` never truncates here
        let end_ms = (pcm16.len() as f64 * 1000.0 / f64::from(sample_rate.max(1))) as u32;
        let text = "stub transcription".to_string();
        Ok(Transcript {
            backend_id: self.backend_id().to_string(),
            segments: vec![TranscriptSegment {
                start_ms: 0,
                end_ms,
                text: text.clone(),
            }],
            text,
            millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    fn backend_id(&self) -> &'static str {
        "stub-stt"
    }
}

/// whisper.cpp STT. `Send + Sync` via an internal mutex (one inference at a
/// time; whisper state is `!Sync` without it).
#[derive(Debug)]
pub struct WhisperStt {
    ctx: Mutex<WhisperContext>,
}

impl WhisperStt {
    /// Load whisper model weights from `model_path` (ggml `.bin`).
    pub fn load(model_path: &Path) -> Result<Self, SttError> {
        if !model_path.is_file() {
            return Err(SttError::Io(format!(
                "model not found: {}",
                model_path.display()
            )));
        }
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .map_err(|e| SttError::Model(e.to_string()))?;
        Ok(Self {
            ctx: Mutex::new(ctx),
        })
    }
}

impl SttPipeline for WhisperStt {
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
        let started = Instant::now();
        let mut float = vec![0.0f32; pcm16.len()];
        convert_integer_to_float_audio(pcm16, &mut float)
            .map_err(|e| SttError::Format(e.to_string()))?;
        let audio = resample_f32(&float, sample_rate, 16_000); // whisper wants 16 kHz mono
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(
            std::thread::available_parallelism()
                .map_or(1, |n| i32::try_from(n.get()).unwrap_or(i32::MAX)),
        );
        params.set_language(req.language.as_deref());
        // `/v1/audio/translations`: whisper's translate task decodes the
        // (non-English) source audio straight into English text.
        params.set_translate(req.translate);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        let ctx = self
            .ctx
            .lock()
            .map_err(|e| SttError::Inference(format!("context poisoned: {e}")))?;
        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::Inference(e.to_string()))?;
        state
            .full(params, &audio)
            .map_err(|e| SttError::Inference(e.to_string()))?;
        let n = usize::try_from(state.full_n_segments()).unwrap_or(0);
        let mut text = String::new();
        let mut segments = Vec::with_capacity(n);
        for i in 0..n {
            let seg = state
                .get_segment(i32::try_from(i).unwrap_or(i32::MAX))
                .ok_or_else(|| SttError::Inference(format!("missing segment {i}")))?;
            let seg_text = seg
                .to_str_lossy()
                .map_err(|e| SttError::Inference(e.to_string()))?;
            // whisper timestamps are centiseconds → ms.
            let start_ms =
                u32::try_from(u64::try_from(seg.start_timestamp().max(0)).unwrap_or(u64::MAX) * 10)
                    .unwrap_or(u32::MAX);
            let end_ms =
                u32::try_from(u64::try_from(seg.end_timestamp().max(0)).unwrap_or(u64::MAX) * 10)
                    .unwrap_or(u32::MAX);
            text.push_str(&seg_text);
            segments.push(TranscriptSegment {
                start_ms,
                end_ms,
                text: seg_text.trim().to_string(),
            });
        }
        Ok(Transcript {
            backend_id: self.backend_id().to_string(),
            text: text.trim().to_string(),
            segments,
            millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    fn backend_id(&self) -> &'static str {
        #[cfg(feature = "stt-vulkan")]
        {
            "whisper-vulkan"
        }
        #[cfg(not(feature = "stt-vulkan"))]
        {
            "whisper-cpu"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> TranscribeRequest {
        TranscribeRequest {
            model: "stub".into(),
            language: None,
            translate: false,
        }
    }

    #[test]
    fn stub_roundtrip() {
        let stt = StubStt;
        assert_eq!(stt.backend_id(), "stub-stt");
        let pcm = vec![0i16; 16_000];
        let t = stt.transcribe(&pcm, 16_000, &req()).unwrap();
        assert_eq!(t.backend_id, "stub-stt");
        assert_eq!(t.text, "stub transcription");
        assert_eq!(t.segments.len(), 1);
        assert_eq!((t.segments[0].start_ms, t.segments[0].end_ms), (0, 1000));
    }

    #[test]
    fn stub_rejects_empty() {
        assert!(matches!(
            StubStt.transcribe(&[], 16_000, &req()),
            Err(SttError::Empty)
        ));
    }

    #[test]
    fn whisper_load_missing_file_errors() {
        let err = WhisperStt::load(Path::new("/nonexistent/ggml-tiny.bin")).unwrap_err();
        assert!(matches!(err, SttError::Io(_)));
    }
}
