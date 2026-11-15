//! TTS backends: deterministic [`StubTts`] sine + any-tts [`KokoroTts`].

use super::{SynthesizeRequest, SynthesizedAudio, TtsError, TtsPipeline, resample_linear_f32};
use std::f32::consts::PI;
use std::path::Path;

/// Deterministic hermetic TTS: 1 s 440 Hz sine at half amplitude.
#[derive(Debug, Default)]
pub struct StubTts;

impl TtsPipeline for StubTts {
    fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizedAudio, TtsError> {
        if req.text.trim().is_empty() {
            return Err(TtsError::Empty);
        }
        let rate = req.sample_rate.unwrap_or(24_000);
        if rate == 0 {
            return Err(TtsError::Format("sample rate must be non-zero".into()));
        }
        let pcm16 = (0..rate as usize)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (0.5 * (2.0 * PI * t * 440.0).sin() * f32::from(i16::MAX)) as i16
            })
            .collect();
        Ok(SynthesizedAudio { sample_rate: rate, pcm16 })
    }

    fn backend_id(&self) -> &'static str {
        "stub-tts"
    }
}

/// Kokoro TTS via any-tts (CPU candle backend). `model_path` is the model
/// directory, or a weight file inside it (parent dir is used).
pub struct KokoroTts {
    model: Box<dyn any_tts::TtsModel>,
}

impl std::fmt::Debug for KokoroTts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KokoroTts").finish_non_exhaustive()
    }
}

impl KokoroTts {
    /// Load Kokoro weights from disk (no downloads: offline resolve only).
    pub fn load(model_path: &Path) -> Result<Self, TtsError> {
        if !model_path.exists() {
            return Err(TtsError::Io(format!("model not found: {}", model_path.display())));
        }
        let dir = if model_path.is_dir() {
            model_path.to_path_buf()
        } else {
            model_path.parent().map(Path::to_path_buf).unwrap_or_else(|| Path::new(".").to_path_buf())
        };
        let config = any_tts::TtsConfig::new(any_tts::ModelType::Kokoro)
            .with_model_path(dir.display().to_string())
            .with_device(any_tts::DeviceSelection::Cpu);
        let model = any_tts::load_model(config).map_err(|e| TtsError::Model(e.to_string()))?;
        Ok(Self { model })
    }
}

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16).collect()
}

impl TtsPipeline for KokoroTts {
    fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizedAudio, TtsError> {
        if req.text.trim().is_empty() {
            return Err(TtsError::Empty);
        }
        if req.sample_rate == Some(0) {
            return Err(TtsError::Format("sample rate must be non-zero".into()));
        }
        let mut synth = any_tts::SynthesisRequest::new(req.text.clone());
        if let Some(voice) = req.voice.as_deref() {
            synth = synth.with_voice(voice);
        }
        let audio = self.model.synthesize(&synth).map_err(|e| TtsError::Inference(e.to_string()))?;
        if audio.sample_rate == 0 || audio.samples.is_empty() {
            return Err(TtsError::Inference("backend returned no audio".into()));
        }
        // Honor the requested rate only for trivial integer-ratio resamples.
        let native = audio.sample_rate;
        let (pcm16, sample_rate) = match req.sample_rate {
            Some(want)
                if want != native
                    && (want % native == 0 || native % want == 0) =>
            {
                (f32_to_i16(&resample_linear_f32(&audio.samples, native, want)), want)
            }
            _ => (f32_to_i16(&audio.samples), native),
        };
        Ok(SynthesizedAudio { sample_rate, pcm16 })
    }

    fn backend_id(&self) -> &'static str {
        "kokoro-cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(text: &str) -> SynthesizeRequest {
        SynthesizeRequest { model: "stub".into(), text: text.into(), voice: None, sample_rate: None }
    }

    #[test]
    fn stub_roundtrip() {
        let tts = StubTts;
        assert_eq!(tts.backend_id(), "stub-tts");
        let out = tts.synthesize(&req("hello")).unwrap();
        assert_eq!(out.sample_rate, 24_000);
        assert_eq!(out.pcm16.len(), 24_000);
        assert!(out.pcm16.iter().any(|&s| s != 0)); // oscillates
        // deterministic: same input, same output
        assert_eq!(out.pcm16, tts.synthesize(&req("hello")).unwrap().pcm16);
        // honors requested rate
        let out8 = tts.synthesize(&SynthesizeRequest { sample_rate: Some(8000), ..req("hi") }).unwrap();
        assert_eq!((out8.sample_rate, out8.pcm16.len()), (8000, 8000));
    }

    #[test]
    fn stub_rejects_empty() {
        assert!(matches!(StubTts.synthesize(&req("")), Err(TtsError::Empty)));
        assert!(matches!(StubTts.synthesize(&req("   ")), Err(TtsError::Empty)));
    }

    #[test]
    fn kokoro_load_missing_file_errors() {
        let err = KokoroTts::load(Path::new("/nonexistent/kokoro")).unwrap_err();
        assert!(matches!(err, TtsError::Io(_)));
    }
}
