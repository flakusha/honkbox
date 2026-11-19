//! Silero VAD behind the non-default `vad` feature (ort, CPU).
//! Expects a Silero v5 ONNX model (`input`/`state`/`sr` in, `output`/`stateN` out).

use super::VadError;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

/// 16 kHz mono, 512-sample (32 ms) Silero window.
const SAMPLE_RATE: u32 = 16_000;
const WINDOW: usize = 512;
/// Speech probability threshold.
const THRESHOLD: f32 = 0.5;

/// Speech span in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadSpan {
    /// Speech start in milliseconds.
    pub start_ms: u32,
    /// Speech end in milliseconds.
    pub end_ms: u32,
}

/// Silero VAD session. `Send + Sync` via an internal mutex (ort `run` takes
/// `&mut self`).
#[derive(Debug)]
pub struct SileroVad {
    session: Mutex<ort::session::Session>,
}

impl SileroVad {
    /// Load a Silero ONNX model from `model_path`.
    pub fn load(model_path: &Path) -> Result<Self, VadError> {
        if !model_path.is_file() {
            return Err(VadError::Io(format!(
                "vad model not found: {}",
                model_path.display()
            )));
        }
        let env = ort::init()
            .with_name("honkbox-silero-vad")
            .build()
            .map_err(|e| VadError::Model(e.to_string()))?;
        let mut builder =
            ort::session::Session::builder(&env).map_err(|e| VadError::Model(e.to_string()))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| VadError::Model(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Speech spans over mono PCM16 at `sample_rate` Hz (resampled to 16 kHz).
    pub fn spans(&self, pcm16: &[i16], sample_rate: u32) -> Result<Vec<VadSpan>, VadError> {
        if pcm16.is_empty() {
            return Err(VadError::Empty);
        }
        if sample_rate == 0 {
            return Err(VadError::Format("sample rate must be non-zero".into()));
        }
        let mono: Vec<f32> = pcm16.iter().map(|s| f32::from(*s) / 32768.0).collect();
        let audio = super::resample_f32(&mono, sample_rate, SAMPLE_RATE);
        let mut session = self
            .session
            .lock()
            .map_err(|e| VadError::Inference(format!("session poisoned: {e}")))?;
        let mut state = vec![0.0f32; 2 * 128];
        let mut spans = Vec::new();
        let mut open: Option<u32> = None;
        let mut chunks = audio.chunks(WINDOW).enumerate();
        while let Some((idx, chunk)) = chunks.next() {
            let mut window = vec![0.0f32; WINDOW];
            window[..chunk.len()].copy_from_slice(chunk); // zero-padded tail
            let input = Tensor::from_array(([1usize, WINDOW], window.into_boxed_slice()))
                .map_err(|e| VadError::Inference(e.to_string()))?
                .into_dyn();
            let state_in = Tensor::from_array(([2usize, 1, 128], state.clone().into_boxed_slice()))
                .map_err(|e| VadError::Inference(e.to_string()))?
                .into_dyn();
            let sr =
                Tensor::from_array(([1usize], vec![i64::from(SAMPLE_RATE)].into_boxed_slice()))
                    .map_err(|e| VadError::Inference(e.to_string()))?
                    .into_dyn();
            let outputs = session
                .run(ort::inputs!["input" => input, "state" => state_in, "sr" => sr])
                .map_err(|e| VadError::Inference(e.to_string()))?;
            let prob = outputs
                .get("output")
                .and_then(|v| v.try_extract_tensor::<f32>().ok())
                .and_then(|(_, d)| d.first().copied())
                .ok_or_else(|| VadError::Inference("vad model missing `output`".into()))?;
            let next: Vec<f32> = outputs
                .get("stateN")
                .and_then(|v| v.try_extract_tensor::<f32>().ok())
                .map(|(_, d)| d.to_vec())
                .ok_or_else(|| VadError::Inference("vad model missing `stateN`".into()))?;
            if next.len() != state.len() {
                return Err(VadError::Format("unexpected vad state shape".into()));
            }
            state = next;
            let t = idx as u32 * 32; // 512 samples @16 kHz = 32 ms
            if prob >= THRESHOLD {
                if open.is_none() {
                    open = Some(t);
                }
            } else if let Some(start) = open.take() {
                spans.push(VadSpan {
                    start_ms: start,
                    end_ms: t,
                });
            }
        }
        if let Some(start) = open.take() {
            let total = (audio.len() as u64 * 1000 / u64::from(SAMPLE_RATE)) as u32;
            spans.push(VadSpan {
                start_ms: start,
                end_ms: total,
            });
        }
        Ok(spans)
    }
}
