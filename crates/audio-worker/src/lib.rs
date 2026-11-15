//! Sync speech-to-text / text-to-speech pipelines (Whisper, Kokoro) + WAV helpers.
//!
//! CPU-only by default. `stt-vulkan` (default) builds Whisper with the Vulkan
//! backend; `vad` pulls in `ort` for Silero VAD (non-default, CPU build clean
//! of ONNX/CUDA otherwise).

mod stt;
mod tts;
#[cfg(feature = "vad")]
mod vad;

pub use stt::{StubStt, WhisperStt};
pub use tts::{KokoroTts, StubTts};
#[cfg(feature = "vad")]
pub use vad::{SileroVad, VadSpan};

use std::fmt;

macro_rules! audio_error {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub enum $name {
            /// Filesystem failure (missing model file, unreadable path, ...).
            Io(String),
            /// Model file rejected or backend failed to load it.
            Model(String),
            /// Backend ran but inference itself failed.
            Inference(String),
            /// Malformed container, bad rate, or otherwise invalid input.
            Format(String),
            /// Empty input where audio or text was required.
            Empty,
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Self::Io(e) => write!(f, "io error: {e}"),
                    Self::Model(e) => write!(f, "model error: {e}"),
                    Self::Inference(e) => write!(f, "inference error: {e}"),
                    Self::Format(e) => write!(f, "format error: {e}"),
                    Self::Empty => write!(f, "empty input"),
                }
            }
        }

        impl std::error::Error for $name {}

        impl From<std::io::Error> for $name {
            fn from(e: std::io::Error) -> Self {
                Self::Io(e.to_string())
            }
        }
    };
}

audio_error!(SttError, "Speech-to-text failure.");
audio_error!(TtsError, "Text-to-speech failure.");
audio_error!(VadError, "Voice-activity-detection failure.");

/// STT request: `model` selects the backend weights, `sample_rate` travels
/// separately on [`SttPipeline::transcribe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscribeRequest {
    /// Model selector (path or registry id, backend-defined).
    pub model: String,
    /// BCP-47 language hint (`None` = auto-detect).
    pub language: Option<String>,
}

/// One timestamped STT segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptSegment {
    /// Segment start in milliseconds.
    pub start_ms: u32,
    /// Segment end in milliseconds.
    pub end_ms: u32,
    /// Segment text.
    pub text: String,
}

/// STT result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Backend that produced this transcript (see `backend_id`).
    pub backend_id: String,
    /// Full concatenated text.
    pub text: String,
    /// Timestamped segments.
    pub segments: Vec<TranscriptSegment>,
    /// Wall-clock inference time in milliseconds.
    pub millis: u64,
}

/// TTS request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizeRequest {
    /// Model selector (path or registry id, backend-defined).
    pub model: String,
    /// Text to speak.
    pub text: String,
    /// Voice preset (`None` = backend default).
    pub voice: Option<String>,
    /// Desired output rate; backends honor trivial (integer-ratio) resamples
    /// and otherwise return their native rate.
    pub sample_rate: Option<u32>,
}

/// Worker-internal mono PCM16 result (not an API type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizedAudio {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Mono PCM16 samples.
    pub pcm16: Vec<i16>,
}

/// Blocking speech-to-text pipeline over mono PCM16 audio.
///
/// Backend ids: `"stub-stt"` | `"whisper-vulkan"` | `"whisper-cpu"`.
pub trait SttPipeline: Send + Sync {
    /// Transcribe mono PCM16 at `sample_rate` Hz.
    fn transcribe(
        &self,
        pcm16: &[i16],
        sample_rate: u32,
        req: &TranscribeRequest,
    ) -> Result<Transcript, SttError>;
    /// Stable backend id.
    fn backend_id(&self) -> &'static str;
}

/// Blocking text-to-speech pipeline producing mono PCM16 audio.
///
/// Backend ids: `"stub-tts"` | `"kokoro-cpu"`.
pub trait TtsPipeline: Send + Sync {
    /// Synthesize `req.text` to mono PCM16.
    fn synthesize(&self, req: &SynthesizeRequest) -> Result<SynthesizedAudio, TtsError>;
    /// Stable backend id.
    fn backend_id(&self) -> &'static str;
}

/// Naive linear resampler for mono f32 PCM (whisper/vad downmix + TTS rate
/// honors; replace with a proper filter only if quality measurably matters).
// ponytail: O(n) lerp, no anti-aliasing; add rubato/sinc if aliasing shows up.
pub(crate) fn resample_linear_f32(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return input.to_vec();
    }
    let out_len = (input.len() as u64 * u64::from(to_rate) / u64::from(from_rate)) as usize;
    (0..out_len.max(1))
        .map(|i| {
            let pos = i as f64 * f64::from(from_rate) / f64::from(to_rate);
            let j = pos.floor() as usize;
            let frac = (pos - j as f64) as f32;
            let a = input[j.min(input.len() - 1)];
            let b = input[(j + 1).min(input.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

/// Encode mono PCM16 as a 16-bit RIFF/WAVE file.
pub fn encode_wav(sample_rate: u32, pcm: &[i16]) -> Vec<u8> {
    let data_len = pcm.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Decode a 16-bit mono PCM WAV file; returns `(sample_rate, samples)`.
/// Rejects non-PCM16, stereo, and malformed RIFF (≤64 lines by contract).
pub fn decode_wav(bytes: &[u8]) -> Result<(u32, Vec<i16>), SttError> {
    let bad = |why: &str| SttError::Format(why.into());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let mut rate: Option<u32> = None;
    let mut data: Option<&[u8]> = None;
    let mut off = 12;
    while off + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = off + 8;
        let end = body.saturating_add(size);
        if end > bytes.len() {
            return Err(bad("chunk overruns file"));
        }
        match &bytes[off..off + 4] {
            b"fmt " => {
                if size < 16 {
                    return Err(bad("fmt chunk too small"));
                }
                let format = u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap());
                let channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
                let bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
                if format != 1 || channels != 1 || bits != 16 {
                    return Err(bad("only mono PCM16 supported"));
                }
                rate = Some(u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap()));
            }
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }
        off = end + (size & 1); // word-align
    }
    let (Some(rate), Some(data)) = (rate, data) else {
        return Err(bad("missing fmt or data chunk"));
    };
    if data.len() % 2 != 0 {
        return Err(bad("odd data length"));
    }
    Ok((
        rate,
        data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_identity() {
        let pcm: Vec<i16> = (0..1600).map(|i| (i * 37 % 1000 - 500) as i16).collect();
        let (rate, back) = decode_wav(&encode_wav(16_000, &pcm)).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(back, pcm);
    }

    #[test]
    fn wav_rejects_non_pcm16_and_garbage() {
        assert!(decode_wav(b"nope").is_err());
        assert!(decode_wav(&[0u8; 8]).is_err());
        // float (format 3) disguised as wav
        let mut wav = encode_wav(16_000, &[1, 2, 3]);
        wav[20] = 3;
        assert!(decode_wav(&wav).is_err());
        // stereo disguised as wav
        let mut wav = encode_wav(16_000, &[1, 2, 3]);
        wav[22] = 2;
        assert!(decode_wav(&wav).is_err());
    }
}

