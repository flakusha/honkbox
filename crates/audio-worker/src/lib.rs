//! Sync speech-to-text / text-to-speech pipelines (Whisper, Kokoro) + WAV helpers.
//!
//! CPU-only by default. `stt-vulkan` (default) builds Whisper with the Vulkan
//! backend; `vad` pulls in `ort` for Silero VAD (non-default, CPU build clean
//! of ONNX/CUDA otherwise).

mod stt;
mod tts;
#[cfg(feature = "vad")]
mod vad;
// audio.cpp engine backend (TASK-audiocpp-stt-provider): compiled only with
// the non-default `audio-cpp` feature; the probe const is ALWAYS defined so
// the capability document cannot drift from the binary (same pattern as
// `VAD_COMPILED`).
#[cfg(feature = "audio-cpp")]
mod audiocpp;

#[cfg(feature = "audio-cpp")]
pub use audiocpp::{AudioCppStt, BACKEND_ID as AUDIOCPP_BACKEND_ID};
pub use stt::{StubStt, WhisperStt};
pub use tts::{KOKORO_VOICES, KokoroTts, StubTts};
#[cfg(feature = "vad")]
pub use vad::{SileroVad, VadSpan};

/// `true` when THIS build compiles Silero VAD (`vad` feature) — the same
/// compile probe the capability advertisement reports as `features.vad`, so
/// the document cannot drift from the binary.
pub const VAD_COMPILED: bool = cfg!(feature = "vad");

/// `true` when THIS build compiles the audio.cpp engine (`audio-cpp`
/// feature) — the compile probe for the capability advertisement
/// (TASK-audiocpp-capabilities); honest absence in default builds.
pub const AUDIOCPP_COMPILED: bool = cfg!(feature = "audio-cpp");

/// Default audio.cpp ASR family (eSpeak-free). Unconditional: the CLI seam
/// names it (clap default + honest error text) even in builds without the
/// `audio-cpp` feature.
pub const AUDIOCPP_DEFAULT_FAMILY: &str = "qwen3_asr";

/// STT families compiled into THIS build (TASK-audiocpp-capabilities).
/// `qwen3_asr` needs both the engine and its weak loader feature:
/// `audio-cpp` alone has no ASR loader, so an engine-only build reports
/// no family.
pub const AUDIOCPP_STT_FAMILIES: &[&str] =
    if cfg!(all(feature = "audio-cpp", feature = "audio-cpp-qwen3-asr")) {
        &["qwen3_asr"]
    } else {
        &[]
    };

/// TTS families compiled into THIS build: none — `make_tts` has no
/// audio.cpp arm, so the list stays empty with every audio.cpp feature on.
pub const AUDIOCPP_TTS_FAMILIES: &[&str] = &[];

use flacenc::error::Verify as _;

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
    /// Translate the output to English (whisper.cpp `translate`; the
    /// `/v1/audio/translations` semantics). Ignored by the stub backend.
    pub translate: bool,
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

/// Lanczos-windowed sinc resampler for mono f32 PCM (STT/VAD downmix and TTS
/// rate honors). The tap count is fixed, so work per output sample is bounded.
const RESAMPLE_HALF_TAPS: usize = 32;

// Anti-alias cutoff = `min(from_rate, to_rate)/2`; the 0.8 guard leaves a
// Lanczos transition band so near-Nyquist tones cannot fold into 16 kHz output.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub(crate) fn resample_f32(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 || from_rate == to_rate {
        return input.to_vec();
    }
    let out_len = (u64::try_from(input.len()).unwrap_or(u64::MAX) * u64::from(to_rate)
        / u64::from(from_rate)) as usize;
    let step = f64::from(from_rate) / f64::from(to_rate);
    let cutoff_cycles = f64::from(from_rate.min(to_rate)) * 0.4 / f64::from(from_rate);
    let last = input.len() - 1;
    // Fixed DC norm: local per-sample sums approach zero for out-of-band input,
    // which would amplify that band instead of filtering it.
    let kernel = |offset: f64| {
        let arg = 2.0 * cutoff_cycles * offset * std::f64::consts::PI;
        let window_arg = offset * std::f64::consts::PI / RESAMPLE_HALF_TAPS as f64;
        let sinc = if arg.abs() < f64::EPSILON {
            1.0
        } else {
            arg.sin() / arg
        };
        let window = if window_arg.abs() < f64::EPSILON {
            1.0
        } else {
            window_arg.sin() / window_arg
        };
        sinc * window
    };
    let norm: f64 = (0..=2 * RESAMPLE_HALF_TAPS)
        .map(|tap| kernel(tap as f64 - RESAMPLE_HALF_TAPS as f64))
        .sum();
    (0..out_len.max(1))
        .map(|i| {
            let pos = i as f64 * step;
            let center = pos as usize;
            let first = center.saturating_sub(RESAMPLE_HALF_TAPS);
            let end = center.saturating_add(RESAMPLE_HALF_TAPS).min(last);
            let acc: f64 = input[first..=end]
                .iter()
                .enumerate()
                .map(|(tap, &sample)| kernel(pos - (first + tap) as f64) * f64::from(sample))
                .sum();
            if norm.abs() < f64::EPSILON {
                0.0
            } else {
                (acc / norm) as f32
            }
        })
        .collect()
}

/// Encode mono PCM16 as a 16-bit RIFF/WAVE file.
pub fn encode_wav(sample_rate: u32, pcm: &[i16]) -> Vec<u8> {
    let data_len = pcm.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + u32::try_from(data_len).unwrap_or(u32::MAX)).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&u32::try_from(data_len).unwrap_or(u32::MAX).to_le_bytes());
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
                rate = Some(u32::from_le_bytes(
                    bytes[body + 4..body + 8].try_into().unwrap(),
                ));
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
        data.as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c))
            .collect(),
    ))
}

audio_error!(EncodeError, "Audio encoding failure.");

/// Fixed FLAC block size (frames); 4096 is the encoder's own default scale.
const FLAC_BLOCK_SIZE: usize = 4096;
/// Opus frame length: 20 ms, the crate's documented sweet spot.
const OPUS_FRAME_MS: u32 = 20;
/// Default Opus bitrate for mono speech (matches the crate's own example).
const OPUS_BITRATE_BPS: i32 = 64_000;

/// Encode mono PCM16 as FLAC (lossless) via pure-Rust `flacenc`.
///
/// No system codecs, no shared state: the config is verified per call and the
/// sink is call-local. Kokoro's native 24 kHz (and the stub's 8 kHz) pass
/// through untouched — FLAC stores any rate.
pub fn encode_flac(sample_rate: u32, pcm: &[i16]) -> Result<Vec<u8>, EncodeError> {
    use flacenc::component::BitRepr;
    if pcm.is_empty() {
        return Err(EncodeError::Empty);
    }
    let signal: Vec<i32> = pcm.iter().map(|&s| i32::from(s)).collect();
    let source = flacenc::source::MemSource::from_samples(&signal, 1, 16, sample_rate as usize);
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|err| EncodeError::Format(format!("bad flac config: {err:?}")))?;
    let stream = flacenc::encode_with_fixed_block_size(&config, source, FLAC_BLOCK_SIZE)
        .map_err(|err| EncodeError::Format(format!("flac encode failed: {err:?}")))?;
    let mut sink = flacenc::bitsink::ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|err| EncodeError::Format(format!("flac sink failed: {err:?}")))?;
    Ok(sink.into_inner())
}

/// Samples the last Opus packet must cover so the stream trims exactly to
/// the input length (RFC 7845 §4.4 end-trim).
fn opus_final_duration(final_granule: u64, written_granule: i64) -> Result<u32, EncodeError> {
    let lag = final_granule.saturating_sub(written_granule.unsigned_abs());
    u32::try_from(lag)
        .map_err(|_| EncodeError::Format(format!("opus final granule out of range: {lag}")))
}

/// Encode mono PCM16 as Ogg-encapsulated Opus (RFC 7845, gapless) via
/// pure-Rust `opus-pure`.
///
/// Same honesty contract as the crate's own `encode` example: the encoder's
/// real `pre_skip` flushes the tail, and the last packet carries the exact
/// final granule so a decoder trims back to precisely `pcm.len()` samples.
/// Only the five Opus-native rates are accepted; anything else is a `Format`
/// error (the rate is server-side data, never user input).
pub fn encode_opus(sample_rate: u32, pcm: &[i16]) -> Result<Vec<u8>, EncodeError> {
    if pcm.is_empty() {
        return Err(EncodeError::Empty);
    }
    let rate = i32::try_from(sample_rate)
        .map_err(|_| EncodeError::Format(format!("opus rate out of range: {sample_rate}")))?;
    let mut encoder = opus_pure::OpusEncoder::new(rate, 1, opus_pure::Application::Audio)
        .map_err(|err| EncodeError::Format(format!("opus rejects rate {sample_rate}: {err:?}")))?;
    encoder.bitrate_bps = OPUS_BITRATE_BPS;
    let frame = sample_rate as usize * OPUS_FRAME_MS as usize / 1_000;
    let head = opus_pure::OpusHead::for_encoder(&encoder, sample_rate);
    let mut tags = opus_pure::OpusTags::new();
    tags.push("ENCODER", "honkbox")
        .map_err(|err| EncodeError::Format(format!("opus tags failed: {err:?}")))?;
    // Every Opus rate divides 48 kHz, so granule conversions are exact.
    let ticks = 48_000 / sample_rate as usize;
    let lookahead = (head.pre_skip as usize).div_ceil(ticks);
    let frames = (pcm.len() + lookahead).div_ceil(frame);
    let tail = u64::try_from(pcm.len() * ticks).unwrap_or(u64::MAX);
    let final_granule = u64::from(head.pre_skip) + tail;
    let mut writer = opus_pure::OggOpusWriter::with_tags(Vec::new(), head, tags)
        .map_err(|err| EncodeError::Format(format!("opus ogg failed: {err:?}")))?;
    let mut packet = vec![0u8; opus_pure::MAX_PACKET_BYTES];
    let mut block = vec![0i16; frame];
    for i in 0..frames {
        let start = (i * frame).min(pcm.len());
        let end = (start + frame).min(pcm.len());
        block[..end - start].copy_from_slice(&pcm[start..end]);
        block[end - start..].fill(0);
        let n = encoder
            .encode_s16(&block, frame, &mut packet)
            .map_err(|err| EncodeError::Format(format!("opus packet failed: {err:?}")))?;
        if i + 1 == frames {
            let duration = opus_final_duration(final_granule, writer.granule())?;
            writer
                .write_packet_with_duration(&packet[..n], duration)
                .map_err(|err| EncodeError::Format(format!("opus last packet failed: {err:?}")))?;
        } else {
            writer
                .write_packet(&packet[..n])
                .map_err(|err| EncodeError::Format(format!("opus packet write failed: {err:?}")))?;
        }
    }
    writer
        .finish()
        .map_err(|err| EncodeError::Format(format!("opus finish failed: {err:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_identity() {
        let pcm: Vec<i16> = (0..1600)
            .map(|i| i16::try_from(i * 37 % 1000 - 500).unwrap_or(i16::MAX))
            .collect();
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

    /// FLAC STREAMINFO field reader (big-endian bit layout, no decoder dep).
    /// Returns `(sample_rate, channels, bits_per_sample, total_samples)`.
    fn flac_streaminfo(bytes: &[u8]) -> (u32, u32, u32, u64) {
        assert_eq!(&bytes[0..4], b"fLaC");
        assert_eq!(bytes[4] & 0x7F, 0); // STREAMINFO block type
        // 3-byte big-endian length at bytes 5..8; pad to u32 for the compare.
        let len = u32::from_be_bytes([0, bytes[5], bytes[6], bytes[7]]);
        assert_eq!(len, 34, "STREAMINFO is always 34 bytes");
        let body = u64::from_be_bytes(bytes[18..26].try_into().unwrap());
        let field = |shift: u32, mask: u64| {
            u32::try_from((body >> shift) & mask).expect("masked bitfield fits u32")
        };
        (
            field(44, 0xF_FFFF),
            field(41, 7) + 1,
            field(36, 31) + 1,
            body & 0xF_FFFF_FFFF,
        )
    }

    #[test]
    fn flac_streaminfo_matches_input() {
        for rate in [8_000, 24_000] {
            let pcm: Vec<i16> = (0..5_000)
                .map(|i| i16::try_from(i * 37 % 2000 - 1000).expect("fits i16"))
                .collect();
            let bytes = encode_flac(rate, &pcm).unwrap();
            let total = u64::try_from(pcm.len()).expect("test vec fits u64");
            assert_eq!(flac_streaminfo(&bytes), (rate, 1, 16, total));
        }
    }

    #[test]
    fn flac_rejects_empty_input() {
        assert!(matches!(encode_flac(24_000, &[]), Err(EncodeError::Empty)));
    }

    /// `OpusHead` field reader: returns `(channels, pre_skip, input_rate)`.
    fn opus_head(bytes: &[u8]) -> (u8, u16, u32) {
        let at = bytes
            .windows(8)
            .position(|w| w == b"OpusHead")
            .expect("OpusHead present");
        (
            bytes[at + 9],
            u16::from_le_bytes(bytes[at + 10..at + 12].try_into().unwrap()),
            u32::from_le_bytes(bytes[at + 12..at + 16].try_into().unwrap()),
        )
    }

    fn speech_like(len: usize) -> Vec<i16> {
        (0..len)
            .map(|i| {
                i16::try_from(i32::try_from(i).expect("fits i32") * 53 % 3000 - 1500)
                    .expect("fits i16")
            })
            .collect()
    }

    #[test]
    fn opus_head_matches_input_and_is_ogg() {
        let pcm = speech_like(4_800);
        let bytes = encode_opus(24_000, &pcm).unwrap();
        assert_eq!(&bytes[0..4], b"OggS");
        assert!(bytes.windows(8).any(|w| w == b"OpusTags"));
        let (channels, _pre_skip, rate) = opus_head(&bytes);
        assert_eq!((channels, rate), (1, 24_000));
    }

    #[test]
    fn opus_roundtrip_preserves_sample_count() {
        let pcm = speech_like(4_800);
        let bytes = encode_opus(24_000, &pcm).unwrap();
        let mut reader = opus_pure::OggOpusReader::new(bytes.as_slice()).unwrap();
        let head = reader.head().clone();
        let mut decoder = head.decoder(24_000).unwrap();
        let mut trim = opus_pure::Trim::new(&head, 24_000, 1).unwrap();
        let mut block = vec![0.0f32; opus_pure::MAX_PACKET_SAMPLES];
        for packet in reader.packets() {
            let packet = packet.unwrap();
            let n = decoder
                .decode(&packet.data, opus_pure::MAX_PACKET_SAMPLES, &mut block)
                .unwrap();
            let kept = trim.keep(&packet, &block[..n]);
            assert!(kept.iter().all(|s| s.is_finite()));
        }
        let total = u64::try_from(pcm.len()).expect("test vec fits u64");
        assert_eq!(trim.samples_emitted(), total);
    }

    #[test]
    fn opus_single_packet_stays_decodable() {
        // One frame at most: exercises the end-trim path when the pre-final
        // granule is still the writer's initial value.
        let pcm = speech_like(100);
        let bytes = encode_opus(24_000, &pcm).unwrap();
        let mut reader = opus_pure::OggOpusReader::new(bytes.as_slice()).unwrap();
        let head = reader.head().clone();
        let mut decoder = head.decoder(24_000).unwrap();
        let mut trim = opus_pure::Trim::new(&head, 24_000, 1).unwrap();
        let mut block = vec![0.0f32; opus_pure::MAX_PACKET_SAMPLES];
        for packet in reader.packets() {
            let packet = packet.unwrap();
            let n = decoder
                .decode(&packet.data, opus_pure::MAX_PACKET_SAMPLES, &mut block)
                .unwrap();
            let kept = trim.keep(&packet, &block[..n]);
            assert!(kept.iter().all(|s| s.is_finite()));
        }
        let total = u64::try_from(pcm.len()).expect("test vec fits u64");
        assert_eq!(trim.samples_emitted(), total);
    }

    #[test]
    fn opus_rejects_empty_and_non_native_rates() {
        assert!(matches!(encode_opus(24_000, &[]), Err(EncodeError::Empty)));
        let pcm = vec![0i16; 480];
        assert!(matches!(
            encode_opus(44_100, &pcm),
            Err(EncodeError::Format(_))
        ));
    }

    /// Compile-probe honesty (TASK-audiocpp-stt-provider): the default
    /// build does NOT compile the audio.cpp engine, so the capability probe
    /// must read false — flipping `audio-cpp` into default features stops
    /// being a silent advertisement change.
    #[cfg(not(feature = "audio-cpp"))]
    #[test]
    // The assertion is definitionally constant under `cfg(not)` — that IS
    // the contract being pinned (the default feature set excludes audio.cpp
    // and the probe reads false), so the lint is intentionally allowed.
    #[allow(clippy::assertions_on_constants)]
    fn audiocpp_probe_is_off_without_the_feature() {
        assert!(!AUDIOCPP_COMPILED);
    }

    fn tone(len: usize, rate: f32, freq: f32) -> Vec<f32> {
        let step = 2.0 * std::f32::consts::PI * freq / rate;
        let mut phase = 0.0f32;
        (0..len)
            .map(|_| {
                let sample = phase.sin();
                phase += step;
                sample
            })
            .collect()
    }

    fn peak(samples: &[f32]) -> f32 {
        samples.iter().copied().map(f32::abs).fold(0.0, f32::max)
    }

    #[test]
    fn resample_8k_to_16k_preserves_tone_length_and_finiteness() {
        let input = tone(400, 8_000.0, 1_000.0);
        let out = resample_f32(&input, 8_000, 16_000);
        assert_eq!(out.len(), input.len() * 2);
        assert!(out.iter().all(|s| s.is_finite()));
        let core = &out[128..out.len() - 128];
        let amplitude = peak(core);
        assert!((0.85..=1.05).contains(&amplitude));
    }

    #[test]
    fn resample_44k1_to_16k_attenuates_near_nyquist_tone() {
        let in_band = tone(2_205, 44_100.0, 1_000.0);
        let high = tone(2_205, 44_100.0, 7_000.0);
        let low_out = resample_f32(&in_band, 44_100, 16_000);
        let high_out = resample_f32(&high, 44_100, 16_000);
        assert_eq!(low_out.len(), 800);
        assert_eq!(high_out.len(), low_out.len());
        assert!(low_out.iter().chain(&high_out).all(|s| s.is_finite()));
        let low_peak = peak(&low_out[64..low_out.len() - 64]);
        let high_peak = peak(&high_out[64..high_out.len() - 64]);
        assert!((0.85..=1.05).contains(&low_peak));
        assert!(high_peak < low_peak * 0.25, "{high_peak} vs {low_peak}");
    }

    #[test]
    fn resample_equal_rate_is_identity() {
        let input = [0.0, 0.5, -0.5, 1.0];
        assert_eq!(resample_f32(&input, 16_000, 16_000), input.to_vec());
    }

    #[test]
    fn resample_zero_and_empty_inputs_are_identity() {
        assert_eq!(resample_f32(&[], 8_000, 16_000), Vec::<f32>::new());
        let input = [0.25, -0.75];
        assert_eq!(resample_f32(&input, 0, 16_000), input.to_vec());
        assert_eq!(resample_f32(&input, 8_000, 0), input.to_vec());
    }

    #[test]
    fn resample_adversarial_inputs_stay_finite_and_bounded() {
        let single = [1.0];
        let dc = [1.0; 64];
        let alternating: Vec<f32> = (0usize..64)
            .map(|i| if i.is_multiple_of(2) { 1.0 } else { -1.0 })
            .collect();
        let cases = [
            (&single[..], 44_100, 16_000),
            (&dc[..], 44_100, 16_000),
            (&alternating[..], 8_000, 16_000),
        ];
        for (input, from, to) in cases {
            let out = resample_f32(input, from, to);
            let expected = usize::try_from(
                u64::try_from(input.len()).unwrap_or(u64::MAX) * u64::from(to) / u64::from(from),
            )
            .unwrap_or(usize::MAX)
            .max(1);
            assert_eq!(out.len(), expected);
            assert!(out.iter().all(|s| s.is_finite()));
            assert!(peak(&out) <= 1.1, "overshoot: {}", peak(&out));
        }
    }
}
