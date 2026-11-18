# TASK: audio.cpp TTS provider

Status: proposed
Priority: P2
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: `AudioCppTts` implements `TtsPipeline` behind the same non-default `audio-cpp` feature, serving at least one verified eSpeak-free family (MOSS-TTS-Nano / PocketTTS candidate); `/v1/audio/speech` serves it end to end; kokoro and stub paths unchanged.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run codegen:check`; a synthesis round-trip on a local family GGUF writing valid 16-bit WAV through the existing `encode_wav` path; feature-off build contains no audio.cpp code.
Links: `.tmp/research/ReviewedAudioCpp.md` §2, §5, §8; `crates/audio-worker/src/tts.rs`; `.plan/tickets/TASK-audiocpp-vendor-spike.md`
Resources: per-test temp WAV fixtures; one local eSpeak-free TTS GGUF (no CI downloads); no eSpeak system library required at test time

## Change

- Gated on `TASK-audiocpp-vendor-spike.md` SPIKE PASS (which fixes the
  eSpeak-free family list).
- `honkbox-audio-worker`: `tts.rs` gains `AudioCppTts`; `backend_id()`
  `"audiocpp-tts-<family>"`. Voices are family-specific: NO clone of the
  static `KOKORO_VOICES` pattern — the provider advertises per-family voice
  metadata only where the loaded model exposes it, else passes through with
  backend-defined failure (same posture as the kokoro pass-through comment in
  tts.rs).
- eSpeak-dependent families (kokoro_tts, sanotts, inflect_v2, piper_tts and
  whatever the spike adds) are NOT compiled into honkbox's family set and
  `AUDIOCPP_STATIC_ESPEAK` is never set; a request naming one → named 400.
- `backends.rs`: `TtsBackend::AudioCpp` variant.

## Acceptance detail

Output is mono PCM16 via the existing f32→i16 clamp path; sample rate honors
`SynthesizeRequest` via the existing resampler; error paths (missing slug,
unsupported family, generation failure) surface as named `TtsError` variants.
