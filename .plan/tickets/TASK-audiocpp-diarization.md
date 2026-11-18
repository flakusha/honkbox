# TASK: audio.cpp speaker diarization

Status: proposed
Priority: P3
Epic: epic-voice-identity
Owner: unassigned
Acceptance: a diarization surface (speakers × time spans over an uploaded transcript/audio job) backed by an audio.cpp diarization family (Sortformer v2.1 multilingual candidate), capability-gated (`features.audio_cpp.diarization`), closing TASK-diarization-spike.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run codegen:check`; a fixture WAV with ≥2 speakers returns time-bounded speaker spans; absent-family builds return the named unsupported error, never a silent empty result.
Links: `.tmp/research/ReviewedAudioCpp.md` §1, §5, §8; `.plan/tickets/TASK-diarization-spike.md`; `.plan/tickets/TASK-audiocpp-stt-provider.md`
Resources: one local Sortformer GGUF fixture; per-test synthetic multi-speaker WAVs; resample-to-16k via the existing resampler

## Change

- Gated on `TASK-audiocpp-stt-provider.md`; the diarization model loads
  through the same audio.cpp session machinery (family `sortformer_diar_v2`).
- New endpoint or job kind per TASK-diarization-spike's decided shape; spans
  in output-sample time offset across chunks (upstream result semantics),
  converted to seconds server-side.
- Capability flag lives inside the `audio_cpp` feature block from
  TASK-audiocpp-capabilities; honest absence for builds that did not compile
  the family.

## Acceptance detail

Spans are non-overlapping per speaker, cover only detected speech, and are
reported with speaker labels consistent with any speaker-tagged ASR path
later; the capability gate and the 400 path are contract-tested.
