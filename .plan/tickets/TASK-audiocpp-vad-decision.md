# TASK: VAD consolidation decision (ort vs audio.cpp bundled Silero)

Status: proposed
Priority: P3
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: a written decision with evidence — keep `ort`-SileroVad vs switch VAD to audio.cpp's bundled silero_vad (which would retire the `ort`/ONNX Runtime dependency) — plus, if switching, a migration plan keeping `features.vad` semantics identical.
Verification: side-by-side span comparison on identical fixtures (span boundaries within tolerance documented in the findings); `cargo test --locked -p honkbox-audio-worker`, `bun run test:server` for whichever path lands; the loser path is removed with no shims if the switch happens.
Links: `.tmp/research/ReviewedAudioCpp.md` §5, §7; `crates/audio-worker/src/vad.rs`; `.plan/tickets/TASK-audio-vad-surface.md`; `.plan/tickets/TASK-audiocpp-vendor-spike.md`
Resources: per-test WAV fixtures only; bundled silero weights from whichever engine is under test; no new downloads

## Change

- Gated on `TASK-audiocpp-stt-provider.md` landing (audio.cpp core already
  in the binary); otherwise this ticket stays parked.
- Compare on: span agreement on the fixture corpus, latency, memory, build
  time impact of carrying audio.cpp core **solely for VAD** (the honest cost
  if no other audio.cpp family is compiled), and the `vad` feature's
  non-default posture.
- If the switch lands: `vad.rs` re-routes to audio.cpp sessions, `ort` leaves
  Cargo.toml, `VAD_COMPILED` semantics and `/v1/capabilities` `features.vad`
  unchanged.

## Acceptance detail

Default behavior (feature off) is byte-identical either way; the decision
record cites measured numbers, not vibes, and names the rev/versions
compared.
