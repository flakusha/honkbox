# TASK: Surface Silero VAD

Status: done (2026-11-20, audio-vad/streaming-STT lane)
Priority: P1
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: `audio-worker/vad` is forwarded from the server feature set and used by streaming STT chunking, not exposed as a standalone toy route.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run lint:rs`
Links: `.tmp/research/ReviewedSupport-Audio.md` §Expansion table; `audio-worker/Cargo.toml:8-11`
Resources: per-test audio fixture dirs; no shared `ort` session

## Change

- Forward `audio-worker/vad` in `crates/server/Cargo.toml`; keep it opt-in
  until `ort =2.0.0-rc.15` clears license/advisory gates.
- Expose VAD decisions internally to the STT chunker with bounded windows.
- Do not add a `/v1/audio/vad` endpoint; consumption is via streaming STT.

## Acceptance detail

A build without the feature has `audio.vad = false` in capabilities; a build
with it chunks speech without unbounded buffering.
