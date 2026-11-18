# TASK: audio.cpp capability advertisement

Status: proposed
Priority: P2
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: `/v1/capabilities` advertises `features.audio_cpp = { built, stt_families, tts_families }` where `built` is the compile probe and the family lists are compile-time constants of the enabled cargo features; a build without the feature reports honest absence (no key drift); contract test cross-checks the document against the OpenAPI paths.
Verification: `bun run test:server`, `bun run codegen:check`, `cargo test --locked -p honkbox-server`; feature-on and feature-off snapshots both asserted in the contract test.
Links: `.tmp/research/ReviewedAudioCpp.md` §6, §8; `crates/server/src/routes.rs` (FeatureFlags, routes.rs:670-694); `crates/audio-worker/src/lib.rs` (VAD_COMPILED, lib.rs:20); `.plan/tickets/TASK-audiocpp-vendor-spike.md`
Resources: none beyond existing test infrastructure

## Change

- Gated on `TASK-audiocpp-vendor-spike.md` SPIKE PASS; lands with or right
  after the provider tickets.
- `honkbox-audio-worker` exports `AUDIOCPP_COMPILED` plus the compile-time
  family tables (mirroring how `crate::formats::sd_ingest_names()` is the
  single source of truth for formats).
- `FeatureFlags` gains the `audio_cpp` struct; the document cannot drift from
  the binary because both sides read the same `cfg!` probes.
- Unknown-family requests get the named-400 treatment; the capability list is
  the contract the 400 cites.

## Acceptance detail

Both snapshots (feature on/off) are contract-tested; a family listed in
`/v1/capabilities` but not compiled is a test failure (drift guard), matching
the existing controlnet/lora honesty posture.
