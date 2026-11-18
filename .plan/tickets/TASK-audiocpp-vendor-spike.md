# TASK: audio.cpp vendor/linkage spike

Status: proposed
Priority: P1
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: a probe crate under `.tmp/audiocpp-spike/` builds `audio-cpp-sys` (core-models, CPU, Linux x64) offline-first, runs one trivial bundled-Silero VAD inference, and documents (a) exactly what the crates.io tarball is missing (submodule content) and whether the `vendor.manifest.toml` nested-submodule pattern can fill it, (b) observed multi-static-ggml symbol behavior vs the whisper-rs-sys precedent, (c) the eSpeak-free family set.
Verification: probe `cargo build` log paths recorded in the spike report; bundled VAD inference produces spans on a synthetic buffer; `nm` survey of the built archives for `ggml_*` definitions; per-family eSpeak grep over the upstream tree.
Links: `.tmp/research/ReviewedAudioCpp.md` §3, §4, §7
Resources: off-workspace probe with its own Cargo.toml; no honkbox workspace gates; no model downloads beyond upstream bundled VAD assets

## Change

- Spike only — no production code. The P2/P3 audio.cpp tickets
  (`TASK-audiocpp-stt-provider`, `TASK-audiocpp-tts-provider`,
  `TASK-audiocpp-capabilities`, `TASK-audiocpp-streaming-stt`,
  `TASK-audiocpp-diarization`, `TASK-audiocpp-vad-decision`) are **gated on
  this ticket's verdict**.
- Pin `audio-cpp` / `audio-cpp-sys` at the latest crates.io release (0.9.0,
  published 2026-11-06; upstream submodule rev recorded in findings).
- Probe the offline-first story: fetch the crate, observe the exact missing
  submodule error, then fill the source and record whether a
  `[[vendor.crate]]` + nested-landing entry mirrors the existing
  llama-cpp-sys-4 pattern closely enough for `scripts/vendor-patches.ts`.
- Empirically observe archive/symbol behavior: `nm` the built static archives
  for `ggml_*` definitions and run a two-archive duplicate-symbol link
  experiment to demonstrate which definition wins and in what order.
- Enumerate the eSpeak-free family set by grepping the upstream tree for
  `EspeakPhonemizer` usage per model frontend; candidates to stay
  eSpeak-free: qwen3_asr, citrinet_asr, moonshine_asr, moss_tts_nano,
  pocket_tts (verify each, don't assume).

## Acceptance detail

Verdict is binary in the ticket: SPIKE PASS (with family set + linkage
recommendation: static in-process vs upstream prebuilt shared lib) or BLOCKED
(with exactly what is missing). Findings land in
`.tmp/research/ReviewedAudioCpp.md`; this ticket flips to done-with-findings
or blocked-with-details.
