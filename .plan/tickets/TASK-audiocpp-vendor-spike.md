# TASK: audio.cpp vendor/linkage spike

Status: done (SPIKE PASS — findings in `.tmp/research/ReviewedAudioCpp.md` §9)
Priority: P1
Epic: epic-audio-streaming
Owner: honkbox-audiocpp-feasibility
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

## Verdict + evidence (landed 2026-11-18)

**SPIKE PASS.** All observation targets hit (full evidence in
`.tmp/research/ReviewedAudioCpp.md` §9; probe under `.tmp/audiocpp-spike/`):

- **Offline-first**: crates.io tarball path auto-clones UNPINNED upstream
  `main` (network + non-reproducible, build.rs `ensure_audio_src` strategy 3);
  the vendor pattern CAN fill it — `[[vendor.crate]]` for audio-cpp-sys +
  `[[vendor.nested]] path = "audio-cpp-sys-0.9.0/audio.cpp"` (nested entries
  land at `vendor/<path>` generically, `scripts/vendor-patches.ts:430-439`)
  hits build.rs strategy 1 (pre-placed tree). Demonstrated end-to-end with a
  pinned v0.9.5 tree (`61c7838baf1f7f34769b5b7bd369c0d78edf9535`, ggml 0.12.0).
- **Build + inference**: `core-models` (engine + bundled silero/marblenet VAD)
  built cold ≈5–6 min on a 7950X, CPU Linux x64, gcc 16.2/cmake 4.3/ninja;
  probe ran bundled-Silero VAD — 4 spans (conf=1) on the bundled speech
  sample, 0 on a synthetic 440 Hz tone (correct speech gating);
  `SPIKE-VAD-OK`. Zero downloads.
- **Three-ggml**: two-archive experiment (GNU ld 2.47) shows silent
  winner-takes-all absorption by archive order (markers 1/1 vs 2/2 on order
  flip); forced pulls fail LOUDLY (`multiple definition of 'ggml_init'`).
  nm survey: audio.cpp `libggml-base.a` defines `ggml_init` + ~2.3k symbol
  lines; `libengine_runtime.a` defines none. Recommendation: static
  in-process (repo convention) with pinned revs; shared-lib prebuilt
  (`AUDIOCPP_BUILD_C_API`) documented as the isolation fallback.
- **eSpeak-free set** (source grep, v0.9.5 tree): consumers are exactly
  kokoro_tts, sanotts, inflect_v2, zipvoice, piper_tts, kitten_tts; every
  honkbox candidate family greps clean (qwen3_asr, moonshine_asr,
  citrinet_asr, parakeet_tdt, confucius4_r2t2, moss, pocket_tts_small,
  qwen3_tts, sortformer_diar(_v2), nemotron_3_diar, moss_transcribe_diarize,
  silero_vad, marblenet_vad). `deny.toml` unchanged.
- **Unresolved (handed to the gated P2/P3 tickets)**: Vulkan/CUDA backend
  builds untested (CPU validated); winner-ggml identity in the real
  honkbox-server link unmeasured until a provider build exists (run `nm`
  once at provider time); per-family weight-license diligence via upstream
  audio.cpp `docs/model_licenses.md` at
  https://github.com/0xShug0/audio.cpp/blob/61c7838baf1f7f34769b5b7bd369c0d78edf9535/docs/model_licenses.md
  at enablement time; the binding exposes its OWN capi.h ABI over upstream
  internals (not upstream's `audiocpp.h`) — pin the binding crate, not just
  upstream.
  binding crate, not just upstream.

**Gate released**: the six P2/P3 audio.cpp tickets may proceed per the §8
sequencing (capabilities + STT provider first).
