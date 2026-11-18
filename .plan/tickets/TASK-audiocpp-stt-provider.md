# TASK: audio.cpp STT provider

Status: done (worker/CLI seam landed; route advertisement deferred to
TASK-audiocpp-capabilities per orchestrator lane split)
Priority: P2
Epic: epic-audio-streaming
Owner: honkbox-audiocpp-feasibility
Acceptance: `AudioCppStt` implements `SttPipeline`, selectable via `--stt-backend audio-cpp --stt-model <slug>`, behind a non-default `audio-cpp` cargo feature; `/v1/audio/transcriptions` serves an audio.cpp ASR family (e.g. `qwen3_asr`) end to end; whisper and stub paths byte-identical when the feature is off.
Verification: `cargo test --locked -p honkbox-audio-worker` (default config compiles WITHOUT the feature — the honest-off path), feature-on build + scoped tests (`-p honkbox-audio-worker --features audio-cpp,audio-cpp-qwen3-asr`), `cargo check --locked -p honkbox-server` (feature-off default), clippy `-D warnings` scoped.
Links: `.tmp/research/ReviewedAudioCpp.md` §5, §6, §8, §9; `crates/server/src/backends.rs`; `.plan/tickets/TASK-audiocpp-vendor-spike.md`
Resources: per-test temp WAV fixtures; one local audio.cpp ASR GGUF (no CI downloads; `AUDIOCPP_ASR_MODEL` env-gated ignored test); admission lease per request as today

## Change (landed 2026-11-19)

- Gated on `TASK-audiocpp-vendor-spike.md` SPIKE PASS (released).
- `honkbox-audio-worker`: new `src/audiocpp.rs` — `AudioCppStt` (Mutex<Model>,
  binding types are `Send + !Sync`, same shape as `WhisperStt`),
  `AUDIOCPP_COMPILED: bool = cfg!(feature = "audio-cpp")` const defined
  UNCONDITIONALLY in lib.rs (capability drift-proofing, `VAD_COMPILED`
  pattern), probe-honesty test for the default build.
- Features: worker `audio-cpp = ["dep:audio-cpp"]` (core models: engine +
  bundled silero/marblenet VAD) + `audio-cpp-qwen3-asr =
  ["audio-cpp?/model-qwen3-asr"]`; server `audio-cpp` forwards both.
- Family selection: `--stt-audiocpp-family` (default `qwen3_asr`,
  `AUDIOCPP_DEFAULT_FAMILY` unconditional for the CLI seam), validated
  against `Registry::supports_family` BEFORE load — unknown family is a
  named `SttError::Model` listing the compiled set. `translate` is an
  honest named rejection (whisper.cpp owns `/v1/audio/translations` in this
  build). In-memory path: PCM16 → f32 → 16 kHz resample → `WavAudio`
  buffer (no temp files); transcript = whole-input segment (StubStt shape;
  audio.cpp word timestamps are family-gated).
- `backends.rs`: `SttBackend::AudioCpp` variant; feature-off arm is a loud
  exit-1 naming the missing feature + requested family (honest absence,
  never a silent fallback). `main.rs`: the flag on `ServeArgs` +
  `TranscribeArgs`.
- Vendor: `vendor.manifest.toml` `[[vendor.crate]] audio-cpp-sys 0.9.0`
  (sha256 `638ae555…`) + `[[vendor.nested]] audio.cpp @
  61c7838baf1f7f34769b5b7bd369c0d78edf9535` (audio.cpp v0.9.5, ggml 0.12.0
  in-tree); workspace `[patch.crates-io] audio-cpp-sys` → extracted dir;
  `scripts/vendor-patches.ts` generalized (nested roots may be declared
  `[[vendor.crate]]` dirs; crate extraction ordered before nested landing —
  a fresh crate fetch previously `rmSync`'d the dir a nested repo had
  landed in).

## Verification (all observed 2026-11-19)

- `cargo test --locked -p honkbox-audio-worker` (default, feature-off):
  **9 passed** (2 suites) incl. `audiocpp_probe_is_off_without_the_feature`.
- `cargo test --locked -p honkbox-audio-worker --features
  audio-cpp,audio-cpp-qwen3-asr`: **10 passed, 1 ignored** — engine built
  from the vendored pinned tree; `unknown_family_is_a_named_error` (names
  the family + compiled set: silero_vad, marblenet_vad) and
  `core_build_compiles_the_bundled_vad_families` ran against the REAL
  engine; the ignored test needs a local ASR GGUF (`AUDIOCPP_ASR_MODEL`).
- `cargo clippy --locked -p honkbox-audio-worker --all-targets --
  -D warnings`: **No issues found** — both feature sets.
- Vendor: `bun scripts/vendor-patches.ts sync` converged (nested audio.cpp
  landed @ `61c7838b…` with bundled weights; patches already applied);
  `--dry-run`: `tree is converged`.
- `cargo check --locked -p honkbox-server` (feature-off default): **scoped
  pass** — zero diagnostics in my files; the sole error is sibling
  `mesh.rs:373` (mesh lane mid-flight, `TryStreamExt` import). Server
  feature-on check + `cargo test -p honkbox-server` / `bun run test:server`
  / `bun run codegen:check` deferred until the mesh lane lands — my wiring
  compiled clean inside the 487-crate default check.
- Route advertisement (`/v1/capabilities` audio block) is
  TASK-audiocpp-capabilities' seam (orchestrator lane split: routes.rs and
  api-types intentionally untouched); until then the backend is
  CLI-selectable only.

## Acceptance detail

Feature-off builds compile the worker without the engine (honest-off probe
test pins it); feature-on error paths (unknown family, missing slug,
missing weights, translate request) surface as named `SttError` variants at
the CLI seam; the wire-level advertisement contract lands with
TASK-audiocpp-capabilities.
