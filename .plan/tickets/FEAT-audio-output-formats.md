# FEAT: Opus/FLAC audio output

Owner: omp (2026-11-19 user call: Opus/FLAC desirable now)
Priority: P0
Epic: epic-output-formats
Status: finished
Acceptance: `/v1/audio/speech` supports at least one additional container (Opus or FLAC) alongside WAV/PCM16.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`, `bun run lint:rs`
Links: `.tmp/research/ReviewedOutput-Formats.md` §2
Resources: per-test temp output files; no shared audio fixture path

## Verification (2026-11-19)

- `cargo test -p honkbox-audio-worker --lib`: 19/19 (incl. new
  `flac_streaminfo_matches_input`, `opus_head_matches_input_and_is_ogg`,
  `opus_roundtrip_preserves_sample_count`). Failures found and fixed:
  Opus frame `rate/20` (50 ms, invalid) → `rate*20/1000`; FLAC
  STREAMINFO 3-byte length misread as u32.
- `cargo test -p honkbox-server --test audio`: exit 0; speech subset 6/6
  incl. new `speech_flac_and_opus_containers` (magic + MIME). Fixed E0277:
  `?` in `audio_speech.rs` handler → explicit error returns.
- `cargo test -p honkbox-api-types`: 16 passed; openapi.json regenerated
  (40/40 routes, flac/opus docs present); `dprint fmt` applied to gen dir.
- `cargo deny check licenses`: ok. Scoped clippy queued (shared lock).
- 21:05 closeout: worker tests 20/20 (new `opus_single_packet_stays_decodable`
  pins the end-trim boundary; roundtrip sample-count exact there too).
  `cargo clippy -p honkbox-audio-worker --all-targets -- -D warnings`: clean
  (the actual deny-gate). `cargo fmt --check -p honkbox-audio-worker`: clean.
  No API-surface change since the 40/40 openapi regen — no regen needed.

## Change

- Worker boundary is mono PCM16 (`SynthesizedAudio`); add a
  `(sample_rate, &[i16]) -> Vec<u8>` encoder in the existing dispatch.
- Pure-Rust Opus crates exist but are young (0.1.x) — vet maturity before
  adding a dep; `deny.toml` license/advisory gates apply.
- MP3 encode is explicitly out (in-tree `nanomp3` is decode-only; C binding
  would be needed). Lossy WebP is likewise out of scope here.

## Acceptance detail

Response MIME matches container; no shared global encoder state; cancellation
and body-size limits still apply. Do not add MP3 in this ticket.
