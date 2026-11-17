# FEAT: Opus/FLAC audio output

Status: proposed
Priority: P2
Epic: epic-output-formats
Owner: unassigned
Acceptance: `/v1/audio/speech` supports at least one additional container (Opus or FLAC) alongside WAV/PCM16.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`, `bun run lint:rs`
Links: `.tmp/research/ReviewedOutput-Formats.md` §2
Resources: per-test temp output files; no shared audio fixture path

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
