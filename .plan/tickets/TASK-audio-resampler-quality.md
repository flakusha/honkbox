# TASK: Resampler quality

Status: proposed
Priority: P2
Epic: epic-audio-streaming
Owner: unassigned
Acceptance: audio resampling uses a windowed-sinc or audited crate with anti-alias filtering, verified at 8k→16k and 44.1k→16k.
Verification: `cargo test --locked -p honkbox-audio-worker`, `bun run lint:rs`, `bun run deny`
Links: `.tmp/research/ReviewedSupport-Audio.md` §Current; `audio-worker/src/lib.rs:129-210`
Resources: per-test generated sine/sweep fixtures in runner tmp; no shared audio device

## Change

- Replace the linear resampler at `audio-worker/src/lib.rs:129-210` with a
  bounded windowed-sinc implementation or a vetted MIT/Apache crate.
- Keep the worker crate zero-heavy-dep where possible; any dep must pass
  `deny.toml` license/advisory gates.
- Add deterministic unit tests for the two named conversion rates.

## Acceptance detail

No aliasing at Nyquist in the tested conversions; output length and sample rate
remain compatible with `encode_wav` and the TTS boundary.
