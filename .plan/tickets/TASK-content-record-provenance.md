# TASK: Embed generation records in delivered images

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: every emitted PNG/JPEG/WebP carries prompt, seed, model hash and
honkbox version in a standard metadata slot, and an unsigned C2PA-shaped
manifest is written to the job bundle from the same emission point.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server
--test images`, `bun run codegen:check`; a bytes-level test reads the chunks
back from a generated file.
Links: `.tmp/research/content-experience-gaps-2026-11-20.md` §1 G1, §3 G8,
`TASK-determinism-contract.md`.
Resources: per-test temp output dir.

## Change

- Add one bounded metadata writer next to `jobs::write_atomic`, using the
  existing `image` crate for PNG `tEXt`/`iTXt` where possible.
- Reuse the determinism record already produced for the bundle; do not invent
  a second provenance format.
- Manifest starts unsigned; signing follows when a certificate policy exists.

## Non-goals

Visible watermark, consumer-side collection UI, certificate management.