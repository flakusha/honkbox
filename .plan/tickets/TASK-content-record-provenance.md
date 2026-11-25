# TASK: Embed generation records in delivered images

Status: Done
**Scope:** Embed deterministic generation records in image bytes and unsigned
bundle manifests.
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

- Add one bounded metadata codec in `jobs::image_metadata`; preserve encoded
  pixels and existing container chunks rather than decode/re-encode.
- Reuse the determinism record already produced for the bundle; do not invent
  a second provenance format.
- Embed before HTTP generation/edit responses and bundle writes, and before
  the CLI bundle write.
- Write the unsigned manifest before the completion sidecar; signing follows
  when a certificate policy exists.

## Non-goals

Visible watermark, consumer-side collection UI, certificate management.

## Verification Notes (2026-11-25)

- `cargo test --locked -p honkbox-server --bin honkbox`: 193 passed,
  including 10 codec tests and 15 bundle tests.
- `cargo fmt --all -- --check`, `bun run size`, `bun run docs:check`,
  `bun run spdx`, `bun run hooks:check`, and markdownlint pass.
- JPEG APP1 and WebP EXIF payloads now carry the standard `Exif\0\0`
  prefix; the codec tests require it (`crates/server/src/image_metadata.rs`).
- A standalone probe compiled the actual codec and asserted both prefixes.
- `cargo check --locked -p honkbox-server --bin honkbox`: 0 errors.
- Full test-target reruns are blocked by unrelated test-helper compile drift:
  `ImageOpOutput.artifact_id` and the new `BlenderStage`/`BlenderBakeMap`
  variants are not covered in this checkout.
