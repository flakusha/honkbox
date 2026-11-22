# TASK: Caption/alt-text output for generated images

Status: proposed
Priority: P2
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: an image-ops-style caption operation can describe an image through
the existing VLM route, writes a bounded `description` into the job bundle and
is exposed as an additive status/bundle field for consumers to store as
accessible alt text.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run lint:rs`.
Links: `.tmp/research/content-experience-gaps-2026-11-20.md` §2 G4,
`TASK-recognition-vlm-ocr-profile.md`, `FEAT-ocr-ingest.md`.
Resources: per-test temp dir; no fixed port.

## Change

- Reuse the VLM/ocr seam; no new captioner dependency or model download.
- Keep caption generation separate from image tagging so either can run alone,
  but allow one model load when both are requested.
- Add the sidecar field without changing existing bundle consumers.

## Non-goals

Perceptual hashing, consumer asset DB schema, model-assisted caption training.