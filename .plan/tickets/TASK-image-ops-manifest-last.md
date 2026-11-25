# TASK: Manifest-last image-op publication

Status: In Progress
**Scope:** Manifest-last image-op bundle publication with verified manifest gate
Priority: P1
Epic: epic-editor-integration
Owner: unassigned
Acceptance: image-op bundles use the Blender manifest convention (ordered entries, command hash, request identity, tool/recipe versions, warnings, outcome) with atomic writes and `manifest.json` last; a terminal result without a verified manifest reports `publication_incomplete`, never implicit success.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server --test image_ops`, `bun run codegen:check`
Links: `.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md` §H-DCC1, `TASK-image-ops-job-route.md`, `TASK-blender-terminal-hardening.md`
Resources: per-test job dirs; no shared GPU slot or fixed port

## Change

- Converge image-op bundles on the Blender manifest convention: ordered
  `{name, kind, artifact_id, sha256, bytes}` entries, command hash,
  `request_id`/job identity, tool/recipe versions, warnings, outcome.
- Atomic artifact writes via the existing `jobs.rs` helpers +
  `Registry::put_bytes`; `manifest.json` written last. No second bundle
  format, no new runtime, no new route.
- Terminal result without a verified manifest is `publication_incomplete`.
  Consumers gate on the manifest, never on SSE or the response alone.

## Acceptance detail

- Structural tests only (manifest order, sha chain, last-write ordering,
  incomplete terminal state); no byte-equality claim across runs.
- Partial outputs stay explicit; cancel between units writes no bundle,
  same as the existing route.
