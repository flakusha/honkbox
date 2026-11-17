# TASK: Image-ops job route

Status: proposed
Priority: P1
Epic: epic-editor-integration
Owner: unassigned
Acceptance: `POST /v1/image-ops` returns a job id, emits progress on `/v1/events`, and writes the result bundle via the existing jobs path.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedImageEdit-CommandProtocol.md`
Resources: per-test job dirs; no shared GPU slot or fixed port

## Change

- Depends on `TASK-image-ops-command-dto.md`.
- Register a new `JobKind`, acquire admission, and honor cooperative cancel.
- Sidecar records command hash, input/output hashes, seed, tool version,
  attempts, and outcome; no token or full URL.

## Acceptance detail

A plugin can submit, poll, and cancel one op without probing trial requests;
capabilities list supported ops.
