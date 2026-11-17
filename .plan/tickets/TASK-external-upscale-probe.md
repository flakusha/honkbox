# TASK: External upscale probe

Status: proposed
Priority: P2
Epic: epic-editor-integration
Owner: unassigned
Acceptance: one local super-resolution binary (Real-ESRGAN or waifu2x) runs through the process supervisor with version/license/argv recorded in the sidecar.
Verification: opt-in local fixture; `bun run lint:rs`, `bun run deny`
Links: `.tmp/research/ReviewedImageEdit-Operations.md` §Tier 2
Resources: per-test output dirs; process group killed in `finally`/drop; no CI download

## Change

- Depends on `TASK-image-ops-job-route.md` and `TASK-process-supervisor.md`.
- Cap scale factor, edge, and batch size; no Rust linkage to LGPL libraries.
- Record binary path/version/license and exact argv; cancel kills the group.

## Acceptance detail

A failed run leaves no orphan process; output dimensions match the bounded
request; no byte-equality claim across models.
