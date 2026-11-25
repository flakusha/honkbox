# TASK: Blender terminal hardening

Status: In Progress
**Scope:** Terminal status, manifest-before-result, partial outputs and idempotency for Blender jobs
Priority: P0
Epic: epic-blender-assets
Owner: unassigned
Acceptance: failed/cancelled Blender jobs have fetchable terminal status without requiring a manifest; `job_result` fires only after durable artifacts/manifest/status commit; partial outputs explicit; write errors propagate; duplicate submit is idempotent
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server --test blender_jobs`, `bun run codegen:check`
Links: `.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md` (H-DCC4); `.plan/tickets/FEAT-blender-job-route.md`; `crates/server/src/routes/blender_exec.rs`; `crates/server/src/jobs_registry.rs`
Resources: per-test temp data dirs; stub Blender recipe; no shared GPU slot; one lease per test

## Change

- Terminal status (`failed`/`cancelled`) is fetchable without a manifest; absence of manifest never masks terminal state.
- `job_result` follows durable artifact/manifest/status commit, never precedes it; consumers gate publication on the manifest.
- Partial outputs are explicit (named partial, never promoted as success); write-error propagation on every artifact path.
- Idempotency/retention policy: duplicate submit executes once; changed-command conflict is explicit; retention bounds unchanged.
- No new job kind, no `/godot`/`/bevy`/`/export` route.

## Acceptance detail

1. Failed and cancelled jobs return terminal status with no manifest present.
2. Live `job_result` never precedes final manifest write; SSE remains a discovery hint.
3. Partial outputs are listed as partial and publish no success asset.
4. Duplicate submit with identical `request_id` executes once; changed command is an explicit conflict.
5. Cancel kills the process group; restart reconciles without duplicate publication.

## Non-goals

- New engine SDK, build, MCP, or second orchestration.
- Byte-identical pixels across Blender versions/hardware.
