# TASK: Bridge shared DTO schemas for salvo

Status: proposed
Priority: P1
Epic: epic-salvo-migration
Owner: unassigned
Acceptance: shared DTOs in `crates/api-types/src/lib.rs` expose a schema usable
by both the existing `utoipa::ToSchema` derive and `salvo::oapi::ToSchema`
(e.g. both derives, or one mechanical bridge) so axum and salvo routes can
coexist during incremental cutover; `ts_rs::TS` output and the generated
TypeScript/OpenAPI contract stay unchanged; `bun run codegen:check` stays green.
No hand edits to generated `packages/web-ui/src/gen/` files, and no schema drift
between the two frameworks is left unrecorded.
Verification:
`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo check --locked -p honkbox-server`,
`bun run check`, `bun run codegen:check`.
Links: `.tmp/research/salvo-migration-and-support-2026-11-24.md` §"One DTO seam
does two jobs", `TASK-salvo-migration-spike.md`,
`TASK-salvo-migration-decision.md`, `epics/epic-salvo-migration.md`.
Resources: Rust target dir `/tmp/cargo-honkbox-salvo`; generated frontend files
are read-only outputs; no ports, no DB, no GPU.

## Change

- Inventory the shared DTO derives and pick the smallest coexistence pattern
  demonstrated by the spike.
- Apply the pattern to the DTO seam without changing wire shapes or `ts_rs` output.
- Re-run the codegen gate; if the artifact changes, regenerate it, never hand-edit it.

## Non-goals

Changing wire schemas, adding a new codegen pipeline, or migrating routes/tests.
