# TASK: Cut salvo routes and server tests over

Status: proposed
Priority: P1
Epic: epic-salvo-migration
Owner: unassigned
Acceptance: the ~53 mounted routes with `#[utoipa::path]` are served through
salvo `#[endpoint]` handlers with unchanged paths, methods, status codes and
response bodies; the 40+ server test files using `tower::ServiceExt::oneshot`
move to the listener-on-`127.0.0.1:0` pattern with unique-port ownership and
deterministic `Drop` teardown, preserving the parallel-safe test contract; the
generated OpenAPI/TypeScript artifacts match the new route surface and
`bun run codegen:check` is green. No hand edits to `packages/web-ui/src/gen/`.
Verification:
`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo test --locked -p honkbox-server --test health`,
`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo test --locked -p honkbox-server --test apps`,
`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo check --locked -p honkbox-server`,
`bun run check`, `bun run codegen:check`.
Links: `.tmp/research/salvo-migration-and-support-2026-11-24.md` §"Migration
surface in honkbox master" and §"Tests", `TASK-salvo-migration-spike.md`,
`TASK-salvo-dto-schema-bridge.md`, `epics/epic-salvo-migration.md`.
Resources: Rust target dir `/tmp/cargo-honkbox-salvo`; each test owns a unique
listener on `127.0.0.1:0` with `Drop` teardown — no shared or fixed ports; no
GPU. Split into small per-route/per-file diffs per `.plan/README.md`.

## Change

- Cut routes over in small batches, keeping axum routes as long as both stacks
  must coexist; rely on the DTO bridge for shared schemas.
- Convert each oneshot test file to an owned port-0 listener with teardown,
  preserving assertions and parallel-safety.
- Regenerate OpenAPI/TypeScript artifacts from the resulting route surface.

## Non-goals

Middleware parity (separate ticket), behavior changes to any endpoint, and
manual edits to generated files.
