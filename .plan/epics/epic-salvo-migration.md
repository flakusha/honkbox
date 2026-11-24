# EPIC: Incremental axum → salvo cutover

Status: in progress
Priority: P1
Owner: unassigned
Research: `.tmp/research/salvo-migration-and-support-2026-11-24.md`,
`TASK-salvo-migration-decision.md` (2026-11-24 user override).

## Scope

Move honkbox's HTTP surface from axum 0.8.9 to salvo 1.x incrementally, starting
with the bounded spike described by the research and proceeding only on its
evidence. The migration touches four coupled seams: ~53 mounted routes with
`#[utoipa::path]` annotations, the shared `crates/api-types/src/lib.rs` DTO seam
(`utoipa::ToSchema` + `ts_rs::TS`), 40+ server test files using
`tower::ServiceExt::oneshot`, and the OpenAPI/codegen pipeline feeding
`packages/web-ui/src/gen/`. Each seam is a separate ticket; no flag day.

## Tickets

- `TASK-salvo-migration-spike.md` — in progress, P1: one `salvo::oapi` route,
  one port-0 listener test with `Drop` teardown, one generated salvo OpenAPI
  artifact, and a recorded before/after `bun run check` cost comparison.
- `TASK-salvo-dto-schema-bridge.md` — proposed, P1: coexist
  `utoipa::ToSchema` and `salvo::oapi::ToSchema` on shared DTOs without hand
  edits to generated files.
- `TASK-salvo-route-and-test-cutover.md` — proposed, P1: cut ~53 route
  annotations and 40+ oneshot test files over to salvo's endpoint/listener
  pattern, preserving route behavior and parallel-safe test contracts.
- `TASK-salvo-middleware-and-codegen.md` — proposed, P1: replace tower-http
  layers with salvo middleware and move OpenAPI generation to the salvo oapi
  artifact while keeping the codegen gate green.

## Acceptance

The spike records all four required evidence items (route, test, generated
artifact, check-cost comparison) without hand-editing generated files. On that
evidence, follow-on work lands with: shared DTO schemas valid for both
frameworks and `ts_rs` output unchanged; every route and oneshot test cut over
with per-test port ownership and deterministic teardown; middleware parity and
salvo-generated OpenAPI with `bun run codegen:check` green. No check result is
claimed before it is run.

## Verification

- `CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo test --locked -p honkbox-server --test health`
- `bun run check`
- `bun run codegen:check`

## Non-goals

HTTP/3, WebTransport, a flag-day rewrite, and hand edits to generated
`packages/web-ui/src/gen/` files.
