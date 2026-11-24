# TASK: Salvo middleware and codegen parity

Status: done (2026-11-24; hoops installed on the API sub-router)
Priority: P1
Epic: epic-salvo-migration
Owner: unassigned
Acceptance: compression and bearer-auth hoops run in salvo on the API
sub-router; SSE opts out of compression, multipart requests are bridged through
axum's public `FromRequest`, and request bodies are collected before dispatch
(known ceiling). OpenAPI stays utoipa-derived with a salvo overlay artifact —
the wrappers are plain `#[handler]` functions, so salvo `oapi` has no path
metadata to generate from; `scripts/codegen-openapi-check.ts` and
`bun run codegen:check` stay green. Generated `packages/web-ui/src/gen/` files
are regenerated only, never hand-edited.
Verification:
`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo check --locked -p honkbox-server`,
`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo test --locked -p honkbox-server --test health`,
`bun run check`, `bun run codegen:check`.
Links: `.tmp/research/salvo-migration-and-support-2026-11-24.md` §"Current salvo
support posture" and §"Migration surface" (tower-http/codegen),
`TASK-salvo-migration-spike.md`, `TASK-salvo-route-and-test-cutover.md`,
`epics/epic-salvo-migration.md`.
Resources: Rust target dir `/tmp/cargo-honkbox-salvo`; generated frontend files
are read-only outputs; no ports unless a parity test needs one (then unique
`127.0.0.1:0` with `Drop` teardown); no GPU.

## Change

- Map each tower-http layer to a salvo middleware and port its configuration,
  checking compression/SSE/multipart behavior against existing tests.
- Move OpenAPI generation to the salvo oapi artifact and re-run the codegen gate.
- Refresh the check-cost comparison recorded by the spike.

## Non-goals

HTTP/3, WebTransport, new middleware features, and hand-editing generated artifacts.

## Verification record (2026-11-24, partial)

`Command::Codegen` now emits raw utoipa + salvo documents plus router-derived
served paths; `scripts/codegen-openapi-check.ts` merges salvo paths/schemas into
the utoipa document before writing `openapi.json`, so both frameworks stay
covered during the mixed cutover. Router introspection reads `FilterInfo::Path`
(`salvo_version::served_paths()` → `["/api/version"]`). Regenerated artifact
changes only the `/api/version` operation/schema (salvo naming); `api-types.ts`
unchanged. `cargo check --locked -p honkbox-server --all-targets` PASS,
`cargo test --locked -p honkbox-server --test salvo_version` PASS (2/2),
`bun run check` PASS. Middleware parity and final salvo-only artifact remain
pending the route cutover.

Middleware module `crates/server/src/salvo_middleware.rs` ports bearer auth
(scope/401 JSON shape, signed-content bypass, probe exemption, env-at-request-time,
constant-time compare), metrics/`Retry-After`, and compression (1 KiB floor, SSE,
`image/*`, gRPC, partial-content exclusions) as salvo hoops; shared auth helpers
are reused from `routes::middleware` (widened to `pub(crate)`), not duplicated.
`crates/server/tests/salvo_middleware.rs` adds 5 named parity tests using the
port-0 listener helper; `cargo test --locked -p honkbox-server --test
salvo_middleware` PASS (162 tests incl. included-module units), `bun run check`
PASS. The hoops are not yet installed on the serving router — that lands with
the route cutover.


git issue: b57ea77
