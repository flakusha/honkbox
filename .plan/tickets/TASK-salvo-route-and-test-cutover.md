# TASK: Cut salvo routes and server tests over

Status: in progress (2026-11-24)
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

## Approach note (2026-11-24)

All 64 mounted handlers (31 files) are inventoried in
`.tmp/salvo_inventory_out.txt`. Route modules and mounted handlers were widened
to `pub(crate)` so a crate-root `salvo_routes` module can call them. Six
file-disjoint batches port the router surface to `salvo_routes/*.rs`; ordinary
handlers are reused through `salvo_support::respond` (axum `IntoResponse`
rendered byte-for-byte by salvo), while SSE/multipart/streaming routes need
native salvo paths. Existing axum oneshot tests stay as handler-level
regression coverage; new salvo parity tests use `tests/support/listener.rs`
(port 0, graceful stop + Drop). The final flip installs `salvo_app::router()`.

## Cutover record (2026-11-24)

All 64 mounted paths now live in six `salvo_routes/*.rs` routers merged by
`salvo_app::router()`. Ordinary handlers are reused through
`salvo_support::respond` (status/headers/body preserved byte-for-byte); SSE and
streamed job/artifact bodies use salvo `ResBody::Stream`; multipart requests
are rebuilt through axum's public `FromRequest` impl so the existing parsers
stay authoritative. `main.rs::serve` binds with `salvo::conn::tcp::TcpListener`
and serves `salvo_app::router(&Arc<AppConfig>)`; the axum `app()` builder stays
only for the existing integration tests. Smoke test against a real server on
127.0.0.1:18081: `/health`, `/api/version`, `/v1/queue`, `/v1/models`,
`/v1/capabilities` → 200, unknown path → 404, gzip applied to
`/v1/openapi.json` (196,325 B → 75,943 B, `content-encoding: gzip`).
Known ceiling: request bodies are collected before dispatch (no incremental
multipart/artifact streaming), and SSE identity headers depend on the bridge.
