# TASK: Salvo middleware and codegen parity

Status: in progress (2026-11-24; codegen union landed, middleware pending)
Priority: P1
Epic: epic-salvo-migration
Owner: unassigned
Acceptance: tower-http layers (compression and auth-adjacent middleware) are
replaced by the equivalent salvo middleware with observable parity for
compression/SSE/multipart behavior; OpenAPI is produced by the salvo `oapi`
artifact path instead of utoipa while `scripts/codegen-openapi-check.ts` and
`bun run codegen:check` stay green; generated `packages/web-ui/src/gen/` files
are regenerated only, never hand-edited; the spike's recorded `bun run check`
cost comparison is updated with the post-parity figure.
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
