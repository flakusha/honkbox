# TASK: Decide on salvo.rs as the server framework

Status: Done
Priority: P3
Epic: epic-salvo-migration
Owner: unassigned
Acceptance: the 2026-11-24 user direction overrides the prior "do not migrate"
ruling; the earlier acceptance (retained below as research history) is
superseded. The decision is now evidence-gated on
`TASK-salvo-migration-spike.md`: one `salvo::oapi` route, one
listener-on-`127.0.0.1:0` test with unique-port ownership and `Drop` teardown,
one generated salvo OpenAPI artifact, and a recorded before/after
`bun run check` cost comparison. Migration then proceeds seam by seam: DTO
bridge, route/test cutover, middleware/codegen. No check result or shipped
work is claimed before it is run.
Verification: research record `.tmp/research/salvo-migration-and-support-2026-11-24.md`
(upstream docs + local grep at master `07c5adf`). The spike owns the
build/test commands
(`CARGO_TARGET_DIR=/tmp/cargo-honkbox-salvo cargo test --locked -p honkbox-server --test salvo_version`,
`bun run check`,
`bun run codegen:check`).
Links: `.tmp/research/salvo-migration-and-support-2026-11-24.md`;
`epics/epic-salvo-migration.md`; `TASK-salvo-migration-spike.md`;
`../research/media-transport-federation-2026-11-20.md` §2;
`TASK-candle-backend-decision.md`.
Resources: none for the decision itself; the spike owns one `127.0.0.1:0`
listener per test with `Drop` teardown and the `/tmp/cargo-honkbox-salvo`
target dir; no GPU.

## Prior ruling (2026-11-20 / 2026-11-24, superseded 2026-11-24)

The 2026-11-24 decision stood — do not migrate honkbox from axum 0.8.9 to
salvo. The 2026-11-20 ruling is unchanged: salvo's HTTP/3/WebTransport
advantage was explicitly ruled out for a local-first/LAN server, while the
migration would rewrite ~53 `#[utoipa::path]` annotations, the shared
`utoipa::ToSchema` DTO derives, the `bun run codegen:check` pipeline and the
`tower::ServiceExt::oneshot` harness across 40+ server test files. Salvo's
`oapi` has no utoipa-6 compatibility layer in upstream docs.

## Revisit triggers

1. honkbox serves over the WAN with a measured QUIC/WebTransport need; or
2. axum cannot satisfy a concrete requirement salvo already ships; or
3. a mechanical migration path exists for utoipa annotations and oneshot tests — being exercised by `TASK-salvo-migration-spike.md` (2026-11-24).

## If overridden

Do not flag-day the migration. This is now the active plan, owned by
`TASK-salvo-migration-spike.md`: port one route with `salvo::oapi` and one test
to the listener-on-port-0 pattern, regenerate `openapi.json`, and compare
`bun run check` before committing the rest. Each test must own a unique port
with `Drop` teardown to keep the parallel-safe test contract. Follow-on seams
are tracked in `epics/epic-salvo-migration.md`.

**Resolved:** 2026-11-25 registry-driven close: git issue 7108624 (registry tip: 74b6257 Konstantin Fedotov Auto-closed: appended .md marker marks TASK-SALVO-MIGRATION-DECISION done)
