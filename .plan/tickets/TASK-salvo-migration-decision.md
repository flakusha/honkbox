# TASK: Decide on salvo.rs as the server framework

Status: postponed
Priority: P3
Epic: epic-docs-tooling
Owner: unassigned
Acceptance: the 2026-11-24 decision stands — do not migrate honkbox from axum
0.8.9 to salvo. The 2026-11-20 ruling is unchanged: salvo's HTTP/3/WebTransport
advantage was explicitly ruled out for a local-first/LAN server, while the
migration would rewrite ~53 `#[utoipa::path]` annotations, the shared
`utoipa::ToSchema` DTO derives, the `bun run codegen:check` pipeline and the
`tower::ServiceExt::oneshot` harness across 40+ server test files. Salvo's
`oapi` has no utoipa-6 compatibility layer in upstream docs.
Verification: research record `.tmp/research/salvo-migration-and-support-2026-11-24.md`
(upstream docs + local grep at master `07c5adf`). No build or test was needed
for a decision.
Links: `../research/media-transport-federation-2026-11-20.md` §2;
`TASK-candle-backend-decision.md`.
Resources: none. Reopen only on a revisit trigger below.

## Revisit triggers

1. honkbox serves over the WAN with a measured QUIC/WebTransport need; or
2. axum cannot satisfy a concrete requirement salvo already ships; or
3. a mechanical migration path exists for utoipa annotations and oneshot tests.

## If overridden

Do not flag-day the migration. Port one route with `salvo::oapi` and one test to
the listener-on-port-0 pattern, regenerate `openapi.json`, and compare
`bun run check` before committing the rest. Each test must own a unique port
with `Drop` teardown to keep the parallel-safe test contract.
