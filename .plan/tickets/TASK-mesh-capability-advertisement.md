# TASK: Advertise external mesh providers

Status: done (2026-11-20, mesh discovery lane)
Priority: P1
Epic: epic-mesh-integration
Owner: unassigned
Acceptance: `/v1/capabilities` reports configured mesh upstreams, discovered model ids, and provider availability without duplicating the model catalog.
Verification: `bun run codegen:check`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedMesh-ExternalAPIAdvertising.md`, `FEAT-capabilities-api.md`
Resources: per-test config + catalog dirs; mesh stub on assigned port

## Change

- Extend the capabilities DTO with `external.mesh` (enabled, upstream count,
  discovered model ids, last-observation timestamp).
- Derive flags from config + bounded discovery cache, never README prose.
- Keep `api_version` semantics; additive optional fields only.

## Acceptance detail

A client can tell whether an external provider is configured and reachable
without probing `/v1/chat/completions`.
