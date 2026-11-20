# TASK: External API catalog links

Status: done
Priority: P3
Epic: epic-mesh-integration
Owner: unassigned
Acceptance: discoverability metadata links to `/v1/capabilities` and `/v1/openapi.json` using RFC 8631/9727 patterns without a parallel schema.
Verification: `bun run codegen:check`, documentation review
Links: `.tmp/research/ReviewedMesh-ExternalAPIAdvertising.md` §RFCs
Resources: none (metadata ticket)

## Change

- Add `service-desc`/`service-meta` link relations pointing at the canonical
  OpenAPI document and capabilities endpoint.
- Optionally publish an `api-catalog` document that references those endpoints;
  do not duplicate the route or feature inventory.
- Keep `/api/version` byte-compatible.

## Acceptance detail

A client following the links reaches the same contract it already uses; no
second source of truth is introduced.

## Verification Notes (2026-11-20)

`GET /api/version` now emits one `Link` header with `service-desc` →
`/v1/openapi.json` and `service-meta` → `/v1/capabilities` (RFC 8631/9727
relations), built as a static `concat!` in
`crates/server/src/routes/health.rs`. The `Json<Version>` body is unchanged; the
optional `api-catalog` document was deliberately not added — it would be a
second inventory with no consumer.

`crates/server/tests/capabilities.rs` adds
`version_advertises_canonical_contract_links` and keeps
`api_version_is_the_wire_contract` unchanged. `cargo test --locked -p
honkbox-server --test capabilities`: 152 passed, 0 failed. `bun run check`
green (clippy, dprint, oxlint, tsc, hooks, `audit-gpu-slot`, codegen:check,
49 mounted routes).
