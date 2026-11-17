# TASK: External API catalog links

Status: proposed
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
