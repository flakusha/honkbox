# FEAT: Server capabilities API

Status: proposed
Priority: P0
Epic: epic-capabilities-contract
Owner: unassigned
Acceptance: `GET /v1/capabilities` returns a machine-readable contract with `api_version`, server info, feature flags, limits, auth mode, and OpenAPI/models pointers, derived from build features/config/catalog.
Verification: `bun run codegen:check`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-Capabilities.md` §Proposed contract
Resources: per-test config + temp catalog dirs; no shared port beyond test server offset

## Change

- Add a DTO in `crates/api-types` with all fields required; register it with
  `utoipa` so `/v1/openapi.json` includes the schema.
- Mount behind the same bearer layer as `/v1/models`; keep `/api/version`
  byte-compatible.
- `api_version` is the wire-contract version, independent of crate `version`.
- Flags come from compile features (`llm-vision`, `vad`), runtime config, and
  catalog scan — never README prose. `controlnet` stays `false` while the 501
  exists; `video` stays `false` until a real job class lands.
- Refresh catalog-derived counts on `/v1/models/refresh` and download commit.

## Acceptance detail

A client can tell whether vision, ControlNet, video, jobs/events, and limits
exist without probing routes or parsing error codes.
