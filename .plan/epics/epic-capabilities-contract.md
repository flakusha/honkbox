# EPIC: Truthful capabilities and version contract

Status: proposed
Priority: P0
Owner: unassigned
Research: `.tmp/research/model-identity-license-download-2026-11-25.md`,
`.tmp/research/media-integration-defaults-and-gaps-2026-11-24.md`,
`.tmp/research/master-next-items-while-ash-cooks-2026-11-24.md`

## Scope

Client applications must be able to ask what honkbox can call instead of probing
routes or reading README. Add a server-wide capability/version surface derived
from compile features, runtime config, and catalog state — never from prose.
Keep the existing `/api/version` response byte-compatible.

## Tickets

- `TASK-support-doc-truth.md` — remove stale video/vision claims (P0).
- `FEAT-capabilities-api.md` — `GET /v1/capabilities` DTO + route + auth (P0).
- `TASK-capabilities-contract-test.md` — flag/route agreement test (P0).
- `TASK-api-version-policy.md` — when to bump `api_version` (P1).
- `TASK-truthful-reserved-capability-rows.md` — reserved `blender`/`3d` queue rows are capacity labels, not callable job kinds (P1).
- `BUG-openapi-embeddings-unlisted.md` — every mounted route appears in the generated OpenAPI document (P1, done).
- `FEAT-model-modality-advertisement.md` — model input/output modalities agree with the live catalog and capability flags (P2).
- `TASK-audiocpp-capabilities.md` — `features.audio_cpp` compile-probed flags and family lists (P2).
- `TASK-download-options-contract.md` — advertise only implemented provider/option support after its validator and staged commit contract land (P1).
- `TASK-model-provenance-contract.md` — advertise identity status/confidence/reason, component coverage and existing-license-gate policy results; never promote unknown to identified/licensed (P1).

## Acceptance

Every advertised `true` flag has a usable route in that build; a 501 or missing
feature gate makes the flag `false`. `/v1/openapi.json` lists the new DTO.
Additive download/provenance blocks are derived from the same validators and
persisted registry verdicts clients consume: reserved providers/options are not
listed as supported, and ambiguous/unknown identity or policy is visible as such
rather than advertised as licensed or callable. `/v1/capabilities` remains behind
the existing read-scope auth hoop on both routers; advertising or omitting a
capability is never the authorization boundary, and negative-auth tests prove
no state or registry mutation.

## Verification

`bun run codegen:check`, `bun run test:server`, `cargo test --locked -p honkbox-server`,
`bun run lint:rs`; Axum/Salvo auth parity and rejected-request no-mutation fixtures.

## Non-goals

Duplicating the OpenAPI path list, semantic version negotiation, MCP tools, or
advertising capabilities merely because weights exist.
