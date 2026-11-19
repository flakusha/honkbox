# EPIC: Model pipeline and multimodal feedback loops

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/ReviewedCreative-BakingModels.md`,
`ReviewedCreative-FeedbackLoop.md`

## Scope

Bake/model validation as external Blender jobs plus a versioned JSON scene spec
that lets a multimodal LLM critique fixed multi-view renders. loop-lore owns the
agent loop; honkbox owns artifacts, renders, and capability truth.

## Tickets

- `TASK-bake-job-contract.md` — bake inputs/outputs/log contract (P2).
- `TASK-blender-bake-extended-maps.md` — metallic/AO/height + thumbnail delta on the bake contract (P0).
- `TASK-model-bundle-validation.md` — `gltf-transform` gate + extension allowlist (P2).
- `TASK-lod-naming-contract.md` — deterministic LOD names/metadata (P3).
- `TASK-scene-spec-schema.md` — versioned JSON scene spec (P2).
- `TASK-render-critique-loop.md` — fixed multi-view render + critique fixture (P3).

## Acceptance

A bundle fails validation with a named error before engine import; the scene
spec is replayable without an LLM; no byte-equality claim across renderers.

## Verification

`bun run test:server`, `bun run test:core`, external `gltf-transform` and Blender
fixtures; no engine SDK or model download in CI.

## Non-goals

In-process `bpy`, FBX/USD parsers, default Draco, and honkbox-owned agent state.
