# EPIC: Model pipeline and deterministic multimodal feedback

Status: in progress
Priority: P1
Owner: unassigned
Research: `.tmp/research/creative-pipeline-promotion-2026-11-25.md`,
`.tmp/research/invoke-ai-patterns-memory-rendering-discovery-2026-11-24.md`,
`.tmp/research/texture-creation-workflows.md`,
`.tmp/research/loop-lore-godot-actionables.md`,
`.tmp/research/gen-flows-props-characters-locations-2026-11-20.md`

## Scope

External Blender/OIIO material preparation and model validation, a versioned
hash-addressed scene contract, bounded fixed-view renders, and a typed critique
sidecar that replays without an LLM. honkbox owns artifacts, validation,
renders, provenance, and capability truth. loop-lore owns authoring UI and any
future agent policy; neither side needs an agent framework to replay a bundle.

## Current landed state (source-verified 2026-11-25)

- **Landed foundations:** `FEAT-model-modality-advertisement.md` publishes
  nested `modality:{input,output}` and the legacy flat list. Image input for an
  LLM requires compiled vision support plus a sibling mmproj unless an operator
  override exists; SD image output does not imply critique/vision input.
- **Landed foundations:** content-addressed artifact registry, job bundle/event
  seam, canonical JSON/hash, and derived seed streams are available for replay.
- **Landed contract/tooling:** `TASK-bake-job-contract.md` and internal
  `BakeContract`/`BakeManifest` exist; `TASK-texture-postprocess-oiio.md` and
  `TASK-material-map-validation.md` provide crate/tooling map validation and
  ORM/mip/colorspace sidecars.
- **Partial, not end-to-end:** internal extended bake/thumbnail types and
  fake-recipe role tests exist, but current public Blender DTOs and the real
  recipe expose only color/normal/roughness plus turntable. Reconcile
  `TASK-blender-bake-extended-maps.md`; do not file a duplicate.
- **Proposed:** scene-spec admission/canonical persistence, typed fixed-view
  rendering, critique sidecar validation, and deterministic replay. No current
  scene-spec/critique route or job kind exists.
- **Proposed downstream:** `TASK-model-bundle-validation.md`,
  `TASK-lod-naming-contract.md`, and `FEAT-game-engine-export-bundles.md` remain
  separate validation/export work.

## Tickets

- `TASK-bake-job-contract.md` — bake inputs/outputs/log contract (P2, done for the internal contract lane).
- `TASK-blender-bake-extended-maps.md` — metallic/AO/height + thumbnail delta (P0, in progress; public DTO/real-recipe reconciliation remains as recorded above).
- `TASK-texture-postprocess-oiio.md` — mip/tile/colorspace/ORM sidecar (P0, done in crate/tooling; upstream of scene materials).
- `TASK-material-map-validation.md` — map roles/dimensions/colorspaces/hashes (P2, done in crate/tooling; consume rather than duplicate).
- `TASK-scene-spec-schema.md` — `scene-spec/v1`, semantic pre-admission validation, canonical persistence, deterministic view/seed plan (P2, proposed).
- `TASK-render-critique-loop.md` — bounded fixed views + `critique/v1` typed edits and no-LLM replay (P1, proposed).
- `TASK-model-bundle-validation.md` — `gltf-transform` gate + extension allowlist (P2, proposed).
- `TASK-lod-naming-contract.md` — deterministic LOD names/metadata (P3, proposed).
- `FEAT-game-engine-export-bundles.md` — engine-agnostic bundle packaging (P2, proposed; `epic-game-engine-export`).

## Dependency order

1. Keep modality advertisement, artifacts/jobs/events, and deterministic
   canonicalization as shared prerequisites.
2. Land `TASK-scene-spec-schema.md` first: shape + semantic validation, stored
   hash closure, safe relative paths, no wall-clock input, canonical digest.
3. Land the fixed-view render half of `TASK-render-critique-loop.md` on that
   contract; use the existing Blender supervisor/lease and do not confuse a
   turntable strip with named front/side/top views.
4. Land typed sidecar validation and deterministic edit replay before any
   optional vision critic. A future critic must be selected from truthful
   catalog image-input capabilities and produce the same typed sidecar.
5. Feed validated models/materials into existing bundle validation/export and
   LOD tickets. Bake/OIIO remain upstream and are not reimplemented here.

## Acceptance

- Invalid scene/spec/sidecar/bundle inputs fail with named errors before job
  admission or engine import; unresolved hashes and unsafe paths never reach a
  tool process.
- Fixed front/side/top renders and every edit/replay version are retained with
  canonical scene, tool/recipe pins, ordered view ids, derived seeds, artifact
  ids, and sha256 hashes.
- Replay requires no LLM, network, agent framework, or GUI and repeats the
  recorded structural plan. No byte-equality claim across renderers.
- Resource isolation is per job: temp staging/bundle, Blender/validator process
  group, lease, bounded threads/RSS/time/logs/images/prompts/output, and
  stage/view-boundary cancellation.

## Verification

`cargo test --locked -p honkbox-external-tools`,
`cargo test --locked -p honkbox-server --test blender_jobs`,
`cargo test --locked -p honkbox-server --test capabilities`,
`bun run test:server`, `bun run codegen:check`; opt-in Blender fixture and
external `gltf-transform inspect/validate <bundle.glb>`. No engine SDK or model
download in CI.

## Non-goals

In-process `bpy`, FBX/USD parsers, default Draco, honkbox-owned agent state,
autonomous/unbounded critique, arbitrary tool calls, duplicate bake/OIIO/LOD/
export tickets, scene-editor UI, model downloads in tests, or cross-renderer
byte equality.
