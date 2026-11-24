# EPIC: Blender asset workstream

Status: in progress
Priority: P1
Owner: main session
Research: `.tmp/research/shaders-materials-uiux-asset-pipeline-2026-11-20.md`,
`.tmp/research/texture-creation-workflows.md`,
`.tmp/research/loop-lore-godot-actionables.md`,
`.tmp/research/gen-flows-props-characters-locations-2026-11-20.md`,
`.tmp/research/creative-pipeline-promotion-2026-11-25.md`

## Scope

Headless Blender jobs for UV prep, retopology, texture attachment, baking,
preview rendering, rigging, weight transfer, and export. Built-in operators
first; OSS addons after license recording and a headless operator probe; paid
addons detect-only. honkbox launches and manages Blender as an external
process; addons live inside Blender and are never incorporated into honkbox.

## Current landed state (source-verified 2026-11-25)

- **Landed:** Blender/addon/operator discovery and capability truth in
  `crates/external-tools/src/lib.rs`, `blender_addons.rs`, and
  `blender_addons.py`. UI-only or unproven addons are not offered.
- **Landed:** typed `AssetJob`/closed internal stages, embedded
  `blender_asset.py`, recipe/source hashes, manifests, stage progress, and
  stage-boundary cancellation in `blender_assets.rs`.
- **Landed:** `POST /v1/jobs/blender`, artifact-id/sha input resolution,
  pre-admission engine/version/stage/output gates, `ResourceKind::Blender`
  lease, supervised process group, atomic job bundle commit, artifact
  promotion, and `job_result` events in `routes/blender*.rs`.
- **Landed runtime preview:** `turntable {frames,resolution}` renders a
  bounding-box orbit strip; the route commits it as WebP. It is not a typed
  front/side/top scene render.
- **Contract-only gap:** internal `BakeMap`/`AssetStage` planning types mention
  metallic/AO/height and thumbnail, and fake-recipe tests cover their manifest
  roles. Current public `BlenderStage`/`BlenderBakeMap` DTOs and the real Python
  recipe do not expose or execute those variants (`BAKE_TYPES` has only
  color/normal/roughness; no thumbnail dispatch). The extended-bake ticket's
  “done” header therefore does not establish end-to-end HTTP/runtime support.
- **Related landed tooling, not a Blender-route stage:** OIIO `.tx`/ORM and
  sidecar logic exists in `texture_postprocess.rs`; `ToolId::Oiio` exists but is
  omitted from the generic `ToolId::ALL` discovery list. Do not claim a wired
  Blender→OIIO job chain.

## Tickets

- `FEAT-blender-job-route.md` — typed `POST /v1/jobs/blender` job kind + route (P1, done; source-verified route, supervised run, artifact-store input, lifecycle/events, and committed GLB/turntable manifest).
- `FEAT-blender-asset-preflight.md` — headless map/geometry normalization into engine-ready assets with a deterministic manifest (P1, done for the internal recipe lane).
- `TASK-blender-addon-capability-probe.md` — headless addon probe; built-in/free-OSS/paid-detect-only tiers with versions and licenses (P1, done).
- `TASK-blender-bake-extended-maps.md` — extended PBR bake maps + thumbnail delta (P0; ticket header says done, but current public DTO/runtime gap above must be reconciled in this existing ticket rather than duplicated).
- `TASK-texture-postprocess-oiio.md` — mip/tile/colorspace/ORM sidecar (P0, crate/tooling done; separate from the Blender HTTP stage list).
- `FEAT-blender-rigging.md` — OSS-first rig jobs with bone/weight reports and base-rig-only warning (P2, proposed).
- `TASK-unirig-probe.md` — UniRig probe + wrapper-license decision (P2, in progress: probe surface landed; real rig run and wrapper decision outstanding).

## Dependency order

1. Keep discovery/preflight/job route as the shared execution foundation.
2. Reconcile the existing extended-bake ticket end to end (public DTO → route
   bounds → real recipe → manifest); do not add a second bake ticket.
3. Keep OIIO consumption in its existing post-process ticket and record the
   discovery-list gap there/with capability ownership; do not fold ORM logic
   into Blender recipes.
4. Scene-spec validation (`TASK-scene-spec-schema.md`, `epic-model-feedback`)
   depends on the landed artifact/admission/supervisor seams, then fixed-view
   rendering (`TASK-render-critique-loop.md`) extends the recipe. These are not
   Blender bake/export duplicates.
5. Rigging follows asset validation; UniRig remains gated on its real fixture
   and wrapper-license decision.
6. Engine bundle/import remains owned by `FEAT-game-engine-export-bundles.md`
   and `TASK-model-bundle-validation.md`; honkbox emits, engines verify.

## Acceptance

- One route-admitted bake fixture produces GLB, turntable, and `manifest.json`
  with sha256 + size per artifact, progress events, and working stage-boundary
  cancel.
- Extended maps/thumbnails count as complete only when requestable through the
  public typed API and executed by the real embedded recipe, with roles and
  colorspaces in the sidecar.
- Retopology records `seed` and Blender/addon versions; automatic rigs are
  labelled base-rig-only (no hero-character promise).
- External tools stay outside `Cargo.lock`; `deny.toml` policy unchanged.

## Verification

`cargo check --locked -p honkbox-server`,
`cargo test --locked -p honkbox-external-tools`,
`cargo test --locked -p honkbox-server --test blender_jobs`,
`bun run codegen:check`, and opt-in headless fixture:
`blender --background --factory-startup -noaudio --python <recipe.py> --python-exit-code 1 -- <job.json> <result.json>`.

## Non-goals

Embedded `bpy`, a scene editor, MCP tools, shipping a bundled Blender runtime,
another bake/OIIO/export ticket, agent frameworks, or a Blender→engine SDK.
