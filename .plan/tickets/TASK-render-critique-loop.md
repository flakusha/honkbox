# TASK: Fixed multi-view render and typed critique replay

Status: proposed
Priority: P1
Epic: epic-model-feedback
Owner: unassigned
Acceptance: a validated scene spec produces bounded named front/side/top renders and a `critique/v1` typed sidecar; recorded edits and views replay without an LLM, network, agent framework, or GUI, with every spec/render/sidecar version retained by content hash.
Verification: `cargo test --locked -p honkbox-external-tools`, `cargo test --locked -p honkbox-server --test blender_jobs`, `bun run test:server`, `bun run codegen:check`; opt-in local Blender fixture
Links: `.tmp/research/creative-pipeline-promotion-2026-11-25.md`, `TASK-scene-spec-schema.md`, `FEAT-blender-job-route.md`, `FEAT-model-modality-advertisement.md`, `crates/server/src/determinism.rs`
Resources: per-test temp scene/render dirs; one supervised Blender process under `ResourceKind::Blender`; explicit CPU-thread/RSS/time caps; bounded images/prompts/output; in-process stub critic; no port, live model, network, or shared scene state

## Dependencies

- `TASK-scene-spec-schema.md` is the sole scene input contract and rejects
  invalid specs before admission.
- Reuse the landed Blender route/supervisor, atomic job bundle writes,
  content-addressed artifact registry, `job_result` events, canonical JSON, and
  derived seed streams. Do not add a second job/event system.
- Upstream material production remains owned by
  `TASK-blender-bake-extended-maps.md`, `TASK-texture-postprocess-oiio.md`,
  and `TASK-material-map-validation.md`; this ticket consumes their hashes and
  roles.

## Change

- Add a closed scene-render stage/recipe over `scene-spec/v1`. It constructs the
  declared objects, transforms, materials, cameras, and lights from staged
  artifact bytes, then renders the ordered fixed views. V1 requires front,
  side, and top; additional views are allowed only when declared.
- Pin CPU Cycles, sample count, denoise off, threads, resolution, color
  management, camera/light setup, and recipe/Blender versions. Emit one named
  PNG per view (plus a contact sheet only if bounded) and an ordered render
  manifest containing view id, dimensions, artifact id, sha256, and derived
  seed. Current bounding-box turntable behavior is not a fixed-view contract.
- Define `critique/v1` as closed JSON containing `scene_spec_sha256`, engine/
  recipe versions, ordered `renders`, and typed `findings`. A finding has a
  stable id, severity, category, bounded message, and optional known view/object
  ids. An optional `edits` list uses closed operations only: set transform,
  set material scalar, set material texture slot, set light property, or set
  camera property. No arbitrary JSON pointers, code, shell, tool calls, or
  callbacks.
- Validate the sidecar before use: exact schema/hash refs, known view/object/
  material/camera/light ids, allowed operation/property/value ranges, no
  duplicate finding/edit ids, no unknown fields, and all collection/byte caps.
  Apply edits in recorded order to a cloned canonical spec, revalidate the
  entire result through the scene-spec validator, then rerender the same view
  plan. Invalid edits fail the run; they are never silently dropped.
- Retain each canonical spec, render manifest, PNG, critique sidecar, resulting
  spec, and digest in the job bundle; promote them through `Registry::put_bytes`
  and emit normal `job_result` frames. Replay reads only these artifacts and
  typed operations. Operational job ids/timestamps stay outside replay inputs.
- An optional future critic may be selected only from `/v1/models` entries that
  actually advertise image input. Its output must decode to the same typed
  sidecar. A missing/ineligible critic leaves replay fully functional and must
  not cause honkbox to download or start a model.

## Acceptance detail

- Positive fixture: fixed front/side/top renders have exact declared dimensions
  and ordered manifest entries; a stub sidecar proposing one valid transform
  edit produces a second canonical spec and rerender set without a model or
  network. Replay from bundle artifacts alone repeats the same edit sequence,
  view order, derived seeds, and artifact references.
- Negative schema fixtures reject unknown sidecar/version/finding fields,
  duplicate ids, malformed or mismatched scene/render hashes, unknown refs,
  unapproved operation/property, unsafe values, edit-count overflow, and
  attempts to alter schema/version/engine pins. Invalid output count, oversized
  PNG/sidecar/spec, non-PNG bytes, or manifest/artifact hash mismatch fails
  before promotion.
- Bounds for v1: at most 8 views, 8 objects, 8 materials, 8 lights, 16 assets,
  1024×1024 per view, 8 MiB total rendered output, 1 MiB canonical scene JSON,
  256 KiB critique JSON, 64 findings, 32 edits, 8 critic prompts, and 2 critic
  iterations in the fixture. A supervisor timeout/RSS cap and bounded logs apply
  to every render; cancel takes effect at a stage/view boundary.
- Determinism is structural: same canonical spec + recorded edits + pinned
  recipe/tool versions + ordered views + derived seeds. No pixel-byte equality
  claim across Blender versions, CPUs, or GPUs.

## Non-goals

No autonomous/unbounded loop, agent framework, persistent agent memory,
loop-lore state ownership, GUI automation, arbitrary tool use, ComfyUI rewrite,
model download, hosted critic dependency, scene editor, training, rigging,
bake/export/OIIO duplication, or cross-renderer byte equality.
