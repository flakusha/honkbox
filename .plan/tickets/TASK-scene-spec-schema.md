# TASK: Versioned deterministic scene-spec schema

Status: proposed
Priority: P2
Epic: epic-model-feedback
Owner: unassigned
Acceptance: `scene-spec/v1` validates and canonically encodes hash-addressed assets, ordered objects/transforms, materials, cameras, lights, fixed views, seeds, and an engine target before job admission; the same input produces the same ordered view/seed plan and can be inspected and replayed without an LLM.
Verification: `cargo test --locked -p honkbox-external-tools`, `cargo test --locked -p honkbox-server --test blender_jobs`, `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/creative-pipeline-promotion-2026-11-25.md`, `FEAT-blender-job-route.md`, `FEAT-blender-asset-preflight.md`, `TASK-render-critique-loop.md`, `crates/server/src/determinism.rs`
Resources: per-test temp fixture/staging dirs; one Blender process per job under the Blender lease; explicit CPU-thread, RSS, timeout, input, and output bounds; no shared scene or profile

## Dependencies

- Landed `POST /v1/jobs/blender` validation/admission and supervised execution
  (`FEAT-blender-job-route.md`, `FEAT-blender-asset-preflight.md`).
- Landed content-addressed artifact registry and `canonical_json` /
  `canonical_hash` / `stage_stream` helpers.
- `TASK-render-critique-loop.md` consumes this contract; it must not invent a
  second scene format.

## Change

- Add a closed `scene-spec/v1` schema (Draft 2020-12 shape schema plus typed Rust
  validation) with only: `schema_version`, `seed`, `engine_target`, `assets`,
  `objects`, `materials`, `cameras`, `views`, and `lights`.
- Each asset entry is `{asset_id, sha256, role}` with a stable local id and
  64-lowercase-hex digest. Roles cover mesh and semantic texture slots
  (`albedo`, `normal`, `roughness`, `metalness`, `ao`, `height`). The server
  resolves digests through the artifact registry; the schema never carries an
  absolute path.
- Objects carry stable ids, asset refs, optional parent ids, finite
  `translation [x,y,z]`, `rotation [rx,ry,rz]` (radians, XYZ Euler order), and
  `scale [x,y,z]` in declared scene units, plus material ids. Materials carry
  scalar PBR values and texture slots that reference asset ids. Cameras carry
  finite position/target/up vectors plus projection/lens/clip values; lights
  carry type, position/direction, color, energy, and bounded attenuation values.
  Views carry stable ids, camera ids, ordered render targets, and bounded
  integer dimensions.
- `engine_target` is a closed value (`blender_cycles_cpu` for v1); record the
  required Blender version and recipe digest in the job engine contract, not
  as wall-clock data in the scene. Scene-relative output names are safe path
  components resolved only beneath the per-job staging directory.
- Validate schema shape, `additionalProperties: false`, collection/byte/
  dimension limits, and unique ids before artifact resolution. Then validate
  reference closure, allowed asset role/format, id == sha256, stored-byte caps,
  acyclic parenting, positive scale, finite numbers, known cameras/materials/
  lights, and safe relative paths. Any failure returns a named 400/404/409/413
  before `admit_then_register`, lease acquisition, bundle creation, or Blender
  spawn.
- Serialize through the existing canonical JSON writer and preserve array
  order. Hash the canonical bytes; derive per-view/per-stage seeds with
  `stage_stream(seed, "scene-spec/v1/<stage>/<view_id>")`. Persist the canonical
  spec, its digest, and resolved input hashes as content-addressed bundle
  artifacts. Do not include job/bundle ids, timestamps, durations, queue data,
  or other wall-clock fields.

## Acceptance detail

- Positive fixture: one cube asset plus a material, camera, light, and
  front/side/top views validates; canonical bytes and scene digest are equal
  across repeated admission; ordered view ids and derived seeds are equal.
- Negative fixtures reject unsupported `schema_version` or `engine_target`,
  unknown fields at every level, duplicate ids, unknown asset/object/material/
  camera/light refs, missing stored artifacts, digest/id mismatch, wrong asset
  role/format, oversized spec/asset/output, zero or excessive dimensions,
  non-finite transforms, zero/negative scale, parenting cycles, absolute paths,
  `..` traversal, and backslash/encoded path escapes. Assert no job registration,
  no Blender lease, and no recipe spawn for each rejection.
- Two isolated runs with the same canonical spec, input hashes, Blender version,
  and recipe digest produce the same ordered multi-view manifest, artifact
  roles, dimensions, and seed plan. Pixel or byte equality across Blender
  versions/hardware is not required; changed tool/recipe pins must be visible.
- Render implementation belongs to `TASK-render-critique-loop.md`; this ticket
  is complete when validation/canonical persistence and the deterministic view
  plan are testable without a model.

## Non-goals

No scene editor, arbitrary Python/GLSL, animation/timeline, camera-path UI,
engine SDK, in-process `bpy`, duplicate bake/OIIO/material/export contract, or
byte-equality promise across renderers.
