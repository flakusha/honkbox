# TASK: Blender addon capability probe

Status: done
Priority: P1
Epic: epic-blender-assets
Owner: unassigned
Acceptance: installed Blender addons are probed headlessly and classified into built-in / free-OSS / paid-detect-only tiers with versions and licenses recorded.
Verification: headless `blender -b --python-expr` probe fixture; `bun run check`
Links: `.plan/epics/epic-blender-assets.md`; research origin `../loop-lore/.tmp/research/blender-bridge-implementation-2026-11-19.md`
Resources: throwaway Blender profile per test; no shared addon install

## Change

- Probe `bpy.ops` availability with `-b`; detect UI-context-dependent operators
  and mark them unsupported headless (hard stop, not a workaround target).
- Record addon name, version, and license in job metadata. GPL addons stay
  inside Blender (user decision); never vendor into the Rust workspace.
- Built-in operators (quadriflow, voxel/decimate, bake, UV) are the verified
  baseline; paid addons are detect-only until proven.

## Acceptance detail

A missing addon yields a 501 naming it; a UI-only operator is not offered as a
headless job; the probe is deterministic for a given Blender install.

## Landed (2026-11-19)

- `crates/external-tools/src/blender_addons.py` (the probe script) and
  `crates/external-tools/src/blender_addons.rs` (parsing, tiers, headless
  verdict). The Blender side reports FACTS only; the tiers and the headless
  verdict are decided in Rust, so both are unit-tested without a Blender
  install.
- Tiers: `built_in` (ships with Blender), `free_oss` (declared license carries
  a recognized free/open token), `paid_detect_only` (third-party without a
  proven free license — detect only). Version and declared license are recorded
  per add-on; GPL add-ons stay inside Blender and are never vendored.
- Headless verdict: an operator is offered only when it exists in the build and
  its `poll()` passes with a prepared scene in object or edit mode. UI-only
  operators (`view3d.*`) fail both polls and are never offered. Operators are
  never executed — execution can write files, and a probe must not.
- Advertisement: enabled add-ons and headless operators become capabilities
  (`addon:<id>`, `op:<id>`). The EXISTING
  `GET /v1/external-tools/blender?capability=addon:quadriflow` check then
  answers 501 naming the missing add-on, and a UI-only operator is never in the
  list. No route file and no `routes.rs` registration was touched.
- The records themselves (tier, version, license) live in the crate:
  `Registry::blender_addons()` / `Registry::blender_operators()`. The HTTP
  surface for those rows and the job-metadata copy land with
  `FEAT-blender-job-route.md`, which owns the job sidecar.
- Determinism: add-ons and operators are sorted by id, so the same install
  always yields the same list.
- Honest gap found while probing: Blender 4.5 removed `object.quadriflow`, so
  the probe reports it absent instead of advertising a missing operator.

Verification: `cargo test --locked -p honkbox-external-tools` (22 passed, 1
ignored) and the ignored test against the local Blender 4.5.15 LTS:
`cargo test --locked -p honkbox-external-tools -- --ignored`.

Live HTTP check (server on :8099, real Blender 4.5.15):
`GET /v1/external-tools/blender` advertises `addon:cycles`,
`op:uv.smart_project`, ... and no `op:object.quadriflow`;
`?capability=addon:quadriflow` -> 501 `external tool blender unavailable:
capability addon:quadriflow`; `?capability=op:view3d.view_axis` -> 501.
