# TASK: Blender addon capability probe

Status: proposed
Priority: P1
Epic: epic-blender-assets
Owner: unassigned
Acceptance: installed Blender addons are probed headlessly and classified into built-in / free-OSS / paid-detect-only tiers with versions and licenses recorded.
Verification: headless `blender -b --python-expr` probe fixture; `bun run check`
Links: `.tmp/research/ReviewedFullCycle-Blender-Addons.md`
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
