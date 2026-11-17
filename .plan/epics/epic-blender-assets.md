# EPIC: Blender asset workstream

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/ReviewedFullCycle-Blender-Workstream.md`,
`ReviewedFullCycle-Blender-Addons.md`, `ReviewedFullCycle-Blender-Rigging.md`

## Scope

Headless Blender jobs for texturing, UV prep, retopology, baking, rigging,
weight transfer, and export. Built-in operators first; OSS addons after license
recording and a headless operator probe; paid addons detect-only. honkbox
launches and manages Blender as an external process; addons live inside Blender
and are never incorporated into honkbox.

## Acceptance

- One bake fixture job produces PNG maps + JSON sidecar in the job bundle with
  progress events and working cancel.
- Retopology records `seed` and Blender/addon versions; automatic rigs are
  labelled base-rig-only (no hero-character promise).
- External tools stay outside `Cargo.lock`; `deny.toml` policy unchanged.

## Verification

`cargo test --locked -p honkbox-server`, `bun run codegen:check`, headless
fixture run: `blender -b -P <script> --python-exit-code 1`.

## Non-goals

Embedded `bpy`, a scene editor, MCP tools, and shipping a bundled Blender
runtime.
