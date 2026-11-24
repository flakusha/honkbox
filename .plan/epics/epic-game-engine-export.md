# EPIC: Game-engine export bundles

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/loop-lore-godot-actionables.md`,
`.tmp/research/shaders-materials-uiux-asset-pipeline-2026-11-20.md`

## Scope

honkbox produces engine-agnostic, versioned asset bundles (glTF/GLB + PBR
textures + rigs/animations + splats + deterministic manifest) and stops there.
loop-lore owns engine project state, export profiles, and MCP calls. Godot and
Bevy are first targets; Ren'Py via pinned SDK CLI; Unity/Unreal optional
external-only. Yield after those five.

## Tickets

- `FEAT-game-engine-export-bundles.md` — versioned GLB + PBR + manifest.
- `TASK-engine-bundle-verification.md` — Godot/Bevy/Ren'Py external verification.
- `TASK-godot-importer-fixture.md` — engine-owned `EditorImportPlugin` fixture
  (cross-app; honkbox emits the manifest, it does not ship editor code).
- `TASK-cc0-pack-fetch.md` — curated CC0 pack fetch + local sha/license rows (P1, done 2026-11-20).
- `TASK-shader-validate-hook.md` — headless `.gdshader` compile hook + 501 (P3, in progress: real Godot binary unverified).
- `TASK-qal-guard.md` — refuse no-redistribution packs before any request (done 2026-11-20).

## Acceptance

- A bundle validates against an explicit glTF extension allowlist; checksums
  and engine/asset versions are recorded in the manifest.
- No engine SDK, engine build command, or MCP concept appears in honkbox code
  or DTOs.
- A Godot headless import of a produced bundle is demonstrated in verification
  (external to honkbox).

## Verification

`gltf-transform inspect/validate <bundle.glb>`, `bun run check`,
`cargo test --locked -p honkbox-server`; Godot: `godot --headless --path <proj>
--import` (external fixture).

## Non-goals

Running `godot --export-pack`, Ren'Py `launcher distribute`, or Unity/Unreal
builds inside honkbox; MCP servers in honkbox.
