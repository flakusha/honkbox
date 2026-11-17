# EPIC: Game-engine export bundles

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/ReviewedFullCycle-GameEngines.md`,
`ReviewedFullCycle-LoopLore-Acceleration.md`

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
