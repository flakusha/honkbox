# TASK: Engine bundle verification fixtures

Status: proposed
Priority: P2
Epic: epic-game-engine-export
Owner: unassigned
Acceptance: one exported bundle imports headlessly in Godot and validates against the Bevy extension allowlist; Ren'Py distribution build exercised with a pinned SDK.
Verification: `gltf-transform inspect/validate`, `godot --headless --path <proj> --import`, Ren'Py `launcher distribute` (all external to honkbox)
Links: `.tmp/research/ReviewedFullCycle-GameEngines.md`
Resources: throwaway engine projects and temp export dirs per test; no shared engine install state

## Change

- Fixture bundle from `FEAT-game-engine-export-bundles.md` (GLB + PBR maps +
  manifest) imported with `godot --headless --import`; assert clean import.
- Validate Bevy profile extension allowlist (no Draco/meshopt/
  `KHR_animation_pointer`/`KHR_gaussian_splatting`).
- Ren'Py CLI is unstable across releases — pin SDK and probe `--help` before
  `launcher distribute`.

## Acceptance detail

Verification is external to honkbox; no engine SDK, build command, or MCP
concept enters honkbox code or DTOs.

## Verification Notes (2026-11-20)

Status remains `proposed`: external-fixture acceptance is not satisfiable here.
`which godot godot4 renpy gltf-transform unirig` finds only `/usr/bin/godot`;
Ren'Py, gltf-transform and UniRig are absent. No engine project, SDK pin, or
external command was added to honkbox code or DTOs. Keep the ticket blocked
until a pinned Ren'Py SDK and the Bevy allowlist checker are available; Godot
presence alone is not enough for the stated acceptance.
