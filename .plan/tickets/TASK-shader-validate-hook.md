# TASK: Headless shader-compile check hook

Status: done (2026-11-24, Godot family now gets a throwaway headless project plus `RenderingServer.shader_set_code`; real Godot 4.7.2 emitted `SHADER ERROR` + `E <line>` for an invalid shader and stayed clean for a valid one)
Priority: P3
Epic: epic-game-engine-export
Owner: unassigned
Acceptance: `.gdshader` text validates by compile before it ships in a bundle; failures name the shader + line.
Verification: 8 `honkbox-external-tools` shader_validate tests pass, including a headless-args fixture that asserts the staged project and script exist; real Godot 4.7.2 probe verified invalid/valid verdicts without opening a window; `bun run test:server` green in the ash worktree.
Links: `.tmp/research/shaders-materials-uiux-asset-pipeline-2026-11-20.md` §1, `TASK-engine-bundle-verification.md`
Resources: per-test tmp dirs; no Godot project state in honkbox

## Change

- External-binary hook (Godot precedent: never linked, never in lockfile):
  compile-check `.gdshader`/GLSL text sourced from godotshaders.com or
  generated, record source URL + author + result in the bundle.
- Shaders stay loop-lore versioned text; honkbox never executes them.

## Acceptance detail

- A shader that fails compile blocks the bundle with file + line named.
  No Godot engine, export, or MCP concept enters honkbox code or DTOs.
