# TASK: Headless shader-compile check hook

Status: in progress (2026-11-20, hook + 501 path landed; Godot-family call now uses `--headless --quit` so tests never open a desktop window; real Godot compile verdict still unverified)
Priority: P3
Epic: epic-game-engine-export
Owner: unassigned
Acceptance: `.gdshader` text validates by compile before it ships in a bundle; failures name the shader + line.
Verification: external `godot --headless` fixture run; `bun run test:server`
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
