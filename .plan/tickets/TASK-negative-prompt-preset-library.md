# TASK: Versioned prompt/style preset library

Status: In Progress
**Scope:** Versioned negative-prompt/style preset library + X-Honkbox-Preset composition
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: operators can list and fetch versioned negative-prompt/style
presets from the model registry; presets compose into generation requests
without a new client-side format, and an unknown preset id is a named 400.
Verification: `bun run test:server`, `bun run codegen:check`,
`bun run audit:gpu-slot` (presets must not acquire a slot).
Links: `.tmp/research/content-experience-gaps-2026-11-20.md` §5 G13.
Resources: read-only registry files; no fixed port.

## Change

- Add a small preset directory next to models with a versioned manifest and a
  read-only route pair; no write endpoint in this slice.
- Reuse existing request validation for composed prompts.

## Non-goals

Preset editor UI, remote preset sync, per-user presets.