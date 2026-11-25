# TASK: Versioned prompt/style preset library

Status: Done
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

## Verification Notes (2026-11-25)

- `crates/registry/src/presets.rs` parses
  `<data_dir>/models/presets/manifest.json` as a versioned, bounded,
  validated, id-sorted document; a missing file means no presets.
- `GET /v1/presets` and `GET /v1/presets/{id}` are read-only. Unknown ids
  return 400 `unknown_preset`; unreadable manifests return 500
  `preset_manifest_invalid`.
- `X-Honkbox-Preset` composes style/negative text into the existing
  `ImagesGenerationRequest` before validation, so no DTO or client format
  changes.
- Gate run this turn: `cargo fmt --all` clean, `giwt plan check-links`
  clean. Product tests and `codegen:check` were not run under the
  no-build instruction.

**Resolved:** 2026-11-25 registry-driven close: git issue 7e6abca (registry tip: 682d2ef Konstantin Fedotov Close issue)
