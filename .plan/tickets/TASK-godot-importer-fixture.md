# TASK: Godot importer fixture

Status: proposed
Priority: P2
Epic: epic-game-engine-export
Owner: unassigned
Acceptance: a minimal Godot `plugin.cfg` + `EditorImportPlugin` reads a honkbox bundle manifest, validates checksums, and instantiates a resource in a throwaway project.
Verification: `godot --headless --path <proj> --import` (external fixture); no honkbox code gate
Links: `.tmp/research/ReviewedExpansion-GameEnginePlugins.md` §Suggested next steps
Resources: throwaway Godot project and import cache per test; no shared engine install state

## Change

- Owned by the engine lane (loop-lore or a separate engine-tools repo); honkbox
  learns no GDScript and ships no editor binary.
- Register `.honkbox.json`/manifest extension via `add_import_plugin()`;
  `_import()` produces a Godot resource and validates checksums from the bundle
  manifest — the single contract, not duplicated validation in Rust.
- Keep `TASK-engine-bundle-verification.md` as the honkbox-side acceptance gate.

## Acceptance detail

No engine SDK or plugin code enters honkbox; a malformed checksum fails the
Godot import with a named error.
