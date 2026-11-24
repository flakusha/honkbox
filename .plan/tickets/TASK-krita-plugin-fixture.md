# TASK: Krita pykrita plugin fixture

Status: proposed
Priority: P2
Epic: epic-editor-integration
Owner: unassigned
Acceptance: a Krita Python plugin reads the active document, sends one bounded image-op command, and imports the returned PNG result as a new layer (ORA return only after `TASK-ora-output` lands).
Verification: local Krita GUI fixture; no CI editor install
Links: `.tmp/research/dcc-closed-loop-honkbox-2026-11-25.md`, `.tmp/research/media-image-intermediates-and-enhancement-2026-11-24.md` (supersedes historical `ReviewedImageEdit-Clients.md` §Krita)
Resources: throwaway Krita resource dir per test; no shared document state

## Change

- Ship `pykrita/<name>/` + `.desktop`, register an `Extension` action via
  `Krita.instance()`.
- Export the active layer to a bounded PNG, POST to honkbox, then import the
  result via `Krita.instance().activeDocument()`.
- Avoid `kritarunner` for the interactive path (known GUI conflict).

## Acceptance detail

The plugin works inside a running Krita session; no Qt/Krita code enters Rust.
