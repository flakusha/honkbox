# TASK: Krita pykrita plugin fixture

Status: proposed
Priority: P2
Epic: epic-editor-integration
Owner: unassigned
Acceptance: a Krita Python plugin reads the active document, sends one bounded image-op command, and imports the returned PNG/ORA result as a layer.
Verification: local Krita GUI fixture; no CI editor install
Links: `.tmp/research/ReviewedImageEdit-Clients.md` §Krita
Resources: throwaway Krita resource dir per test; no shared document state

## Change

- Ship `pykrita/<name>/` + `.desktop`, register an `Extension` action via
  `Krita.instance()`.
- Export the active layer to a bounded PNG, POST to honkbox, then import the
  result via `Krita.instance().activeDocument()`.
- Avoid `kritarunner` for the interactive path (known GUI conflict).

## Acceptance detail

The plugin works inside a running Krita session; no Qt/Krita code enters Rust.
