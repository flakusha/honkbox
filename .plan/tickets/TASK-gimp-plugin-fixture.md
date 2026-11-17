# TASK: GIMP 3 Python plugin fixture

Status: proposed
Priority: P2
Epic: epic-editor-integration
Owner: unassigned
Acceptance: a GIMP 3 Python plug-in sends one bounded image-op command to honkbox and inserts the returned layer into the open image.
Verification: local GIMP 3 fixture; no CI editor install
Links: `.tmp/research/ReviewedImageEdit-Clients.md` §GIMP 3
Resources: throwaway GIMP profile and image per test; no shared PDB instance

## Change

- Register a `Gimp.ImageProcedure`, use `Gegl.init(None)` for non-interactive runs.
- POST a small PNG data URI with bearer auth; poll the job and insert result as
  a new layer.
- Do not link `libgimp` from Rust or automate the GUI.

## Acceptance detail

Cancel in GIMP propagates to the honkbox job; the plugin shows named errors for
unknown ops and auth failure.
