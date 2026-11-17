# EPIC: GIMP/Krita editor integration

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/ReviewedImageEdit-*.md`

## Scope

Editor plugins in GIMP/Krita send bounded commands to honkbox and insert
returned pixels or OpenRaster layers. honkbox provides high-performance ops as
jobs; it never embeds GTK/Qt/libgimp or automates the editor GUI.

## Tickets

- `TASK-image-ops-command-dto.md` — bounded op/input/output DTO (P1).
- `TASK-image-ops-job-route.md` — `POST /v1/image-ops` + events (P1).
- `TASK-cheap-ops-spike.md` — resize/crop/color/format batch ops (P1).
- `TASK-ora-output.md` — layered ORA result bundle (P2).
- `TASK-gimp-plugin-fixture.md` — GIMP 3 Python PDB client fixture (P2).
- `TASK-krita-plugin-fixture.md` — Krita pykrita client fixture (P2).
- `TASK-external-upscale-probe.md` — Real-ESRGAN/waifu2x external process (P2).

## Acceptance

Commands are idempotent, bounded, and cancelable; sidecars record command hash,
input/output hashes, tool version, and outcome; plugins never probe by trial.

## Verification

`bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`;
small local PNG/ORA fixtures.

## Non-goals

Native `.xcf`/`.kra` parsing, linking LGPL libvips/GEGL, and GUI automation.
