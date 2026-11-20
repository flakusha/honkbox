# EPIC: Blender asset workstream

Status: in progress
Priority: P1
Owner: main session
Research: `.tmp/research/ReviewedFullCycle-Blender-Workstream.md`,
`ReviewedFullCycle-Blender-Addons.md`, `ReviewedFullCycle-Blender-Rigging.md`

## Scope

Headless Blender jobs for texturing, UV prep, retopology, baking, rigging,
weight transfer, and export. Built-in operators first; OSS addons after license
recording and a headless operator probe; paid addons detect-only. honkbox
launches and manages Blender as an external process; addons live inside Blender
and are never incorporated into honkbox.

## Tickets

- `FEAT-blender-job-route.md` — typed `POST /v1/jobs/blender` job kind + route (P1, LANDED 2026-11-19: the typed route, the supervised run and the closed stage vocabulary live in `crates/server/src/routes/blender*.rs`; the mesh comes from the artifact store by id and the lifecycle runs through `/v1/jobs` + `/v1/events`; the live HTTP fixture wrote `output.glb`, `output_turntable.webp`, `manifest.json` and `status.json: done`).
- `FEAT-blender-asset-preflight.md` — headless map/geometry normalization into engine-ready assets with a deterministic manifest (P1, LANDED 2026-11-19: the typed job API, the embedded recipe and the manifest live in `crates/external-tools`; the HTTP job route is the next ticket).
- `TASK-blender-addon-capability-probe.md` — headless addon probe; built-in/free-OSS/paid-detect-only tiers with versions and licenses (P1, LANDED 2026-11-19: crate-side probe + capability advertisement; the tier rows reach HTTP and job metadata with the job route).
- `FEAT-blender-rigging.md` — OSS-first rig jobs with bone/weight reports and base-rig-only warning (P2).
- `TASK-blender-bake-extended-maps.md` — extended PBR bake maps + manifest roles (P0, done 2026-11-20).
- `TASK-unirig-probe.md` — UniRig probe + wrapper-license decision (P2, in progress: real rig run outstanding).

## Acceptance

- One bake fixture job produces GLB, turntable and a `manifest.json` with a
  sha256 + size per artifact in the job bundle, with progress events and
  working stage-boundary cancel.
- Retopology records `seed` and Blender/addon versions; automatic rigs are
  labelled base-rig-only (no hero-character promise).
- External tools stay outside `Cargo.lock`; `deny.toml` policy unchanged.

## Verification

`cargo check --locked -p honkbox-server`, `cargo test --locked -p honkbox-server`,
`bun run codegen:check`, headless fixture run:
`blender --background --factory-startup -noaudio --python <recipe.py> --python-exit-code 1 -- <job.json> <result.json>`.

## Non-goals

Embedded `bpy`, a scene editor, MCP tools, and shipping a bundled Blender
runtime.
