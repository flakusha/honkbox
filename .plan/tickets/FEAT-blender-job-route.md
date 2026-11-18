# FEAT: Typed Blender job route

Status: proposed
Priority: P1
Epic: epic-blender-assets
Owner: unassigned
Acceptance: `POST /v1/jobs/blender` admits a typed job with stages, params, engine version, output limits and a manifest; lifecycle runs through `/v1/jobs` and `/v1/events`; a fixture job writes GLB + manifest + turntable artifact with cancel between stages.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`, `bun run codegen:check`, headless fixture `blender --background --factory-startup -noaudio --python job.py -- <args>`.
Links: `../loop-lore/.tmp/research/blender-fullcycle-production-editing-2026-11-18.md` §3.2; loop-lore `SOL-blender-job-boundary-and-contract.md`, `TASK-blender-headless-node-bridge.md`; `crates/server/src/jobs_registry.rs:78-111`; `crates/residency/src/lib.rs:54-100`; `TASK-process-supervisor.md`; `FEAT-blender-asset-preflight.md`.
Resources: per-job temp dirs and assigned ports; one Blender process per job, killed in drop/finally; no shared patch cache.

## Change

- Extend the closed `JobKind` vocabulary in `crates/server/src/jobs_registry.rs` with `blender`. Reserved `ResourceKind::{Blender,ThreeD}` admission rows alone do not admit jobs.
- Add the typed DTO in `crates/api-types` (ts-rs + utoipa) and the route in `crates/server/src/routes.rs`: input `{assetId|objectKey, format: glb|obj|ply, sha256}`, `stages` from a closed list, `params`, `engine {blenderVersion: "4.5.15", scriptVersion: <sha>}`, `output {format: "glb", maxBytes}`, `manifest: 1` → `{jobId, status: "queued"}`.
- Reuse `TASK-process-supervisor.md` for process lifecycle and `FEAT-blender-asset-preflight.md` for stage recipes. Every artifact write is `.part` + fsync + rename; cancel only between stages, never mid-bake.
- Emit lifecycle through the existing jobs/events surface; metrics use bounded `kind=blender` labels with no user IDs, paths or prompts; enforce poly-count and byte caps at the route.
- Gate the route on `FeatureFlags`, not on reserved queue rows in `GET /v1/capabilities` (see `TASK-truthful-reserved-capability-rows.md`).
- Pipeline stays local-first: an on-host/LAN Blender process. A hosted endpoint is a later BYO-key fallback, never the default.

## Acceptance detail

A queued job reaches a terminal state with `output.glb`, `manifest.json` and `turntable.webp`; a cancelled job leaves no orphan process and releases its lease; oversized or wrong-sha input is rejected before the process starts. Automatic retopo/rig output is labelled base-mesh only.
