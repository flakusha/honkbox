# FEAT: Typed Blender job route

Status: done
Priority: P1
Epic: epic-blender-assets
Owner: main session
Acceptance: `POST /v1/jobs/blender` admits a typed job with stages, params, engine version, output limits and a manifest; the mesh is a stored artifact id (64-hex sha256), `engine.scriptVersion` is optional; lifecycle runs through `/v1/jobs` and `/v1/events`; a fixture job writes `output.glb`, `manifest.json` and `output_turntable.webp` with cancel between stages.
Verification: `cargo check --locked -p honkbox-server` (clean tree), `cargo test --locked -p honkbox-server`, `bun run codegen:check`, `cargo test --locked -p honkbox-external-tools -- --ignored`, headless fixture `bash .tmp/e2e-job.sh`.
Links: `../loop-lore/.tmp/research/blender-fullcycle-production-editing-2026-11-18.md` §3.2; loop-lore `SOL-blender-job-boundary-and-contract.md`, `TASK-blender-headless-node-bridge.md`; `crates/server/src/jobs_registry.rs:78-111`; `crates/residency/src/lib.rs:54-100`; `TASK-process-supervisor.md`; `FEAT-blender-asset-preflight.md`.
Resources: per-job temp dirs and assigned ports; one Blender process per job, killed in drop/finally; no shared patch cache.

## Change

- Extend the closed `JobKind` vocabulary in `crates/server/src/jobs_registry.rs` with `blender`. Reserved `ResourceKind::{Blender,ThreeD}` admission rows alone do not admit jobs.
- Add the typed DTO in `crates/api-types` (ts-rs + utoipa) and the route in `crates/server/src/routes.rs`: input `{assetId: <64-hex sha256>, format: glb|obj|ply}`, `stages` from a closed list, `params`, `engine {blenderVersion: "4.5.15", scriptVersion?: <sha>}`, `output {format: "glb", maxBytes}`, `manifest: 1` → `{jobId, status: "queued"}`.
- Reuse `TASK-process-supervisor.md` for process lifecycle and `FEAT-blender-asset-preflight.md` for stage recipes. Every artifact write is `.part` + fsync + rename; cancel only between stages, never mid-bake.
- Emit lifecycle through the existing jobs/events surface; metrics use bounded `kind=blender` labels with no user IDs, paths or prompts; enforce poly-count and byte caps at the route.
- Gate the route on `FeatureFlags`, not on reserved queue rows in `GET /v1/capabilities` (see `TASK-truthful-reserved-capability-rows.md`).
- Pipeline stays local-first: an on-host/LAN Blender process. A hosted endpoint is a later BYO-key fallback, never the default.

## Acceptance detail

A queued job reaches a terminal state with `output.glb`, `manifest.json` and `output_turntable.webp`; a cancelled job leaves no orphan process and releases its lease; oversized or wrong-sha input is rejected before the process starts. Automatic retopo/rig output is labelled base-mesh only.

## Implementation (2026-11-19)

- Route: `crates/server/src/routes/blender.rs` (the handler, the engine gate,
  the request validation), `crates/server/src/routes/blender_stages.rs` (the
  closed stage vocabulary + its bounds), `crates/server/src/routes/blender_exec.rs`
  (the supervised run). `routes.rs` registers the route and declares the
  three modules; the `supervisor` module moved from `main.rs` into
  `routes.rs` (the `#[path]`-included test harnesses pull in `routes.rs`,
  and the Blender route is the supervisor's first caller).
- The mesh is resolved from the artifact store by id — the route never takes
  a filesystem path — and the artifact id IS the content digest, so a wrong
  digest is a 400 before any process starts.
- Gates before the process: request shape (400), the mesh in the store (404),
  the mesh size (413), the engine contract (409/501), the admission lease
  (429), the data dir (503). A refused request never enters the registry.
- The run: the recipe files are written into the job's staging directory,
  Blender runs under the process supervisor (its own process group, a
  wall-clock bound, a group-wide teardown) and the artifacts are committed
  into the job bundle (`.part` + fsync + rename). Cancel is a STAGE-BOUNDARY
  cancel: the recipe checks a marker file between stages, so a bake is never
  interrupted mid-write.
- The turntable is a stage of its own (`turntable { frames, resolution }`):
  the recipe renders N views around the object and tiles them into one PNG;
  the bundle keeps it as webp (the workspace has no animated webp encoder,
  so the turntable ships as one still).
- The engine contract pins the Blender version; the recipe digest is
  optional (`scriptVersion` absent = the server runs its current recipe,
  present = a recipe change is a named 409).
- Tests: `crates/server/tests/blender_jobs.rs` (the request shape, the store
  misses, the digest mismatch, the no-engine 501 that registers no job, the
  OpenAPI path, the capability row) + the inline unit tests of the request
  validation. The recipe itself is exercised by the real-Blender fixture
  (`cargo test --locked -p honkbox-external-tools -- --ignored`).

## Verification (2026-11-19, takeover)

- `cargo fmt --all -- --check` — clean; dprint check of both generated files — clean.
- `cargo test --locked -p honkbox-server` (project-local CARGO_TARGET_DIR):
  all suites green.
- Live HTTP fixture `bash .tmp/e2e-job.sh` against local Blender 4.5.15:
  `202` accepted, job reached `status.json: done` with `output.glb`,
  `output_turntable.webp` and a path-free `manifest.json` (sha256 per artifact).
- `bun run codegen:check` — api-types tests passed and OpenAPI covers every
  mounted route; the first run hit a full `/tmp` zram device (environmental),
  rerun with the project-local target dir. Final diff check runs after the
  generated files are staged.
- Clean-tree `cargo check --locked -p honkbox-server` — 0 errors after the
  P0-1 compile plumbing landed; `cargo clippy --workspace --all-targets --
  -D warnings` — 0 errors. The bounded `honkbox_jobs_total` grid is covered
  by an inline test in `crates/server/src/metrics.rs`, and the route-side
  duplicate increment is gone (one count per admitted job).

