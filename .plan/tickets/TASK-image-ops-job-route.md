# TASK: Image-ops job route

Status: done (2026-11-18, LANE-impl-wave)
Priority: P1
Epic: epic-editor-integration
Owner: LANE-impl-wave
Acceptance: `POST /v1/image-ops` returns a job id, emits progress on `/v1/events`, and writes the result bundle via the existing jobs path.
Verification: `bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: `.tmp/research/ReviewedImageEdit-CommandProtocol.md`
Resources: per-test job dirs; no shared GPU slot or fixed port

## Change

- Depends on `TASK-image-ops-command-dto.md`.
- Register a new `JobKind`, acquire admission, and honor cooperative cancel.
- Sidecar records command hash, input/output hashes, seed, tool version,
  attempts, and outcome; no token or full URL.

## Acceptance detail

A plugin can submit, poll, and cancel one op without probing trial requests;
capabilities list supported ops.

## Implementation note (2026-11-18)

- `POST /v1/image-ops` mounted inside the bearer `route_layer` like every
  sibling. New `JobKind::ImageOps` (wire label `image_ops`) registered while
  the request lives — `X-Job-Id` header + `GET /v1/jobs` + lifecycle frames
  on the existing `/v1/events` stream (`job_registered`/`job_finished`
  observed by test).
- Admission: the ledger's CPU-only `rag` bucket (0 GPU slots) — the fixed
  8-kind vocabulary has no image-ops kind; same documented nearest-bucket
  precedent as audio mapping to `llm`.
- Cooperative cancel honored BETWEEN inputs (honest `interruptible` = batch
  greater than 1; cancel mid-batch answers 409 `job_cancelled` envelope, no
  bundle written).
- Work runs on the blocking pool via the in-process `image` crate
  (TASK-cheap-ops-spike Tier 0): Lanczos3 resize, pixel-exact crop, lossless
  90/180/270 rotation, grayscale, container re-encode — deterministic
  outputs (fixed filter + sd-core's fixed encoders, quality 90 default).
- Result bundle via the existing jobs path: `<data_dir>/jobs/{ulid}/` with
  the same atomic part-write+fsync+rename discipline and the same GC
  lifecycle. `ops.json` sidecar records command_sha256, per-input
  source+sha256, per-output file+sha256, `seed: null` (pixel ops are
  seedless; null keeps the record shape uniform with generation bundles),
  tool name/version/engine, attempts (1), outcome (`done`), elapsed_ms.
  `command.json` carries the bounded command — never payload bytes, tokens,
  or URLs. The response returns b64 outputs + sha256 + the bundle ULID that
  artifact inputs reference.
- Capabilities: `GET /v1/capabilities` gained additive `image_ops.ops` (the
  closed vocabulary, derived from `ImageOpKind::ALL`).
- Tests: new `tests/image_ops.rs` (13 tests: happy path + bundle sidecar
  fields, determinism, idempotent replay, request_id conflict, unknown
  fields/ops, reject-not-clamp bounds, crop bounds, rotate/grayscale/format,
  batch + cap, malformed inputs, artifact round-trip + no-probe, cancel
  between units with both outcomes accepted (never flaky), events
  lifecycle, capabilities ops list).
- Scope disclosure: the route executes synchronously (job id + lifecycle
  events + pollable registry + cancel like every GPU sibling); there is no
  separate result-fetch route — outputs ride the response and the bundle is
  on disk (artifact HTTP serving is TASK-binary-artifact-api's scope).
