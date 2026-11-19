# FEAT: Artifact delivery and job-result events

Status: done
Priority: P1
Epic: epic-binary-formats
Acceptance: `POST /v1/images/generations` with `X-Honkbox-Delivery: artifact` answers `{artifact_id, content_type, size_bytes, job_id}` instead of `b64_json`; blender bundle outputs are promoted into the artifact registry at commit; `GET /v1/events` emits `job_progress` and `job_result {artifact_id, sha256, size_bytes}` frames; `GET /v1/capabilities` advertises `features.artifact_delivery` + a data-dir-present signal.
Links: loop-lore `.plan/tickets/TASK-honkbox-generation-result-delivery-and-asset-ingest.md` (the consumer); `epic-capabilities-contract`; `epic-blender-assets`.
Verification: `bun run check`, `bun run test:server`, `bun run codegen:check`

## Change

- `routes/images_inputs.rs`: honor `X-Honkbox-Delivery: inline|artifact` (default `inline` = current `b64_json`, backwards compatible). In `artifact` mode stream rendered bytes into `crate::artifacts` via `Registry::put_bytes(bytes, content_type)` and answer `ImagesGenerationResponse { created, data: [], artifacts: [{artifact_id, content_type, size_bytes, job_id}], bundle }` — no `b64_json` in artifact mode.
- `routes/blender_exec.rs:commit`: store each committed artifact in the artifact registry via `Registry::put_bytes`; put `artifact_id` in each `manifest.json` artifact entry and emit `job_result` for each promoted output. Registry promotion is chosen for content-addressing; `GET /v1/jobs/{id}/content/{name}` already exists and must be name-derived from the manifest so custom prefixes and bake maps are reachable.
- `jobs_registry::global().emit_progress(id, step, total_steps)` and `emit_result(id, artifact_id, sha256, size_bytes)` feed the existing SSE broadcast. Wire frames: `job_progress {event,id,kind,step,pct}` and `job_result {event,id,kind,artifact_id,sha256,size_bytes}`. `SdPipeline::generate_with_progress(req, cancel, &dyn Fn(u32,u32))` defaults to `generate` so backends can adopt it later. No replay — `GET /v1/jobs` remains the documented recovery.
- `routes/capabilities.rs`: add `features.artifact_delivery: bool` (= data dir present) and additive optional `storage: { data_dir_present: bool }` so loop-lore can probe before using artifact mode.
- `images_generate.rs:25-40`: artifact-mode bundle/artifact write fails loud (today the bundle write is best-effort warn-only).

## Wire contract (frozen for both repos)

- `X-Honkbox-Delivery: artifact` (default `inline`); artifact response fields:
  `artifact_id`, `content_type`, `size_bytes`, `job_id`.
- SSE frames: `job_progress {event,id,kind,step,pct}` and
  `job_result {event,id,kind,artifact_id,sha256,size_bytes}`.
- Capabilities: `features.artifact_delivery`, `storage.data_dir_present`.

## Rejected

- Output-path request param: a remote peer must never name a filesystem path (same rule as the blender route); traversal vector.
- Callback URL: second auth scheme + SSRF surface; use a pre-registered endpoint id if ever needed.
- honkbox writing into loop-lore storage: couples honkbox to loop-lore's DB/upload layout; storage ownership stays in loop-lore's `createAsset`.

## Verification (2026-11-20)

`bun run check`, `bun run test:server`, `bun run test:core` green; generated TS/OpenAPI regenerated and drift-checked. Live loopback smoke against a bearer-auth stub server: artifact-mode image returned one artifact + job id, SSE `job_result` carried the matching sha256/size, mid-run cancel accepted, `/v1/jobs`, `/v1/events` and `/v1/metrics` reachable. Stub backend emits no `job_progress` steps by design; frame-label and registry tests cover that contract.
