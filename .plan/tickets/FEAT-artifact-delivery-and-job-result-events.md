# FEAT: Artifact delivery and job-result events

Status: proposed
Priority: P1
Epic: epic-binary-formats
Acceptance: `POST /v1/images/generations` with `X-Honkbox-Delivery: artifact` answers `{artifact_id, content_type, size_bytes, job_id}` instead of `b64_json`; blender bundle outputs are promoted into the artifact registry at commit; `GET /v1/events` emits `job_progress` and `job_result {artifact_id, sha256, size_bytes}` frames; `GET /v1/capabilities` advertises `features.artifact_delivery` + a data-dir-present signal.
Links: loop-lore `.plan/tickets/TASK-honkbox-generation-result-delivery-and-asset-ingest.md` (the consumer); `epic-capabilities-contract`; `epic-blender-assets`.

## Change

- `routes/images_inputs.rs:81`: honor the `X-Honkbox-Delivery` header (default `inline` = current `b64_json`, backwards compatible). In `artifact` mode stream the rendered bytes (`images_generate.rs:403`) into `crate::artifacts` and answer with the id + `job_id`/`bundle` correlation instead of base64.
- `routes/blender_exec.rs:commit`: store each committed artifact in the artifact registry too; put the ids in `manifest.json` and in the `job_result` frame. (Strictly better than serving `<data_dir>/jobs/**` bytes over HTTP — no new route, wire format stays content-addressed.)
- `routes/jobs.rs` SSE: add `job_progress {id, kind, step, pct?}` from the SD sampling loop + blender stage loop, and `job_result {artifact_id, sha256, size_bytes}` on `job_finished`. No replay/`Last-Event-ID` — document poll-against-`/v1/jobs` as the recovery.
- `routes/capabilities.rs`: add `features.artifact_delivery` + data-dir-present flag so loop-lore can probe before using artifact mode.
- `images_generate.rs:25-40`: artifact-mode bundle/artifact write fails loud (today the bundle write is best-effort warn-only).

## Rejected

- Output-path request param: a remote peer must never name a filesystem path (same rule as the blender route); traversal vector.
- Callback URL: second auth scheme + SSRF surface; use a pre-registered endpoint id if ever needed.
- honkbox writing into loop-lore storage: couples honkbox to loop-lore's DB/upload layout; storage ownership stays in loop-lore's `createAsset`.
