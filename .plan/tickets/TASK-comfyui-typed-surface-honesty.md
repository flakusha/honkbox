# TASK: make the typed ComfyUI reads honest and usable

Status: done (2026-11-24)
- Implemented: `ComfyUiArtifact` DTO + generated TS/OpenAPI; history outputs are
  validated artifact refs; `/view` accepts a safe `subfolder`; progress/status
  come from upstream history messages; detached status reports
  `available: false` / `detached-unverified`; upstream failures map to 502/503
  with no path or error chain. Unit + router tests added.
- Verified 2026-11-24: full `cargo test --locked -p honkbox-server` (29 test
  binaries), `codegen:check`, workspace clippy, oxlint/tsc/dprint/API-docs gates.
Priority: P1
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: `/comfyui/history/{id}` names artifacts `/comfyui/view` can fetch, and every DTO claim matches what the handlers emit in managed and detached mode.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/comfyui-endpoint-review-2026-11-24.md` §4; `FEAT-comfyui-workflow-pipeline.md`
Resources: fixture history JSON; temp dirs per test; no shared ComfyUI queue state

## Change

- Extract `filename`/`subfolder`/`type` from upstream history outputs into a
  typed artifact list; keep raw node-graph/workflow JSON out. The artifacts
  must be directly usable as `/comfyui/view` query parameters.
- Populate `ComfyUiProgress` from upstream status messages when present; keep
  `None` when absent and say so in the DTO docs. Align the documented status
  vocabulary with the emitted values (`queued`/`completed`/`success`/
  `error`/`running` as actually produced).
- Detached mode: stop claiming `available: true` from the cached external-tools
  row without a live check (or mark the status unverified). Distinguish
  "upstream reloading" from "upstream dead" in the 501 reason without leaking
  paths or the error chain.
- Preserve the current security bounds: `safe_component`, `type` allowlist,
  query encoding, 16 MiB body cap, 1 MiB JSON cap, octet-stream content type,
  loopback-only upstream.

## Acceptance detail

- A fixture history entry with one `SaveImage` output yields one typed
  artifact whose fields round-trip through `/comfyui/view` validation.
- No status value documented in `api-types` is unreachable, and no reachable
  value is undocumented; TS generated files match the Rust DTOs.
- Tests cover both managed and detached truth sources.
