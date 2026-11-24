# FEAT: ComfyUI workflow pipelines

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: submit/cancel/events/artifacts for pinned API-format workflow templates through an external ComfyUI server.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/ReviewedFullCycle-ComfyUI-Pipelines.md`,
`comfyui-serve-compat-spike-2026-11-20.md`, `epic-comfyui-managed-runtime.md`
Resources: per-test prompt IDs and temp output dirs; no shared ComfyUI queue state
Prerequisite: honkbox's managed ComfyUI app and the four typed `/comfyui/*`
reads are landed; this ticket adds the template write surface above them, not
a raw ComfyUI wire proxy.

## Change

- New `WorkflowPipeline` seam mirroring `SdPipeline`; no raw user workflow JSON.
  Templates are versioned assets with a recorded hash.
- Validate `class_type`s once against `GET /object_info`; inject typed params
  into named slots; fail with 501 naming a missing node.
- Submit via `POST /prompt`; track `prompt_id`; use `/ws` for progress and
  `/history/{prompt_id}` + `/view` as the source of truth for outputs.
- Cancel queued work with `POST /queue`; running work with `POST /interrupt`;
  idle unload with `POST /free`. Hold a lease for the whole execution.

## Acceptance detail

All output nodes are copied into the job bundle with node IDs and filenames;
WS disconnect does not lose the result; no ComfyUI server path is returned to
clients. Workflows are code — only admin-installed templates.
