# TASK: ComfyUI-compatible serve subset spike

Status: proposed
Priority: P3
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: a recorded decision on whether honkbox serves a ComfyUI-compatible subset (`POST /prompt`, `GET /history/{id}`, `GET /view`, `/ws` progress) backed by `SdPipeline`, and what the refusal surface is if not.
Verification: documentation decision review; `bun run codegen:check` if a DTO is sketched
Links: ComfyUI `server.py` + `docs.comfy.org` comms routes/messages (push completion via `executing node===null`, `/history` poll fallback); loop-lore `src/generation/providers/comfyui*.ts` (consumer: submit/poll/download/interrupt); llama-swap `/comfyui` passthrough docs (the shape being paralleled)
Resources: none (spike; fixture workflow JSON only if a probe is built)

## Change

- Assess mapping honkbox job model onto ComfyUI semantics: prompt_id ↔ job ULID, node outputs ↔ bundle artifacts, `SaveImage` requirement (ComfyUI persists nothing without a save node; honkbox bundles everything), `/interrupt` ↔ job cancel, `/free` ↔ unload.
- Assess auth gap: ComfyUI servers are unauthenticated by default; honkbox mounts behind bearer — record how a compat subset stays behind the bearer layer without breaking ComfyUI clients.
- Non-goal: full node-graph execution — the subset only serves pipeline-backed templates, same constraint as `FEAT-comfyui-workflow-pipeline` in reverse.

## Acceptance detail

Spike concludes serve-a-subset, serve-nothing (document why), or serve-behind-adapter; a follow-up ticket is filed only for the chosen path. No route code in this ticket.
