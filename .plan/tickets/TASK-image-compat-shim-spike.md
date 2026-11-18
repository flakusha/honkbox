# TASK: sd.cpp/sdapi image-compat shim spike

Status: proposed
Priority: P3
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: a recorded decision on whether honkbox serves an sd.cpp-compatible subset (`/sdapi/v1/txt2img|img2img` sync, `/sdcpp/v1/capabilities`, job polling) backed by `SdPipeline`, and what stays honkbox-native if not.
Verification: documentation decision review; `bun run codegen:check` if a DTO is sketched
Links: stable-diffusion.cpp `examples/server/api.md` (three families: native `/sdcpp/v1` jobs, sync `/sdapi/v1`, OpenAI `/v1/images`); loop-lore `src/generation/image-engine/{sdapi,sdcpp,openai}.ts` (consumers per family); `TASK-loop-lore-provider-smoke` (already covers `/v1/images/generations` interop)
Resources: none (spike; no model downloads)

## Change

- Assess which family maps cleanly: `/v1/images/generations|edits` already exists (smoke-covered); the gap is the sync `/sdapi/v1` pair (response-on-finish + `/sdapi/v1/progress`, `denoising_strength` clamp, hr-fix fields) and native `/sdcpp/v1` job polling (`queued|generating|completed|failed|cancelled`, `completed-ttl` pruning, `409 model_busy` on load).
- Assess `<lora:…>` prompt tags (explicitly unsupported by sd.cpp — structured `lora` fields instead) against honkbox's LoRA path, and the `503 model_not_loaded`/`model_loading` states against the residency ledger.
- Non-goal: video (`/sdcpp/v1/vid_gen`, `/v1/video`) and upscale — separate epics own those.

## Acceptance detail

Spike concludes shim-a-family, shim-nothing (document why), or extend-`/v1/images` instead; a follow-up ticket is filed only for the chosen path. No route code in this ticket.
