# FEAT: ComfyUI video-to-audio template family
Status: proposed
Priority: P1
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: pinned V2A and video+text-to-audio templates submit through the typed workflow pipeline, validate their required node classes against `/object_info`, and copy WAV/FLAC outputs into the job bundle with no raw workflow JSON or ComfyUI server paths exposed.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`; opt-in live ComfyUI smoke with operator-supplied weights stays ignored by default
Links: `.tmp/research/video-to-audio-modalities-2026-11-24.md`, `FEAT-comfyui-workflow-pipeline.md`, `epic-video-support.md`
Resources: per-test prompt ids and temp dirs; no shared ComfyUI queue state

## Change

- Depends on `FEAT-comfyui-workflow-pipeline.md`. No native V2A worker, no new model registry family, no Python dependency.
- Add two versioned templates with recorded hashes: MMAudio (`MMAudioSampler`) for V2A first, ControlFoley `TC-V2A` shape for video+text-to-audio second. HunyuanVideoFoley stays an operator alternative under its own license; FoleyCrafter is not a first proof.
- Template manifest declares `required_class_types`, typed parameters (`video_ref`, optional `prompt`, `seed`, `duration`, `audio_format`), and output contract. Reject unknown parameters before submit.
- Stage input video into ComfyUI `input/` through the job bundle. Never accept an HTTP-supplied absolute path into a workflow.
- Output WAV/FLAC; Opus only where the workflow uses core `SaveAudioOpus`. Copy artifacts into the bundle with node id and filename, never return a ComfyUI path.

## Acceptance detail

- Validate required `class_type`s via `/object_info` before `POST /prompt`; a missing class returns 501 naming it, not an opaque upstream error.
- Hold one lease for the whole run; always send a targeted `/interrupt` body; filter WS frames by `prompt_id`; recover from `/history` + `/view` if the socket drops.
- Node auto-fetch and weight downloads stay disabled or pre-staged by the operator. honkbox bundles no weights and no node packs.
- No raw `POST /prompt`, `/ws` or generic workflow JSON surface is added.
