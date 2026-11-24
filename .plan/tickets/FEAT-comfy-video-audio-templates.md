# FEAT: ComfyUI video and audio template family

Status: proposed
Priority: P1
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: pinned native LTX-2.5/H3 audio-video graphs and LTX-2.3 Foley V2A submit through the typed workflow pipeline with operator-supplied weights, exact required-class preflight, separated component/weight license evidence, bounded declared-output capture and no raw workflow or ComfyUI path exposure.
Verification: `cargo test --locked -p honkbox-server --test comfy_workflows`, `cargo test --locked -p honkbox-server --test apps`, `bun run codegen:check`, `bun run check`; opt-in live ComfyUI smoke with operator-supplied weights stays ignored by default
Links: `.tmp/research/media-masters-audio-video-2026-11-25.md`, `.tmp/research/video-to-audio-modalities-2026-11-24.md`, `FEAT-comfyui-workflow-pipeline.md`, `TASK-comfy-custom-node-capability-gate.md`, `FEAT-media-master-and-derived-artifacts.md`, `FEAT-video-source-audio-reuse.md`, `epic-video-support.md`
Resources: per-test fake ComfyUI servers on port 0, prompt ids and temp template/output dirs; one dedicated operator-pinned live instance only for ignored smoke; no shared ComfyUI queue, global `/free`, CI weights or Python dependency in honkbox

## Context

[FACT] honkbox has no native LTX/H3 video worker, frame timeline, audio extraction
or muxer. The `ltx2` scheduler in `sd-core` is an image scheduler name, not video
support. This ticket is an external ComfyUI adapter; native video remains gated
by `FEAT-video-output.md` and `epic-video-support.md`.

[FACT] Current ComfyUI docs publish native LTX-2.5 T2V/I2V/FLF2V/DFR workflows
with synchronized audio, and native MiniMax H3 T2V/I2V/R2V workflows with stereo
audio. LTX-2.3 Foley V2A is a separate, custom-node path that freezes an existing
muted video and generates audio.

[FACT] “Native workflow” does not remove node gating. Official LTX-2.5 graphs
contain LTXV*, `MultimodalGuider` and `VHS_VideoCombine` classes; the pinned API
graph's complete class set is authoritative. `/object_info/{class}` proves only
that a class is visible with an input schema, not pack version, digest, approval,
provenance or license.

## Change

### Pinned template family

Depends on `FEAT-comfyui-workflow-pipeline.md` and
`TASK-comfy-custom-node-capability-gate.md`. Add immutable, versioned API-format
templates with exact graph/manifest SHA-256 and a closed typed slot schema. No
browser/workflow JSON is accepted at runtime.

Ranked first proofs:

1. **LTX-2.5 native T2V and I2V** for synchronized audio-video. Start from the
   official ComfyUI templates, pin the exact API graph and model set, and keep
   LTX-2.5 assets separate from LTX-2.3. Hugging Face repository access and every
   weight are acquired by the operator before launch.
2. **MiniMax H3 native T2V/I2V** for synchronized stereo audio-video; add R2V
   only after its larger reference/audio slot and output bounds are proven.
   Current docs require ComfyUI 0.30.0+ for the base templates. Admission requires
   an explicit operator license/territory/revenue decision.
3. **LTX-2.3 Foley V2A** for an existing muted video plus a Foley prompt. Pin the
   Foley LoRA/base/text-encoder combination and the official API workflow. It
   requires `ComfyUI-LTXVideo` audio/V2A nodes and VideoHelperSuite
   `VHS_LoadVideo`; use bounded windows and preserve the source frames.

MMAudio (`MMAudioSampler`) may remain an operator-supplied V2A alternative after
its pack, weights and license are independently pinned.
ControlFoley/HunyuanVideoFoley are later alternatives, not first proofs.
FoleyCrafter is not a first proof.

Each mode gets its own pruned API graph and exact `required_class_types[]` set.
Optional turbo/prompt-enhancer/reference branches are removed when disabled;
otherwise their classes and weights are required too. Unknown parameters,
URLs, filesystem paths and arbitrary JSON are rejected before upstream admission.

### Inputs, weights and outputs

- `video_ref`, `image_refs`, `audio_refs`, prompt, seed, bounded duration/frame
  count, output selection and closed enums are typed slots. Existing honkbox
  artifact ids are the only media references; staging uses generated safe names
  and never a client absolute path.
- Base weights, VAEs, text encoders, LoRAs/adapters, embeddings and node packs
  are user-supplied and pre-staged through the operator's ComfyUI model paths.
  honkbox performs no download, mirror, install, Manager call, `pip`, `git` or
  generic URL fetch. LTX-2.5 gated-repository access is an operator prerequisite.
- Model/weight slots resolve only reviewed artifact ids or closed catalog
  references. No model path or upstream URL crosses the public API.
- Declare outputs per graph. Capture only declared `type=output` references via
  bounded `/view` streaming, hash while writing, and register the same bytes in
  the artifact store. Write `manifest.json`/media roles last.
- Native LTX/H3 `SaveVideo` output is a master only in the provenance sense; set
  `lossless` from the actual container. Do not call H.264/H.265 MP4 lossless.
  Separate audio is derived only when the graph emits it. Do not promise
  WAV/FLAC extraction from a muxed MP4 without a separately reviewed demux path.
- LTX-2.3 Foley may emit the original frames plus generated audio. Preserve the
  source video hash, generated-audio hash, window bounds and mux/merge relation;
  never silently replace source audio outside the selected mode.

### Component and weight license evidence

Template admission requires separate operator evidence for:

- ComfyUI/core code and each custom node pack;
- base model weights and quantized variants;
- text encoder/VAE and each LoRA/adapter/embedding;
- output use and any commercial grant.

LTX code is Apache-2.0, but LTX-2.x weights use the LTX-2.x Community License
amended 2026-11-11. The license excludes weights/parameters from “Apache Code”;
entities at or above $10M annual revenue need a paid commercial license for
commercial use. Never state that Apache code makes LTX weights Apache-licensed.

MiniMax H3 weights, inference code and derivatives use the H3 Community License,
not Apache-2.0. It excludes the EU/UK/Republic of Korea/USA and requires prior
written MiniMax authorization above $20M yearly commercial revenue. Its
Qwen3-VL-32B encoder is Apache-2.0 as a separate component; that does not relicense
H3. Retain the actual commercial grant when the community territory/revenue terms
do not apply.

ComfyUI is GPL-3.0 and remains an operator-owned external process. honkbox links
no ComfyUI code and bundles no node pack, weight or adapter. Missing, conflicting
or unverifiable license evidence blocks template admission; `/object_info` cannot
supply it.

### Execution contract

- Preflight every required class with targeted, bounded
  `GET /object_info/{class}` before `POST /prompt`. Missing class or `{}` returns
  501 naming it and makes zero prompt calls; non-200/invalid JSON is 502 and also
  makes zero prompt calls.
- Hold the workflow pipeline's managed permit or detached Comfy lease for the
  whole run. Use one fresh prompt id/client id and a private WebSocket; accept
  only matching frames.
- Recover through `/history/{prompt_id}` + `/view` after socket loss. Unknown
  submission is reconciled by the same id, never blindly resubmitted.
- Cancel only the owned id with targeted `/queue` and `/interrupt`; never clear
  the queue, send an empty/global interrupt or call `/free`.
- Runtime policy is `auto_install: false`, `runtime_fetch: false`. The operator
  policy note owns pack refs/digests/patches, whitelists and blocked-egress audit.

## Acceptance detail

1. Fake-server happy paths for one LTX-2.5 graph, one H3 graph and one LTX-2.3
   Foley graph assert exact template/graph hashes, typed slots, generated input
   names, one prompt/client id, declared-output copy, source/output hashes and
   media roles.
2. Present/absent/empty/faulty class catalogs prove preflight outcomes. A missing
   class returns 501 with its name and zero `/prompt` calls; faulty catalogs return
   502 with zero prompt calls. Optional disabled branches are absent from both
   graph and required-class set.
3. A class visible upstream but absent from operator policy is rejected. Public
   capabilities and job records never claim pack version, digest, approval or
   provenance from class visibility.
4. Missing weight/license evidence, a path/URL slot, runtime-fetch/install class,
   raw graph request, unknown parameter, LTX-2.3/2.5 asset mismatch or digest
   mismatch fails before submission.
5. Socket loss recovers the same prompt exactly once; cancel targets only the
   owned id; a decoy prompt remains queued/running and its files unchanged. No
   queue-clear, empty interrupt or `/free` request is observed.
6. Oversized input/output, excessive output count, undeclared node/media key,
   traversal/symlink-shaped reference or invalid upload response fails without a
   publishable manifest. Existing bundle/artifact caps remain unchanged.
7. A `SaveVideo`-only graph yields a truthful muxed-video master/final record,
   not a fabricated WAV/FLAC artifact. A graph with separate declared audio
   yields hashed audio with correct lineage.
8. Live smoke is ignored unless `HONKBOX_COMFY_LIVE=1`, uses a dedicated
   operator-pinned loopback instance and pre-staged licensed weights, records
   sanitized component hashes/policy evidence, and proves no install/download/
   fetch endpoint or unexpected egress occurred.

## Non-goals

- Native honkbox LTX/H3 video, a Python sidecar, new model registry family,
  bundled weights/node packs, generic workflow proxy or ComfyUI Manager.
- Automatic audio extraction/demux/mux, FFmpeg/OIIO integration, realtime video,
  unlimited duration/frames/edge or shared upstream queues.
- Treating Apache code, Qwen3-VL or class visibility as permission for LTX/H3
  weights or custom node packs.

## Verification

Commands to run after implementation (not run for this ticket):

```sh
cargo test --locked -p honkbox-server --test comfy_workflows
cargo test --locked -p honkbox-server --test apps
bun run codegen:check
bun run check
HONKBOX_COMFY_LIVE=1 cargo test --locked -p honkbox-server \
  --test comfy_workflows -- --ignored --nocapture
```
