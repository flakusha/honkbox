# EPIC: Native SD backend completeness

Status: in progress
Priority: P0
Owner: unassigned
Research: `.tmp/research/model-support-sdxl-krea2-2026-11-24.md`,
`.tmp/research/native-video-backend-refresh-2026-11-25.md`,
`.tmp/research/ash-promotion-refresh-2026-11-25.md`

## Scope

The image-completeness work is landed: ControlNet execution, native cancellation,
image-edit LoRAs, explicit inpaint mask semantics, and base-model tags. Remaining
work is honest Ash promotion on master and a bounded native-video FFI proof.
Backend changes belong in the `vendor_patches/` stack, `sd-core`, or the Ash crate,
not server-only special cases.

The pinned stable-diffusion.cpp revision already contains the video C API and
MiniMax-H3/LTX-2 implementations. This epic does not propose a vendor upgrade;
it proves the current pin before any video surface is admitted.

## Current landed state (2026-11-25)

- `TASK-sd-controlnet-execution.md`, `TASK-sd-ffi-cancel.md`,
  `FEAT-sd-edit-loras.md`, `TASK-sd-inpaint-mask-semantics.md`, and
  `TASK-sd-base-model-tags.md` are done.
- `FEAT-ash-vulkan-backend.md` is in progress at crate scope. Reduced SD1.5
  generation, SD2 header/spec acceptance and SDXL inspection exist; master CLI
  source is not manifest-reachable and HTTP selection/capability truth remain open.
- The pinned vendor header/source exposes video frames/audio/fps and H3/LTX
  families; the high-level Rust wrapper and honkbox routes remain image-only.

## Tickets

- `TASK-sd-controlnet-execution.md` — P0, done.
- `TASK-sd-ffi-cancel.md` — P0, done.
- `FEAT-sd-edit-loras.md` — P1, done.
- `TASK-sd-inpaint-mask-semantics.md` — P1, done 2026-11-24.
- `TASK-sd-base-model-tags.md` — P2, done 2026-11-24.
- `BUG-ash-sd15-numerics.md` — P0, proposed; canonical timestep layout and
  ResBlock timestep activation before any Ash promotion.
- `TASK-ash-verification-gates.md` — P1, proposed; restore package/CI/hook
  commands and gate shader provenance.
- `TASK-ash-backend-promote.md` — P0, proposed; compile-reachable shared CLI/HTTP
  constructor and truthful capability output.
- `TASK-ash-unsupported-conditioning-guards.md` — P1, proposed; typed refusal at
  the direct pipeline boundary, reusing the landed route error code.
- `FEAT-ash-sd2-spec.md` — P1, proposed; source-cited SD2 spec plus one synthetic
  predictor/DDIM step, not full generation.
- `TASK-ash-family-readiness-markers.md` — P2, proposed; execution-scope markers
  without a whole-family availability boolean.
- `TASK-ash-rrdb-upscaler-spike.md` — P2, proposed; one RRDB block and PixelShuffle
  shape/repeatability evidence, not a full upscaler.
- `TASK-video-ffi-spike.md` — P2, proposed; current-pin support probe + one short
  local clip; no HTTP route or capability advertisement.
- `FEAT-video-output.md` — P3, postponed until the spike proves one named family
  on the current pin.

## Dependency order

1. Keep landed image completeness independent.
2. Fix Ash SD1.5 numerics and restore verification/provenance gates; then make
   Ash compile-reachable and capability-truthful. Guard unsupported conditioning
   and mark family execution scope without advertising availability.
3. Prove SD2 one-step numerics and the RRDB/PixelShuffle spike as separate,
   bounded evidence; neither implies full-family support.
4. Complete the license-gated current-pin video FFI spike.
5. Only after a green clip, define owned video output in `sd-core`; job admission,
   routes and capability advertisement remain gated in `epic-video-support`.

## Acceptance

The image-completeness tickets stay green. Ash SD1.5 numerics match the vendored
reference, verification/provenance gates run, and Ash selection/capability
output match the compiled constructor. Direct callers receive named conditioning
refusals, and family diagnostics never imply unsupported execution. The video
spike records exact family/mode/frame/fps/edge evidence and does not change the
HTTP surface.

## Verification

`bun run test:core`, `bun run test:server`, `bun run lint:rs`; feature-on/off Ash
server tests; one opt-in local checkpoint fixture for ControlNet/cancel/Ash smoke
and, separately, one license-cleared local video fixture. No CI model downloads.

## Non-goals

Bundling third-party weights, IP-Adapter/PhotoMaker embedding, byte-level
cross-GPU equality, a stable-diffusion.cpp vendor bump, full SD2/SDXL generation,
Ash LLM, or any video route/capability flag before the current-pin spike succeeds.
