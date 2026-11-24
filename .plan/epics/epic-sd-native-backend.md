# EPIC: Native SD backend completeness

Status: proposed
Priority: P0
Owner: unassigned
Research: `.tmp/research/ReviewedSupport-SD.md`,
`ReviewedSupport-CurrentState.md`

## Scope

Close the gap between the request surface and the vendored stable-diffusion.cpp
seam: real ControlNet execution, honest mid-flight cancellation, LoRA on edits,
mask semantics, and base-model compatibility tags. All work belongs in the
`vendor_patches/` stack + `sd-core`, not in server-only special cases.

## Tickets

- `TASK-sd-controlnet-execution.md` — wrap control-net load/has/unload (P0).
- `TASK-sd-ffi-cancel.md` — wrap `sd_cancel_generation` (P0).
- `FEAT-sd-edit-loras.md` — thread LoRAs into `/v1/images/edits` (P1).
- `TASK-sd-inpaint-mask-semantics.md` — P1, landed 2026-11-24: explicit
  replace/inpaint mode, mask sha256 in sidecar, mismatched dims rejected.
- `TASK-sd-base-model-tags.md` — P2, landed 2026-11-24: normalized safetensors
  base tags, known-mismatch 400, unknown warnings, sidecar tags.

## Acceptance

A valid ControlNet request no longer returns 501; cancel responses state
interruptibility honestly; new fields keep `reject-not-clamp` and the 2048 px cap.

## Verification

`bun run test:core`, `bun run test:server`, `bun run lint:rs`; one opt-in local
checkpoint fixture for ControlNet/cancel behavior. No CI model downloads.

## Non-goals

Bundling third-party weights, IP-Adapter/PhotoMaker embedding, and byte-level
cross-GPU equality.
