# TASK: Accept and render ControlNet requests

Status: done
Priority: P0
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: `/v1/images/generations` validates `controlnet`/`control_image`/`controlnet_strength`, resolves the catalog `controlnet` entry, threads a valid pair into the SD pipeline request (`control_net`/`control_image`/`controlnet_strength`), and answers 501 `controlnet_unsupported` (never a silent ignore) for pipelines that cannot carry control. Full execution through the real engine is `TASK-sd-controlnet-execution.md`, not this ticket.
Verification: done — `crates/server/tests/jobs.rs`: `controlnet_rejections_are_precise_400s`, `controlnet_valid_inputs_answer_501_and_empties_fall_through`, `controlnet_valid_pair_is_threaded_into_the_sd_cpp_pipeline`, `controlnet_strength_default_and_rejection`; inline `crates/sd-core/src/lib.rs`: `control_pair_accepts_only_complete_pairs`; vendor plumbing in `vendor_patches/0004-gen-img-control-image.patch`; resolve/refusal logic at `crates/server/src/routes.rs:3055` (`resolve_controlnet`).
Links: `.plan/tickets/TASK-sd-controlnet-execution.md` (full execution path); `vendor_patches/0004-gen-img-control-image.patch`
Resources: per-test control image temp files; no CI downloads

## Change

- Add `controlnet` (catalog slug), `control_image` (base64 PNG) and
  `controlnet_strength` in `(0, 1]` to the image generation request;
  reject-not-clamp for out-of-range strength and half-pairs.
- Resolve the slug against the `controlnet` catalog kind; refuse with 501
  `controlnet_unsupported` when the active pipeline cannot carry control.
- Plumb `control_net`/`control_image`/`controlnet_strength` through
  `sd-core` `Txt2ImgRequest` and the vendored engine via
  `vendor_patches/0004-gen-img-control-image.patch`.

## Acceptance detail

A valid control pair is threaded into the sd.cpp pipeline request; invalid
or partial input returns named 400s before any GPU work; a pipeline that
cannot carry control answers the honest 501, never an uncontrolled render.
