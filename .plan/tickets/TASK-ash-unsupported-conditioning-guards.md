# TASK: Guard Ash conditioning at the pipeline boundary

Status: proposed
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: ignored conditioning is refused by `AshSdPipeline` before device admission, checkpoint reads or sampling; direct callers receive a typed error and HTTP conversion preserves named 501 `conditioning_unsupported`.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd conditioning_guard`, `cargo test --locked -p honkbox-sd-core`, `cargo test --locked -p honkbox-server conditioning_unsupported`.
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §3.2; `crates/server/src/routes/images_generate.rs:73-104`; `crates/ash-backend/src/sd.rs:280-395`.
Resources: CPU-only synthetic requests and a fake error-returning pipeline; no model, GPU lease, download or shared mutable fixture.

## Verified 2026-11-25

- **[FACT]** The route-level candidate already landed:
  `reject_unhonored_conditioning` returns 501 `conditioning_unsupported`, and
  `post_images_generations` invokes it at `images_generate.rs:368-375`.
  ControlNet has its own resolver. Do not re-file that route guard as new work.
- **[FACT]** `AshSdPipeline::generate` does not inspect `loras`, `hires`,
  `init_image`, `mask_image`, `strength`, or ControlNet fields. A direct
  `SdPipeline` caller can therefore ask for conditioning that the reduced
  SD1.5 path does not apply.
- **[FACT]** `SdError` currently has `TooLarge`, `Encode`, `Backend` and
  `Cancelled`, but no typed unsupported-conditioning case
  (`crates/sd-core/src/lib.rs:405-434`).
- **[UNKNOWN]** No build or runtime check was run in this planning pass.

## Dependencies

- None for the pure request classifier, typed error and server mapping tests.
- Real HTTP exposure of `AshSdPipeline` remains owned by
  `TASK-ash-backend-promote`; this ticket neither adds an HTTP constructor nor
  claims the backend is selectable.

## Change

- Keep the existing route guard and error code. Add one pure Ash request check
  reused by `AshSdPipeline::generate` before `SD_SPEC.validate`,
  `admit_checkpoint`, tokenizer loading or any dispatch.
- Add a typed `SdError::UnsupportedConditioning` carrying the first unsupported
  field name. Map that variant to the existing OpenAI 501
  `conditioning_unsupported` envelope; do not flatten it into `Backend`/500.
- Cover non-empty `loras`, `hires`, `init_image`, `mask_image`, `strength`,
  `control_net`, `control_image`, `controlnet_strength`, and `control_type`.
  Keep deterministic field precedence and include the field in the message.

## Acceptance criteria

- [ ] A table-driven CPU test proves each unsupported field is named and that a
  plain supported txt2img request passes the classifier.
- [ ] `AshSdPipeline::generate` calls the check before device admission,
  checkpoint reads, tokenizer access and sampling; refusal cannot produce an
  image or partially honor the request.
- [ ] A fake `SdPipeline` returning the new error proves HTTP status 501, code
  `conditioning_unsupported`, and a message naming the field.
- [ ] Existing route-level Ash guard tests and ControlNet refusal tests remain
  green; no second error code or availability flag is introduced.
- [ ] Missing Ash HTTP selection/capability truth is still reported under
  `TASK-ash-backend-promote`, not marked complete by this ticket.

## Verification

Run from the repository root after implementation:

```sh
cargo test --locked -p honkbox-ash-backend --features ash-sd conditioning_guard
cargo test --locked -p honkbox-sd-core
cargo test --locked -p honkbox-server conditioning_unsupported
cargo test --locked -p honkbox-server
```

No GPU or checkpoint is needed for these checks. The main agent/integrator runs
the commands; this filing pass runs none.

## Non-goals

Implementing LoRA, hires, img2img/inpaint or ControlNet; changing recipe or
sampler semantics; adding HTTP Ash selection; advertising family availability;
returning a fake or silently uncontrolled image.


## Verification evidence (2026-11-24)

`reject_unhonored_conditioning` (`crates/server/src/routes/images_generate.rs:79-104`) returns `conditioning_unsupported` for `ash-sd` on `hires_fix`, `loras` and image inputs. The generate route calls it before cache/admission (`images_generate.rs:368`); the edits route calls it with `image_inputs=true` (`images_edit.rs:342`). `crates/server/tests/images.rs::ash_backend_refuses_unhonored_conditioning` covers hires/loras and `crates/server/tests/edits.rs::ash_backend_refuses_image_inputs` covers image/mask. ControlNet keeps its own guard. `ImagesGenerationRequest` has no `init_image`/`mask_image` fields: those inputs belong to `/v1/images/edits`, where they are already guarded.
