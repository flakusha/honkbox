# TASK: Execute ControlNet requests

Status: proposed
Priority: P0
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: a valid ControlNet request runs through the vendored engine instead of returning 501 `controlnet_unsupported`.
Verification: `bun run test:core`, `bun run test:server`, `cargo test --locked -p honkbox-server`
Links: `.tmp/research/ReviewedSupport-SD.md` §Partial; `routes.rs:2924-2991`
Resources: per-test control image temp files; one opt-in local checkpoint fixture, no CI downloads

## Change

- Add control fields to `SdPipeline::Txt2ImgRequest` (`sd-core:283-325`).
- In `vendor_patches/`, wrap `sd_ctx_load_control_net`,
  `sd_ctx_unload_control_net`, `sd_ctx_has_control_net`, and `sd_control_type_t`
  (C header `:546-549,567-570`); expose through `diffusion-rs` + `sd-core`.
- Map catalog slug → weight path + control type; keep existing 400 validation
  for half-pairs, bad base64, non-PNG, unknown slug.
- Delete the 501 branch and update `/v1/capabilities` to report `controlnet: true`
  only when the feature and weights are present.

## Acceptance detail

A ControlNet job produces an image through the real backend; a missing weight
returns a named error, never a 501; structural parity checks stay green.
