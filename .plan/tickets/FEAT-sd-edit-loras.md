# FEAT: LoRA on image edits

Status: done
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: `/v1/images/edits` accepts the same validated `LoraRef` list as txt2img instead of hardcoding `loras: None`.
Verification: done — `crates/server/tests/edits.rs`: `edits_unknown_lora_slug_is_400`, `edits_too_many_loras_is_400`, `edits_lora_strength_out_of_range_is_400`, `edits_mixed_lora_dirs_is_400`, `edits_loras_warn_and_record`, `edits_warnings_absent_without_loras`, `edits_malformed_loras_field_is_400`; the multipart field parses the same `ImagesLora` JSON shape and validation runs entirely through the shared `resolve_loras` (`crates/server/src/routes.rs:2877`) — no edits-side duplicate checks; sidecar recording reuses `crates/server/src/jobs.rs` (`BundleInputs.loras`).
Links: `.tmp/research/ReviewedSupport-SD.md` expansion table; `routes.rs:3580`
Resources: per-test temp images/masks; no shared LoRA cache

## Change

- Gate the multipart branch with `loras` parsed identically to txt2img
  (`api-types/src/lib.rs:950-961`): max 2, strength `[-2,2]`, catalog slug lookup.
- Preserve the existing reject-not-clamp errors and license warning.
- Record LoRA slugs/strengths in the job sidecar for reproducibility.

## Acceptance detail

Edit requests with LoRAs succeed with the same validation as txt2img; invalid
LoRA input returns 400 before any GPU work.
