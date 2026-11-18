# TASK: Per-request LoRAs on `/v1/images/generations`

Status: done
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: `/v1/images/generations` accepts `loras: [{slug, strength}]`, resolves each slug against the catalog `loras` kind, rejects (never clamps) more than two adapters or strengths outside `[-2, 2]`, enforces the single-LoRA-directory constraint, emits advisory license warnings, and records slugs/strengths in the job sidecar.
Verification: done — `crates/server/tests/images.rs`: `unknown_lora_slug_is_400`, `too_many_loras_is_400`, `lora_strength_out_of_range_is_400`, `mixed_lora_dirs_is_400`, `loras_warn_and_record`, `warnings_absent_without_loras`; resolution/warning logic at `crates/server/src/routes.rs:2877` (`resolve_loras`); sidecar recording tested by `crates/server/src/jobs.rs`: `sidecar_records_loras_and_strength`.
Links: `.plan/tickets/FEAT-sd-edit-loras.md` (edits-side LoRA surface)
Resources: per-test temp LoRA catalog entries under fake roots; no shared LoRA cache

## Change

- Add `loras: [{slug, strength}]` to the image generation request (max 2
  adapters, strength `[-2, 2]`, reject-not-clamp).
- Resolve slugs against the catalog; all LoRAs in a request must live in
  one directory; emit an advisory license warning naming each slug.
- Record resolved slugs/strengths in the job bundle sidecar for
  reproducibility.

## Acceptance detail

Valid LoRA lists resolve, warn, and land in the sidecar; invalid slugs,
counts, strengths, or mixed directories return named 400s before any GPU
work; without `loras` the `warnings` key is omitted entirely.
