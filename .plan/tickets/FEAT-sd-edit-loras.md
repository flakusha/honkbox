# FEAT: LoRA on image edits

Status: proposed
Priority: P1
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: `/v1/images/edits` accepts the same validated `LoraRef` list as txt2img instead of hardcoding `loras: None`.
Verification: `bun run test:server`, `bun run codegen:check`, `cargo test --locked -p honkbox-server`
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
