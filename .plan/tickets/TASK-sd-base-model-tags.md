# TASK: Base-model compatibility tags

Status: proposed
Priority: P2
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: registry scan records base-model tags for checkpoints and LoRAs, and a mismatched pair warns or is rejected.
Verification: `bun run test:registry`, `bun run codegen:check`, `cargo test --locked -p honkbox-registry`
Links: `.tmp/research/ReviewedSupport-SD.md` expansion table; `api-types/src/lib.rs:959-961`
Resources: per-test model directories with stub GGUF/safetensors headers; no shared registry

## Change

- Extend scan-time metadata with `base_model` tags; keep registry std+serde.
- Compare checkpoint and LoRA tags at request validation; warn for unknown
  tags, reject a known mismatch with a named 400.
- Surface tags in `/v1/models` and the job sidecar.

## Acceptance detail

A known SDXL LoRA on an SD1.5 checkpoint is rejected before GPU work; unknown
tags degrade to a warning, never a silent success.
