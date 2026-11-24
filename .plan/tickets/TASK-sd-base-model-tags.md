# TASK: Base-model compatibility tags

Status: done (2026-11-24)
- Scan-time safetensors `__metadata__` base tags normalized to
  `sd15`/`sdxl`/`sd3` (unknown text preserved), surfaced through
  `ModelMetadata::Safetensors.base_model` in `/v1/models`.
- Known checkpoint/LoRA mismatch is a named 400 `base_model_mismatch` before
  GPU work; one-sided or unknown tags warn. Both tags are written to the job
  sidecar (`checkpoint_base_model`, `loras[].base_model`).
- Verified 2026-11-24: registry 26/26; server edits 186/186 (mismatch, warning,
  sidecar cases), determinism/asset-packs/capabilities/bin suites green;
  clippy `-D warnings` clean for server/registry/api-types; codegen refreshed.
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
