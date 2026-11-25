# TASK: Prove Qwen-Image-2.1 segment attention and prefix KV

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: CPU-only tests prove Qwen-Image-2.1 mixed-granularity block-causal attention and per-layer conditioning-prefix K/V reuse against an uncached reference; the existing Qwen Research License gate refuses commercial or undeclared-family admission before any loader, lease or device work; no generation or real weights are involved.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/qwen-image21-attention.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-compute qwen_image21_attention_proof`; `TMPDIR="$(mktemp -d "$PWD/.tmp/qwen-image21-license.XXXXXX")" cargo test --locked -p honkbox-server --test qwen_image21_license`; `bun run lint:rs`.
Links: `.tmp/research/ash-qwen-image21-implementation-candidate-2026-11-25.md` §§4–9; `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S5/S7/S9 and §6; `crates/ash-backend/src/attention.rs`; `crates/server/src/license_gate.rs`; `crates/server/src/config.rs:260-272`; `vendor/diffusion-rs/sys/stable-diffusion.cpp/src/model/diffusion/qwen_image_2_1.hpp:185-245,333-365,506-519`.
Resources: deterministic in-memory f32 Q/K/V and unique gitignored `.tmp/qwen-image21-*` directories; in-process server tests with fake paths and test-local counters; no GPU, checkpoint, tokenizer, VAE, model download, port, database, shared cache or multi-GPU.

## Verified 2026-11-25

- **[FACT]** Ash attention supports concatenated multi-head Q/Kᵀ/V and additive `run_masked`; it has no segment-mask builder, persistent prefix state or Qwen position layout (`attention.rs:9-29,64-148,239-334`). Older single-head/no-mask claims are stale.
- **[FACT]** The pinned MIT reference evaluates each segment query against keys/values through that segment end. Text segments use an internal causal mask; image segments are bidirectional; earlier segments remain visible (`qwen_image_2_1.hpp:237-245,506-519`).
- **[FACT]** The reference stores each layer prefix K/V during STORE and loads it during REUSE, then evaluates only target queries against prefix plus target (`qwen_image_2_1.hpp:194-235,333-365`). This is diffusion-step prefix reuse, not autoregressive decoding or response caching.
- **[FACT]** The existing license gate returns 412 `model_license_non_commercial` for curated non-commercial records in commercial mode, but an absent curated record currently becomes local-only (`license_gate.rs:109-170`). Commercial mode is the default (`config.rs:260-272`).
- **[FACT]** Research records the Qwen Research License (2026-09-20) as non-commercial unless separately licensed. A commercial-evidence representation is not present in `LicenseRecord`; it remains an owner decision.
- **[UNKNOWN]** Prefix-cache numerical tolerance, device memory/driver behavior, exact checkpoint geometry and real image parity are unmeasured. This ticket deliberately performs no device work.

## Dependencies

- `TASK-ash-native-families-index.md` owns S5/S7/S9 scope and SD-track order.
- `TASK-ash-autoregressive-state.md` supplies the shared stateful allocation/cache substrate and invalidation/lifetime conventions. This ticket consumes it; it does not create another cache framework.
- `TASK-ash-flow-dit-substrate.md` supplies single-stream DiT layout, modulation and position-buffer conventions. This proof consumes its segment/position contract without implementing a DiT block or scheduler.
- The completed `TASK-model-license-registry.md` and existing `license_gate` are the policy owners. Family admission/provenance remains shared; no Qwen-only license service is added.
- Existing Ash correctness, verification, selection and readiness tickets remain downstream execution gates, not attention-proof scope.

## Change

### 1. Fail-closed family license admission

- Add a curated Qwen-Image-2.1 family record with `commercial_terms=non-commercial` and authoritative terms URL. Route that family through the existing gate before pipeline resolution/load, residency admission, bundle writes or device construction.
- Remove the generic undeclared-local fallback from this declared family boundary: a local Qwen-Image declaration without a curated record fails closed. Do not infer family identity from a filename or claim content recognition of arbitrary files this ticket never reads.
- Test commercial default, explicit non-commercial mode, missing record, missing terms, unsupported edit/LoRA/hires fields and malformed declaration. Refusal is 412 with the existing named code and leaves loader/admission/device counters at zero.
- Keep commercial use refused until the owner defines auditable separate-license evidence in the shared record. An environment variable, test shortcut or absent record must not bypass policy.

### 2. Mixed-granularity block-causal attention

- Add a validated segment-layout/mask builder over the existing additive attention API. For ordered non-empty segments, a query in segment `i` may attend every key through `segment[i].end`; within text segments only keys `<= query`, within image/reference/target segments the whole segment is visible.
- Reject empty/overflowing layouts, zero-length segments, gaps, overlaps, reversed bounds, out-of-range positions, head/dimension mismatch and non-finite Q/K/V before dispatch. Use checked arithmetic and the existing `-inf` mask convention.
- Prove `[text, reference, text, target]` with a hand-computed multi-head fixture. Text queries cannot see future text; reference and target queries are bidirectional within their segment; no segment sees a later segment; every later segment sees its full prefix.
- Add leakage negatives: changing future text or any later segment cannot change an earlier output; changing a future key inside the same image segment must change it; changing an allowed prefix key must change later outputs. Include one-token segments, unequal lengths and two-head split/concat.

### 3. Per-layer prefix K/V reuse only

- Model STORE then REUSE for two target evaluations per layer using synthetic prefix and target tensors. Cached target output must match full uncached attention within the declared f32 tolerance; prefix tensors are stored once per layer and never mutated by target evaluation.
- Key validity by exact prefix content identity, layer, model/fixture identity, sequence/segment layout, shape/head layout and dtype/precision. Changed prefix bytes, shape, layout, precision or layer must miss.
- Missing, corrupt, truncated, partial or stale prefix state falls back to complete uncached execution; allocation failure does the same. No partial prefix is reused and no output/response is cached.
- Count checked prefix K/V byte lengths and prove two layers do not alias. Keep all state request/job-scoped through the shared substrate; no cross-request reuse, new registry, second cache or family-specific residency system.

## Acceptance criteria

- [ ] Table-driven CPU tests match the independent masked reference for all segment types and prove the future-key leakage negatives.
- [ ] Validation rejects malformed layouts and non-finite tensors before attention dispatch with named errors.
- [ ] STORE/REUSE reproduces uncached target outputs for every synthetic layer and repeated target evaluation within the declared tolerance.
- [ ] Prefix invalidation, corruption, partial state, allocation failure and cross-layer aliasing have explicit tests; fallback is complete uncached execution.
- [ ] Commercial, missing-record and unsupported-conditioning admissions return named 412s before loader, lease or device counters change; explicit non-commercial policy reaches only the synthetic proof boundary.
- [ ] Tests initialize no Vulkan device, load no checkpoint and require no real weights; the ticket makes no generation, memory, performance or availability claim.

## Verification

```sh
mkdir -p .tmp
TMPDIR="$(mktemp -d "$PWD/.tmp/qwen-image21-attention.XXXXXX")" \
  cargo test --locked -p honkbox-ash-backend --features ash-compute \
    qwen_image21_attention_proof
TMPDIR="$(mktemp -d "$PWD/.tmp/qwen-image21-license.XXXXXX")" \
  cargo test --locked -p honkbox-server --test qwen_image21_license
bun run lint:rs
```

CPU attention tests must pass without a Vulkan loader. Both normal targets run in parallel: each test owns unique temp paths/counters and deterministic cleanup, with no shared globals or ordering dependence. Server tests use fake paths, never `global()` residency or a real device. No command here was run during planning.

## Non-goals

Qwen-Image generation or execution promotion; DiT/VAE/Qwen3-VL implementation; 3-axis RoPE; flow/Turbo scheduling; reference-image editing; real checkpoints or weights in CI; automatic downloads/conversion; new routes/DTOs, cache/registry/residency/license systems, dependencies or bundled assets; multi-GPU; quality, latency or VRAM claims. Any later device proof is separate and must pass the license gate first, verify every artifact SHA-256/license, and acquire one exclusive single-device lease before device creation.
