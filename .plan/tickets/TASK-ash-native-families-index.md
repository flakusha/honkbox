# TASK: Reconcile Ash native-family roster and readiness

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: one audited roster links every S0-S9 and family owner exactly, preserves both track orders and deferred candidates, and records readiness with the existing marker vocabulary without implementing or advertising any family.
Verification: `giwt plan check-links`, `bunx markdownlint-cli2 .plan/epics/epic-ash-native-families.md .plan/tickets/TASK-ash-native-families-index.md`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md`; `.tmp/research/ash-family-evidence-inventory-2026-11-25.md`; [epic-ash-native-families.md](../epics/epic-ash-native-families.md); [TASK-ash-family-readiness-markers.md](TASK-ash-family-readiness-markers.md).
Resources: read-only current-source/research reconciliation; unique repo-local scratch directory only if a future mechanical checker needs fixtures; no GPU, checkpoint, network, model download, or shared cache.

## Verified 2026-11-25

- **[FACT]** The audited implementation index defines S0-S9, two dependency
  tracks, eight candidates, stop conditions, and a one-job/one-device boundary.
- **[FACT]** The evidence inventory makes current source authoritative, requires
  user-supplied weights, and prohibits a second GGUF parser, deterministic cache,
  residency system, model registry, or availability mechanism.
- **[FACT]** Existing Ash source has reduced SD1.5 execution, SD2 header/spec
  acceptance, SDXL inspection, multi-head attention, safetensors reads, and DDIM.
  It does not establish GGUF/decoder/DiT family execution.
- **[FACT]** Known SD1.5 numerics, verification, promotion, and unsupported
  conditioning work already have owners. SD2 and SDXL inspection must not be
  re-filed as a new-family deliverable.
- **[UNKNOWN]** Build/runtime parity, quality, latency, VRAM, real-checkpoint
  tolerances, exact upstream revisions, and operator license declarations for
  every candidate.

## Dependencies

This ticket coordinates planning records only. It may start now and does not
authorize family execution.

### Existing correctness and evidence owners

- [BUG-ash-sd15-numerics.md](BUG-ash-sd15-numerics.md)
- [TASK-ash-verification-gates.md](TASK-ash-verification-gates.md)
- [TASK-ash-backend-promote.md](TASK-ash-backend-promote.md)
- [TASK-ash-unsupported-conditioning-guards.md](TASK-ash-unsupported-conditioning-guards.md)
- [FEAT-ash-sd2-spec.md](FEAT-ash-sd2-spec.md)
- [TASK-ash-family-readiness-markers.md](TASK-ash-family-readiness-markers.md)
- [TASK-ash-rrdb-upscaler-spike.md](TASK-ash-rrdb-upscaler-spike.md)
- [TASK-model-provenance-contract.md](TASK-model-provenance-contract.md)
- [TASK-model-license-registry.md](TASK-model-license-registry.md)

### Shared substrate owners

- S1: [TASK-ash-gguf-reader.md](TASK-ash-gguf-reader.md)
- S2: [TASK-ash-gguf-dequant.md](TASK-ash-gguf-dequant.md)
- S3: [TASK-ash-tokenizer-framework.md](TASK-ash-tokenizer-framework.md)
- S4: [TASK-ash-decoder-ops.md](TASK-ash-decoder-ops.md)
- S5: [TASK-ash-autoregressive-state.md](TASK-ash-autoregressive-state.md)
- S6: [TASK-ash-sampling-contract.md](TASK-ash-sampling-contract.md)
- S7: [TASK-ash-flow-dit-substrate.md](TASK-ash-flow-dit-substrate.md)
- S8: [TASK-ash-vae-family-adapters.md](TASK-ash-vae-family-adapters.md)
- S9: [TASK-ash-llm-evidence-integration.md](TASK-ash-llm-evidence-integration.md)

### Bounded family owners

- [FEAT-ash-moss-image-family.md](FEAT-ash-moss-image-family.md)
- [FEAT-ash-embeddinggemma-2.md](FEAT-ash-embeddinggemma-2.md)
- [FEAT-ash-lfm25-text-decoder.md](FEAT-ash-lfm25-text-decoder.md)
- [FEAT-ash-qwen35-dense.md](FEAT-ash-qwen35-dense.md)
- [TASK-ash-qwen-image21-attention-proof.md](TASK-ash-qwen-image21-attention-proof.md)

## Change

- Maintain one roster in the epic and one dependency list here, using the exact
  filenames above. Reconcile additions/removals against current code and both
  cited research documents; do not infer readiness from generated indexes.
- Preserve the post-S0 SD order: shared S7/S8 flow-DiT/VAE proof → MOSS-Image →
  conditional Qwen-Image-2.1 attention/prefix-cache proof.
- Preserve the post-S0 LLM order: shared S1-S6 GGUF/tokenizer/decoder/state/
  sampling substrate → embeddinggemma-2 and LFM2.5 smallest proofs → Qwen3.5
  dense → separately reconsidered Gemma 4/larger variants.
- Keep Krea 2 conditional/deferred, MiniMax H3 “do not start native Ash,” and
  Inkling-2-Mini late/conditional. Record their unmet dependencies and license or
  architecture gates without creating execution tickets in this wave.
- Reuse [TASK-ash-family-readiness-markers.md](TASK-ash-family-readiness-markers.md)
  vocabulary: `header=accepted`, `keys=family-spec-required` or the named subset
  audit, `execution=inspection-only`, `execution=blocked-spec-only`, and
  `execution=reduced-sd15-path`. A `runnable` marker requires evidence from the
  owning execution ticket; header acceptance and op-admission counts never imply
  whole-family availability.
- Route admission evidence through S9,
  [TASK-model-provenance-contract.md](TASK-model-provenance-contract.md), and
  [TASK-model-license-registry.md](TASK-model-license-registry.md). Do not create
  family-specific inspector, cache, registry, residency, license-policy, or
  availability tickets.
- Reconcile each owner’s smallest slice, stop condition, resources, and focused
  verification commands before marking a dependency satisfied. Missing real
  harness names or checkpoint evidence remain `[UNKNOWN]`, not assumed passed.

## Acceptance criteria

- [ ] Every audited S0-S9 and bounded family filename appears exactly as an owned
  dependency in both the epic and this index; reciprocal links resolve.
- [ ] Both track orders match the audited research and remain behind S0; deferred
  Krea 2, MiniMax H3, Inkling, and Gemma 4/larger variants have no implied start.
- [ ] Each family has one smallest proof and explicit stop/admission gates; no
  family ticket duplicates a shared substrate owner.
- [ ] Readiness text uses the existing marker vocabulary and distinguishes
  accepted headers, subset key audits, inspection-only/blocked/reduced execution,
  and evidence-backed runnable execution.
- [ ] The roster names no duplicate inspector, parser, cache, registry, residency,
  license-policy, or availability owner.
- [ ] Generated planning artifacts are identified as generated and are not
  hand-edited; stale generated indexes are not used as reconciliation evidence.
- [ ] This ticket changes planning records only and implements no inference
  family, product route, model loader, shader, dependency, or runtime claim.

## Verification

```sh
giwt plan check-links
bunx markdownlint-cli2 \
  .plan/epics/epic-ash-native-families.md \
  .plan/tickets/TASK-ash-native-families-index.md
```

These are planning/link checks, not runtime verification. No build, test, GPU,
checkpoint, network, formatter, or model command is run by this reconciliation
ticket. Family tickets own their exact `--locked` commands and opt-in
single-device real-checkpoint evidence.

## Generated artifacts

`.plan/epics-index.md`, `.plan/tickets/index.json`, `.plan/feature-matrix.md`,
`.plan/interconnect.md`, and `.plan/code-map.json` are generated and explicitly
must not be hand-edited. Regenerate them only through the repaired owner,
[TASK-plan-index-generator.md](TASK-plan-index-generator.md), once available.

## Non-goals

Implementing any family or shared kernel; making Ash selectable; adding an HTTP
surface or capability claim; duplicating model discovery/provenance/license
systems; downloading or bundling weights; multi-GPU execution; benchmarks;
quality claims; or treating accepted metadata as runnable support.