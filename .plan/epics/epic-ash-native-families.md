# EPIC: Ash native inference families

Status: proposed
Priority: P1
Owner: unassigned
Acceptance: shared S1-S9 substrates have passing focused proofs before any family starts; each family proves only its named smallest slice; family diagnostics reuse readiness markers and never imply availability from a header.
Verification: run each linked ticket’s exact focused commands with `--locked`, plus `giwt plan check-links`; real-checkpoint proofs remain opt-in, single-device, and unexecuted by the filing pass.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md`; `.tmp/research/ash-family-evidence-inventory-2026-11-25.md`; [TASK-ash-native-families-index.md](../tickets/TASK-ash-native-families-index.md).
Resources: synthetic fixtures in unique `mktemp -d "$PWD/.tmp/ash-native-families.XXXXXX"` directories; opt-in GPU proofs use an exclusive single-device lease and user-supplied read-only checkpoints with recorded SHA-256 and license.

## Scope

Promote the audited family ordering into owned, bounded proofs. Build each shared
seam once; family tickets consume it and add only family-specific deltas. This is
a proposed implementation epic, not a claim that any new family builds, runs,
meets parity, or fits a device.

Current source is authoritative. Research documents are evidence, not shippable
specifications. **[UNKNOWN]** all new-family build, GPU parity, quality, latency,
VRAM, and real-checkpoint tolerances until their tickets produce evidence.

## S0 gate: existing correctness and promotion work

Neither track starts before the existing owners establish truthful selection and
correctness gates:

- [BUG-ash-sd15-numerics.md](../tickets/BUG-ash-sd15-numerics.md) — timestep
  layout and missing ResBlock SiLU CPU-reference regression.
- [TASK-ash-verification-gates.md](../tickets/TASK-ash-verification-gates.md) —
  reachable verification and provenance gates.
- [TASK-ash-backend-promote.md](../tickets/TASK-ash-backend-promote.md) —
  compile-reachable CLI/HTTP selection and truthful capability advertisement.
- [TASK-ash-unsupported-conditioning-guards.md](../tickets/TASK-ash-unsupported-conditioning-guards.md)
  — named refusal for unsupported direct-pipeline conditioning.

[FEAT-ash-sd2-spec.md](../tickets/FEAT-ash-sd2-spec.md) remains the current SD
execution priority. It is not a technical prerequisite for unrelated DiT/LLM
substrate work, but it precedes MOSS-Image family execution.
[TASK-ash-rrdb-upscaler-spike.md](../tickets/TASK-ash-rrdb-upscaler-spike.md)
is independent and gates neither track.

## Shared substrate owners

All owners are unassigned until planned. S1-S9 are dependencies, not permission
to skip S0 or family admission. One job uses one Vulkan physical device.

| ID | Owner ticket | Required proof | Stop condition |
|---|---|---|---|
| S1 | [TASK-ash-gguf-reader.md](../tickets/TASK-ash-gguf-reader.md) | Bounded generated GGUF metadata/tensor fixtures, malformed and oversized inputs, alias-aware payload access | Stop on unbounded reads, guessed offsets, duplicate parser ownership, or fixture mismatch. |
| S2 | [TASK-ash-gguf-dequant.md](../tickets/TASK-ash-gguf-dequant.md) | CPU reference per selected checkpoint dtype, shape/range failures, no silent fallback | Stop if a required dtype or real checkpoint layout lacks a reference. |
| S3 | [TASK-ash-tokenizer-framework.md](../tickets/TASK-ash-tokenizer-framework.md) | Shared interface and golden IDs from user assets, unknown-token/long-input cases | Stop if family templates or tokenizer assets are missing or unlicensed. |
| S4 | [TASK-ash-decoder-ops.md](../tickets/TASK-ash-decoder-ops.md) | Synthetic CPU-reference RMSNorm/RoPE/SwiGLU/QK-norm/short-convolution shapes and errors | Stop on shape divergence or unavailable required capabilities. |
| S5 | [TASK-ash-autoregressive-state.md](../tickets/TASK-ash-autoregressive-state.md) | KV/recurrent state, causal/sliding/packed masks, prefill/decode and cancellation equivalence | Stop on cache-boundary or cancellation mismatch. |
| S6 | [TASK-ash-sampling-contract.md](../tickets/TASK-ash-sampling-contract.md) | Deterministic greedy/stop/stream-cancel contract through the existing worker seam | Stop on partial output after cancel or unowned sampling semantics. |
| S7 | [TASK-ash-flow-dit-substrate.md](../tickets/TASK-ash-flow-dit-substrate.md) | Synthetic single-stream DiT block, timestep/position/flow contracts and identity controls | Stop on CPU/GPU divergence or silent scheduler assumptions. |
| S8 | [TASK-ash-vae-family-adapters.md](../tickets/TASK-ash-vae-family-adapters.md) | Explicit per-family key/spec maps and synthetic encode/decode checks | Stop rather than silently reuse SD1.5 scale or channels. |
| S9 | [TASK-ash-llm-evidence-integration.md](../tickets/TASK-ash-llm-evidence-integration.md) | Integrate bounded LLM evidence with existing admission, job, and provenance seams | Stop on missing component hashes/license decision or duplicate registry/policy ownership. |

S9 consumes the existing [TASK-model-provenance-contract.md](../tickets/TASK-model-provenance-contract.md)
and [TASK-model-license-registry.md](../tickets/TASK-model-license-registry.md).
It does not create another registry, cache, inspector, or license policy engine.

## Track A: SD shared flow-DiT to image families

1. **S7 + S8 substrate.** Prove one synthetic single-stream DiT block and one
   explicit VAE adapter before a large checkpoint.
2. **MOSS-Image.** [FEAT-ash-moss-image-family.md](../tickets/FEAT-ash-moss-image-family.md)
   consumes S7/S8 and the Qwen3.5 encoder proof. Start with a synthetic DiT block
   and Base/SFT spec, not full generation.
3. **Conditional Qwen-Image-2.1.** [TASK-ash-qwen-image21-attention-proof.md](../tickets/TASK-ash-qwen-image21-attention-proof.md)
   proves mixed-granularity attention and prefix KV reuse. Real weights remain
   blocked unless the Qwen Research License non-commercial path or a separate
   license is explicitly recorded and admitted.
4. **Krea 2 — deferred.** Reconsider only after parity-proven Ash image execution,
   MMDiT, Qwen3-VL conditioning, Turbo behavior, and revenue/content-filter/notice
   conditions are owned. No Krea 2 execution ticket is in this wave.

## Track B: LLM shared substrate to text families

1. **S1-S6 substrate.** Generated fixtures precede a small user-supplied GGUF.
2. **Smallest proofs.** [FEAT-ash-embeddinggemma-2.md](../tickets/FEAT-ash-embeddinggemma-2.md)
   and [FEAT-ash-lfm25-text-decoder.md](../tickets/FEAT-ash-lfm25-text-decoder.md)
   each remain bounded text-only proofs. Do not begin with large MoE or Omni
   audio execution.
3. **Qwen3.5 dense.** [FEAT-ash-qwen35-dense.md](../tickets/FEAT-ash-qwen35-dense.md)
   proves Gated DeltaNet plus gated attention; reusable encoder work feeds
   MOSS-Image.
4. **Gemma 4 / larger variants — deferred.** Reconsider after dense parity and
   owned PLE/hybrid local-global KV proofs; larger Qwen/MoE/MTP work needs a
   separate readiness and memory decision.
5. **Inkling-2-Mini — deferred.** A late conditional candidate only. Loop-core,
   relative bias, short convolution, MoE, memory, and upstream-format evidence
   must be resolved independently; no DSpark assumption or execution claim.

**MiniMax H3 / H3-Image — do not start native Ash.** Its packed attention, 3D
RoPE, PDD sampler, VAE/audio branches, memory planning, multi-GPU mismatch, and
territory/commercial terms require a separately funded architecture decision.

## Admission and evidence constraints

- Checkpoints, tokenizers, vocabularies, VAEs, and projectors are user-supplied;
  no downloads, conversion, redistribution, or bundled weights.
- Record exact artifact SHA-256, source revision when known, tokenizer/projector
  hashes, license document, and admission result before device work.
- Restricted, ambiguous, missing-license, territory, revenue, or authorization
  cases fail closed before checkpoint/device work. Research header metadata alone
  never proves availability or license compliance.
- CPU-reference and synthetic-shape tests precede opt-in real-checkpoint tests.
  GPU tests acquire an exclusive single-device lease and use a unique repo-local
  scratch directory; no multi-GPU tensor/sequence parallelism.
- Required capabilities fail explicitly; optional fast paths fall back to the
  portable path. No fake tensors/output, silent conditioning drops, or
  header-derived family availability.
- New shader expressions retain original or attributed `// Provenance:` lines;
  `bun run spdx` remains a release gate.
- [TASK-ash-family-readiness-markers.md](../tickets/TASK-ash-family-readiness-markers.md)
  owns `header=accepted`, `keys=...`, `execution=inspection-only`,
  `execution=blocked-spec-only`, and `execution=reduced-sd15-path` vocabulary.
  A runnable state may be added only when its ticket demonstrates execution.

## Stop conditions

Stop the epic’s family-execution wave if S0 regresses; a shared seam lacks an
independent reference; admission provenance is incomplete; a required capability
is unavailable; cache/mask/sampler/DiT outputs diverge; cancellation emits partial
output; or a family license condition is unresolved. Synthetic success does not
waive these gates. Each family records blockers and unknowns rather than widening
scope.

## Reconciliation and generated artifacts

[TASK-ash-native-families-index.md](../tickets/TASK-ash-native-families-index.md)
owns the roster, dependency graph, readiness reconciliation, and non-duplication
review. `.plan/epics-index.md`, `.plan/tickets/index.json`,
`.plan/feature-matrix.md`, `.plan/interconnect.md`, and `.plan/code-map.json` are
generated artifacts and are explicitly not hand-edited. Until
[TASK-plan-index-generator.md](../tickets/TASK-plan-index-generator.md) lands,
their staleness is not evidence for or against this epic.

## Non-goals

No family availability claims, full-family completion, quality/latency/VRAM
benchmarks, model downloads or bundling, new dependencies, multi-GPU support,
new HTTP surface, duplicate inspector/cache/registry/admission mechanism, or
execution of deferred Krea 2, MiniMax H3, Inkling, or larger variants in this wave.