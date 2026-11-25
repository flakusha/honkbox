# TASK: Integrate measured Ash LLM evidence into the full AI cycle

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: one versioned evidence path records actual Ash LLM logits, selected/count tokens, cancellation and run-scoped peak allocation bytes; the same run is admitted with a nonzero placement-aware `residency::Demand` before allocation, and its record is referenced by full-cycle native-runtime acceptance without adding a cache, model family or public response field.
Verification: `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-llm-evidence.XXXXXX")" cargo test --locked -p honkbox-llm-worker --lib llm_evidence`; `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-llm-evidence.XXXXXX")" cargo test --locked -p honkbox-ash-backend --features ash-llm llm_evidence`; `TMPDIR="$(mktemp -d "$PWD/.tmp/ash-llm-evidence.XXXXXX")" cargo test --locked -p honkbox-server --test ash_llm_evidence`; the isolated ignored command below; `bun run codegen:check`; `bun run lint:rs`.
Links: `.tmp/research/llm-runtime-memory-2026-11-25.md`; `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S5/S6/S9; `.plan/epics/epic-full-ai-cycle.md`; `crates/llm-worker/src/lib.rs:185-203,287-327,381-458`; `crates/ash-backend/src/llm.rs`; `crates/ash-backend/src/compute.rs:82-131`; `crates/residency/src/lib.rs:126-159,603-685`; `crates/server/src/routes/chat.rs:72-113,240-335`; `crates/server/src/main.rs:1135-1157`.
Resources: deterministic synthetic logits/tokens; unique gitignored `.tmp/ash-llm-evidence.*` output directories; test-local `Ledger`, never `global()`; one private test-only device selector and exclusive single-device lease for ignored hardware tests; read-only user artifact/reference with recorded SHA-256/license; no fixed port, shared cache, download or multi-GPU.

## Verified 2026-11-25

- **[FACT]** `LlmPipeline` exposes complete/stream/cancel methods but no execution-evidence callback. `ChatResponse` carries text, token counts and milliseconds, not logits, selected IDs, cancellation observation or peak bytes (`llm-worker/src/lib.rs:185-203,413-458`).
- **[FACT]** `AshLlmPipeline` is an honest placeholder: every completion returns `Load(NotYet)` and no sink token (`ash-backend/src/llm.rs:29-79`). No Ash LLM execution or measurement exists.
- **[FACT]** `Demand` contains GPU-slot, VRAM, RAM and thread estimates; defaults reserve one slot and zero bytes. `Ledger::try_acquire(kind)` uses per-kind configured demand (`residency/src/lib.rs:126-159,668-685`).
- **[FACT]** Native chat resolves/loads the pipeline before taking its slot lease; CLI chat constructs the pipeline before `try_acquire(Llm)` (`routes/chat.rs:72-113`; `main.rs:1135-1157`). Neither path currently proves byte-bearing pre-allocation admission.
- **[FACT]** The worker KV formula charges every layer as `n_layer * n_embd * 2 * element_bytes`; it excludes weights, projector, logits, decode/vision scratch, recurrent state and allocator headroom (`llm-worker/src/lib.rs:287-327`).
- **[FACT]** `GpuBuffer` allocates/frees Vulkan memory but has no run-scoped allocation counter or peak-byte readout (`compute.rs:82-131`). No repository-wide Ash peak-byte metric was found.
- **[UNKNOWN]** Actual Ash logits/tokens, cancel latency, peak bytes, real-checkpoint parity and any throughput result. Source presence and third-party figures are not honkbox measurements.

## Dependencies

- `TASK-ash-sampling-contract.md` owns deterministic Ash logits-to-token selection, stop policy, streaming and cancellation. This ticket observes that path; it does not implement another sampler.
- `TASK-ash-autoregressive-state.md` owns request-scoped KV/recurrent/convolution state and its byte geometry. Evidence accounts for the resulting allocations without a second state cache.
- `TASK-inference-cache-correctness.md` owns verified artifact identity, the shared weight/working-memory breakdown and demand-bearing lease plumbing. Consume that planner and ledger API; do not duplicate it here.
- `TASK-ash-native-families-index.md` owns S9 ordering. Any real-checkpoint record comes from an already selected text-family owner; this ticket creates no family and does not require a new one.
- `TASK-vulkan-coopmat-baseline.md` retains full op-coverage and same-window kernel-comparison ownership. `TASK-ash-backend-promote.md` retains public selection/capability ownership.

## Change

### 1. One internal, versioned evidence contract

- Add a small `LlmEvidence` record in `honkbox-llm-worker` and one additive observed-execution method on `LlmPipeline`. The default method delegates to existing completion/cancellation and reports measurement `unavailable`; it must not fabricate zero logits, zero peak bytes or successful cancellation. Ash overrides it using the same run core as normal completion.
- Record schema version, backend/family/architecture, ordered artifact digests, engine/code/source pins, device/driver/API, context/seed/options, terminal outcome, prompt/generated/selected token counts and bounded selected token IDs.
- For each sampled step record logits shape/dtype, all-finite status, canonical raw-logits SHA-256, top-1 ID/value and the selected ID. Keep raw logits out of logs and public DTOs; the digest and reference vectors provide the comparison boundary.
- Record prefill/decode/total durations as observations, not speed claims. Record cancellation-observed, cancellation point and emitted token count; cancellation returns `LlmError::Cancelled`, never a partial `Ok` reply.
- Keep the record internal/test-artifact only. Do not add HTTP/CLI response fields, a response cache, an evidence database or a second persistence convention.

### 2. Measure peak bytes at the allocation boundary

- Add run-scoped accounting around Ash Vulkan allocation/free, covering every buffer used by the observed prefill/decode/state/logits path. Track live allocated bytes and high-water bytes separately for device-local and host-visible memory; include allocation size, not tensor payload length alone.
- Attribute counters to one run without global mutable state, shared cross-test counters or retained history. Drop/failure/cancel paths must unwind counts and release allocations.
- State metric scope explicitly: measured Ash allocations exclude driver reservations, allocator overhead and unrelated process memory unless the platform supplies them. Label values `measured`, `derived` or `estimated`; never infer peak bytes from checkpoint size.
- Unit-test allocate/reuse/free, nested buffers, allocation failure and cancellation. The high-water value must survive temporary frees and must not leak into a sibling run.

### 3. Make the observed run byte-bearing before allocation

- Resolve identity and obtain a bounded header/geometry plan first. Use the shared planner from `TASK-inference-cache-correctness.md` to account for weights, projector, KV, recurrent/convolution state, logits, decode scratch and named headroom exactly once in VRAM or RAM according to placement.
- Acquire the resulting per-job `Demand` before model/device allocation. Migrate native chat and CLI chat ordering rather than adding an Ash-only lease. Preserve bounded queue, hard-capacity and exact-release behavior; zero-byte Ash demand is a test failure.
- Record planned `Demand` and measured peak separately. Prior same-geometry evidence may justify conservative planner headroom, but a run must not mutate global demand from its own post-hoc peak or label an admission estimate measured.
- A configured budget that cannot fit weights plus working bytes rejects before loader/device counters change. Success, named load/decode failure, cancellation and SSE disconnect release exactly the committed bytes/slot.

### 4. Full-cycle acceptance harness

- Add CPU contract tests plus `crates/ash-backend/tests/ash_llm_evidence.rs` as an ignored hardware test. It runs the real Ash observed path with tiny generated tensors; an optional user checkpoint uses the same test only when digest, license and reference manifest are supplied.
- Add `crates/server/tests/ash_llm_evidence.rs` with in-process routes, test-local `Ledger`, deterministic recording pipelines and fake paths. It proves observer propagation, nonzero Demand, pre-load rejection, cancel/no-partial-success, sibling isolation and release without a GPU or global residency state.
- Emit a bounded machine-readable record plus concise summary under the test-owned `.tmp/ash-llm-evidence.*` directory. A synthetic record is labeled synthetic; a real record includes every artifact digest/license and reference-generator command/pin. Missing hardware or user weights means `unmeasured`, not pass.
- Reference the accepted record from `epic-full-ai-cycle.md` build-order item 3. Acceptance requires actual `ash-llm` logits/tokens, observed cancellation, nonzero admitted bytes and measured peak bytes. Public fast-path claims still require the separate coopmat/op-coverage evidence; this ticket makes none.

## Acceptance criteria

- [ ] The evidence schema is versioned and canonical; unavailable measurements remain absent/explicit and cannot be mistaken for zero.
- [ ] Ash observed execution records finite/logits-digest/top-1/selected-ID evidence for every bounded step and exact prompt/generated counts.
- [ ] Cancel before prefill, between decode steps and at stream teardown returns `Cancelled`, emits no successful partial response and does not cancel a sibling request.
- [ ] Peak bytes are run-scoped high-water allocation measurements with stated scope; temporary frees, failures and sibling runs cannot corrupt them.
- [ ] The run acquires nonzero placement-aware `Demand` before allocation; over-budget rejection happens before load/device work and every terminal path releases exactly its commitment.
- [ ] CPU tests pass with no Vulkan or real weights; ignored synthetic hardware evidence runs alone under one exclusive device lease.
- [ ] Any optional real-checkpoint evidence verifies recorded SHA-256/license and pinned-reference inputs; its absence leaves real parity explicitly unmeasured.
- [ ] Full-cycle acceptance references the resulting record; no second cache, family, sampler, ledger, registry, public DTO or availability/performance claim is added.

## Verification

```sh
mkdir -p .tmp
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-llm-evidence.XXXXXX")" \
  cargo test --locked -p honkbox-llm-worker --lib llm_evidence
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-llm-evidence.XXXXXX")" \
  cargo test --locked -p honkbox-ash-backend --features ash-llm llm_evidence
TMPDIR="$(mktemp -d "$PWD/.tmp/ash-llm-evidence.XXXXXX")" \
  cargo test --locked -p honkbox-server --test ash_llm_evidence
bun run codegen:check
bun run lint:rs

# Opt-in only; run alone on an idle GPU host, never in CI.
evidence="$(mktemp -d "$PWD/.tmp/ash-llm-hardware.XXXXXX")"
HONKBOX_ASH_LLM_DEVICE_INDEX=0 \
HONKBOX_ASH_LLM_DEVICE_LEASE="$evidence/device.lease" \
HONKBOX_ASH_LLM_EVIDENCE_OUT="$evidence/evidence.json" \
TMPDIR="$evidence" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --test ash_llm_evidence \
  -- --ignored --nocapture --test-threads=1

# Optional real-checkpoint mode of the same ignored test.
HONKBOX_ASH_LLM_MODEL=/read-only/text-model.gguf \
HONKBOX_ASH_LLM_MODEL_SHA256=<recorded-sha256> \
HONKBOX_ASH_LLM_LICENSE=/read-only/model-license.txt \
HONKBOX_ASH_LLM_REFERENCE=/read-only/reference.json \
HONKBOX_ASH_LLM_DEVICE_INDEX=0 \
HONKBOX_ASH_LLM_DEVICE_LEASE="$evidence/real-device.lease" \
HONKBOX_ASH_LLM_EVIDENCE_OUT="$evidence/real-evidence.json" \
TMPDIR="$evidence" cargo test --locked -p honkbox-ash-backend \
  --features ash-llm --test ash_llm_evidence \
  -- --ignored --nocapture --test-threads=1
```

Normal tests run in parallel: each owns unique temp paths and a test-local `Ledger`, with deterministic cleanup and no shared globals or ordering dependence. Ignored hardware tests alone serialize on the exclusive device lease; the test must acquire it before device creation and verify required real-artifact fields before payload/device access. Archive evidence before removing `$evidence`. No command here was run during planning.

## Non-goals

A new model family, sampler, response/prefix cache, identity database, residency ledger, registry or public evidence API; real weights in CI; automatic model download/conversion; bundled weights; multi-GPU; CPU-expert offload; cross-backend byte equality; kernel op-coverage ownership; public Ash selection/default/capability changes; quality, token/s, latency, VRAM-saving or availability claims.
