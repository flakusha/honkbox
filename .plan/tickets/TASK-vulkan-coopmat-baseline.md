# TASK: Capability-gated RDNA3/RDNA4 Vulkan cooperative-matrix baseline

Status: proposed
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: a test-only, capability-gated Ash quantized-matmul baseline runs on RDNA3/RDNA4 with the current scalar f32 path as fallback; full Vulkan op coverage includes `MUL_MAT_ID`/expert routing; numerical, same-window timing, and peak-memory evidence is recorded; no public backend flag or default change ships before review.
Verification: `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute coopmat -- --test-threads=1`; `HONKBOX_ASH_BENCH_DEVICE_INDEX=0 TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.tmp/coopmat-shader-cache-$$" cargo test --locked -p honkbox-ash-backend --features ash-compute --test vulkan_coopmat_baseline -- --ignored --nocapture`; `TMPDIR="$PWD/.tmp" GGML_VK_VISIBLE_DEVICES=0 "$PWD/.tmp/llama-b11160-build/bin/test-backend-ops" test -b Vulkan0`; `bun run lint:rs`
Links: `.tmp/research/llm-runtime-memory-2026-11-25.md`; `.tmp/research/vulkan-inference-catalogue-llm-2026-11-24.md`; `.tmp/research/vulkan-inference-components-2026-11-24.md`; `.plan/tickets/TASK-ash-backend-promote.md`; `crates/ash-backend/src/device.rs:47-86`; `crates/ash-backend/shaders/matmul.comp`; `crates/ash-backend/src/llm.rs:29-84`; `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:2481-2495,2554-2568`; <https://github.com/ggml-org/llama.cpp/releases/tag/b11160>; <https://github.com/ggml-org/llama.cpp/discussions/31958>
Resources: one isolated process per selected GPU; private test-only device selector; separate gitignored `.tmp/` build/shader-cache dirs; `--test-threads=1`; no concurrent honkbox server or GPU job; record GPU/driver/Mesa/RADV/CPU/power state and exact source pins.

## Verified premise

- [FACT] Upstream b11160 (`70c4e15`, 2026-09-24) reports RDNA3/RDNA4 int8
  cooperative-matrix quantized matmul. Discussion #31958 reports silent
  small-batch/routing failures unless the full `test-backend-ops` suite runs.
- [FACT] The vendored ggml Vulkan source already probes signed-int8 subgroup
  cooperative-matrix properties and creates quantized `MUL_MAT`/`MUL_MAT_ID`
  pipelines for RDNA3/RDNA4. Architecture/type coverage differs (for example,
  non-ID Q4_K is excluded on RDNA4 while ID Q4_K is created).
- [UNKNOWN] The current crate tarball is not proven to be b11160, and no
  honkbox run identifies which Vulkan pipeline executed. Pin and log the
  actual source/build before measuring.
- [FACT] Ash currently exposes storage/int8/subgroup capability bits but no
  cooperative-matrix feature/shape query (`device.rs:47-86`). Its only matmul
  shader is scalar f32, one thread per output element.
- [FACT] `AshLlmPipeline` remains an unimplemented placeholder. This ticket
  measures a kernel baseline; it does not claim an Ash LLM backend.
- [FACT] Ash SD is wired into CLI `generate` (`main.rs:976-1023`). HTTP `serve`
  still uses `make_sd` with only Stub/SdCpp (`main.rs:677`;
  `backends.rs:101-128`). HTTP Ash selection stays outside this ticket.

## Change

### 1. Capability probe and deterministic selection

- Extend the existing device capability seam with `VK_KHR_cooperative_matrix`
  availability, enabled feature state, supported shader stages, component
  types, scope, and M/N/K dimensions. Architecture name alone must not select
  the path.
- Add a test-only `coopmat` cargo feature and private benchmark selector
  (`HONKBOX_ASH_BENCH_DEVICE_INDEX`). Neither is a public runtime/CLI flag.
- Log the selected path and the capability tuple. Missing extension, feature,
  shape, shader support, or unsupported quant type selects the existing scalar
  f32 path (or an explicit unsupported error for quant inputs); never silently
  dispatch the wrong shader.
- Unit-test the selector with synthetic capability records, including RDNA3,
  RDNA4, missing feature, wrong component types, wrong scope, and wrong shape.
  These tests run without a GPU.

### 2. Quantized kernel and routing coverage

- Implement the minimum Ash cooperative-matrix quantized matmul needed to
  compare against the scalar fallback. Keep the scalar shader as the reference
  and fallback; do not replace it.
- Cover every quant type advertised by the selected path and both ordinary
  `MUL_MAT` and routed `MUL_MAT_ID`. Routed tests include repeated, out-of-range
  rejected, and boundary expert ids; output placement must match the reference.
- Include tail/non-multiple shapes and non-contiguous/transposed operands where
  the API supports them. Do not infer full coverage from one passing Q4_K case.
- Run the complete Vulkan backend suite from a pinned b11160 source build, not
  only a `MUL_MAT` filter. A filtered pass cannot close this ticket.

### 3. Numerical acceptance before timing

- Use deterministic bounded synthetic matrices and a CPU reference. Require
  finite outputs, exact routing/index behavior, relative L2 error `<= 1e-4`,
  and max absolute error `<= 1e-3 * max(1, max_abs_reference)` against the
  current non-coopmat reference. Fix these tolerances before collecting timing.
- Check quant-block dequantization separately and cover saturated int8/int32
  boundaries so overflow cannot hide behind a loose end-to-end tolerance.
- Add one small real-checkpoint logits comparison when a local checkpoint is
  available; it is opt-in and uses relative L2 `<= 1e-3`. No checkpoint means
  the kernel evidence remains valid but end-to-end parity is explicitly
  unmeasured.

### 4. Same-window measurement

- Add `crates/ash-backend/tests/vulkan_coopmat_baseline.rs` as an ignored,
  opt-in hardware test. It emits a machine-readable evidence record under the
  gitignored scratch directory and a concise report for review.
- Measure decode-sized and prefill-sized matrices separately. Minimum shape
  grid: `n={1,2,3,4,8,16,32,64,128,512}`,
  `m={1,17,256,4096}`, `k={256,14336}`, filtered only when a shape exceeds a
  recorded device limit.
- Create pipelines, descriptors, and buffers outside timing. Warm up each
  path, then alternate scalar/coopmat order within the same window. Record at
  least 15 samples per case, median, min/max or interquartile spread, sample
  count, and raw samples. Cross-window comparisons are not acceptance evidence.
- Record peak device allocation bytes for each path, plus artifact bytes and
  workspace/scratch bytes separately. Do not infer memory from file size.
- Record GPU name/PCI ids, architecture, driver, Mesa/RADV version, Vulkan API,
  CPU, power/clock policy, source commit/build id, compiler flags, and shader
  cache state.

## Acceptance detail

- The capability selector chooses coopmat only for a fully supported tuple and
  chooses scalar fallback otherwise; logs make the decision auditable.
- Full `test-backend-ops test -b Vulkan0` passes on the pinned upstream build,
  including `MUL_MAT_ID`. Local Ash tests pass for all advertised quant/type and
  routing combinations plus tail shapes.
- Numerical tolerances above pass before any performance number is reported.
- The evidence table contains same-window scalar/coopmat median and spread for
  decode and prefill shapes, plus peak bytes for both paths.
- Promotion requires a measured improvement larger than the observed
  same-window run-to-run spread on at least one target workload and no target
  workload regression beyond that spread. Otherwise the result is "baseline
  only; no promotion".
- No public flag, default flip, capability advertisement, or throughput claim
  lands in this ticket.

## Upstream full-suite preparation and verification

Run later in a gitignored scratch tree; do not run during planning:

```sh
mkdir -p "$PWD/.tmp/llama-b11160-src" "$PWD/.tmp/llama-b11160-build"
curl -fsSL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11160.tar.gz \
  | tar -xz -C "$PWD/.tmp/llama-b11160-src" --strip-components=1
cmake -S "$PWD/.tmp/llama-b11160-src" \
  -B "$PWD/.tmp/llama-b11160-build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON
cmake --build "$PWD/.tmp/llama-b11160-build" \
  --target test-backend-ops -j"$(nproc)"
TMPDIR="$PWD/.tmp" GGML_VK_VISIBLE_DEVICES=0 \
  "$PWD/.tmp/llama-b11160-build/bin/test-backend-ops" test -b Vulkan0
```

Repeat the hardware run once per RDNA3/RDNA4 device with no concurrent GPU
work. A machine without either architecture records scalar fallback and skips
the coopmat evidence; it does not fabricate a result.

## Non-goals

- HTTP Ash backend selection or an Ash LLM implementation.
- New public CLI/config flags or a default backend change.
- Cross-backend byte equality, model-weight redistribution, or shader-source
  copying from another project without provenance review.
