# TASK: Add a strict shared GGUF metadata and tensor reader

Status: proposed
Priority: P1
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: generated GGUF fixtures prove bounded v2/v3 metadata and tensor parsing, overflow-safe aligned ranges, typed architecture metadata, and on-demand alias-aware reads without family-specific behavior.
Verification: `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_reader -- --nocapture`, `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute checkpoint::tests -- --nocapture`, `bun scripts/check-ash-provenance.ts`.
Links: `.tmp/research/ash-family-implementation-index-2026-11-25.md` §4 S1 and §5; `.tmp/research/ash-family-evidence-inventory-2026-11-25.md` §2–4; `.tmp/research/ash-lfm-implementation-candidate-2026-11-25.md` §4–6; `.tmp/research/ash-qwen35plus-implementation-candidate-2026-11-25.md` §4–5; `crates/ash-backend/src/checkpoint.rs`; `crates/registry/src/metadata.rs`; `vendor/llama-cpp-sys-4-0.11.0/llama.cpp/ggml/include/gguf.h`; https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
Resources: CPU-only generated fixtures; each test owns a unique directory under `TMPDIR` with PID, atomic sequence and timestamp, removed by RAII teardown; no ports, databases, devices, model files or shared globals.

## Verified 2026-11-25

- **[FACT]** `CheckpointMap` is a strict safetensors-only header-to-tensor map.
  It validates names, shapes, dtypes and ranges, then opens the file and reads
  only a requested payload (`crates/ash-backend/src/checkpoint.rs:75-258`).
- **[FACT]** The registry GGUF path is a deliberately lenient summary probe. It
  reads a few metadata fields and skips values; it has no tensor directory,
  payload ranges or payload access (`crates/registry/src/metadata.rs:117-220`).
  It is not evidence of inference-ready GGUF support.
- **[FACT]** Pinned GGUF v3 is little-endian by default, uses `u64` string
  lengths, places tensor offsets relative to the aligned tensor-data section,
  and defaults `general.alignment` to 32
  (`vendor/llama-cpp-sys-4-0.11.0/llama.cpp/ggml/include/gguf.h:1-46`).
- **[UNKNOWN]** The largest metadata/tokenizer payload that future shared
  consumers will accept, and the complete dtype inventory of any selected
  checkpoint. No real checkpoint was opened for this ticket; the initial caps
  and layout set below are fail-closed implementation boundaries, not model
  availability or readiness claims.

## Dependencies

- `TASK-ash-native-families-index.md` owns shared-seam ordering and the rule
  that family tickets consume S1 rather than filing another GGUF parser.
- Reuse the existing `checkpoint.rs` contract: tensor metadata, exact-name
  lookup, ordered alias fallback, `io::ErrorKind::InvalidData`, checked ranges
  and on-demand payload reads. Do not introduce a second strict-reader API.
- `TASK-ash-gguf-dequant.md` consumes validated descriptors and bytes. S1 does
  not depend on dequantization, tokenizer, decoder, sampling or any family.
- The lenient registry probe remains summary-only in this ticket. Any later
  consolidation must preserve one strict parser without adding a dependency or
  making catalog probing allocate tensor payloads.

## Change

- Implement one strict reader in the existing Ash checkpoint module; a
  `checkpoint/gguf.rs` child is acceptable if it keeps `checkpoint.rs` small.
  Expose GGUF through the same tensor metadata, lookup, read and error
  conventions as `CheckpointMap`, with GGUF metadata alongside that map.
- Accept GGUF versions 2 and 3 as little-endian. Parse the prologue, all KV
  value types 0–12, nested arrays, strings and the complete tensor-info table.
  Reject bad magic/version/types, invalid UTF-8, invalid metadata keys,
  duplicate keys/tensor names, invalid booleans and unsupported dimensions.
- Bound work before allocation: at most 16 MiB of prologue/KV/tensor-info
  bytes, 4096 KVs, 64 KiB per string, four array nesting levels, and array byte
  sizes that fit both the remaining metadata budget and `usize`. Counts must
  fit the byte budget before entering a loop; `u64::MAX` declarations must not
  cause allocation, iteration or seek overflow. Exceeding a cap is a named
  invalid-data error, never a partial map.
- Parse `general.architecture` as the required lowercase ASCII architecture
  identifier and `general.alignment` as an optional nonzero `uint32` multiple
  of 8. Preserve unknown metadata with its declared type. Provide typed generic
  access to common `<arch>.context_length`, embedding/block/FFN/head fields
  where present; accept the specification-supported integer widths and reject
  wrong types without coercion. Family schemas and defaults stay outside S1.
- Compute the tensor-data start with checked alignment after tensor info. Parse
  dimensions in GGUF `ne` order without reversing them; reject zero dimensions,
  excessive rank, element/byte overflow and unknown dtype IDs. A storage-layout
  table, separate from numeric decoding, initially covers F32 (4 bytes/value),
  F16/BF16 (2) and Q4_K (256 values/144-byte block). Other dtype IDs fail by
  name; S2 may extend this same table when adding decode support.
- Validate every tensor offset is aligned, every checked range is inside the
  observed file length, and no tensor ranges overlap. Alignment gaps are valid;
  gaps must not be mistaken for tensors. Construction reads metadata only and
  never decodes or copies payloads.
- Add exact-name payload reads plus an alias-aware read where the requested
  exact name wins and otherwise caller-supplied aliases are tried in order.
  The shared reader contains no family alias map. Each read seeks to the
  validated range, checks the observed file length again and reads exactly the
  tensor byte length.
- Add tiny deterministic fixture builders and tests for valid v2/v3 files,
  default/custom alignment, every scalar KV type, nested/string arrays,
  metadata-only files, multiple tensors, exact/alias precedence and requested
  payload bytes. Add malformed, truncated-at-each-section and oversized cases
  for counts, strings, arrays, dimensions, offsets, block alignment, overlap
  and arithmetic overflow.

## Acceptance criteria

- [ ] A valid generated GGUF map exposes architecture metadata, tensor names,
  dtypes, `ne` shapes, relative offsets and exact byte lengths; no payload is
  decoded during construction.
- [ ] Default alignment 32 and a custom valid alignment produce the same
  checked absolute ranges; zero, non-multiple-of-8 and overflowing alignment
  values fail by name.
- [ ] Unsupported versions/types/dtypes, duplicate names, invalid UTF-8/keys,
  invalid booleans, zero/excessive dimensions and malformed arrays fail
  closed with `InvalidData` and useful context.
- [ ] Oversized declared counts, strings, arrays and shapes fail before large
  allocation or unbounded iteration, including `u64::MAX` and checked-multiply
  overflow fixtures.
- [ ] Unaligned, overlapping, header-pointing and past-EOF tensor ranges fail;
  legal alignment padding and metadata-only files remain valid.
- [ ] Exact lookup wins over aliases, ordered aliases resolve deterministically,
  missing tensors fail by name, and reads return only the requested bytes.
- [ ] Existing safetensors map/read/alias tests still pass unchanged.
- [ ] Results describe only strict container parsing; they do not claim a
  family tensor map, dequantization, tokenizer, model execution or readiness.

## Verification

```sh
TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_reader -- --nocapture
TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute checkpoint::tests -- --nocapture
bun scripts/check-ash-provenance.ts
```

These are proposed commands, not run by the filing pass. Tests are CPU-only and
parallel-safe; no Vulkan device or real checkpoint is required.

## Non-goals

Family-specific tensor maps or metadata defaults; tokenizer/chat-template
interpretation; sampling, decoder ops or model graphs; numeric dequantization;
checkpoint download, conversion or bundling; a second parser, cache, residency
system, registry or availability mechanism; HTTP/CLI selection; multi-GPU;
performance, VRAM, quality, compatibility or readiness claims.