# TASK: Vulkan GGUF dequant kernel against the host oracle

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: a host-side pure-Rust Q4_K dequant reference is checked byte-exact against llama.cpp outputs on a fixture tensor, and a Vulkan dequant kernel compares against the host oracle, never against the other backend directly.
Verification: `TMPDIR="$PWD/.tmp" cargo test --locked -p honkbox-ash-backend --features ash-compute gguf_dequant`
Links: `.tmp/research/vulkan-colibri-hyperstream-2026-11-25.md` §7.5, `.tmp/research/cuda-capyflash-infr-2026-11-25.md` §9.2, `TASK-ash-gguf-dequant.md`, `crates/ash-backend/src/checkpoint.rs`
Resources: fixture tensor, CPU-only tests; no GPU required for the oracle step, no checkpoint download

## Attribution

Oracle-ladder validation (oracle on CPU, kernel vs oracle, pipeline vs golden) synthesized from `izembaki-hash/HyperStream` dequant-throughput claims (Apache-2.0, author-reported 51.1 GB/s, unverified) and `kryptic-sh/infr` pager practice (MIT). Q4_K math provenance follows `TASK-ash-gguf-dequant.md` (pinned MIT ggml, notice retained). Method: white-room. Host reference and kernel are honkbox-original; fixture expected values are never produced by the production decoder.

## Change

- Host-side dequant reference (pure Rust, no Vulkan) for Q4_K first, checked byte-exact against llama.cpp dequant outputs on a fixture tensor.
- Vulkan dequant kernel later compares against the host oracle, never against the other backend directly. Host-dequant stays the honest fallback.
- Q4_K family first (most common in the wild); other dtypes fail by name per the existing ticket.

## Acceptance criteria

- [ ] Host oracle matches llama.cpp outputs byte-exact on the fixture tensor.
- [ ] Kernel output matches the host oracle; no kernel-vs-foreign-backend comparison anywhere.
- [ ] `TASK-ash-gguf-dequant.md` non-goals (GPU kernels) are satisfied here without reopening that ticket.
- [ ] No checkpoint download, GPU lease, or new dependency for the oracle tests.
