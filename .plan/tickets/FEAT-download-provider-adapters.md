# FEAT: download provider adapters over the options contract
Status: proposed
Priority: P2
Epic: epic-external-tools
Owner: unassigned
Acceptance: HF, llama.cpp and ComfyUI-node source forms resolve through the provider-neutral contract with explicit option support, checksum/resume/cancel semantics preserved where the upstream allows it, and no client-controlled flag passthrough to a managed engine; all weights remain user-supplied and no CI downloads are added.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run test:core`
Links: `TASK-download-options-contract.md`, `.tmp/research/video-audio-reuse-and-download-options-2026-11-24.md`, `crates/server/src/downloads.rs`
Resources: per-test temp model dirs and fake HF/HTTP endpoints; no live Hub token, no shared cache

## Change

- Stage 1 HF: revision (full SHA), tree API filename discovery, `allow_patterns`/`ignore_patterns`, `local_dir` mapping and LFS sha256. Keep the current single-file path as the default.
- Stage 2 llama.cpp: map `-hf <repo>:<quant>`, `--hf-repo` + `--hf-file`, `-mu/--model-url` and `-dr/--docker-repo` into provider options; clients never pass raw flags to a managed server. Auth stays in the operator env.
- Stage 3 ComfyUI-node: support only pre-staged, declared source/weight roots and documented mirror env vars. Do not call ComfyUI-Manager, install node packs, or trigger a node runtime auto-fetch.
- Each adapter rejects options it cannot honor; no adapter silently downgrades checksum or destination confinement.

## Acceptance detail

- A fake HF endpoint proves revision-pinned resolution and sha256 mismatch failure; no real model bytes enter the repo.
- A fake llama.cpp source proves `--model-url` handling without forwarding a client string into a process argv.
- ComfyUI adapter is proven by a manifest/fixture path only; no live custom-node install occurs.
