# TASK: provider-neutral download options contract
Status: proposed
Priority: P1
Epic: epic-external-tools
Owner: unassigned
Acceptance: `DownloadRequest` gains a provider-neutral option vocabulary with explicit rejection of unsupported combinations, while the existing `repo`+`file` and direct `url` forms keep their current behavior byte-for-byte; no secret value enters a DTO, status response, log or metric.
Verification: `bun run test:server`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/video-audio-reuse-and-download-options-2026-11-24.md`, `crates/server/src/downloads.rs`, `epic-external-tools.md`
Resources: per-test temp model dirs, fake HTTP servers and env overrides for token names; no real Hub traffic in CI

## Change

- Extend the request contract with: `provider` (`hf` | `direct` | `comfyui_node` | `llama_server`), `revision` (full SHA when a commit), `path` plus `allow_patterns`/`ignore_patterns`, `destination_root`, `token_env` (NAME only), `expected_sha256`, `dry_run`, and an explicit `resume` preference. Keep `.part` + meta, ETag resume, cancel and single-flight unchanged.
- Default `provider` to `hf` for `repo`+`file` and `direct` for `url`; reject unknown providers and options a provider cannot honor before enqueueing.
- Token values are read only by the adapter from the named env var at fetch time, mirroring the bearer-key pattern; never echo them in `DownloadState`.
- `dry_run` reports the resolved file list/sizes without writing files and does not create a download id that can be cancelled.

## Acceptance detail

- Existing HF LFS sha256 verification and 200/206 resume tests still pass with no new field supplied.
- Tests cover unsupported option combinations, token-env indirection, and a dry-run that leaves no `.part` behind.
- No provider adapter, network fetch or CI model download is part of this ticket.
