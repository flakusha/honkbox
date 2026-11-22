# BUG: job cancel cannot interrupt a stalled attached LLM stream

Status: done (2026-11-22)
- Verified: `cargo test --locked -p honkbox-server --bin honkbox cancel` (10 passed) and full `bun run check && bun run test` green (`.tmp/gates-attached-job-cancel.log`).
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: cancelling a live attached-LLM job through the job-registry cancel path interrupts the upstream generation immediately, not only at the next token boundary; the upstream socket actually closes, the GPU lease is released, and the request maps to the existing `408 cancelled` outcome. A stalled upstream (no tokens for the whole wait) must be cancellable within the cancel path's own bound rather than the 300 s upstream timeout.
Verification: `cargo test --locked -p honkbox-server`; extend `cancel_interrupts_stalled_stream` (`crates/server/src/llm_attached.rs:834-867`) so the fake upstream asserts its client socket closed after cancellation, not just that the local future returned quickly.
Links: `db95fb3` (attached cancellation commit); `crates/server/src/routes/chat.rs:283`; `crates/server/src/routes/openai_stream.rs:93`; `crates/server/src/llm_attached.rs:25,269-277,308-321,363-387,786,834-867`; `crates/server/src/routes/openai_constraints.rs:57-61`; `.tmp/intercom/FIXED-llama-server-attach.md`.
Resources: in-process fake upstream and stub LLM pipeline; no GPU, no model download.

## Evidence

Review of `db95fb3` confirmed client-disconnect cancellation is fixed: `AttachedLlm` no longer holds a shared counter or overrides `abort` (`llm_attached.rs:30-37`), `LlmPipeline::abort` is a default no-op (`crates/llm-worker/src/lib.rs:440`), and a per-request runtime plus `select!` on a cancel oneshot drops the SSE future on disconnect (`llm_attached.rs:269-277,308-321,363-387`). `Cancelled` maps to `408 cancelled` (`openai_constraints.rs:57-61`).

Two gaps remain:

- Job cancellation sets `RequestCancel` only when a token reaches the sink (`chat.rs:283`, `openai_stream.rs:93`). A stalled upstream produces no token, so the cancel flag is never set and generation plus lease persist until the 300 s bound (`llm_attached.rs:25`). Wire the job-registry cancel path to the same per-request cancel handle.
- `cancel_interrupts_stalled_stream` (`llm_attached.rs:834-867`) asserts only client-side latency (<1 s); the local flag check at `:283` satisfies that while the fake server keeps sleeping 2 s (`:786`). Assert the fake server observes the client socket closing.

## Fix

- `routes/chat.rs::watch_job_cancel` awaits the job registry's `CancellationToken` on the server runtime and sets the stream's `RequestCancel` immediately; both chat and OpenAI SSE bodies carry the watcher and abort it on drop (`CancelOnDrop`). The existing `cancel_signal` poll then closes the upstream socket at the next poll, without waiting for a token boundary.
- `stalling_server` now reports whether the fake upstream observed the client socket close; `cancel_interrupts_stalled_stream` asserts that flag, and a focused `job_cancel_watcher` unit test proves the bridge reaches only its own request.

## Non-goals

Non-streaming `complete` has no cancel seam (`crates/llm-worker/src/lib.rs:416`; `chat.rs:100`, `openai_chat.rs:251`) — pre-existing and out of scope. The one-runtime-per-streamed-request shape (`llm_attached.rs:269-273`) stays as-is; revisit only if thread cost becomes measurable.