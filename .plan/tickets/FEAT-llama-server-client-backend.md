# FEAT: Consume an attached llama-server through the LLM seam

Status: done (2026-11-22; isolated adapter tests + full server tests + workspace clippy green)
Priority: P1
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: when an attached `llama-server` has proved `openai_compat`, an opt-in backend serves the existing chat contract (`/v1/chat`, `/v1/chat/completions`, SSE streams, `abort`) through `/v1/chat/completions` on that server, post-honkbox-bearer and loopback-only. Capabilities advertise chat through this backend only when it is selected and healthy.
Verification: isolated adapter harness → 6/6 pass (`.tmp/llama-harness/test2.log`);
`cargo test --locked --offline -p honkbox-server` → all suites pass and
`cargo clippy --locked --offline --workspace --all-targets -- -D warnings` clean
(`.tmp/integration-review-server-test.log`, `.tmp/integration-review-clippy.log`).
Links: `.tmp/research/cli-app-management-llama-server-2026-11-22.md`, `TASK-llama-server-attach.md`
Resources: no new Rust dependency without a dated `deny.toml` review.

## Implementation note (2026-11-22)

Transport: reqwest blocking client, constructed by `AttachedLlm::new`.
reqwest 0.13 spawns its own current-thread runtime for blocking clients
(`blocking/client.rs::ClientHandle::new`), so construction inside axum's
runtime is safe and every `LlmPipeline` call stays sync. The upstream stream
is read with `BufReader::read_line` under a 1 MiB body cap; `abort` bumps a
generation atomic and the reader drops, closing the upstream connection.

Selection: `--llm-backend attached` (opt-in; default remains `stub`) with the
endpoint from `[tools] llama_server_url` and an optional `LLAMA_API_KEY` env.
The constructor proves `/health` 200 + `/v1/models.data` before returning, so
an unproved backend is a loud exit-1 at startup and no capability row can
claim a server that did not answer. No DTO or route change: the existing
`external_tools.openai_compat` row is the capability surface; adding a second
chat flag to `FeatureFlags` would need a codegen change for no new truth.

Error classes: upstream 400 maps to `invalid_request` 400, 503/loading to 503,
other upstream and transport failures to 502 via `LlmError::Upstream`.
Images are refused with the named `vision_unsupported` 400 rather than
silently dropped. Request/response bodies are bounded; upstream error text is
capped at 512 chars.

## Change

- Choose the transport before coding: either add the pre-existing `reqwest`
  `blocking` feature and implement a sync `LlmPipeline` adapter, or add a thin
  async typed proxy route. The trait is sync and tokio has no `process`
  feature here, so an async client cannot simply be called from
  `complete_stream`; write the bridge decision into the implementation note.
- Wire the opt-in selection at `crates/server/src/backends.rs::make_llm`
  (same place as `LlmBackend::LlamaCpp`); extend `LlmBackend` with a CLI/attach
  variant. Default stays `Stub` — an operator opts in explicitly.
- Map request DTOs (messages, tools, grammar/JSON schema, sampling, images)
  and streamed SSE chunks back through `ChatResponse` / `complete_stream` sink.
  `abort` must cancel the upstream stream, not just flip a local atomic.
- Keep the app's API key separate from honkbox's bearer; never expose the app
  as a public route, never echo its errors into the capability document.

## Acceptance detail

- Named 400/502/503 responses when the attached server disappears mid-stream;
  no panics, no blank 200s, no partial replies presented as complete.
- Request/response bodies bounded; client disconnect stops upstream work.
- The `FeatureFlags`/capability row stays false while the backend is selected
  but unhealthy, matching the existing honesty rule (`epic-capabilities-contract`).
- Tool calling and JSON-schema constraints are either mapped faithfully or
  refused with a named error — never silently dropped.

## Non-goals

Starting/stopping the app, model hot-swap, router mode, generic OpenAI provider
routing (the mesh owns provider routing).
