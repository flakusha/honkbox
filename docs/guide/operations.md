# Operations: jobs, events, topology

## Jobs & cancellation

Every admitted job is stamped with an `X-Job-Id` response header and
listed while live by `GET /v1/jobs`; a request refused before admission (bad
body, model without the required capability) never registers a job and carries
no header.

`POST /v1/jobs/{id}/cancel` and `DELETE /v1/jobs/{id}` have the same contract:
for every live id they mark the cooperative-stop token and return HTTP 200
`{"cancelled":true}`; an unknown or already-ended id returns 404. `cancelled`
means the token was marked, not that the work stopped.

`interruptible` is a separate field on `JobInfo` and `JobEvent`. Image jobs can
stop between images; sd.cpp also aborts a running single-image generation
mid-sampling, returning 409 `job_cancelled` with no partial render. Streaming
chat stops at the next token boundary. Sync chat/completions, embeddings, and
audio calls without a stop hook run to completion. Thus `job_finished` with
`cancelled:true` records a marked token, not necessarily early termination.

## Job events (`GET /v1/events`)

A read-only Server-Sent-Events feed for every registered job kind. Five job
events are each self-labeled in the JSON `event` field and the SSE `event:` line:

- `job_registered` — the route admitted the request; it may still be queued.
- `job_cancelled` — the cooperative-stop token was marked; not a completion event.
- `job_finished` — the request ended, with any outcome.
- `job_progress` — work advanced: `step`, `total_steps`, and optional `pct`.
- `job_result` — an artifact became available; a job may emit several, and this
  is not a completion signal.

Each `JobEvent` also rides in the SSE `id:` line as its monotonic per-process
`seq`. Keep-alive comments warm the connection.

A parseable `?since=<seq>` takes precedence over `Last-Event-ID`; otherwise the
header is used. With no parseable cursor the stream is live-only, never a 400.
A valid cursor replays only events with a greater `seq`, oldest first, from the
retained 256-event window, then continues live. `GET /v1/jobs` remains the
current-state snapshot.

A cursor older than that window, or a live subscriber that falls behind, emits
an `events_dropped` gap marker instead of the missed events. Its data is
`{"event":"events_dropped","skipped":N}`; it is not a `JobEvent`.

`JobEvent` fields (optional fields are omitted when not applicable):

| Field | Meaning |
| --- | --- |
| `seq` | Monotonic per-process sequence; also the SSE `id:`. |
| `event` | One of the five job event names above. |
| `id` | Live registry id (`job_<ulid>`), matching `X-Job-Id`. |
| `kind` | `chat`, `embeddings`, `images`, `transcribe`, `synthesize`, `translations`, `image_ops`, `blender`, or `ocr`. |
| `model` | Model slug when known. |
| `started_at_ms` | Registration time in Unix milliseconds. |
| `interruptible` | Whether the route can honor cancellation before completion. |
| `cancelled` | Whether a cancel had been requested at this event's point in time. |
| `step`, `total_steps`, `pct` | `job_progress`: completed units, total units (`0` when unknown), and percent `0..=100` (absent when total is `0`). |
| `artifact_id`, `sha256`, `size_bytes` | `job_result`: content-addressed id, artifact SHA-256, and byte length. |
| `artifact_name`, `artifact_kind` | `job_result`: producer name and artifact kind, distinct from job `kind`. |
| `artifact_index`, `artifact_primary` | `job_result`: zero-based emission position and preferred-artifact flag. |
| `annotations` | Optional NSFW annotations attached to the output. |

Sources: [`JobEvent` DTO](../../crates/api-types/src/lib.rs),
[`GET /v1/events` handler](../../crates/server/src/routes/jobs.rs); the served
schema is `GET /v1/openapi.json`.

## Release verification

For a release checkout, run the two master verification gates:

```sh
bun run codegen:check
bun run audit:gpu-slot
```

`codegen:check` proves generated TS bindings are fresh against `api-types`.
`audit:gpu-slot` fails when a server test takes the GPU slot without serializing
on its lock.

## Deployment topology

Three owners can want the same GPU, so keep them apart:

- honkbox's built-in LLM seam (`stub`, `llama.cpp`, attached `llama-server`)
  — honkbox owns its swap ledger and residency lease.
- an externally managed `llama-swap` / `llama-server` — the operator owns that
  process and its VRAM; honkbox only calls it.
- the managed-app path (`FEAT-cli-app-manager`) — honkbox owns the process lifetime for declared `llama_server`/`comfyui` apps via config-only launch modes (`path`, `installation`, `script`), with no auto-start/restart.

Never point two of them at one bind address. Router mode stays an explicit
non-goal — a second GPU/VRAM scheduler next to honkbox's swap ledger and
residency lease is two schedulers for one device.

## ComfyUI management and reads

`POST /v1/apps/comfyui/start` explicitly starts the declared managed `comfyui`
app; `POST /comfyui/start` is its bearer-gated, admin-scoped convenience alias.
They share the same start branches and `ManagedApp` response and never
auto-start. Already running is 409; no manager, a disabled app, or an unsupported
platform is 501. These start routes are separate from the reads below.
[Source](../../crates/server/src/routes/apps.rs).

`GET /comfyui/status`, `GET /comfyui/queue`,
`GET /comfyui/history/{id}`, and `GET /comfyui/view` form a bearer-gated typed
read surface, not a raw ComfyUI wire proxy; no raw node-graph submission.
History returns typed artifact refs that `/view` can fetch; progress is
best-effort and absent when upstream history has none.

Managed ComfyUI runs an operator-owned checkout: `.venv/bin/python main.py` (`installation`) or a fixed launcher (`script`), or a `path` binary. honkbox does not run `uv sync`, install ComfyUI, bump revisions or auto-restart it. The recorded pair (2026-11-20) is fork `85037383c2f7f982a85b71f19601a43d28f53016` on upstream `995456d0188ffb51032c28f41527cfa82d87d8a9`; bump both together, deliberately. Point `[models].comfyui_dir` at the same model store instead of duplicating weights. Readiness probes `/queue`; detached mode reports `available: false` because a startup probe cannot prove a live upstream.
