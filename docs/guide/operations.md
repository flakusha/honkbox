# Operations: jobs, events, topology

## Jobs & cancellation

Every ADMITTED GPU request is stamped with an
`X-Job-Id` response header and listed (while in flight) by `GET /v1/jobs`;
a request refused before admission (bad body, model without the required
capability) never registers a job and carries no header.
`POST /v1/jobs/{id}/cancel` — or the REST-shaped `DELETE /v1/jobs/{id}`,
same contract — marks a job's cooperative-stop token: it actually stops
multi-image generations (between images; the request then answers 409
`job_cancelled`) and streaming chat (at the next token boundary);
single-shot FFI calls (one image, sync chat, embeddings, audio) are not
interruptible mid-flight and run to completion, which the cancel
response reports honestly (`{"cancelled": false}`).

## Job events (`GET /v1/events`)

A read-only Server-Sent-Events feed of the
job registry, all job kinds. Three SSE event types, each also
self-labeled in the JSON `event` field: `job_registered` (a route
admitted the request; may still be queued), `job_cancelled` (a cancel
was REQUESTED — not a completion event), and `job_finished` (the job's
request ended, any outcome). Data shape (`JobEvent`): `id`, `kind`
(`chat`/`embeddings`/`images`/`transcribe`/`synthesize`/`translations`),
`model` (when known), `started_at_ms`, `interruptible`, and `cancelled`
— whether a cancel had been requested at that point (`true` on
`job_finished` for a token-marked job, which for a non-interruptible job
still ran to completion). The stream is transition-only: reconnects
replay nothing (`GET /v1/jobs` is the snapshot), keep-alive comments
keep the connection warm, and a consumer that falls >256 events behind
gets an `events_dropped` frame naming the skipped count instead of a
silent gap.

## Deployment topology

Three owners can want the same GPU, so keep them apart:

- honkbox's built-in LLM seam (`stub`, `llama.cpp`, attached `llama-server`)
  — honkbox owns its swap ledger and residency lease.
- an externally managed `llama-swap` / `llama-server` — the operator owns that
  process and its VRAM; honkbox only calls it.
- the managed-app path (`FEAT-cli-app-manager`) — honkbox owns the process lifetime for declared `llama_server`/`comfyui` apps via config-only launch modes (`path`, `installation`, `script`), with no auto-start/restart.

Never point two of them at one bind address: the example honkbox port `9010`
and an sd.cpp autostart on `9010` collide. Router mode stays an explicit
non-goal — a second GPU/VRAM scheduler next to honkbox's swap ledger and
residency lease is two schedulers for one device.

## ComfyUI read surface

`/comfyui/*` is a bearer-gated typed read surface (`status`/`queue`/`history`/`view`), not a raw ComfyUI wire proxy; no raw node-graph submission. History returns typed artifact refs that `/view` can fetch; progress is best-effort and absent when upstream history has none.

Managed ComfyUI runs an operator-owned checkout: `.venv/bin/python main.py` (`installation`) or a fixed launcher (`script`), or a `path` binary. honkbox does not run `uv sync`, install ComfyUI, bump revisions or auto-restart it. The recorded pair (2026-11-20) is fork `85037383c2f7f982a85b71f19601a43d28f53016` on upstream `995456d0188ffb51032c28f41527cfa82d87d8a9`; bump both together, deliberately. Point `[models].comfyui_dir` at the same model store instead of duplicating weights. Readiness probes `/queue`; detached mode reports `available: false` because a startup probe cannot prove a live upstream.
