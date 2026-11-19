# TASK: loop-lore provider smoke path

Status: done
Priority: P2
Epic: epic-cross-app-integration
Owner: unassigned
Acceptance: a loop-lore Custom provider pointed at honkbox completes one chat turn and one image job with observable job/events state.
Verification: run both apps locally; `curl /health`, `/v1/models`; inspect `/v1/jobs`, `/v1/events`, `/metrics`
Links: `.tmp/research/ReviewedFullCycle-LoopLore-Acceleration.md`
Resources: loopback ports assigned by the harness; per-run temp data dirs for both apps

## Change

- Configure honkbox bearer token and load one model; register loop-lore Custom
  OpenAI-compatible provider at `/v1/chat/completions`.
- Submit one image via `/v1/images/generations` (artifact delivery mode);
  confirm job registration, `job_progress`/`job_result`/finish events, and
  cancel mid-run.
- Compare honkbox queue/lease metrics with loop-lore's scheduler events: one
  side owns GPU admission, never both.

## Acceptance detail

No MCP surface added to honkbox; loop-lore keeps provider routing, circuit
breaker, and requestClass labels. The progress-event clause is satisfied by
`FEAT-artifact-delivery-and-job-result-events`, not by liveness-only frames.

## Verification (2026-11-20)

Loop-lore Custom provider against a real bearer-auth honkbox on loopback: one chat turn, one artifact-mode image job with matching `job_result`, mid-run cancel, `/v1/jobs` snapshot, `/v1/events` stream and `/v1/metrics` all succeeded. Probe reported `queue.gpuSlots = 1`, `owner = honkbox`, `localAdmissionDenied = true`; a loop-lore-held slot was rejected before any dispatch. Harness: `.tmp/honkbox-smoke.ts` (throwaway, gitignored).
