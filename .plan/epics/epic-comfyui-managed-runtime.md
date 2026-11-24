# EPIC: Managed ComfyUI runtime and typed surface

Status: active (2026-11-24; runtime landed, workflow pipeline open)
Priority: P1
Owner: unassigned
Research: `.tmp/research/comfyui-endpoint-review-2026-11-24.md`,
`comfyui-managed-instance-review-2026-11-24.md`,
`comfyui-prior-findings-reconciliation-2026-11-24.md`,
`comfyui-serve-compat-spike-2026-11-20.md`,
`comfyui-uv-pytorch-build-cycle-2026-11-20.md`.

## Scope

The honkbox-owned ComfyUI process (declared in `[apps.comfyui]`, explicit
start/stop) and the bearer-gated typed reads that report it. The wire boundary
stays the serve-compat verdict: no raw `POST /prompt`, `/ws`, `/interrupt` or
node-graph proxy. Workflow submission remains the separate
`FEAT-comfyui-workflow-pipeline.md` seam, built on top of this runtime.

## State of the pieces

- Landed: `FEAT-cli-app-manager.md` — process lifetime, lease, bounded logs,
  process-group teardown; four typed `/comfyui/*` read routes.
- Landed: `TASK-process-supervisor.md` — the one-shot external tool primitive
  (deliberately not reused for daemons).
- Landed 2026-11-24: readiness against ComfyUI's real `/queue` route
  (`BUG-comfyui-readiness-probe.md`).
- Landed 2026-11-24: typed history/status honesty and the operator setup
  contract (`TASK-comfyui-typed-surface-honesty.md`,
  `TASK-comfyui-operator-lifecycle.md`).
- Open: the workflow pipeline, video-audio templates and custom-node gating.

## Tickets

- `BUG-comfyui-readiness-probe.md` — P0, landed 2026-11-24: per-app probe path
  (llama `/health`, ComfyUI `/queue`).
- `TASK-comfyui-typed-surface-honesty.md` — P1, typed outputs/progress/status
  and detached truth.
- `TASK-comfyui-operator-lifecycle.md` — P1, config example, rev pins, shared
  model dirs, honest version.
- `FEAT-comfyui-workflow-pipeline.md` — proposed; depends on this runtime but
  adds no raw wire proxy (already tracked in `epic-full-ai-cycle`).

- `FEAT-comfy-video-audio-templates.md` — proposed; MMAudio V2A then ControlFoley TV2A through the pipeline.
- `TASK-comfy-custom-node-capability-gate.md` — proposed; required `class_type` gating and operator-owned node policy.

## Acceptance

A managed ComfyUI can actually reach `ready` against an unmodified upstream,
`/comfyui/status` agrees with what the read routes can serve, history gives
artifact references `/view` can fetch, and an operator can reproduce the
setup from repo docs with pinned revisions. No lease, endpoint or child
survives a stop/crash.

## Verification

`cargo test --locked -p honkbox-server`, `bun run codegen:check`,
`bun run lint:rs`. Live upstream runs stay opt-in/ignored.

## Non-goals

ComfyUI installer/auto-install, generic command DSL, raw wire compatibility,
a second GPU scheduler, managed app UI.
