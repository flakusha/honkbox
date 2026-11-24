# EPIC: Managed ComfyUI runtime and typed surface

Status: in progress (2026-11-25; runtime/reads landed, workflow pipeline open)
Priority: P1
Owner: unassigned
Research: `.tmp/research/comfyui-workflow-sidecar-2026-11-25.md`,
`.tmp/research/comfyui-endpoint-review-2026-11-24.md`,
`.tmp/research/comfyui-managed-instance-review-2026-11-24.md`,
`.tmp/research/comfyui-prior-findings-reconciliation-2026-11-24.md`,
`.tmp/research/comfyui-custom-nodes-integration-2026-11-24.md`,
`.tmp/research/comfyui-serve-compat-spike-2026-11-20.md`,
`.tmp/research/comfyui-uv-pytorch-build-cycle-2026-11-20.md`.

## Scope

The honkbox-owned ComfyUI process (declared in `[apps.comfyui]`, explicit
start/stop), its bearer-gated typed reads, and the closed server-side workflow
adapter above them. The wire boundary stays the serve-compat verdict: no raw
`POST /prompt`, `/ws`, `/interrupt`, `/queue` write or node-graph proxy.
Workflow submission uses immutable admin templates and typed slots through
`POST /v1/jobs/comfyui` (the specialized Blender-route pattern); upstream
protocol details never become public honkbox routes.

## State of the pieces

- Landed: `FEAT-cli-app-manager.md` — process lifetime, bounded logs,
  process-group teardown, resource lease and no auto-start/restart.
- Landed: `TASK-process-supervisor.md` — the one-shot external tool primitive
  (deliberately not reused for daemons).
- Landed 2026-11-24: readiness against ComfyUI's real `/queue` route
  (`BUG-comfyui-readiness-probe.md`).
- Landed 2026-11-24: typed history/status honesty and the operator setup
  contract (`TASK-comfyui-typed-surface-honesty.md`,
  `TASK-comfyui-operator-lifecycle.md`). History now returns typed status,
  progress and view-ready image/gif references; detached discovery is
  `detached-unverified`; raw history graphs are not exposed.
- Open: `FEAT-comfyui-workflow-pipeline.md`, its P0 custom-node gate and the
  video-audio templates.

## Typed execution boundary

- Admin-installed API-format templates record immutable id/version, canonical
  manifest SHA-256 and graph SHA-256. Clients supply only typed scalar/enum/
  artifact slots, never a graph, URL or upstream path.
- Required `class_type`s are preflighted with targeted `/object_info/{class}`
  lookup before `/prompt`; missing class is 501 naming it.
- The private adapter owns `prompt_id`, `client_id`, `/ws` filtering,
  `/history/{prompt_id}` + `/view` recovery and targeted cancellation.
- Only manifest-declared output references are copied into atomic job bundles
  and artifacts. WS previews, UI metadata, raw prompts and server paths are
  not artifact truth.
- A managed process already holds `ResourceKind::Comfy` for its lifetime;
  workflows reuse that lease with a one-permit execution gate. Detached
  workflows take one lease per job. No double acquisition or second scheduler.
- Node/weight install, ref pins, patches, signed policy and runtime-fetch
  controls remain operator-owned. honkbox forbids its own auto-install,
  Manager use, runtime fetch and generic workflow proxy.

## Tickets

- `BUG-comfyui-readiness-probe.md` — P0, landed 2026-11-24: per-app probe path
  (llama `/health`, ComfyUI `/queue`).
- `TASK-comfyui-typed-surface-honesty.md` — P1, landed: typed outputs/progress/
  status and detached truth.
- `TASK-comfyui-operator-lifecycle.md` — P1: config example, rev pins, shared
  model dirs, honest version and operator policy note.
- `FEAT-comfyui-workflow-pipeline.md` — P0, proposed; owns template manifest,
  typed slots, submission/observation/recovery, cancel, bounds and artifacts.
  This is the single ticket; `epic-full-ai-cycle` references the same feature.
- `TASK-comfy-custom-node-capability-gate.md` — P0, proposed; class visibility
  versus pack/ref/digest truth and no auto-install/runtime fetch.
- `FEAT-comfy-video-audio-templates.md` — P1, proposed; pinned native
  LTX-2.5/H3 audio-video graphs and LTX-2.3 Foley V2A through the typed pipeline.

## Native H3/LTX adapter path

A future native H3/LTX media pipeline is an alternative execution adapter for
the same typed job/slot/bundle/admission contract, not an alternative ComfyUI
wire format. It must preserve class/capability preflight equivalents, license
and territory gates, immutable provenance and the no-proxy boundary.

[FACT] no native H3/LTX pipeline exists today; video remains in
`epic-video-support`. LTX-2.x weights use the Community License amended
2026-11-11 ($10M commercial-entity threshold), and MiniMax H3 weights exclude
EU/UK/South Korea/USA and require authorization above $20M annual revenue.
These facts select or constrain an adapter; they never justify weakening the
typed workflow contract.

## Acceptance

A managed ComfyUI reaches `ready` against an unmodified upstream, `/comfyui/status`
agrees with what the read routes can serve, and history references can be
copied by `/view` under the existing path/body bounds. An operator reproduces
setup from pinned revisions and pre-staged weights.

The workflow ticket adds observable fake-server acceptance for exact ids,
class preflight, WS filtering, history recovery, bounded upload/output copy,
targeted cancellation, honest lease use and atomic bundles. Live smoke is
ignored by default, requires a dedicated pinned instance and unique ids, and
never uses a shared ComfyUI queue. No lease, endpoint or child survives a
stop/crash; no raw workflow graph or host path is returned.

## Verification

```sh
cargo test --locked -p honkbox-server --test comfy_workflows
cargo test --locked -p honkbox-server --test apps
cargo test --locked -p honkbox-server --test job_content
bun run codegen:check
bun run check
HONKBOX_COMFY_LIVE=1 cargo test --locked -p honkbox-server \
  --test comfy_workflows -- --ignored --nocapture
```

## Non-goals

ComfyUI installer/auto-install, runtime node/weight fetch, generic command DSL,
raw wire compatibility, a second GPU scheduler, managed app UI.
