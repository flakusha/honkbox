# EPIC: External tool registry and process supervision

Status: in progress (2026-11-25 — registry/supervisor landed; CLI-app and
typed workflow follow-ups open)
Priority: P0
Owner: unassigned
Research: `.tmp/research/comfyui-workflow-sidecar-2026-11-25.md`,
`.tmp/research/comfyui-custom-nodes-integration-2026-11-24.md`,
`.tmp/research/cli-app-management-llama-server-2026-11-22.md`,
`.tmp/research/multi-gpu-2026-11-20.md`,
`.tmp/research/comfyui-lane-fields-2026-11-25.md`,
`.tmp/research/video-ffi-fields-2026-11-25.md`,
`.tmp/research/api-provisioning-extended-fields-2026-11-25.md`,
`.tmp/research/creative-provenance-fields-2026-11-25.md`.

## Scope

One capability-aware registry for Blender, ComfyUI, and future external engines
plus a single process supervisor: discover versions/devices, start/stop, pass
args, enforce timeout and CPU/RAM limits, kill on cancel, collect artifacts.
“Managing apps” is not a feature of its own — it is this plumbing.

ComfyUI execution is a separate typed adapter epic:
`epic-comfyui-managed-runtime.md`. The registry still exposes coarse
`Capability::Workflow`; it does not become a workflow graph store or a
custom-node package manager.
Editor/DCC tools (Krita/GIMP/Inkscape/Aseprite/Substance/Audacity) gain
capability rows only when a real adapter exists (`dcc-closed-loop-honkbox-2026-11-25.md`
H-DCC3); aspirational tools are never advertised.

## Tickets

- `TASK-external-tool-registry.md` — capability discovery + 501 on missing tool.
- `TASK-process-supervisor.md` — lifecycle, limits, cancel, artifact dir.
- `TASK-llama-server-attach.md` — attach a user-installed `llama-server`
  (landed; six-tool capability contract green).
- `FEAT-cli-app-manager.md` — managed start/stop lifecycle (landed 2026-11-22).
  ComfyUI follow-ups are tracked in `epic-comfyui-managed-runtime.md`.
- `FEAT-llama-server-client-backend.md` — consume the attached server through
  the LLM seam (landed; also tracked in `epic-llm-expansion`).
- `TASK-multi-gpu-device-axis.md` — device ids and affinity on ledger slots,
  keeping the single-slot default unchanged.
- `TASK-fizgig-probe-honesty.md` — bounded retry, no blocked axum worker,
  honest capability rows; follow-up to the attach/probe slices.
- `TASK-training-reservation-class.md` — dedicated training GPU class so
  capability rows never reserve a hidden route.
- `TASK-download-options-contract.md` — provider-neutral download option
  vocabulary; no new fetch behavior.
- `FEAT-download-provider-adapters.md` — staged HF / llama.cpp / ComfyUI-node
  adapters over the contract. Any ComfyUI-node work remains operator-owned
  acquisition/admission; it must not perform automatic runtime install/fetch.
- `TASK-shared-model-library.md` — shared model library wave-1 follow-up.

Cross-epic execution contracts (owned in `epic-comfyui-managed-runtime.md`):

- `FEAT-comfyui-workflow-pipeline.md` — first typed write adapter above the
  coarse `Workflow` capability: immutable API templates, typed slots,
  preflight, private WS/history/view recovery and artifacts.
- `TASK-comfy-custom-node-capability-gate.md` — node-class visibility is not
  pack version/ref/digest truth; no honkbox auto-install, Manager or runtime
  fetch.

## Capability and policy truth

- `/system_stats` plus non-empty `/object_info` proves a coarse workflow surface
  exists. It does not prove every template or node pack can run.
- Targeted `/object_info/{class}` preflight proves current class visibility and
  input schema only. It cannot prove pack name, version, ref, digest or approval.
- Pack/weight install, ref pins, patches, signed policy and runtime-fetch
  controls stay in the operator-owned ComfyUI environment. honkbox does not
  clone, import or execute a pack, and adds no raw workflow proxy.
- The separate GPL ComfyUI process remains an operator-installed loopback
  service; no engine source is linked into the Rust workspace. LTX/H3 weight
  license, territory and revenue gates are admission facts, not capability
  flags.

## Native H3/LTX alternative adapter path

A future native H3/LTX media adapter is an alternative typed execution backend
behind the same job/slot/bundle/admission contract. It is not permission to
weaken the typed ComfyUI boundary or to add arbitrary graph/path passthrough.
It must preserve equivalent capability checks, immutable provenance and
license gates.

No native H3/LTX pipeline exists today; video work remains in
`epic-video-support`. LTX-2.x weights use the Community License amended
2026-11-11 ($10M commercial threshold); MiniMax H3 excludes EU/UK/South
Korea/USA and requires authorization above $20M annual revenue.

## Acceptance

All external engines report capabilities through one DTO; no engine is linked
into the Rust workspace; leases and cancellation flow through the resource
ledger (`TASK-resource-ledger-bounded-queue.md`).

A ComfyUI `Workflow` row is not presented as template/pack approval. Fake-server
workflow tests own their ports, UUIDs and local queues; no shared ComfyUI queue
state is used. Managed execution reuses the runtime's existing Comfy lease;
detached execution holds one lease through artifact commit. Live ComfyUI smoke
is opt-in, uses a dedicated pinned instance and never performs install/fetch.

## Verification

```sh
cargo test --locked -p honkbox-server --test comfy_workflows
cargo test --locked -p honkbox-server --test apps
bun run codegen:check
bun run check
HONKBOX_COMFY_LIVE=1 cargo test --locked -p honkbox-server \
  --test comfy_workflows -- --ignored --nocapture
```

## Non-goals

giwt-style manager UI; MCP tools; embedded `bpy`; raw workflow compatibility;
honkbox-owned custom-node installer or runtime package manager.
