# TASK: ComfyUI custom-node capability gating and operator policy note

Status: proposed
Priority: P0
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: required node classes fail closed without claiming pack provenance or
adding a honkbox node manager.

- Workflow templates fail closed when a required node class is absent from
  targeted `/object_info/{class}` lookup; 501 names the missing class and no
  `/prompt` request occurs.
- Node-class visibility is never treated as pack version, ref, digest,
  approval or provenance truth; the coarse `Workflow` capability remains
  unchanged.
- honkbox forbids its own auto-install, runtime fetch, Manager, `pip`, `git`,
  generic URL download and raw workflow proxy paths.
- Template policy requires operator-owned pre-staged nodes/weights and
  disabled runtime auto-fetch; the operator note states the non-verifiable
  boundary honestly.
- Fake-server tests cover present, absent, empty and faulty catalogs; no live
  install or shared ComfyUI queue is used.
Verification: `cargo test --locked -p honkbox-server --test comfy_workflows`,
`cargo test --locked -p honkbox-server --test apps`, `bun run codegen:check`,
`bun run check`; live smoke is opt-in/ignored.
Links: workflow research, prior custom-node findings, workflow ticket and operator docs.

- Research: `.tmp/research/comfyui-workflow-sidecar-2026-11-25.md`
- Prior research: `.tmp/research/comfyui-custom-nodes-integration-2026-11-24.md`,
  `.tmp/research/comfyui-prior-findings-reconciliation-2026-11-24.md`
- Workflow: `.plan/tickets/FEAT-comfyui-workflow-pipeline.md`
- Operator lifecycle: `.plan/tickets/TASK-comfyui-operator-lifecycle.md`
- Epic: `.plan/epics/epic-comfyui-managed-runtime.md`
- API object-info:
  <https://docs.comfy.org/development/comfyui-server/comms_routes>
Resources: isolated fake catalogs/templates and operator-owned pack/weight policy.

- Per-test fake `/object_info` payloads and temp template/policy dirs
- Operator-owned ComfyUI fork manifests, ref pins, patches and policy audit
- Existing README/operator setup docs; no new package-manager surface
- No honkbox node manager, new global capability flag, Python runtime or
  shared ComfyUI state

## Context

[FACT] `probe_comfyui` currently treats `/system_stats` plus non-empty
`/object_info` as `Capability::Workflow` (`external-tools/src/lib.rs:508-594`).
That coarse row says a workflow surface exists; it does not certify a graph.

[FACT] `/object_info` exposes registered node classes and their input specs.
It does not expose the supplying pack's git ref, version, digest or approval.

[FACT] custom nodes are arbitrary Python executed by the trusted ComfyUI
process. The operator fork already owns manifest refs/extras/patches, signed
governance, node whitelists and audit tooling. honkbox must not grow a second
manager.

## Change

## Class preflight in the typed workflow seam

- Add `required_class_types[]` and operator-enumerated
  `forbidden_class_types[]` to the immutable workflow template manifest.
  Required classes must equal the API-format graph's complete `class_type`
  set and be disjoint from the forbidden set.
- Before `/prompt`, query `GET /object_info/{class}` for each required class,
  percent-encoded and bounded. Accept only an exact class key in the returned
  object. A missing class or `{}` is 501 naming the class; non-200 or invalid
  JSON is 502. Do not confuse a faulty catalog with absence.
- Keep the external-tools `Workflow` capability as-is. Add no global
  `comfyui_custom_nodes` flag, no per-class inventory API and no false claim
  that a probe approves templates.
- Class visibility is a compatibility check only. Record it in the job bundle
  separately from operator pack/weight provenance and license evidence.

## Forbid honkbox auto-install and runtime fetch

- honkbox MUST NOT run `pip`, `git clone`, custom-node install scripts,
  ComfyUI-Manager APIs, registry downloads, weight downloads or generic URL
  fetches. No Manager extension route or generic workflow JSON path is added.
- Workflow slot schemas do not accept URLs or model paths; only typed scalars,
  enum values and existing honkbox artifact ids. Classes the operator identifies
  as runtime-fetching/install must appear in `forbidden_class_types` and are
  rejected at template load; honkbox does not claim to infer hidden behavior.
- Template manifests assert
  `runtime_policy { auto_install: false, runtime_fetch: false }`. Installation
  rejects any other values. This is an operator assertion, not remotely
  provable behavior.
- Operator must pre-stage node packs and weights, keep ref/digest pins and
  patches, disable runtime auto-fetch (including ControlFoley/HunyuanVideoFoley
  paths), and use `extra_model_paths.yaml` for user-supplied weights.
- honkbox launch configuration remains unchanged. It does not fetch or patch
  the operator's Python environment, and it does not add new config fields.

## Operator policy note

Extend the existing operator setup docs (README/example config) from this
ticket rather than creating a second honkbox package manager or reopening the
landed `TASK-comfyui-operator-lifecycle.md`. The note must distinguish:

| Evidence | Honest meaning |
|---|---|
| `GET /system_stats` | ComfyUI version/devices, not pack identity |
| `GET /object_info/{class}` | Class exists now with that input schema |
| `GET /extensions` | JS asset directories, not pack provenance |
| Operator manifest/audit | Declared refs, patches, digests and policy, not HTTP proof |
| Live egress audit | Observed no fetch during that smoke run, not a permanent guarantee |

Required note content:

- Operator manifest fields `url`, `ref`, `extras`, `patches`; immutable refs and
  recorded digests; source review before update.
- `WAS_BLOCK_AUTO_INSTALL` and signed governance policy with
  `custom_nodes_deps.py audit`; audit failures block promotion.
- Node whitelist/default-deny policy and explicit review of any class absent
  from the policy. `/object_info` cannot supply this decision.
- Runtime auto-fetch disabled for packs that otherwise fetch control/video
  helpers; weights and embeddings pre-staged before ComfyUI launch.
- Weight license/territory/revenue gate recorded before admission. LTX-2.x
  weights use the Community License amended 2026-11-11 with a $10M commercial
  threshold; MiniMax H3 excludes EU/UK/South Korea/USA and requires
  authorization above $20M annual revenue.
- Live smoke with blocked egress and a retained sanitized audit proving that
  no Manager/install/download/fetch endpoint was requested during the run.
- HTTP cannot prove hidden network behavior or pack authenticity. The note
  must say so instead of presenting class visibility as provenance.

## Acceptance criteria

1. Fake catalog with the exact required class plus input schema permits
   preflight and records visibility only.
2. Missing class returns 501 naming it and makes zero `/prompt` calls.
3. Empty `{}` catalog returns the same honest missing-class result.
4. Non-200, truncated JSON and wrong response type return 502 and make zero
   `/prompt` calls.
5. Manifest/graph class-set mismatch, empty `class_type`, undeclared required
   class or runtime policy other than `false/false` fails installation before
   a template can be submitted.
6. A class visible upstream but present in the operator-enumerated
   `forbidden_class_types` is rejected at load; class visibility never grants
   policy approval. External signed-policy audit remains operator-owned.
7. Request log contains no `pip`, `git`, Manager, install, registry, generic
   download or fetch endpoint. No URL/path slot is accepted.
8. Public capability/job responses never claim pack version, digest or
   provenance from `/object_info`.
9. Each test owns its fake server, response fixtures, template dir, UUIDs,
   local Registry/Ledger where execution is exercised, and RAII cleanup. No
   static/shared queue, mutable process global or live ComfyUI process.

## Verification

Commands to run after implementation (not run for this ticket):

```sh
cargo test --locked -p honkbox-server --test comfy_workflows
cargo test --locked -p honkbox-server --test apps
bun run codegen:check
bun run check
HONKBOX_COMFY_LIVE=1 cargo test --locked -p honkbox-server \
  --test comfy_workflows -- --ignored --nocapture
```
