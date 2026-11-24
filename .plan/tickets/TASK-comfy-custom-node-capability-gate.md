# TASK: ComfyUI custom-node capability gating and operator policy note
Status: proposed
Priority: P1
Epic: epic-comfyui-managed-runtime
Owner: unassigned
Acceptance: workflow templates fail closed when a required node class is absent from `/object_info`, the ComfyUI capability row is not treated as proof that every template can run, and operator docs state the custom-node install/policy contract without honkbox gaining a node manager.
Verification: `bun run test:server`, `bun run codegen:check`
Links: `.tmp/research/comfyui-custom-nodes-integration-2026-11-24.md`, `FEAT-comfyui-workflow-pipeline.md`, `epic-external-tools.md`
Resources: per-test fake `/object_info` payloads and temp dirs; no shared ComfyUI state

## Change

- Add a template-level required `class_type` check against `/object_info` before submit; missing classes produce 501 naming the class. Keep the external-tools `Workflow` capability as-is rather than adding a new global flag.
- Keep custom-node installation, ref pins, patches, signed governance policy and node whitelists in the operator-owned ComfyUI fork. honkbox never clones, imports or executes a node pack.
- Document: manifest `url`/`ref`/`extras`/`patches`, `WAS_BLOCK_AUTO_INSTALL`, signed policy with `custom_nodes_deps.py audit`, disabled runtime auto-fetch for ControlFoley/HunyuanVideoFoley, and `extra_model_paths.yaml` for user-supplied weights.
- State the honest limit: HTTP exposes node classes and input specs, not pack version or digest. No pack inventory API is added to honkbox.

## Acceptance detail

- Tests cover a present node set, an absent required node and an empty `/object_info` catalog; no live ComfyUI is required.
- No `pip`, `git clone`, Python runtime, Manager API or generic workflow JSON path is introduced.
- Loops back to `TASK-comfyui-operator-lifecycle.md` for the operator note; no new config fields.
