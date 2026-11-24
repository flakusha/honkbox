# `.plan/` — task tracking

Status: planning scaffold in use; plain git worktrees, no giwt adoption.
Current-wave research index:
`.tmp/research/plan-research-promotion-index-2026-11-25.md`.

## Authority

`AGENTS.md` > current repository code > this README > epics > tickets > docs.
Current behavior is what the code does; ticket status is a claim to verify.
`.tmp/research/` is gitignored evidence, not a shippable specification.

## Layout

| Path | Purpose |
| --- | --- |
| `tickets/` | `TASK\|FEAT\|BUG-<slug>.md` implementation/feature/fix specs |
| `epics/` | `epic-<slug>.md` cross-cutting initiatives |
| `epics-index.md`, `index.json`, `feature-matrix.md`, `interconnect.md`, `code-map.json` | Generated planning artifacts; currently stale until `TASK-plan-index-generator.md` lands |

## Ticket header

```text
Status: proposed | in progress | done | blocked | postponed
Priority: P0 | P1 | P2 | P3
Epic: epic-<slug> (or none)
Owner: <name or unassigned>
Acceptance: <observable result>
Verification: <exact commands>
Links: <research/docs/DTO/route references>
Resources: <per-test tmp dirs/ports/DB names owned>
```

## Conventions

- Promote research only with an owner, observable acceptance, exact verification,
  and a resource contract. Keep code-derived claims source-cited.
- Frontend tickets include `bun run check`, `bun run codegen:check`, and a UI
  surface check. `packages/web-ui/src/gen/` is generated, never hand-edited.
- Tests are parallel-safe: unique tmp dirs, port 0 or explicit offsets, per-test
  DB/collection names, deterministic teardown, no ordering dependence. Serial
  runs are shakeout only.
- Use plain git worktrees. Hook hardening is tracked by
  `TASK-precommit-gates.md`; hook tests use per-test temp git repos and no shared
  index/hooksPath.
- Keep ticket diffs small: one route/DTO seam or one engine adapter per ticket.
- Do not hand-edit generated plan indexes. `giwt plan check-links` checks Markdown
  links, not code-formatted `.tmp/research/` path existence.

## Current epics (24)

- [epic-full-ai-cycle.md](epics/epic-full-ai-cycle.md)
- [epic-blender-assets.md](epics/epic-blender-assets.md)
- [epic-game-engine-export.md](epics/epic-game-engine-export.md)
- [epic-output-formats.md](epics/epic-output-formats.md)
- [epic-3d-capture.md](epics/epic-3d-capture.md)
- [epic-external-tools.md](epics/epic-external-tools.md)
- [epic-cross-app-integration.md](epics/epic-cross-app-integration.md)
- [epic-docs-tooling.md](epics/epic-docs-tooling.md)
- [epic-capabilities-contract.md](epics/epic-capabilities-contract.md)
- [epic-llm-expansion.md](epics/epic-llm-expansion.md)
- [epic-sd-native-backend.md](epics/epic-sd-native-backend.md)
- [epic-video-support.md](epics/epic-video-support.md)
- [epic-audio-streaming.md](epics/epic-audio-streaming.md)
- [epic-media-timeline.md](epics/epic-media-timeline.md)
- [epic-voice-identity.md](epics/epic-voice-identity.md)
- [epic-mesh-integration.md](epics/epic-mesh-integration.md)
- [epic-vector-materials.md](epics/epic-vector-materials.md)
- [epic-model-feedback.md](epics/epic-model-feedback.md)
- [epic-editor-integration.md](epics/epic-editor-integration.md)
- [epic-binary-formats.md](epics/epic-binary-formats.md)
- [epic-trust-safety.md](epics/epic-trust-safety.md)
- [epic-authn-authz.md](epics/epic-authn-authz.md)
- [epic-comfyui-managed-runtime.md](epics/epic-comfyui-managed-runtime.md)
- [epic-salvo-migration.md](epics/epic-salvo-migration.md)

## 2026-11-25 research promotion map

| Research document | Promoted / reconciled plan surface |
| --- | --- |
| `native-video-backend-refresh-2026-11-25.md` | Native/video epics, current-pin `TASK-video-ffi-spike`, postponed `FEAT-video-output` |
| `ash-promotion-refresh-2026-11-25.md` | Ash feature/promotion truth, four bounded follow-ups, native epic build order |
| `llm-runtime-memory-2026-11-25.md` | P0 `TASK-inference-cache-correctness`, P1 `TASK-vulkan-coopmat-baseline`, LLM/full-cycle epics |
| `model-identity-license-download-2026-11-25.md` | Registry provenance/license tickets and revision-aware download contracts |
| `creative-pipeline-promotion-2026-11-25.md` | Scene-spec/fixed-view/critique tickets; false-done Blender bake correction |
| `legacy-reviewed-research-map-2026-11-25.md` | 46 missing historical research names mapped to current replacements or `historical-only` |
| `media-masters-audio-video-2026-11-25.md`, `media-handling-consolidated-2026-11-24.md` and topic notes | Output/audio/binary/template epics and format-specific tickets |
| `comfyui-workflow-sidecar-2026-11-25.md` and ComfyUI notes | Managed runtime, typed workflow/write path, custom-node capability gates |
| `salvo-migration-and-support-2026-11-24.md` | `epic-salvo-migration` + `TASK-salvo-migration-spike` |
| `dcc-closed-loop-honkbox-2026-11-25.md` | Editor/DCC honkbox items (H-DCC1–H-DCC6); tightened Krita/GIMP/ORA tickets; editor + cross-app + external-tools epics |

Older existing research (`content-experience-gaps-2026-11-20.md`,
`python-sidecar-pytorch-2026-11-20.md`,
`splat-generation-rendering-idol-vkgs-2026-11-20.md`,
`gen-flows-props-characters-locations-2026-11-20.md`,
`shaders-materials-uiux-asset-pipeline-2026-11-20.md`,
`product-licenses-include-vs-api-2026-11-20.md`,
`fizgig-training-python-sidecar-2026-11-22.md`,
`cli-app-management-llama-server-2026-11-22.md`,
`cli-app-launch-management-other-apps-2026-11-22.md`,
`llama-swap-loop-lore-gaps-2026-11-22.md`,
`multi-gpu-2026-11-20.md`, `media-transport-federation-2026-11-20.md`)
keeps its existing plan homes. The consolidated index records the current wave;
the legacy map prevents deleted `Reviewed*.md` files from being treated as
available evidence.

## Current build order (heavy inference first)

1. **P0 correctness:** Ash compile reachability/HTTP selection/capability truth;
   inference-cache identity plus byte-bearing LLM admission.
2. **P1 provenance/runtime:** persistent bounded model provenance and license
   decisions; staged revision-aware multi-file downloads through the existing
   supervisor.
3. **P1 native evidence:** SD2 one-step spec, direct Ash conditioning refusal,
   capability-gated RDNA3/RDNA4 cooperative-matrix baseline.
4. **P2 orchestration/media:** typed ComfyUI workflow write path, media master/
   derived artifact roles, and current-pin native video FFI spike.
5. **P2 creative:** scene-spec schema, bounded fixed-view rendering, typed
   critique/replay; readiness markers and RRDB/PixelShuffle spike.
6. **P3 optional surfaces:** video mux, vision critic, timeline rendering, voice
   cloning after consent/license policy.
7. **P2 planning integrity:** `TASK-plan-index-generator` before generated plan
   artifacts are treated as authoritative.

## Validation

- `giwt plan check-links`
- `bunx markdownlint-cli2 <changed .plan markdown files>`
- `bun run codegen:check` after DTO/OpenAPI changes
- Product changes use the narrow per-ticket commands, then `bun run check` /
  `bun run test` as appropriate.

Generated indexes remain stale until their generator is repaired. Do not infer
ticket completeness, research-file existence, or capability truth from them.
