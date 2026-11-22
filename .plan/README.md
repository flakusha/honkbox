# `.plan/` — task tracking

Research: `.tmp/research/ReviewedDocs-Ticketing.md`. Prior art:
`../loop-lore/.plan/README.md`. Minimal shape; no giwt, generated artifacts,
or matrices yet.

## Authority

`AGENTS.md` > `.plan/` > `src/` > `docs/`. Current behavior is what `src/` does;
specs and docs may drift.

## Layout

| Path | Purpose |
| --- | --- |
| `tickets/` | `TASK-|FEAT-|BUG-<slug>.md` implementation/feature/fix specs |
| `epics/` | `epic-<slug>.md` cross-cutting initiatives |

## Ticket header

```
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

- Research stays in gitignored `.tmp/research/`. Promote it here only with an
  owner, acceptance criteria, and a verification command.
- Frontend tickets must include `bun run check`, `bun run codegen:check`, and
  a UI surface check; `packages/web-ui/src/gen/` is generated, never hand-edited.
- Tests are designed for parallel execution: unique tmp dirs / port 0 or offsets
  / per-test DB or collection names, deterministic teardown in `afterEach`/
  `finally`, no ordering dependence. State the resource contract in `Resources:`. Serial runs are shakeout only, not the end goal.
- Use plain git worktrees; giwt is not adopted. Hook hardening is tracked by
  `tickets/TASK-precommit-gates.md`; hook tests must satisfy the parallel-safe
  test contracts above (per-test temp git repos, no shared index/hooksPath).
- Keep ticket diffs small: one route/DTO seam or one engine adapter per ticket.

## Current epics

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

## Research promotion map

Most `.tmp/research/` docs have a plan home below or were triaged as no new honkbox work; scratch stays reference only.

| Research doc | Promoted to |
| --- | --- |
| `ReviewedFullCycle-Index.md` | `epic-full-ai-cycle` + build order |
| `ReviewedFullCycle-Resource-Queues.md` | `TASK-resource-ledger-bounded-queue`, `epic-external-tools` |
| `ReviewedFullCycle-Determinism.md` | `TASK-determinism-contract` |
| `ReviewedFullCycle-OCR.md` | `FEAT-ocr-ingest` |
| `ReviewedFullCycle-RAG.md` | `FEAT-rag-pipeline` |
| `ReviewedFullCycle-ComfyUI-Pipelines.md` | `FEAT-comfyui-workflow-pipeline` |
| `ReviewedFullCycle-Tool-Management.md` | `epic-external-tools` |
| `ReviewedFullCycle-Blender-Workstream.md`, `ReviewedFullCycle-Blender-Addons.md`, `ReviewedFullCycle-Blender-Rigging.md` | `epic-blender-assets` + 3 tickets |
| `ReviewedFullCycle-3D.md` | `epic-3d-capture` + 2 tickets |
| `ReviewedFullCycle-GameEngines.md` | `epic-game-engine-export` + 2 tickets |
| `ReviewedFullCycle-LoopLore-Acceleration.md` | `epic-cross-app-integration` |
| `ReviewedOutput-Formats.md` | `epic-output-formats` + 6 tickets |
| `ReviewedDocs-Ticketing.md`, `ReviewedDocs-Organization.md` | this file, `docs/README.md`, `epic-docs-tooling` |
| `ReviewedSupport-Index.md` | support epics + build order below |
| `ReviewedSupport-CurrentState.md` | `TASK-support-doc-truth`, `FEAT-capabilities-api` |
| `ReviewedSupport-LLM.md` | `epic-llm-expansion` (3 tickets) |
| `ReviewedSupport-SD.md` | `epic-sd-native-backend` (5 tickets) |
| `ReviewedSupport-Video.md` | `epic-video-support` + migrated `FEAT-video-output.md` |
| `ReviewedSupport-Audio.md` | `epic-audio-streaming` + existing audio-format tickets |
| `ReviewedSupport-Capabilities.md` | `epic-capabilities-contract` (4 tickets) |
| `ReviewedSupport-NextSteps.md` | ordered build order below |
| `ReviewedExpansion-Index.md` | expansion build order below |
| `ReviewedExpansion-TimelineEditing.md` | `epic-media-timeline` + 2 tickets |
| `ReviewedExpansion-GameEnginePlugins.md` | `epic-game-engine-export` + import fixture ticket |
| `ReviewedExpansion-RemoteVisionURLs.md` | `epic-llm-expansion` + URL fetch ticket |
| `ReviewedExpansion-SpeculativeDecoding.md` | `epic-llm-expansion` + decoding spike |
| `ReviewedExpansion-VoiceIdentity.md` | `epic-voice-identity` + 2 tickets |
| `ReviewedMesh-*.md` (5 docs), `ReviewedCreative-MeshRemoteAPI.md` | `epic-mesh-integration` + 7 tickets |
| `ReviewedCreative-Index.md`, `ReviewedCreative-SVG.md`, `ReviewedCreative-TexturesMaterials.md` | `epic-vector-materials` + 5 tickets |
| `ReviewedCreative-BakingModels.md`, `ReviewedCreative-FeedbackLoop.md` | `epic-model-feedback` + 5 tickets |
| `ReviewedImageEdit-*.md` (4 docs) | `epic-editor-integration` + 7 tickets |
| `ReviewedBinaryFormats.md` | `epic-binary-formats` + 6 tickets |
| `content-experience-gaps-2026-11-20.md` | `epic-trust-safety` + auth tickets (`epic-authn-authz`) |
| `python-sidecar-pytorch-2026-11-20.md` | `TASK-python-sidecar-subprocess` |
| `comfyui-uv-pytorch-build-cycle-2026-11-20.md` | `FEAT-comfyui-workflow-pipeline` (uv env + sync discipline) |
| `splat-generation-rendering-idol-vkgs-2026-11-20.md` | `epic-3d-capture` + 5 tickets (idol spike, vkgs probe, turntable, flythrough, camera-path schema) |
| `gen-flows-props-characters-locations-2026-11-20.md` | `epic-3d-capture` + 4 tickets (props, panorama, hunyuanworld, mesh-watch) + `TASK-unirig-probe` (`epic-blender-assets`) |
| `shaders-materials-uiux-asset-pipeline-2026-11-20.md` | `TASK-cc0-pack-fetch` + `TASK-shader-validate-hook` (`epic-game-engine-export`) |
| `product-licenses-include-vs-api-2026-11-20.md` | `epic-binary-formats` + 3 tickets (license registry, qal guard, nc gate) |
| `fizgig-training-python-sidecar-2026-11-22.md` | `TASK-fizgig-cli-probe` + `FEAT-fizgig-lora-training-jobs` + 6 follow-up tickets (family models, artifact cap, probe honesty, admission tests, training class, weight-license gate) |
| `cli-app-management-llama-server-2026-11-22.md` | `TASK-llama-server-attach` + `FEAT-llama-server-client-backend` + `FEAT-cli-app-manager` + `FEAT-residency-control-plane` |
| `cli-app-launch-management-other-apps-2026-11-22.md` | `FEAT-fizgig-lora-training-jobs` + `BUG-fizgig-family-model-resolution` + `BUG-fizgig-artifact-cap-and-gc` |
| `organic-growth-2026-11-20.md` | `epic-full-ai-cycle` + `TASK-generation-replay-endpoint` + `TASK-operability-surfaces` |
| `multi-gpu-2026-11-20.md` | `epic-external-tools` + `TASK-multi-gpu-device-axis` |
| `next-tickets-implementation-plan-2026-11-22.md` | `BUG-capabilities-rerank-flag-without-ranker`, `TASK-job-result-artifact-selection-metadata`, `TASK-images-edits-artifact-delivery-parity`, `TASK-audio-voices-loaded-backend-truth` |
| `harness-giwt-vcs-integration-2026-11-20.md` | `TASK-capabilities-golden-fixture`; giwt adoption intentionally not adopted |
| `media-transport-federation-2026-11-20.md` | `TASK-exr-format-support` |
| `ash-vulkan-backend-licensing-extensibility-2026-11-22.md` | `TASK-ash-backend-promote` |

## Support expansion order (2026-11-17)

P0: flag/doc truth, capabilities API + contract test, native ControlNet and
cancel. P1: LLM cache, edit LoRA, mask semantics, VAD + streaming STT.
P1 also includes the API-version policy. P2: video FFI spike → job class →
i2v frames, base-model tags, tool-choice contract. P3: video mux. Full
dependency graph stays in
`.tmp/research/ReviewedSupport-NextSteps.md`; scratch is reference only.

Expansion additions (`ReviewedExpansion-Index.md`): P2 OTIO sidecar, Godot
importer fixture, remote vision URL fetch, speculative decoding spike, and
diarization spike; P3 external edit-process spike and voice-clone policy.

Final research wave (2026-11-17): P1 format registry, container safety,
capability advertisement, editor command DTO/route, cheap pixel ops, and mesh
capability advertisement. P2 mesh client/discovery/reroute/stream-safety, SVG/PBR
validation, bake/model gates, scene spec, ORA, editor plugin fixtures, upscale
probe, and binary artifact transport. P3 generators, catalog links, LOD naming,
critique loop, and the mesh-lane federated-provider assessment (likely
non-feature).
