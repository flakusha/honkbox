<!-- SPDX-License-Identifier: AGPL-3.0-or-later-->
<!-- SPDX-FileCopyrightText: 2026 honkbox contributors (formerly sd-rip)-->

# Plan Interconnect

Derived producer/consumer/sibling graph over `.plan/` tickets and epics, the
src/ paths they own, and the artifacts they declare (**Generates:**). Sibling
edges derived from feature-matrix co-membership carry `derived` provenance.
Regenerate with `giwt plan links --write`; freshness is gated by `giwt plan
links --check` / `giwt plan validate`. No timestamps — output is
byte-identical for identical inputs.

- tickets: 178
- epics: 23
- artifacts: 0
- src paths owned: 68
- edges: produces 137, consumes 0, sibling 343, owns 0, related 0, generates 0

## Shared src hubs

| src path | owners |
| --- | --- |
| `src/apps.rs` | `BUG-comfyui-readiness-probe`, `FEAT-cli-app-manager`, `TASK-capabilities-golden-fixture` |
| `src/backends.rs` | `FEAT-llama-server-client-backend`, `TASK-audiocpp-stt-provider` |
| `src/downloads.rs` | `FEAT-download-provider-adapters`, `TASK-download-options-contract`, `TASK-support-doc-truth` |
| `src/formats.rs` | `FEAT-image-output-formats`, `TASK-format-registry`, `TASK-model-format-validation`, `TASK-svg-ora-support` |
| `src/gen` | `FEAT-jobs-snapshot-artifact-ids-and-frame-metadata`, `TASK-job-result-artifact-selection-metadata` |
| `src/gen/openapi.json` | `BUG-openapi-embeddings-unlisted`, `TASK-ash-backend-promote` |
| `src/jobs.rs` | `BUG-fizgig-artifact-cap-and-gc`, `FEAT-sd-edit-loras`, `TASK-determinism-contract`, `TASK-generations-loras`, `TASK-job-result-artifact-selection-metadata` |
| `src/jobs_registry.rs` | `FEAT-blender-job-route`, `TASK-fizgig-admission-contract-tests`, `TASK-images-edits-artifact-delivery-parity`, `TASK-job-result-artifact-selection-metadata`, `TASK-truthful-reserved-capability-rows` |
| `src/lib.rs` | `BUG-attached-job-cancel-stalled-stream`, `BUG-capabilities-rerank-flag-without-ranker`, `FEAT-ash-vulkan-backend`, `FEAT-blender-job-route`, `FEAT-image-output-formats`, `FEAT-residency-control-plane`, `FEAT-sd-edit-loras`, `TASK-api-version-policy`, `TASK-ash-backend-promote`, `TASK-audio-resampler-quality`, `TASK-audio-voices-loaded-backend-truth`, `TASK-audiocpp-capabilities`, `TASK-cheap-ops-spike`, `TASK-controlnet-rendering`, `TASK-face-detection-policy`, `TASK-gpu-slot-audit-hook`, `TASK-job-result-artifact-selection-metadata`, `TASK-llm-tool-choice-contract`, `TASK-llm-vision-default`, `TASK-sd-base-model-tags`, `TASK-support-doc-truth`, `TASK-training-reservation-class`, `TASK-truthful-reserved-capability-rows` |
| `src/main.rs` | `BUG-fizgig-artifact-cap-and-gc`, `TASK-ash-backend-promote`, `TASK-model-format-validation` |
| `src/memory/rerank.ts` | `BUG-capabilities-rerank-flag-without-ranker`, `FEAT-rerank-endpoint` |
| `src/metrics.rs` | `FEAT-blender-job-route`, `TASK-llm-prompt-cache`, `TASK-training-reservation-class` |
| `src/ora.rs` | `TASK-container-safety`, `TASK-svg-ora-support` |
| `src/routes.rs` | `BUG-capabilities-rerank-flag-without-ranker`, `BUG-openapi-embeddings-unlisted`, `FEAT-blender-job-route`, `FEAT-residency-control-plane`, `FEAT-sd-edit-loras`, `TASK-audiocpp-capabilities`, `TASK-controlnet-rendering`, `TASK-generations-loras`, `TASK-gpu-slot-audit-hook`, `TASK-llm-tool-choice-contract`, `TASK-process-supervisor`, `TASK-svg-ora-support`, `TASK-truthful-reserved-capability-rows` |
| `src/routes/blender_exec.rs` | `FEAT-blender-job-route`, `TASK-job-result-artifact-selection-metadata` |
| `src/routes/capabilities.rs` | `BUG-capabilities-rerank-flag-without-ranker`, `FEAT-residency-control-plane`, `TASK-audio-voices-loaded-backend-truth`, `TASK-fizgig-probe-honesty` |
| `src/routes/fizgig.rs` | `TASK-fizgig-admission-contract-tests`, `TASK-fizgig-probe-honesty` |
| `src/routes/fizgig_run.rs` | `BUG-fizgig-artifact-cap-and-gc`, `BUG-fizgig-family-model-resolution`, `TASK-fizgig-admission-contract-tests` |
| `src/routes/images_edit.rs` | `TASK-images-edits-artifact-delivery-parity`, `TASK-sd-inpaint-mask-semantics` |
| `src/routes/python_sidecar.rs` | `FEAT-fizgig-lora-training-jobs`, `TASK-fizgig-cli-probe` |
| `src/swap.rs` | `FEAT-residency-control-plane`, `TASK-llm-prompt-cache` |
| `src/tts.rs` | `TASK-audio-voices-loaded-backend-truth`, `TASK-audiocpp-tts-provider` |

## Edges

| from | kind | to | via | derived |
| --- | --- | --- | --- | --- |
| `BUG-attached-job-cancel-stalled-stream` | produces | `src/lib.rs` | - | - |
| `BUG-attached-job-cancel-stalled-stream` | produces | `src/llm_attached.rs` | - | - |
| `BUG-attached-job-cancel-stalled-stream` | produces | `src/routes/chat.rs` | - | - |
| `BUG-attached-job-cancel-stalled-stream` | produces | `src/routes/openai_constraints.rs` | - | - |
| `BUG-attached-job-cancel-stalled-stream` | produces | `src/routes/openai_stream.rs` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/integrations/honkbox/capabilities.ts` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/lib.rs` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/memory/rerank.ts` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/routes.rs` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/routes/capabilities.rs` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/routes/rerank.rs` | - | - |
| `BUG-capabilities-rerank-flag-without-ranker` | produces | `src/runtime_config.rs` | - | - |
| `BUG-comfyui-readiness-probe` | produces | `src/apps.rs` | - | - |
| `BUG-fizgig-artifact-cap-and-gc` | produces | `src/jobs.rs` | - | - |
| `BUG-fizgig-artifact-cap-and-gc` | produces | `src/main.rs` | - | - |
| `BUG-fizgig-artifact-cap-and-gc` | produces | `src/routes/fizgig_run.rs` | - | - |
| `BUG-fizgig-family-model-resolution` | produces | `src/fizgig/families/cache.py` | - | - |
| `BUG-fizgig-family-model-resolution` | produces | `src/routes/fizgig_run.rs` | - | - |
| `BUG-openapi-embeddings-unlisted` | produces | `src/gen/openapi.json` | - | - |
| `BUG-openapi-embeddings-unlisted` | produces | `src/routes.rs` | - | - |
| `BUG-openapi-embeddings-unlisted` | produces | `src/routes/openapi.rs` | - | - |
| `FEAT-ash-vulkan-backend` | produces | `src/lib.rs` | - | - |
| `FEAT-blender-asset-preflight` | produces | `src/blender_assets.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/jobs_registry.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/lib.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/metrics.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/routes.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/routes/blender.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/routes/blender_exec.rs` | - | - |
| `FEAT-blender-job-route` | produces | `src/routes/blender_stages.rs` | - | - |
| `FEAT-cli-app-manager` | produces | `src/apps.rs` | - | - |
| `FEAT-download-provider-adapters` | produces | `src/downloads.rs` | - | - |
| `FEAT-fizgig-lora-training-jobs` | produces | `src/routes/python_sidecar.rs` | - | - |
| `FEAT-image-output-formats` | produces | `src/formats.rs` | - | - |
| `FEAT-image-output-formats` | produces | `src/lib.rs` | - | - |
| `FEAT-image-output-formats` | produces | `src/routes/image_ops_exec.rs` | - | - |
| `FEAT-job-bundle-zip` | produces | `src/routes/job_content.rs` | - | - |
| `FEAT-jobs-snapshot-artifact-ids-and-frame-metadata` | produces | `src/gen` | - | - |
| `FEAT-llama-server-client-backend` | produces | `src/backends.rs` | - | - |
| `FEAT-rerank-endpoint` | produces | `src/memory/rerank.ts` | - | - |
| `FEAT-residency-control-plane` | produces | `src/lib.rs` | - | - |
| `FEAT-residency-control-plane` | produces | `src/routes.rs` | - | - |
| `FEAT-residency-control-plane` | produces | `src/routes/capabilities.rs` | - | - |
| `FEAT-residency-control-plane` | produces | `src/routes/models.rs` | - | - |
| `FEAT-residency-control-plane` | produces | `src/routes/runtime.rs` | - | - |
| `FEAT-residency-control-plane` | produces | `src/swap.rs` | - | - |
| `FEAT-sd-edit-loras` | produces | `src/jobs.rs` | - | - |
| `FEAT-sd-edit-loras` | produces | `src/lib.rs` | - | - |
| `FEAT-sd-edit-loras` | produces | `src/routes.rs` | - | - |
| `TASK-api-version-policy` | produces | `src/lib.rs` | - | - |
| `TASK-ash-backend-promote` | produces | `src/gen/api-types.ts` | - | - |
| `TASK-ash-backend-promote` | produces | `src/gen/openapi.json` | - | - |
| `TASK-ash-backend-promote` | produces | `src/lib.rs` | - | - |
| `TASK-ash-backend-promote` | produces | `src/main.rs` | - | - |
| `TASK-ash-backend-promote` | produces | `src/sdxl.rs` | - | - |
| `TASK-audio-resampler-quality` | produces | `src/lib.rs` | - | - |
| `TASK-audio-voices-loaded-backend-truth` | produces | `src/lib.rs` | - | - |
| `TASK-audio-voices-loaded-backend-truth` | produces | `src/routes/audio_speech.rs` | - | - |
| `TASK-audio-voices-loaded-backend-truth` | produces | `src/routes/audio_voices.rs` | - | - |
| `TASK-audio-voices-loaded-backend-truth` | produces | `src/routes/capabilities.rs` | - | - |
| `TASK-audio-voices-loaded-backend-truth` | produces | `src/tts.rs` | - | - |
| `TASK-audiocpp-capabilities` | produces | `src/lib.rs` | - | - |
| `TASK-audiocpp-capabilities` | produces | `src/routes.rs` | - | - |
| `TASK-audiocpp-stt-provider` | produces | `src/audiocpp.rs` | - | - |
| `TASK-audiocpp-stt-provider` | produces | `src/backends.rs` | - | - |
| `TASK-audiocpp-tts-provider` | produces | `src/tts.rs` | - | - |
| `TASK-audiocpp-vad-decision` | produces | `src/vad.rs` | - | - |
| `TASK-blender-addon-capability-probe` | produces | `src/blender_addons.py` | - | - |
| `TASK-blender-addon-capability-probe` | produces | `src/blender_addons.rs` | - | - |
| `TASK-capabilities-golden-fixture` | produces | `src/apps.rs` | - | - |
| `TASK-capabilities-golden-fixture` | produces | `src/nsfw.rs` | - | - |
| `TASK-capabilities-golden-fixture` | produces | `src/routes/apps.rs` | - | - |
| `TASK-cheap-ops-spike` | produces | `src/lib.rs` | - | - |
| `TASK-comfyui-serve-compat-spike` | produces | `src/generation/providers/comfyui` | - | - |
| `TASK-container-safety` | produces | `src/container.rs` | - | - |
| `TASK-container-safety` | produces | `src/ora.rs` | - | - |
| `TASK-controlnet-rendering` | produces | `src/lib.rs` | - | - |
| `TASK-controlnet-rendering` | produces | `src/routes.rs` | - | - |
| `TASK-determinism-contract` | produces | `src/determinism.rs` | - | - |
| `TASK-determinism-contract` | produces | `src/jobs.rs` | - | - |
| `TASK-device-selection-rank-review` | produces | `src/device.rs` | - | - |
| `TASK-download-options-contract` | produces | `src/downloads.rs` | - | - |
| `TASK-external-api-catalog-links` | produces | `src/routes/health.rs` | - | - |
| `TASK-face-detection-policy` | produces | `src/lib.rs` | - | - |
| `TASK-fizgig-admission-contract-tests` | produces | `src/jobs_registry.rs` | - | - |
| `TASK-fizgig-admission-contract-tests` | produces | `src/routes/fizgig.rs` | - | - |
| `TASK-fizgig-admission-contract-tests` | produces | `src/routes/fizgig_run.rs` | - | - |
| `TASK-fizgig-cli-probe` | produces | `src/fizgig/families/train.py` | - | - |
| `TASK-fizgig-cli-probe` | produces | `src/routes/python_sidecar.rs` | - | - |
| `TASK-fizgig-probe-honesty` | produces | `src/routes/capabilities.rs` | - | - |
| `TASK-fizgig-probe-honesty` | produces | `src/routes/fizgig.rs` | - | - |
| `TASK-format-registry` | produces | `src/formats.rs` | - | - |
| `TASK-generations-loras` | produces | `src/jobs.rs` | - | - |
| `TASK-generations-loras` | produces | `src/routes.rs` | - | - |
| `TASK-gpu-slot-audit-hook` | produces | `src/lib.rs` | - | - |
| `TASK-gpu-slot-audit-hook` | produces | `src/routes.rs` | - | - |
| `TASK-images-edits-artifact-delivery-parity` | produces | `src/image-edit/providers/honkbox-provider.ts` | - | - |
| `TASK-images-edits-artifact-delivery-parity` | produces | `src/jobs_registry.rs` | - | - |
| `TASK-images-edits-artifact-delivery-parity` | produces | `src/routes/images_edit.rs` | - | - |
| `TASK-images-edits-artifact-delivery-parity` | produces | `src/routes/images_inputs.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/artifacts.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/gen` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/integrations/honkbox/result-ingest.ts` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/jobs.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/jobs_registry.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/lib.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/routes/blender_exec.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/routes/images_generate.rs` | - | - |
| `TASK-llama-server-attach` | produces | `src/llama_server.rs` | - | - |
| `TASK-llm-prompt-cache` | produces | `src/config.rs` | - | - |
| `TASK-llm-prompt-cache` | produces | `src/metrics.rs` | - | - |
| `TASK-llm-prompt-cache` | produces | `src/swap.rs` | - | - |
| `TASK-llm-tool-choice-contract` | produces | `src/lib.rs` | - | - |
| `TASK-llm-tool-choice-contract` | produces | `src/routes.rs` | - | - |
| `TASK-llm-vision-default` | produces | `src/lib.rs` | - | - |
| `TASK-llm-vision-default` | produces | `src/playground.ts` | - | - |
| `TASK-model-format-validation` | produces | `src/formats.rs` | - | - |
| `TASK-model-format-validation` | produces | `src/main.rs` | - | - |
| `TASK-model-format-validation` | produces | `src/model_formats.rs` | - | - |
| `TASK-process-supervisor` | produces | `src/routes.rs` | - | - |
| `TASK-process-supervisor` | produces | `src/supervisor.rs` | - | - |
| `TASK-sd-base-model-tags` | produces | `src/lib.rs` | - | - |
| `TASK-sd-inpaint-mask-semantics` | produces | `src/routes/images_edit.rs` | - | - |
| `TASK-support-doc-truth` | produces | `src/db.rs` | - | - |
| `TASK-support-doc-truth` | produces | `src/downloads.rs` | - | - |
| `TASK-support-doc-truth` | produces | `src/lib.rs` | - | - |
| `TASK-svg-ora-support` | produces | `src/formats.rs` | - | - |
| `TASK-svg-ora-support` | produces | `src/ora.rs` | - | - |
| `TASK-svg-ora-support` | produces | `src/raster.rs` | - | - |
| `TASK-svg-ora-support` | produces | `src/routes.rs` | - | - |
| `TASK-svg-ora-support` | produces | `src/svg.rs` | - | - |
| `TASK-training-reservation-class` | produces | `src/lib.rs` | - | - |
| `TASK-training-reservation-class` | produces | `src/metrics.rs` | - | - |
| `TASK-truthful-reserved-capability-rows` | produces | `src/jobs_registry.rs` | - | - |
| `TASK-truthful-reserved-capability-rows` | produces | `src/lib.rs` | - | - |
| `TASK-truthful-reserved-capability-rows` | produces | `src/routes.rs` | - | - |
| `epic-blender-assets` | produces | `src/routes/blender` | - | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `BUG-capabilities-rerank-flag-without-ranker` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-ash-vulkan-backend` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-blender-job-route` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-residency-control-plane` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `BUG-openapi-embeddings-unlisted` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-ash-vulkan-backend` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-blender-job-route` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-rerank-endpoint` | `src/memory/rerank.ts` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-residency-control-plane` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs; src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `BUG-comfyui-readiness-probe` | sibling | `FEAT-cli-app-manager` | `src/apps.rs` | - |
| `BUG-comfyui-readiness-probe` | sibling | `TASK-capabilities-golden-fixture` | `src/apps.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `BUG-fizgig-family-model-resolution` | `src/routes/fizgig_run.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `FEAT-sd-edit-loras` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-ash-backend-promote` | `src/main.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-determinism-contract` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-fizgig-admission-contract-tests` | `src/routes/fizgig_run.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-generations-loras` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-model-format-validation` | `src/main.rs` | - |
| `BUG-fizgig-family-model-resolution` | sibling | `TASK-fizgig-admission-contract-tests` | `src/routes/fizgig_run.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-blender-job-route` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-residency-control-plane` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-sd-edit-loras` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-ash-backend-promote` | `src/gen/openapi.json` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-audiocpp-capabilities` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-controlnet-rendering` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-gpu-slot-audit-hook` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-llm-tool-choice-contract` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-blender-job-route` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-residency-control-plane` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-residency-control-plane` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-fizgig-admission-contract-tests` | `src/jobs_registry.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-images-edits-artifact-delivery-parity` | `src/jobs_registry.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs_registry.rs; src/lib.rs; src/routes/blender_exec.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-prompt-cache` | `src/metrics.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-training-reservation-class` | `src/lib.rs; src/metrics.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs; src/lib.rs; src/routes.rs` | - |
| `FEAT-cli-app-manager` | sibling | `TASK-capabilities-golden-fixture` | `src/apps.rs` | - |
| `FEAT-download-provider-adapters` | sibling | `TASK-download-options-contract` | `src/downloads.rs` | - |
| `FEAT-download-provider-adapters` | sibling | `TASK-support-doc-truth` | `src/downloads.rs` | - |
| `FEAT-fizgig-lora-training-jobs` | sibling | `TASK-fizgig-cli-probe` | `src/routes/python_sidecar.rs` | - |
| `FEAT-image-output-formats` | sibling | `FEAT-residency-control-plane` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-format-registry` | `src/formats.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-model-format-validation` | `src/formats.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-svg-ora-support` | `src/formats.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `FEAT-jobs-snapshot-artifact-ids-and-frame-metadata` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/gen` | - |
| `FEAT-llama-server-client-backend` | sibling | `TASK-audiocpp-stt-provider` | `src/backends.rs` | - |
| `FEAT-residency-control-plane` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs; src/routes/capabilities.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-prompt-cache` | `src/swap.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-determinism-contract` | `src/jobs.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-generations-loras` | `src/jobs.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs; src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-ash-backend-promote` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-model-format-validation` | `src/main.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-audiocpp-tts-provider` | `src/tts.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-container-safety` | sibling | `TASK-svg-ora-support` | `src/ora.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-generations-loras` | `src/jobs.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-support-doc-truth` | `src/downloads.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-fizgig-admission-contract-tests` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/fizgig.rs` | - |
| `TASK-fizgig-admission-contract-tests` | sibling | `TASK-images-edits-artifact-delivery-parity` | `src/jobs_registry.rs` | - |
| `TASK-fizgig-admission-contract-tests` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs_registry.rs` | - |
| `TASK-fizgig-admission-contract-tests` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs` | - |
| `TASK-format-registry` | sibling | `TASK-model-format-validation` | `src/formats.rs` | - |
| `TASK-format-registry` | sibling | `TASK-svg-ora-support` | `src/formats.rs` | - |
| `TASK-generations-loras` | sibling | `TASK-gpu-slot-audit-hook` | `src/routes.rs` | - |
| `TASK-generations-loras` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs` | - |
| `TASK-generations-loras` | sibling | `TASK-llm-tool-choice-contract` | `src/routes.rs` | - |
| `TASK-generations-loras` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-generations-loras` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-generations-loras` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-images-edits-artifact-delivery-parity` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs_registry.rs` | - |
| `TASK-images-edits-artifact-delivery-parity` | sibling | `TASK-sd-inpaint-mask-semantics` | `src/routes/images_edit.rs` | - |
| `TASK-images-edits-artifact-delivery-parity` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs; src/lib.rs` | - |
| `TASK-llm-prompt-cache` | sibling | `TASK-training-reservation-class` | `src/metrics.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-model-format-validation` | sibling | `TASK-svg-ora-support` | `src/formats.rs` | - |
| `TASK-process-supervisor` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-process-supervisor` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-support-doc-truth` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-support-doc-truth` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-svg-ora-support` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `TASK-training-reservation-class` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
