<!-- SPDX-License-Identifier: AGPL-3.0-or-later-->
<!-- SPDX-FileCopyrightText: 2026 honkbox contributors (formerly sd-rip)-->

# Plan Interconnect

Derived producer/consumer/sibling graph over `.plan/` tickets and epics, the
src/ paths they own, and the artifacts they declare (**Generates:**). Sibling
edges derived from feature-matrix co-membership carry `derived` provenance.
Regenerate with `giwt plan links --write`; freshness is gated by `giwt plan
links --check` / `giwt plan validate`. No timestamps — output is
byte-identical for identical inputs.

- tickets: 232
- epics: 25
- artifacts: 0
- src paths owned: 117
- edges: produces 273, consumes 0, sibling 998, owns 0, related 0, generates 0

## Shared src hubs

| src path | owners |
| --- | --- |
| `src/apps.rs` | `BUG-comfyui-readiness-probe`, `FEAT-cli-app-manager`, `FEAT-comfyui-workflow-pipeline`, `TASK-capabilities-golden-fixture` |
| `src/artifacts.rs` | `FEAT-comfyui-workflow-pipeline`, `TASK-job-result-artifact-selection-metadata` |
| `src/attention.rs` | `FEAT-ash-moss-image-family`, `FEAT-ash-qwen35-dense`, `TASK-ash-autoregressive-state`, `TASK-ash-flow-dit-substrate`, `TASK-ash-qwen-image21-attention-proof` |
| `src/backend.rs` | `TASK-ash-sampling-contract`, `TASK-inference-cache-correctness`, `TASK-llm-constrained-output-eval`, `TASK-llm-vision-ingest-contract` |
| `src/backends.rs` | `FEAT-llama-server-client-backend`, `TASK-audiocpp-stt-provider` |
| `src/checkpoint.rs` | `FEAT-ash-embeddinggemma-2`, `FEAT-ash-moss-image-family`, `TASK-ash-gguf-reader`, `TASK-ash-vae-family-adapters`, `TASK-ash-vulkan-dequant-kernel` |
| `src/compute.rs` | `TASK-ash-decoder-ops`, `TASK-ash-llm-evidence-integration`, `TASK-ash-rrdb-upscaler-spike`, `TASK-ash-shader-pipeline-cache` |
| `src/config.rs` | `FEAT-comfyui-workflow-pipeline`, `TASK-ash-qwen-image21-attention-proof`, `TASK-llm-prompt-cache` |
| `src/determinism.rs` | `FEAT-comfyui-workflow-pipeline`, `TASK-determinism-contract`, `TASK-inference-cache-correctness`, `TASK-render-critique-loop`, `TASK-scene-spec-schema` |
| `src/device.rs` | `TASK-ash-shader-pipeline-cache`, `TASK-device-selection-rank-review`, `TASK-vulkan-coopmat-baseline` |
| `src/downloads.rs` | `FEAT-download-provider-adapters`, `TASK-download-options-contract`, `TASK-support-doc-truth` |
| `src/formats.rs` | `FEAT-image-output-formats`, `TASK-exr-format-support`, `TASK-format-registry`, `TASK-model-format-validation`, `TASK-svg-ora-support` |
| `src/gen` | `FEAT-jobs-snapshot-artifact-ids-and-frame-metadata`, `TASK-job-result-artifact-selection-metadata` |
| `src/image_metadata.rs` | `TASK-content-record-provenance`, `TASK-img-meta-comfyui-pair`, `TASK-img-meta-png-parameters`, `TASK-img-meta-webp-exif` |
| `src/jobs.rs` | `BUG-fizgig-artifact-cap-and-gc`, `FEAT-comfyui-workflow-pipeline`, `FEAT-sd-edit-loras`, `TASK-determinism-contract`, `TASK-generations-loras`, `TASK-job-result-artifact-selection-metadata` |
| `src/jobs_registry.rs` | `FEAT-blender-job-route`, `FEAT-comfyui-workflow-pipeline`, `TASK-blender-terminal-hardening`, `TASK-fizgig-admission-contract-tests`, `TASK-images-edits-artifact-delivery-parity`, `TASK-job-result-artifact-selection-metadata`, `TASK-truthful-reserved-capability-rows` |
| `src/lib.rs` | `BUG-attached-job-cancel-stalled-stream`, `BUG-capabilities-rerank-flag-without-ranker`, `FEAT-ash-vulkan-backend`, `FEAT-blender-job-route`, `FEAT-comfyui-workflow-pipeline`, `FEAT-image-output-formats`, `FEAT-residency-control-plane`, `FEAT-sd-edit-loras`, `TASK-api-version-policy`, `TASK-ash-llm-evidence-integration`, `TASK-ash-sampling-contract`, `TASK-ash-unsupported-conditioning-guards`, `TASK-audio-resampler-quality`, `TASK-audio-residency-contention-policy`, `TASK-audio-voices-loaded-backend-truth`, `TASK-audiocpp-capabilities`, `TASK-blender-bake-extended-maps`, `TASK-capability-honesty-gate`, `TASK-cheap-ops-spike`, `TASK-comfy-custom-node-capability-gate`, `TASK-controlnet-rendering`, `TASK-download-options-contract`, `TASK-face-detection-policy`, `TASK-gpu-slot-audit-hook`, `TASK-inference-cache-correctness`, `TASK-job-result-artifact-selection-metadata`, `TASK-llm-kv-spill-policy`, `TASK-llm-load-plan-admission`, `TASK-llm-moe-tier-offload`, `TASK-llm-speculative-model-draft`, `TASK-llm-tool-choice-contract`, `TASK-llm-vision-default`, `TASK-llm-vision-ingest-contract`, `TASK-salvo-dto-schema-bridge`, `TASK-sd-base-model-tags`, `TASK-support-doc-truth`, `TASK-training-reservation-class`, `TASK-truthful-reserved-capability-rows`, `epic-blender-assets`, `epic-salvo-migration` |
| `src/llm.rs` | `FEAT-ash-embeddinggemma-2`, `FEAT-ash-lfm25-text-decoder`, `FEAT-ash-moss-image-family`, `FEAT-ash-qwen35-dense`, `TASK-ash-autoregressive-state`, `TASK-ash-llm-evidence-integration`, `TASK-ash-sampling-contract`, `TASK-vulkan-coopmat-baseline` |
| `src/llm_attached.rs` | `BUG-attached-job-cancel-stalled-stream`, `FEAT-ash-lfm25-text-decoder` |
| `src/main.rs` | `BUG-fizgig-artifact-cap-and-gc`, `TASK-ash-backend-promote`, `TASK-ash-family-readiness-markers`, `TASK-ash-llm-evidence-integration`, `TASK-model-format-validation` |
| `src/memory/rerank.ts` | `BUG-capabilities-rerank-flag-without-ranker`, `FEAT-rerank-endpoint` |
| `src/metrics.rs` | `FEAT-blender-job-route`, `FEAT-comfyui-workflow-pipeline`, `TASK-llm-prompt-cache`, `TASK-training-reservation-class` |
| `src/model.rs` | `FEAT-ash-moss-image-family`, `FEAT-ash-qwen35-dense`, `TASK-ash-decoder-ops`, `TASK-ash-flow-dit-substrate`, `TASK-ash-sampling-contract` |
| `src/ora.rs` | `TASK-container-safety`, `TASK-svg-ora-support` |
| `src/routes.rs` | `BUG-capabilities-rerank-flag-without-ranker`, `BUG-openapi-embeddings-unlisted`, `FEAT-blender-job-route`, `FEAT-comfyui-workflow-pipeline`, `FEAT-residency-control-plane`, `FEAT-sd-edit-loras`, `TASK-audiocpp-capabilities`, `TASK-capability-honesty-gate`, `TASK-controlnet-rendering`, `TASK-generations-loras`, `TASK-gpu-slot-audit-hook`, `TASK-llm-tool-choice-contract`, `TASK-process-supervisor`, `TASK-svg-ora-support`, `TASK-truthful-reserved-capability-rows` |
| `src/routes/apps.rs` | `FEAT-comfyui-workflow-pipeline`, `TASK-capabilities-golden-fixture` |
| `src/routes/blender.rs` | `FEAT-blender-job-route`, `TASK-capability-honesty-gate` |
| `src/routes/blender_exec.rs` | `FEAT-blender-job-route`, `FEAT-comfyui-workflow-pipeline`, `TASK-blender-terminal-hardening`, `TASK-job-result-artifact-selection-metadata` |
| `src/routes/capabilities.rs` | `BUG-capabilities-rerank-flag-without-ranker`, `FEAT-comfyui-workflow-pipeline`, `FEAT-residency-control-plane`, `TASK-audio-voices-loaded-backend-truth`, `TASK-capability-honesty-gate`, `TASK-fizgig-probe-honesty` |
| `src/routes/chat.rs` | `BUG-attached-job-cancel-stalled-stream`, `TASK-ash-llm-evidence-integration` |
| `src/routes/fizgig.rs` | `TASK-fizgig-admission-contract-tests`, `TASK-fizgig-probe-honesty` |
| `src/routes/fizgig_run.rs` | `BUG-fizgig-artifact-cap-and-gc`, `BUG-fizgig-family-model-resolution`, `TASK-fizgig-admission-contract-tests` |
| `src/routes/image_ops_exec.rs` | `FEAT-image-output-formats`, `TASK-exr-format-support`, `TASK-img-meta-exif-orientation` |
| `src/routes/images_edit.rs` | `TASK-images-edits-artifact-delivery-parity`, `TASK-sd-inpaint-mask-semantics` |
| `src/routes/images_generate.rs` | `TASK-ash-unsupported-conditioning-guards`, `TASK-inference-cache-correctness`, `TASK-job-result-artifact-selection-metadata` |
| `src/routes/job_content.rs` | `FEAT-comfyui-workflow-pipeline`, `FEAT-job-bundle-zip` |
| `src/routes/python_sidecar.rs` | `FEAT-fizgig-lora-training-jobs`, `TASK-fizgig-cli-probe` |
| `src/sampler.rs` | `FEAT-ash-moss-image-family`, `TASK-ash-diffusion-scheduler-host`, `TASK-ash-flow-dit-substrate` |
| `src/sd.rs` | `FEAT-ash-sd2-spec`, `TASK-ash-family-readiness-markers`, `TASK-ash-tokenizer-framework`, `TASK-ash-unsupported-conditioning-guards` |
| `src/swap.rs` | `FEAT-residency-control-plane`, `TASK-inference-cache-correctness`, `TASK-llm-prompt-cache`, `TASK-plan-index-generator` |
| `src/timestep.rs` | `BUG-ash-sd15-numerics`, `TASK-ash-flow-dit-substrate` |
| `src/tokenizer.rs` | `FEAT-ash-moss-image-family`, `TASK-ash-tokenizer-framework` |
| `src/tts.rs` | `TASK-audio-voices-loaded-backend-truth`, `TASK-audiocpp-tts-provider` |
| `src/unet_real.rs` | `BUG-ash-sd15-numerics`, `TASK-ash-block-level-goldens` |
| `src/vae_real.rs` | `FEAT-ash-moss-image-family`, `TASK-ash-vae-family-adapters` |

## Edges

| from | kind | to | via | derived |
| --- | --- | --- | --- | --- |
| `BUG-ash-sd15-numerics` | produces | `src/ggml-cpu/ops.cpp` | - | - |
| `BUG-ash-sd15-numerics` | produces | `src/model/common/block.hpp` | - | - |
| `BUG-ash-sd15-numerics` | produces | `src/timestep.rs` | - | - |
| `BUG-ash-sd15-numerics` | produces | `src/unet_real.rs` | - | - |
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
| `FEAT-ash-embeddinggemma-2` | produces | `src/checkpoint.rs` | - | - |
| `FEAT-ash-embeddinggemma-2` | produces | `src/embed.rs` | - | - |
| `FEAT-ash-embeddinggemma-2` | produces | `src/llm.rs` | - | - |
| `FEAT-ash-lfm25-text-decoder` | produces | `src/llm.rs` | - | - |
| `FEAT-ash-lfm25-text-decoder` | produces | `src/llm_attached.rs` | - | - |
| `FEAT-ash-lfm25-text-decoder` | produces | `src/models/lfm-base.cpp` | - | - |
| `FEAT-ash-lfm25-text-decoder` | produces | `src/models/lfm2.cpp` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/attention.rs` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/catalog.json` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/checkpoint.rs` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/core/ggml_extend.cpp` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/llm.rs` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/model.rs` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/model/diffusion/moss_image.hpp` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/pipeline/diffusion_engine.cpp` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/sampler.rs` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/tokenizer.rs` | - | - |
| `FEAT-ash-moss-image-family` | produces | `src/vae_real.rs` | - | - |
| `FEAT-ash-qwen35-dense` | produces | `src/attention.rs` | - | - |
| `FEAT-ash-qwen35-dense` | produces | `src/llm.rs` | - | - |
| `FEAT-ash-qwen35-dense` | produces | `src/llm/qwen35.rs` | - | - |
| `FEAT-ash-qwen35-dense` | produces | `src/model.rs` | - | - |
| `FEAT-ash-qwen35-dense` | produces | `src/models/delta-net-base.cpp` | - | - |
| `FEAT-ash-qwen35-dense` | produces | `src/models/qwen35.cpp` | - | - |
| `FEAT-ash-sd2-spec` | produces | `src/model_spec.rs` | - | - |
| `FEAT-ash-sd2-spec` | produces | `src/sd.rs` | - | - |
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
| `FEAT-comfyui-workflow-pipeline` | produces | `src/apps.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/artifacts.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/config.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/determinism.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/jobs.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/jobs_registry.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/lib.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/metrics.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/routes.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/routes/apps.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/routes/blender_exec.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/routes/capabilities.rs` | - | - |
| `FEAT-comfyui-workflow-pipeline` | produces | `src/routes/job_content.rs` | - | - |
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
| `TASK-ash-autoregressive-state` | produces | `src/attention.rs` | - | - |
| `TASK-ash-autoregressive-state` | produces | `src/llm.rs` | - | - |
| `TASK-ash-backend-promote` | produces | `src/main.rs` | - | - |
| `TASK-ash-block-level-goldens` | produces | `src/unet_real.rs` | - | - |
| `TASK-ash-block-trait` | produces | `src/unet_attention.rs` | - | - |
| `TASK-ash-block-trait` | produces | `src/unet_input.rs` | - | - |
| `TASK-ash-block-trait` | produces | `src/unet_traversal.rs` | - | - |
| `TASK-ash-decoder-ops` | produces | `src/compute.rs` | - | - |
| `TASK-ash-decoder-ops` | produces | `src/mlp.rs` | - | - |
| `TASK-ash-decoder-ops` | produces | `src/model.rs` | - | - |
| `TASK-ash-decoder-ops` | produces | `src/norm.rs` | - | - |
| `TASK-ash-diffusion-scheduler-host` | produces | `src/ddim_sampler.rs` | - | - |
| `TASK-ash-diffusion-scheduler-host` | produces | `src/sampler.rs` | - | - |
| `TASK-ash-family-readiness-markers` | produces | `src/checkpoint_family.rs` | - | - |
| `TASK-ash-family-readiness-markers` | produces | `src/main.rs` | - | - |
| `TASK-ash-family-readiness-markers` | produces | `src/sd.rs` | - | - |
| `TASK-ash-flow-dit-substrate` | produces | `src/attention.rs` | - | - |
| `TASK-ash-flow-dit-substrate` | produces | `src/model.rs` | - | - |
| `TASK-ash-flow-dit-substrate` | produces | `src/sampler.rs` | - | - |
| `TASK-ash-flow-dit-substrate` | produces | `src/timestep.rs` | - | - |
| `TASK-ash-gguf-dequant` | produces | `src/ggml-common.h` | - | - |
| `TASK-ash-gguf-dequant` | produces | `src/ggml-quants.c` | - | - |
| `TASK-ash-gguf-dequant` | produces | `src/ggml.c` | - | - |
| `TASK-ash-gguf-dequant` | produces | `src/matmul.rs` | - | - |
| `TASK-ash-gguf-dequant` | produces | `src/unet_checkpoint.rs` | - | - |
| `TASK-ash-gguf-reader` | produces | `src/checkpoint.rs` | - | - |
| `TASK-ash-gguf-reader` | produces | `src/metadata.rs` | - | - |
| `TASK-ash-llm-evidence-integration` | produces | `src/compute.rs` | - | - |
| `TASK-ash-llm-evidence-integration` | produces | `src/lib.rs` | - | - |
| `TASK-ash-llm-evidence-integration` | produces | `src/llm.rs` | - | - |
| `TASK-ash-llm-evidence-integration` | produces | `src/main.rs` | - | - |
| `TASK-ash-llm-evidence-integration` | produces | `src/routes/chat.rs` | - | - |
| `TASK-ash-qwen-image21-attention-proof` | produces | `src/attention.rs` | - | - |
| `TASK-ash-qwen-image21-attention-proof` | produces | `src/config.rs` | - | - |
| `TASK-ash-qwen-image21-attention-proof` | produces | `src/license_gate.rs` | - | - |
| `TASK-ash-qwen-image21-attention-proof` | produces | `src/model/diffusion/qwen_image_2_1.hpp` | - | - |
| `TASK-ash-rrdb-upscaler-spike` | produces | `src/compute.rs` | - | - |
| `TASK-ash-rrdb-upscaler-spike` | produces | `src/conv2d.rs` | - | - |
| `TASK-ash-sampling-contract` | produces | `src/backend.rs` | - | - |
| `TASK-ash-sampling-contract` | produces | `src/lib.rs` | - | - |
| `TASK-ash-sampling-contract` | produces | `src/llm.rs` | - | - |
| `TASK-ash-sampling-contract` | produces | `src/model.rs` | - | - |
| `TASK-ash-sampling-contract` | produces | `src/routes/openai_chat.rs` | - | - |
| `TASK-ash-shader-pipeline-cache` | produces | `src/compute.rs` | - | - |
| `TASK-ash-shader-pipeline-cache` | produces | `src/device.rs` | - | - |
| `TASK-ash-taesd-decoder` | produces | `src/vae_decode.rs` | - | - |
| `TASK-ash-tokenizer-framework` | produces | `src/prompt.rs` | - | - |
| `TASK-ash-tokenizer-framework` | produces | `src/sd.rs` | - | - |
| `TASK-ash-tokenizer-framework` | produces | `src/tokenizer.rs` | - | - |
| `TASK-ash-unsupported-conditioning-guards` | produces | `src/lib.rs` | - | - |
| `TASK-ash-unsupported-conditioning-guards` | produces | `src/routes/images_generate.rs` | - | - |
| `TASK-ash-unsupported-conditioning-guards` | produces | `src/sd.rs` | - | - |
| `TASK-ash-vae-family-adapters` | produces | `src/checkpoint.rs` | - | - |
| `TASK-ash-vae-family-adapters` | produces | `src/model/vae/auto_encoder_kl.hpp` | - | - |
| `TASK-ash-vae-family-adapters` | produces | `src/model/vae/wan_vae.hpp` | - | - |
| `TASK-ash-vae-family-adapters` | produces | `src/name_conversion.cpp` | - | - |
| `TASK-ash-vae-family-adapters` | produces | `src/vae_real.rs` | - | - |
| `TASK-ash-vulkan-dequant-kernel` | produces | `src/checkpoint.rs` | - | - |
| `TASK-audio-resampler-quality` | produces | `src/lib.rs` | - | - |
| `TASK-audio-residency-contention-policy` | produces | `src/lib.rs` | - | - |
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
| `TASK-blender-bake-extended-maps` | produces | `src/blender_asset.py` | - | - |
| `TASK-blender-bake-extended-maps` | produces | `src/lib.rs` | - | - |
| `TASK-blender-terminal-hardening` | produces | `src/jobs_registry.rs` | - | - |
| `TASK-blender-terminal-hardening` | produces | `src/routes/blender_exec.rs` | - | - |
| `TASK-capabilities-golden-fixture` | produces | `src/apps.rs` | - | - |
| `TASK-capabilities-golden-fixture` | produces | `src/nsfw.rs` | - | - |
| `TASK-capabilities-golden-fixture` | produces | `src/routes/apps.rs` | - | - |
| `TASK-capability-honesty-gate` | produces | `src/lib.rs` | - | - |
| `TASK-capability-honesty-gate` | produces | `src/routes.rs` | - | - |
| `TASK-capability-honesty-gate` | produces | `src/routes/blender.rs` | - | - |
| `TASK-capability-honesty-gate` | produces | `src/routes/capabilities.rs` | - | - |
| `TASK-cheap-ops-spike` | produces | `src/lib.rs` | - | - |
| `TASK-comfy-custom-node-capability-gate` | produces | `src/lib.rs` | - | - |
| `TASK-comfyui-serve-compat-spike` | produces | `src/generation/providers/comfyui` | - | - |
| `TASK-container-safety` | produces | `src/container.rs` | - | - |
| `TASK-container-safety` | produces | `src/ora.rs` | - | - |
| `TASK-content-record-provenance` | produces | `src/image_metadata.rs` | - | - |
| `TASK-controlnet-rendering` | produces | `src/lib.rs` | - | - |
| `TASK-controlnet-rendering` | produces | `src/routes.rs` | - | - |
| `TASK-determinism-contract` | produces | `src/determinism.rs` | - | - |
| `TASK-determinism-contract` | produces | `src/jobs.rs` | - | - |
| `TASK-device-selection-rank-review` | produces | `src/device.rs` | - | - |
| `TASK-download-options-contract` | produces | `src/downloads.rs` | - | - |
| `TASK-download-options-contract` | produces | `src/lib.rs` | - | - |
| `TASK-exr-format-support` | produces | `src/formats.rs` | - | - |
| `TASK-exr-format-support` | produces | `src/routes/image_ops_exec.rs` | - | - |
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
| `TASK-img-meta-comfyui-pair` | produces | `src/image_metadata.rs` | - | - |
| `TASK-img-meta-exif-orientation` | produces | `src/routes/image_ops_exec.rs` | - | - |
| `TASK-img-meta-png-parameters` | produces | `src/image_metadata.rs` | - | - |
| `TASK-img-meta-webp-exif` | produces | `src/image_metadata.rs` | - | - |
| `TASK-inference-cache-correctness` | produces | `src/backend.rs` | - | - |
| `TASK-inference-cache-correctness` | produces | `src/determinism.rs` | - | - |
| `TASK-inference-cache-correctness` | produces | `src/lib.rs` | - | - |
| `TASK-inference-cache-correctness` | produces | `src/routes/images_generate.rs` | - | - |
| `TASK-inference-cache-correctness` | produces | `src/swap.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/artifacts.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/gen` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/integrations/honkbox/result-ingest.ts` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/jobs.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/jobs_registry.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/lib.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/routes/blender_exec.rs` | - | - |
| `TASK-job-result-artifact-selection-metadata` | produces | `src/routes/images_generate.rs` | - | - |
| `TASK-llama-server-attach` | produces | `src/llama_server.rs` | - | - |
| `TASK-llm-constrained-output-eval` | produces | `src/backend.rs` | - | - |
| `TASK-llm-kv-spill-policy` | produces | `src/lib.rs` | - | - |
| `TASK-llm-load-plan-admission` | produces | `src/lib.rs` | - | - |
| `TASK-llm-moe-tier-offload` | produces | `src/lib.rs` | - | - |
| `TASK-llm-prompt-cache` | produces | `src/config.rs` | - | - |
| `TASK-llm-prompt-cache` | produces | `src/metrics.rs` | - | - |
| `TASK-llm-prompt-cache` | produces | `src/swap.rs` | - | - |
| `TASK-llm-speculative-model-draft` | produces | `src/lib.rs` | - | - |
| `TASK-llm-tool-choice-contract` | produces | `src/lib.rs` | - | - |
| `TASK-llm-tool-choice-contract` | produces | `src/routes.rs` | - | - |
| `TASK-llm-vision-default` | produces | `src/lib.rs` | - | - |
| `TASK-llm-vision-default` | produces | `src/playground.ts` | - | - |
| `TASK-llm-vision-ingest-contract` | produces | `src/backend.rs` | - | - |
| `TASK-llm-vision-ingest-contract` | produces | `src/lib.rs` | - | - |
| `TASK-model-format-validation` | produces | `src/formats.rs` | - | - |
| `TASK-model-format-validation` | produces | `src/main.rs` | - | - |
| `TASK-model-format-validation` | produces | `src/model_formats.rs` | - | - |
| `TASK-plan-index-generator` | produces | `src/swap.rs` | - | - |
| `TASK-process-supervisor` | produces | `src/routes.rs` | - | - |
| `TASK-process-supervisor` | produces | `src/supervisor.rs` | - | - |
| `TASK-render-critique-loop` | produces | `src/determinism.rs` | - | - |
| `TASK-salvo-dto-schema-bridge` | produces | `src/lib.rs` | - | - |
| `TASK-salvo-middleware-and-codegen` | produces | `src/salvo_middleware.rs` | - | - |
| `TASK-salvo-migration-spike` | produces | `src/salvo_version.rs` | - | - |
| `TASK-scene-spec-schema` | produces | `src/determinism.rs` | - | - |
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
| `TASK-vulkan-coopmat-baseline` | produces | `src/device.rs` | - | - |
| `TASK-vulkan-coopmat-baseline` | produces | `src/ggml-vulkan/ggml-vulkan.cpp` | - | - |
| `TASK-vulkan-coopmat-baseline` | produces | `src/llm.rs` | - | - |
| `epic-blender-assets` | produces | `src/lib.rs` | - | - |
| `epic-salvo-migration` | produces | `src/lib.rs` | - | - |
| `BUG-ash-sd15-numerics` | sibling | `TASK-ash-block-level-goldens` | `src/unet_real.rs` | - |
| `BUG-ash-sd15-numerics` | sibling | `TASK-ash-flow-dit-substrate` | `src/timestep.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `BUG-capabilities-rerank-flag-without-ranker` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-ash-lfm25-text-decoder` | `src/llm_attached.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-ash-vulkan-backend` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-blender-job-route` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-residency-control-plane` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs; src/routes/chat.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `BUG-attached-job-cancel-stalled-stream` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `BUG-openapi-embeddings-unlisted` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-ash-vulkan-backend` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-blender-job-route` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-rerank-endpoint` | `src/memory/rerank.ts` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-residency-control-plane` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs; src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `BUG-capabilities-rerank-flag-without-ranker` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `BUG-comfyui-readiness-probe` | sibling | `FEAT-cli-app-manager` | `src/apps.rs` | - |
| `BUG-comfyui-readiness-probe` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/apps.rs` | - |
| `BUG-comfyui-readiness-probe` | sibling | `TASK-capabilities-golden-fixture` | `src/apps.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `BUG-fizgig-family-model-resolution` | `src/routes/fizgig_run.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `FEAT-sd-edit-loras` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-ash-backend-promote` | `src/main.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-ash-family-readiness-markers` | `src/main.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-ash-llm-evidence-integration` | `src/main.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-determinism-contract` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-fizgig-admission-contract-tests` | `src/routes/fizgig_run.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-generations-loras` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs` | - |
| `BUG-fizgig-artifact-cap-and-gc` | sibling | `TASK-model-format-validation` | `src/main.rs` | - |
| `BUG-fizgig-family-model-resolution` | sibling | `TASK-fizgig-admission-contract-tests` | `src/routes/fizgig_run.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-blender-job-route` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-residency-control-plane` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `FEAT-sd-edit-loras` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-audiocpp-capabilities` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-capability-honesty-gate` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-controlnet-rendering` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-gpu-slot-audit-hook` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-llm-tool-choice-contract` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `BUG-openapi-embeddings-unlisted` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `FEAT-ash-lfm25-text-decoder` | `src/llm.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `FEAT-ash-moss-image-family` | `src/checkpoint.rs; src/llm.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `FEAT-ash-qwen35-dense` | `src/llm.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-ash-autoregressive-state` | `src/llm.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-ash-gguf-reader` | `src/checkpoint.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-ash-llm-evidence-integration` | `src/llm.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-ash-sampling-contract` | `src/llm.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-ash-vae-family-adapters` | `src/checkpoint.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-ash-vulkan-dequant-kernel` | `src/checkpoint.rs` | - |
| `FEAT-ash-embeddinggemma-2` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `FEAT-ash-lfm25-text-decoder` | sibling | `FEAT-ash-moss-image-family` | `src/llm.rs` | - |
| `FEAT-ash-lfm25-text-decoder` | sibling | `FEAT-ash-qwen35-dense` | `src/llm.rs` | - |
| `FEAT-ash-lfm25-text-decoder` | sibling | `TASK-ash-autoregressive-state` | `src/llm.rs` | - |
| `FEAT-ash-lfm25-text-decoder` | sibling | `TASK-ash-llm-evidence-integration` | `src/llm.rs` | - |
| `FEAT-ash-lfm25-text-decoder` | sibling | `TASK-ash-sampling-contract` | `src/llm.rs` | - |
| `FEAT-ash-lfm25-text-decoder` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `FEAT-ash-qwen35-dense` | `src/attention.rs; src/llm.rs; src/model.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-autoregressive-state` | `src/attention.rs; src/llm.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-decoder-ops` | `src/model.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-diffusion-scheduler-host` | `src/sampler.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-flow-dit-substrate` | `src/attention.rs; src/model.rs; src/sampler.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-gguf-reader` | `src/checkpoint.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-llm-evidence-integration` | `src/llm.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-qwen-image21-attention-proof` | `src/attention.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-sampling-contract` | `src/llm.rs; src/model.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-tokenizer-framework` | `src/tokenizer.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-vae-family-adapters` | `src/checkpoint.rs; src/vae_real.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-ash-vulkan-dequant-kernel` | `src/checkpoint.rs` | - |
| `FEAT-ash-moss-image-family` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-ash-autoregressive-state` | `src/attention.rs; src/llm.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-ash-decoder-ops` | `src/model.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-ash-flow-dit-substrate` | `src/attention.rs; src/model.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-ash-llm-evidence-integration` | `src/llm.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-ash-qwen-image21-attention-proof` | `src/attention.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-ash-sampling-contract` | `src/llm.rs; src/model.rs` | - |
| `FEAT-ash-qwen35-dense` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `FEAT-ash-sd2-spec` | sibling | `TASK-ash-family-readiness-markers` | `src/sd.rs` | - |
| `FEAT-ash-sd2-spec` | sibling | `TASK-ash-tokenizer-framework` | `src/sd.rs` | - |
| `FEAT-ash-sd2-spec` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/sd.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-blender-job-route` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-residency-control-plane` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `FEAT-ash-vulkan-backend` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/jobs_registry.rs; src/lib.rs; src/metrics.rs; src/routes.rs; src/routes/blender_exec.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-residency-control-plane` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-blender-terminal-hardening` | `src/jobs_registry.rs; src/routes/blender_exec.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes.rs; src/routes/blender.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-fizgig-admission-contract-tests` | `src/jobs_registry.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-images-edits-artifact-delivery-parity` | `src/jobs_registry.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs_registry.rs; src/lib.rs; src/routes/blender_exec.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-prompt-cache` | `src/metrics.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-training-reservation-class` | `src/lib.rs; src/metrics.rs` | - |
| `FEAT-blender-job-route` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs; src/lib.rs; src/routes.rs` | - |
| `FEAT-blender-job-route` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `FEAT-blender-job-route` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `FEAT-cli-app-manager` | sibling | `FEAT-comfyui-workflow-pipeline` | `src/apps.rs` | - |
| `FEAT-cli-app-manager` | sibling | `TASK-capabilities-golden-fixture` | `src/apps.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `FEAT-image-output-formats` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `FEAT-job-bundle-zip` | `src/routes/job_content.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `FEAT-residency-control-plane` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `FEAT-sd-edit-loras` | `src/jobs.rs; src/lib.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-ash-qwen-image21-attention-proof` | `src/config.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs; src/routes/capabilities.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-blender-terminal-hardening` | `src/jobs_registry.rs; src/routes/blender_exec.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-capabilities-golden-fixture` | `src/apps.rs; src/routes/apps.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-determinism-contract` | `src/determinism.rs; src/jobs.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-fizgig-admission-contract-tests` | `src/jobs_registry.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-generations-loras` | `src/jobs.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-images-edits-artifact-delivery-parity` | `src/jobs_registry.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-inference-cache-correctness` | `src/determinism.rs; src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/artifacts.rs; src/jobs.rs; src/jobs_registry.rs; src/lib.rs; src/routes/blender_exec.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-prompt-cache` | `src/config.rs; src/metrics.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-render-critique-loop` | `src/determinism.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-scene-spec-schema` | `src/determinism.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-training-reservation-class` | `src/lib.rs; src/metrics.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs; src/lib.rs; src/routes.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `FEAT-comfyui-workflow-pipeline` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `FEAT-download-provider-adapters` | sibling | `TASK-download-options-contract` | `src/downloads.rs` | - |
| `FEAT-download-provider-adapters` | sibling | `TASK-support-doc-truth` | `src/downloads.rs` | - |
| `FEAT-fizgig-lora-training-jobs` | sibling | `TASK-fizgig-cli-probe` | `src/routes/python_sidecar.rs` | - |
| `FEAT-image-output-formats` | sibling | `FEAT-residency-control-plane` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-exr-format-support` | `src/formats.rs; src/routes/image_ops_exec.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-format-registry` | `src/formats.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-img-meta-exif-orientation` | `src/routes/image_ops_exec.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-model-format-validation` | `src/formats.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-svg-ora-support` | `src/formats.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `FEAT-image-output-formats` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `FEAT-jobs-snapshot-artifact-ids-and-frame-metadata` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/gen` | - |
| `FEAT-llama-server-client-backend` | sibling | `TASK-audiocpp-stt-provider` | `src/backends.rs` | - |
| `FEAT-residency-control-plane` | sibling | `FEAT-sd-edit-loras` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs; src/routes/capabilities.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes.rs; src/routes/capabilities.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs; src/swap.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-prompt-cache` | `src/swap.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-plan-index-generator` | `src/swap.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-residency-control-plane` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `FEAT-residency-control-plane` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-api-version-policy` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-determinism-contract` | `src/jobs.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-generations-loras` | `src/jobs.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs; src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `FEAT-sd-edit-loras` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-ash-llm-evidence-integration` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-api-version-policy` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-ash-autoregressive-state` | sibling | `TASK-ash-flow-dit-substrate` | `src/attention.rs` | - |
| `TASK-ash-autoregressive-state` | sibling | `TASK-ash-llm-evidence-integration` | `src/llm.rs` | - |
| `TASK-ash-autoregressive-state` | sibling | `TASK-ash-qwen-image21-attention-proof` | `src/attention.rs` | - |
| `TASK-ash-autoregressive-state` | sibling | `TASK-ash-sampling-contract` | `src/llm.rs` | - |
| `TASK-ash-autoregressive-state` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-ash-family-readiness-markers` | `src/main.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-ash-llm-evidence-integration` | `src/main.rs` | - |
| `TASK-ash-backend-promote` | sibling | `TASK-model-format-validation` | `src/main.rs` | - |
| `TASK-ash-decoder-ops` | sibling | `TASK-ash-flow-dit-substrate` | `src/model.rs` | - |
| `TASK-ash-decoder-ops` | sibling | `TASK-ash-llm-evidence-integration` | `src/compute.rs` | - |
| `TASK-ash-decoder-ops` | sibling | `TASK-ash-rrdb-upscaler-spike` | `src/compute.rs` | - |
| `TASK-ash-decoder-ops` | sibling | `TASK-ash-sampling-contract` | `src/model.rs` | - |
| `TASK-ash-decoder-ops` | sibling | `TASK-ash-shader-pipeline-cache` | `src/compute.rs` | - |
| `TASK-ash-diffusion-scheduler-host` | sibling | `TASK-ash-flow-dit-substrate` | `src/sampler.rs` | - |
| `TASK-ash-family-readiness-markers` | sibling | `TASK-ash-llm-evidence-integration` | `src/main.rs` | - |
| `TASK-ash-family-readiness-markers` | sibling | `TASK-ash-tokenizer-framework` | `src/sd.rs` | - |
| `TASK-ash-family-readiness-markers` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/sd.rs` | - |
| `TASK-ash-family-readiness-markers` | sibling | `TASK-model-format-validation` | `src/main.rs` | - |
| `TASK-ash-flow-dit-substrate` | sibling | `TASK-ash-qwen-image21-attention-proof` | `src/attention.rs` | - |
| `TASK-ash-flow-dit-substrate` | sibling | `TASK-ash-sampling-contract` | `src/model.rs` | - |
| `TASK-ash-gguf-reader` | sibling | `TASK-ash-vae-family-adapters` | `src/checkpoint.rs` | - |
| `TASK-ash-gguf-reader` | sibling | `TASK-ash-vulkan-dequant-kernel` | `src/checkpoint.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-ash-rrdb-upscaler-spike` | `src/compute.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-ash-sampling-contract` | `src/lib.rs; src/llm.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-ash-shader-pipeline-cache` | `src/compute.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-model-format-validation` | `src/main.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-ash-llm-evidence-integration` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-ash-qwen-image21-attention-proof` | sibling | `TASK-llm-prompt-cache` | `src/config.rs` | - |
| `TASK-ash-rrdb-upscaler-spike` | sibling | `TASK-ash-shader-pipeline-cache` | `src/compute.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-inference-cache-correctness` | `src/backend.rs; src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-constrained-output-eval` | `src/backend.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-llm-vision-ingest-contract` | `src/backend.rs; src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `TASK-vulkan-coopmat-baseline` | `src/llm.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-ash-sampling-contract` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-ash-shader-pipeline-cache` | sibling | `TASK-device-selection-rank-review` | `src/device.rs` | - |
| `TASK-ash-shader-pipeline-cache` | sibling | `TASK-vulkan-coopmat-baseline` | `src/device.rs` | - |
| `TASK-ash-tokenizer-framework` | sibling | `TASK-ash-unsupported-conditioning-guards` | `src/sd.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-audio-resampler-quality` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs; src/routes/images_generate.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs; src/routes/images_generate.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-ash-unsupported-conditioning-guards` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-ash-vae-family-adapters` | sibling | `TASK-ash-vulkan-dequant-kernel` | `src/checkpoint.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-audio-residency-contention-policy` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-audio-resampler-quality` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-audio-voices-loaded-backend-truth` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-audio-residency-contention-policy` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-audiocpp-capabilities` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-audiocpp-tts-provider` | `src/tts.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes/capabilities.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-audio-voices-loaded-backend-truth` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-blender-bake-extended-maps` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-audiocpp-capabilities` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-capability-honesty-gate` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-blender-bake-extended-maps` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-blender-terminal-hardening` | sibling | `TASK-fizgig-admission-contract-tests` | `src/jobs_registry.rs` | - |
| `TASK-blender-terminal-hardening` | sibling | `TASK-images-edits-artifact-delivery-parity` | `src/jobs_registry.rs` | - |
| `TASK-blender-terminal-hardening` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs_registry.rs; src/routes/blender_exec.rs` | - |
| `TASK-blender-terminal-hardening` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-cheap-ops-spike` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-controlnet-rendering` | `src/lib.rs; src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-fizgig-probe-honesty` | `src/routes/capabilities.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-capability-honesty-gate` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-comfy-custom-node-capability-gate` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-cheap-ops-spike` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-controlnet-rendering` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-comfy-custom-node-capability-gate` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-container-safety` | sibling | `TASK-svg-ora-support` | `src/ora.rs` | - |
| `TASK-content-record-provenance` | sibling | `TASK-img-meta-comfyui-pair` | `src/image_metadata.rs` | - |
| `TASK-content-record-provenance` | sibling | `TASK-img-meta-png-parameters` | `src/image_metadata.rs` | - |
| `TASK-content-record-provenance` | sibling | `TASK-img-meta-webp-exif` | `src/image_metadata.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-download-options-contract` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-generations-loras` | `src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs; src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-controlnet-rendering` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-controlnet-rendering` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-generations-loras` | `src/jobs.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-inference-cache-correctness` | `src/determinism.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-render-critique-loop` | `src/determinism.rs` | - |
| `TASK-determinism-contract` | sibling | `TASK-scene-spec-schema` | `src/determinism.rs` | - |
| `TASK-device-selection-rank-review` | sibling | `TASK-vulkan-coopmat-baseline` | `src/device.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-face-detection-policy` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-support-doc-truth` | `src/downloads.rs; src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-download-options-contract` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-exr-format-support` | sibling | `TASK-format-registry` | `src/formats.rs` | - |
| `TASK-exr-format-support` | sibling | `TASK-img-meta-exif-orientation` | `src/routes/image_ops_exec.rs` | - |
| `TASK-exr-format-support` | sibling | `TASK-model-format-validation` | `src/formats.rs` | - |
| `TASK-exr-format-support` | sibling | `TASK-svg-ora-support` | `src/formats.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-gpu-slot-audit-hook` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-face-detection-policy` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
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
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-inference-cache-correctness` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs; src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-gpu-slot-audit-hook` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-images-edits-artifact-delivery-parity` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/jobs_registry.rs` | - |
| `TASK-images-edits-artifact-delivery-parity` | sibling | `TASK-sd-inpaint-mask-semantics` | `src/routes/images_edit.rs` | - |
| `TASK-images-edits-artifact-delivery-parity` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs` | - |
| `TASK-img-meta-comfyui-pair` | sibling | `TASK-img-meta-png-parameters` | `src/image_metadata.rs` | - |
| `TASK-img-meta-comfyui-pair` | sibling | `TASK-img-meta-webp-exif` | `src/image_metadata.rs` | - |
| `TASK-img-meta-png-parameters` | sibling | `TASK-img-meta-webp-exif` | `src/image_metadata.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-job-result-artifact-selection-metadata` | `src/lib.rs; src/routes/images_generate.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-constrained-output-eval` | `src/backend.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-prompt-cache` | `src/swap.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-llm-vision-ingest-contract` | `src/backend.rs; src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-plan-index-generator` | `src/swap.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-render-critique-loop` | `src/determinism.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-scene-spec-schema` | `src/determinism.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-inference-cache-correctness` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-kv-spill-policy` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `TASK-truthful-reserved-capability-rows` | `src/jobs_registry.rs; src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-job-result-artifact-selection-metadata` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-constrained-output-eval` | sibling | `TASK-llm-vision-ingest-contract` | `src/backend.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-llm-load-plan-admission` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-kv-spill-policy` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-llm-moe-tier-offload` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-load-plan-admission` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-llm-speculative-model-draft` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-moe-tier-offload` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-prompt-cache` | sibling | `TASK-plan-index-generator` | `src/swap.rs` | - |
| `TASK-llm-prompt-cache` | sibling | `TASK-training-reservation-class` | `src/metrics.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-llm-tool-choice-contract` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-speculative-model-draft` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-llm-vision-default` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-process-supervisor` | `src/routes.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs; src/routes.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-tool-choice-contract` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-llm-vision-ingest-contract` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-vision-default` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `TASK-salvo-dto-schema-bridge` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-llm-vision-ingest-contract` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-model-format-validation` | sibling | `TASK-svg-ora-support` | `src/formats.rs` | - |
| `TASK-process-supervisor` | sibling | `TASK-svg-ora-support` | `src/routes.rs` | - |
| `TASK-process-supervisor` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `TASK-render-critique-loop` | sibling | `TASK-scene-spec-schema` | `src/determinism.rs` | - |
| `TASK-salvo-dto-schema-bridge` | sibling | `TASK-sd-base-model-tags` | `src/lib.rs` | - |
| `TASK-salvo-dto-schema-bridge` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-salvo-dto-schema-bridge` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-salvo-dto-schema-bridge` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-salvo-dto-schema-bridge` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-salvo-dto-schema-bridge` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `TASK-support-doc-truth` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-sd-base-model-tags` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-support-doc-truth` | sibling | `TASK-training-reservation-class` | `src/lib.rs` | - |
| `TASK-support-doc-truth` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-support-doc-truth` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-support-doc-truth` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-svg-ora-support` | sibling | `TASK-truthful-reserved-capability-rows` | `src/routes.rs` | - |
| `TASK-training-reservation-class` | sibling | `TASK-truthful-reserved-capability-rows` | `src/lib.rs` | - |
| `TASK-training-reservation-class` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-training-reservation-class` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `TASK-truthful-reserved-capability-rows` | sibling | `epic-blender-assets` | `src/lib.rs` | - |
| `TASK-truthful-reserved-capability-rows` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
| `epic-blender-assets` | sibling | `epic-salvo-migration` | `src/lib.rs` | - |
