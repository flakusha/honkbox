# honkbox-ash-backend

Experimental pure-Rust Vulkan inference backend for honkbox, built on the `ash`
crate. It is the ground-up alternative to the default `sd.cpp` shortcut stack
(`honkbox-sd-core`'s `SdCppPipeline`); it shares the `SdPipeline` seam and needs
no HTTP, DTO, or OpenAPI changes.

Design record: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md`.

## Status

- **Experimental and opt-in.** The default build is unchanged. Enable
  `ash-sd` on this crate (or `server/ash-sd` on the
  server, which forwards to it) to select this backend.
- **Real SD 1.5 sampling.** `AshSdPipeline::generate` runs CLIP text encoding,
  a reduced UNet + DDIM loop, and a reduced VAE decoder entirely on Vulkan
  compute. The reduced topology is square-latent only, not pixel-equal to
  sd.cpp; it never returns fake images or silently falls back.
- **No bundled weights.** Checkpoints and CLIP `vocab.json` / `merges.txt` are
  supplied by the caller (CLI: `--clip-vocab-dir <dir>`) and read on demand.
- **Unsafe policy.** `unsafe` is confined to `src/raw.rs`, one audited wrapper
  per Vulkan entry point; the crate root denies `unsafe_code` everywhere else.

## Layout

- `src/raw.rs` — the single sanctioned `unsafe` boundary, thin wrappers over
  Vulkan FFI.
- `src/device.rs` — headless instance, logical device, compute queue, physical
  device selection (discrete > integrated > CPU, then VRAM), and capacity
  admission.
- `src/compute.rs` plus the sibling kernel modules (`matmul`, `attention`,
  `conv2d`, `group_norm`, `silu`, …) — the shared compute-pass vocabulary.
- `src/model.rs` and `src/model_spec.rs` — the `Op` vocabulary and the per-family
  tensor-name/graph spec (the "spec" layer; new families land here, not in
  dispatch or server routes).
- `src/checkpoint.rs`, `src/checkpoint_family.rs`, `src/sd15_keys.rs`,
  `src/shape_infer.rs` — safetensors header parsing, family detection, exact key
  and shape validation, and width inference (the "checkpoint" layer).
- `src/sd.rs` — `AshSdPipeline`, the `SdPipeline` implementation behind
  `ash-sd`.
- `src/llm.rs` — `AshLlmPipeline`, a placeholder `LlmPipeline` behind
  `ash-llm` that returns a named unimplemented error.
- `src/stubs.rs` — STT/TTS seam types behind `ash-stt` /
  `ash-tts`, likewise named-error placeholders.
- `shaders/` — GLSL compute sources; build-time compiled to SPIR-V (see
  `build.rs`), never compiled at runtime.

## Validating checkpoints

From the workspace root:

```
cargo test --locked -p honkbox-ash-backend --features ash-sd
honkbox models inspect-ash comfy:stable-diffusion/v1-5-pruned-emaonly-fp16.safetensors
```

The tests validate headers, keys, shapes, and readiness from the safetensors
metadata; they do not download models. `models inspect-ash` reports the detected
family plus the same shape/readiness summary for a user-supplied checkpoint.
End-to-end sampling with a local checkpoint and tokenizer assets is opt-in via
`bun run test:parity` after setting `HONKBOX_ASH_PARITY_MODEL` and
`HONKBOX_ASH_CLIP_VOCAB_DIR`.

## Supported checkpoint families

- **SD 1.5** — checkpoint validation plus end-to-end Vulkan sampling: full CLIP
  text tower, reduced UNet (three downsampling stages, mid cross-attention,
  nearest-neighbour upsample), DDIM eta=0, and reduced VAE decoder. Square
  latent sizes only; both decoder tails skip blocks present in the full model.
- **SDXL** — header validation is implemented: UNet entry/timestep/label and
  context widths, VAE decoder entry/exit shapes, and both CLIP towers (CLIP-L
  and OpenCLIP bigG) are checked, with group counts and representative
  embeddings reported as readiness. No SDXL sampling yet.

Other families (Flux, SD3, GGUF LLM/audio) are not supported by this backend
yet; the spec layer is where they would be added.

## Upstream code and licenses

`ash` is this crate's only direct third-party dependency. The rest of the
list is either a workspace-wide vendored upstream or a project whose behavior
this crate reimplements; license texts and attribution come from the upstream
sources (also pinned under `vendor/` where applicable):

- [ash](https://github.com/ash-rs/ash/blob/master/LICENSE-APACHE) —
  Apache-2.0; regular Cargo dependency, allowlisted in `deny.toml`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) — MIT
  (ggml-org/llama.cpp); workspace-vendored upstream, not a direct dependency
  of this crate.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/blob/master/LICENSE)
  — MIT (leejet/stable-diffusion.cpp); workspace-vendored upstream.
- [ggml](https://github.com/ggml-org/ggml/blob/master/LICENSE) — MIT;
  workspace-vendored upstream.
- [diffusion-rs](https://github.com/newfla/diffusion-rs/blob/main/LICENSE) —
  MIT (newfla/diffusion-rs); workspace-vendored upstream (already vendored).
- [huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE)
  — Apache-2.0; the SD 1.5 spec in `src/checkpoint.rs` uses diffusers-style
  key names, mapped to original-SD checkpoint names the way vendored
  stable-diffusion.cpp's `convert_diffusers_*_to_original_sd1` does. (The SDXL
  keys in `src/sdxl.rs` follow the ComfyUI checkpoint layout, not diffusers'.)
- [OpenAI CLIP](https://github.com/openai/CLIP/blob/main/LICENSE) — MIT; the
  CLIP BPE tokenizer in `src/tokenizer.rs` follows the published CLIP
  algorithm.
- [OpenAI GPT-2](https://github.com/openai/gpt-2/blob/master/LICENSE) —
  Modified MIT; the byte-level BPE merges and byte↔unicode table in
  `src/tokenizer.rs` follow the published GPT-2 algorithm. No vocab or merge
  files are bundled — the caller supplies `vocab.json` / `merges.txt`.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) —
  GPL-3.0; **subprocess/API only**, never vendored, linked, or copied into the
  tree. The same boundary applies to other GPL-family tools.

[audio.cpp](https://github.com/0xShug0/audio.cpp/blob/main/LICENSE) (Apache-2.0)
is vendored in the workspace for `audio-cpp-sys` but is not a dependency of
this crate, and upstream ships no `NOTICE` file — so there is no NOTICE to
propagate from it here.

Model weights are external artifacts with their own licenses and are never
committed here: SD 1.5 checkpoints are typically OpenRAIL-M, SDXL checkpoints
RAIL++-M. The model registry governs territory, commercial, and gated-use
restrictions; this backend only loads user-supplied files.

honkbox itself is **AGPL-3.0-or-later** via `REUSE.toml`, and `reuse lint` gates
that declaration; there are no per-file SPDX headers. `deny.toml` is
allowlist-only: it allows `AGPL-3.0-or-later` (the workspace's own license id),
MIT, Apache-2.0, and a set of other permissive licenses, and denies everything
not on that list. GPL/LGPL variants and source-available licenses (SSPL,
Elastic, BUSL, non-commercial) are therefore denied as incoming dependencies,
so GPL-family compatibility arguments do not change the subprocess-only
policy. Note that the shipped `NOTICE` (generated by `cargo about`) does not
currently list the `ash` crate, even though `ash` is a dependency of this
crate.

Every shader under `shaders/` carries a first-line provenance comment in the
form `// Provenance: original (<what it was written from>)` — all 13 shipped
shaders are original, written from published formulas, specs, or first
principles, and need no upstream notice. A line-by-line translation of an
upstream kernel is a derivative work instead: it must name its upstream source
in that same comment and retain that source's license notices (MIT: copyright
and permission notice; Apache-2.0: patent/trademark/attribution notices plus
`NOTICE` propagation when the upstream ships one, and modified-file marking).
Do not mine CUDA samples or compiler output for Vulkan kernels — source them
from Khronos specs, papers, or MIT-licensed ggml compute code.

These are engineering gates, not legal advice; redistribution of derived
shaders in a released binary warrants counsel review.
