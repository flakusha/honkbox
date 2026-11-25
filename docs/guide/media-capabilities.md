# Media capabilities: ControlNet, vision chat, audio

For exact request fields, wire error codes, statuses, and ranges, use the checked-in
[OpenAPI document](../../packages/web-ui/src/gen/openapi.json), the live
`GET /v1/openapi.json`, or the generated
HTML reference served at `/api-docs/index.html` (`bun run docs:api`). This guide
covers capability meaning, defaults, and operational caveats.

## ControlNet

`comfy:controlnet/*` weight files are discovered by the model
scan (kind `controlnet`, listed in `GET /v1/models`).
`POST /v1/images/generations` accepts the extension fields `controlnet`
(catalog slug/alias), `control_image` (base64 PNG), and optional
`controlnet_strength` (default 0.9). Validation rejects incomplete pairs,
invalid image bytes, unknown slugs, and out-of-range strengths; it never
silently clamps a request. A valid pair is applied by the sd-cpp pipeline:
the weights load with the model context and the conditioning image (sent
preprocessed, e.g. a depth or edge map; the engine resizes it) steers the
render. A backend that cannot apply control returns an explicit unsupported
response rather than silently ignoring it.

## Vision chat (mmproj, REQ-8)

An LLM weights file with a sibling mmproj
projector (`mmproj*.gguf` in the same directory, the HF multimodal-repo
convention) is discovered automatically at catalog time — such entries
report `capabilities.vision: true` and `modalities: ["text", "vision"]`
in `GET /v1/models`, and the chat pipeline loads the projector alongside
the model. All three chat routes (`POST /v1/chat`, `/v1/chat/stream`,
`/v1/chat/completions`) then accept OpenAI-style content-parts arrays:
`image_url` parts accept inline data URIs only; the server decodes the
bytes locally and never fetches remote URLs. Media markers retain their
position in the message and pass through llama.cpp's mtmd projector before
text generation. Image content on a text-only model (no sibling projector or
operator `capabilities.vision = true` override) is rejected rather than
silently flattened. A projector file itself (`mmproj*.gguf`) is listed by the
scan but is not chat-capable (`capabilities.chat: false`): it is a component,
not a model. Malformed image payloads are rejected rather than ignored.
Default-on: the mtmd path compiles via feature `llm-vision` →
`honkbox-llm-worker/vision`, and the `[patch.crates-io]` llama-cpp-sys-4
vendor patch (workspace `Cargo.toml`, applied by
`bun scripts/vendor-patches.ts sync`) adds the missing `vendor-hash`
link directive its `mtmd-helper.cpp` references — so the default build
serves image input on models with a sibling mmproj projector.
`--no-default-features` builds a text-only server where image parts get
the named `vision_unsupported` 400 — never a silently flattened image.

## Audio formats & translation

`/v1/audio/transcriptions`
`response_format` also accepts `srt` / `vtt`, and
`POST /v1/audio/translations` (multipart `file`, `model?`,
`response_format?`) transcribes with whisper's translate task — output
always English.

## Release feature flags

`bun run install:local` builds the server with Cargo feature `release`
(`ocr-ocrs` + `audio-vad` + `audio-cpp`), while `llm-vision`, `vulkan`, and
`barcode` remain default-on. A successful release build reports these
compile-time `/v1/capabilities` flags as true:

- `features.vision.built` — vision chat support is compiled.
- `features.vad` — Silero VAD support is compiled.
- `features.audio_cpp.built` — the audio.cpp STT/TTS backend is compiled.
- `recognition.ocr` — true only when OCR is compiled, selected, and both
  detection and recognition weights are discovered in an OCR root; otherwise
  the route is unserved even in a release build.

Ordinary development builds omit `audio-cpp`, `audio-vad`, and `ocr-ocrs`, so
they stay lean; those corresponding flags remain false. The heavy audio.cpp
CMake build runs only for the release set. Capability values are derived from
the running server, not solely from its Cargo features: `recognition.ocr`
also requires a selected OCR backend, and model counts depend on discovery.

## Multi-GPU Vulkan

sd.cpp uses Vulkan device 0 by default. On a multi-GPU host this may select the
iGPU: generation can be slow, and larger generations can abort with a
context-lost error. Set `GGML_VK_VISIBLE_DEVICES` to the dGPU index before
starting honkbox, for example:

```sh
export GGML_VK_VISIBLE_DEVICES=1
```

Use the dGPU's actual Vulkan device index; `1` is only an example.
