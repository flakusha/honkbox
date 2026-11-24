# Media capabilities: ControlNet, vision chat, audio

## ControlNet

`comfy:controlnet/*` weight files are discovered by the model
scan (kind `controlnet`, listed in `GET /v1/models`). `POST
/v1/images/generations` accepts a honkbox extension triple — `controlnet`
(catalog slug/alias) + `control_image` (base64 PNG) + optional
`controlnet_strength` in `(0, 1]` (default 0.9; out-of-range is a 400,
never clamped) — and validates it (half-specified pairs, undecodable
base64 / non-PNG bytes, and unknown slugs are 400s). A valid pair is
applied by the sd-cpp pipeline: the weights load with the model context
and the conditioning image (sent preprocessed, e.g. a depth or edge map;
the engine resizes it) steers the render. A backend that cannot apply
control (the stub) is answered with an explicit 501
(`controlnet_unsupported`) rather than silently ignored.

## Vision chat (mmproj, REQ-8)

An LLM weights file with a sibling mmproj
projector (`mmproj*.gguf` in the same directory, the HF multimodal-repo
convention) is discovered automatically at catalog time — such entries
report `capabilities.vision: true` and `modalities: ["text", "vision"]`
in `GET /v1/models`, and the chat pipeline loads the projector alongside
the model. All three chat routes (`POST /v1/chat`, `/v1/chat/stream`,
`/v1/chat/completions`) then accept OpenAI-style content-parts arrays:
`image_url` parts must be `data:<mime>;base64,...` URIs (the server
decodes the bytes inline; remote URLs are never fetched — an explicit
400 `image_url_unsupported`), spliced as media markers at the part's
position, and evaluated through llama.cpp's mtmd projector before the
text tokens. Image content on a text-only model (no sibling projector,
no operator `capabilities.vision = true` override) is a named 400 —
`vision_unsupported` (`OpenAI` envelope on the completions route,
plain text on the native routes), never a silently flattened image. A
projector file itself (`mmproj*.gguf`) is listed by the scan but is not
chat-capable (`capabilities.chat: false`) — it is a component, not a
model. Undecodable payloads answer 400 `invalid_image_data`.
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
