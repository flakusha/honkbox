# honkbox

Local-first generative studio in Rust: Stable Diffusion image generation
plus local LLM chat plus audio (STT/TTS) behind one Axum REST/SSE server,
Vulkan-first GPU with CPU fallback, and a tabbed web UI served over the same
API.

## Prereqs

- Bun >= 1.4.4 (`bun --version`); every JS/TS flow here is bun-only.
- System cargo (see `rust-toolchain.toml`) plus `cargo-watch` for `dev:server`
  (`cargo install cargo-watch`).
- Model dirs (configurable, defaults): LLM `~/.cache/huggingface`, ComfyUI
  `~/git-ai/ai-models/Data/Models`.
- Native build deps for the default `vulkan` feature: `cmake`, a C++ compiler,
  `libclang` (bindgen), Vulkan headers + `glslc` + SPIR-V headers
  (Debian/Ubuntu: `cmake clang libvulkan-dev glslang-tools spirv-headers`).
- `LIBCLANG_PATH`: export ONE working value in every shell before any cargo/bun
  command (bindgen re-runs + rebuilds all of sd.cpp when it changes; codegen and
  tests also build the native backend). If the default LLVM mis-generates
  (`_IO_FILE` size errors), point it at another installed LLVM, e.g.
  `export LIBCLANG_PATH=/usr/lib/llvm/20/lib64`.
- Multi-GPU Vulkan: sd.cpp uses device 0 unless `GGML_VK_VISIBLE_DEVICES` names
  the dGPU index (e.g. `1`); without it generation may be slow or abort with
  context-lost.
- Windows/macOS CI builds use `--no-default-features` (CPU-only sd.cpp; needs
  `cmake` + LLVM for libclang, no Vulkan SDK).

## Quickstart

```sh
bun install
bun run build
bun run dev        # Vite :5173 + server :8080 together
```

Or run the server directly:

```sh
cargo run -p honkbox-server -- serve
cargo run -p honkbox-server -- serve --dev  # serve web-dist from disk

# Short form `hbx` is a shell alias only, not a second binary:
alias hbx=honkbox
```

## Scripts

| Script          | What it does                                        |
| --------------- | --------------------------------------------------- |
| `dev`           | Vite + server together                              |
| `dev:web`       | Vite dev server only (:5173)                        |
| `dev:server`    | Server under `cargo watch` (`serve --dev`)          |
| `build`         | Full build (vendor + web + server)                  |
| `check`         | format + lint + typecheck + lint:rs + gpu-slot audit + codegen:check |
| `format`        | `dprint check`                                      |
| `lint`          | oxlint over `scripts packages`                      |
| `typecheck`     | `tsc --noEmit`                                      |
| `test`          | `bun test` + `cargo test` workspace                 |
| `lint:rs`       | Clippy workspace, warnings denied                   |
| `deny`          | `cargo deny` licenses/bans/advisories               |
| `spdx`          | `reuse lint`                                        |
| `codegen:check` | Generated TS bindings fresh vs `api-types`          |
| `audit:gpu-slot`| Fails when a server test takes the GPU slot without serializing on a lock |
| `ci`            | `check` + `test` + `build`                          |

## Layout

- `crates/api-types` — shared DTOs only (serde + ts-rs + utoipa derives).
- `crates/sd-core` — `SdPipeline` impl, samplers/schedulers.
- `crates/llm-worker` — llama-cpp sessions, GGUF load, token streaming.
- `crates/audio-worker` — whisper STT, Kokoro TTS, Silero VAD.
- `crates/residency` — VRAM ledger + tier presets, `/v1/memory` backing types.
- `crates/registry` — model-dir scan, weight metadata, SQLite catalog cache.
- `crates/recognition-worker` — recognition pipelines (stub + barcode) behind `POST /v1/recognize`.
- `crates/external-tools` — Blender/ComfyUI discovery probes; probed once at startup, `GET /v1/external-tools`.
- `crates/server` — Axum REST+SSE binary, clap CLI, queue, embedded UI, TOML
  config.
- `packages/web-ui` — tabbed frontend (Vite + TS).
- `scripts/` — bun dev/build/codegen helpers run via package scripts.
- `packages/web-ui/src/gen/` — generated TS from `api-types`; do not hand-edit
  (`codegen:check` enforces).

## Config

The whole on-disk state lives under ONE **data dir**, resolved as:

1. `--data-dir` global CLI flag
2. `HONKBOX_DATA_DIR` env
3. `[core] data_dir` in the config file
4. default: `$XDG_STATE_HOME/honkbox`, else `$HOME/.local/state/honkbox` (empty
   or relative XDG/env values are treated as unset)

Derived paths: `<data_dir>/honkbox.toml` (config), `<data_dir>/jobs/`,
`<data_dir>/logs/honkbox.log`, `<data_dir>/registry.sqlite3` (registry cache),
and starter model dirs `<data_dir>/models/{stt,tts}`. The startup log names the
active data dir. Model/discovery roots stay explicit (external weight trees);
temp PNGs stay in `$TMPDIR`.

```toml
[core]
data_dir = "/absolute/path"   # optional; must be absolute

[models]
llm_dirs = ["~/.cache/huggingface"]
comfyui_dir = "~/git-ai/ai-models/Data/Models"

[server]
auth = "none"                 # "none" (default) or "bearer"
queue_wait_secs = 10          # admission-queue bound (REQ-6); 0 = queue off

[auth]
api_key_env = "HONKBOX_API_KEY"   # env var NAME holding the bearer key
```

- Env overrides: `HONKBOX_LLM_DIRS` / `HONKBOX_COMFYUI_DIR` (env wins over
  file), `HONKBOX_QUEUE_WAIT_SECS` for the queue bound.
- Jobs & cancellation: every ADMITTED GPU request is stamped with an
  `X-Job-Id` response header and listed (while in flight) by `GET /v1/jobs`;
  a request refused before admission (bad body, model without the required
  capability) never registers a job and carries no header.
  `POST /v1/jobs/{id}/cancel` — or the REST-shaped `DELETE /v1/jobs/{id}`,
  same contract — marks a job's cooperative-stop token: it actually stops
  multi-image generations (between images; the request then answers 409
  `job_cancelled`) and streaming chat (at the next token boundary);
  single-shot FFI calls (one image, sync chat, embeddings, audio) are not
  interruptible mid-flight and run to completion, which the cancel
  response reports honestly (`{"cancelled": false}`).
- Job events (`GET /v1/events`): a read-only Server-Sent-Events feed of the
  job registry, all job kinds. Three SSE event types, each also
  self-labeled in the JSON `event` field: `job_registered` (a route
  admitted the request; may still be queued), `job_cancelled` (a cancel
  was REQUESTED — not a completion event), and `job_finished` (the job's
  request ended, any outcome). Data shape (`JobEvent`): `id`, `kind`
  (`chat`/`embeddings`/`images`/`transcribe`/`synthesize`/`translations`),
  `model` (when known), `started_at_ms`, `interruptible`, and `cancelled`
  — whether a cancel had been requested at that point (`true` on
  `job_finished` for a token-marked job, which for a non-interruptible job
  still ran to completion). The stream is transition-only: reconnects
  replay nothing (`GET /v1/jobs` is the snapshot), keep-alive comments
  keep the connection warm, and a consumer that falls >256 events behind
  gets an `events_dropped` frame naming the skipped count instead of a
  silent gap.
- ControlNet: `comfy:controlnet/*` weight files are discovered by the model
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
- Vision chat (mmproj, REQ-8): an LLM weights file with a sibling mmproj
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
  Opt-in: `cargo build --features llm-vision` compiles the mtmd path
  (feature `llm-vision` → `honkbox-llm-worker/vision`); without it the
  catalog reports text-only and image parts 400 by name. The gate is a
  vendor packaging caveat, not a missing capability: llama-cpp-sys-4
  0.11.0 static builds cannot link the mtmd bitmap loader (its
  `mtmd-helper.cpp` needs the `vendor-hash` library, which the sys build
  script does not link); once a `[patch.crates-io]` vendor patch adds the
  missing link directive, the feature can move into `default`.
- Auth (`[server] auth`, REQ-7): with `auth = "bearer"`, every route except
  `GET /health`, `GET /metrics`, and the static web UI requires
  `Authorization: Bearer <key>`; failures return a 401 with an OpenAI-shaped
  error body. The key VALUE is read at request time from the env var named by
  `[auth] api_key_env` (rotation needs no restart) and is never logged.
  Fail-closed startup: `auth = "none"` with a non-loopback `--host`
  (`127.0.0.0/8`, `::1`, `localhost` are loopback) refuses to start unless
  `honkbox serve --allow-unauthenticated-bind` is passed.
- Logs: `RUST_LOG` (default `info`); `--log-file <path>` > `HONKBOX_LOG_FILE` >
  `<data_dir>/logs/honkbox.log` (stderr human-readable, file JSONL).
- Registry cache: `HONKBOX_REGISTRY_DB` (path; `off`/empty disables) >
  `<data_dir>/registry.sqlite3`.
- Busy slot (REQ-6): when one job holds the GPU slot, the NEXT request
  waits up to `[server] queue_wait_secs` for it to free (a waiting client
  sees latency, not an error); any further concurrent request gets the
  immediate 429 (`queue_full`) and a waiter past its bound gets 429
  (`queue_timeout`), both with a `Retry-After` hint. `0` restores the old
  fail-fast 429 for every request. Audio: `/v1/audio/transcriptions`
  `response_format` also accepts `srt` / `vtt`, and
  `POST /v1/audio/translations` (multipart `file`, `model?`,
  `response_format?`) transcribes with whisper's translate task — output
  always English.

## Migrating from sd-rip

The binary is now `honkbox` (`sd-rip` is gone); the sd-rip compat release is
over — old paths and `SD_RIP_*` vars are no longer read. Move your state by hand
to the new default location:

```bash
D="${XDG_STATE_HOME:-$HOME/.local/state}/honkbox"
mkdir -p "$D"
mv ~/.honkbox/jobs "$D/jobs"
mv ~/.honkbox/logs "$D/logs"
mv ~/.honkbox/models "$D/models"              # if present
mv ~/.honkbox/registry.sqlite3 "$D/"           # if present
mv ~/.honkbox/honkbox.toml "$D/"
# legacy sd-rip users first: same layout under ~/.sd-rip
```

Or set `--data-dir`/`HONKBOX_DATA_DIR` to an existing `~/.honkbox` instead of
moving anything. Browser model picks keep migrating themselves: `sd-rip.*`
localStorage keys are read once, then rewritten under `honkbox.*`.

## Docs

`AGENTS.md` is the authoritative repository contract and `.plan/` tracks work
items. The API reference is generated from the server source and served live
at `GET /v1/openapi.json` (also checked in under `packages/web-ui/src/gen/`)
— read it instead of a copied endpoint list; it is drift-gated by
`bun run codegen:check`. `bun run doc:rs` builds the Rust API docs.
`.tmp/research/Reviewed*.md` is the local design record (gitignored scratch on
this machine, not a published doc).

## License

`AGPL-3.0-or-later` (`LICENSES/AGPL-3.0-or-later.txt`); compliance via
`REUSE.toml` — no per-file license headers.
