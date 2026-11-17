# honkbox

Local-first generative studio in Rust: Stable Diffusion image/video generation
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
| `check`         | format + lint + typecheck + lint:rs + codegen:check |
| `format`        | `dprint check`                                      |
| `lint`          | oxlint over `scripts packages`                      |
| `typecheck`     | `tsc --noEmit`                                      |
| `test`          | `bun test` + `cargo test` workspace                 |
| `lint:rs`       | Clippy workspace, warnings denied                   |
| `deny`          | `cargo deny` licenses/bans/advisories               |
| `spdx`          | `reuse lint`                                        |
| `notice`        | Regenerate `NOTICE` via `cargo about`               |
| `codegen:check` | Generated TS bindings fresh vs `api-types`          |
| `ci`            | `check` + `test` + `build`                          |

## Layout

- `crates/api-types` — shared DTOs only (serde + ts-rs + utoipa derives).
- `crates/sd-core` — `SdPipeline` impl, samplers/schedulers, video job types.
- `crates/llm-worker` — llama-cpp sessions, GGUF load, token streaming.
- `crates/audio-worker` — whisper STT, Kokoro TTS, Silero VAD.
- `crates/residency` — VRAM ledger + tier presets, `/v1/memory` backing types.
- `crates/registry` — SQLite schema, model-dir scan, hf-hub downloads.
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

[auth]
api_key_env = "HONKBOX_API_KEY"   # env var NAME holding the bearer key
```

- Env overrides: `HONKBOX_LLM_DIRS` / `HONKBOX_COMFYUI_DIR` (env wins over
  file).
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

`.tmp/research/Reviewed*.md` are the design record (baseline, crates, model
config, logging, licensing); read them before changing architecture.

## License

`AGPL-3.0-or-later` (`LICENSES/AGPL-3.0-or-later.txt`); compliance via
`REUSE.toml` — no per-file license headers.
