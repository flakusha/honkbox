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
  command, using the newest installed LLVM library directory; bindgen and native
  rebuilds depend on it. See [configuration](docs/guide/configuration.md#native-build-environment).
- Multi-GPU Vulkan can select the wrong GPU or fail with context-lost; see the
  [Vulkan caveat](docs/guide/media-capabilities.md#multi-gpu-vulkan).
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

## Local install

```sh
bun install
bun run install:local       # vendor sync + web build + release build, then install
honkbox serve               # via ~/.local/bin/honkbox
```

`install:local` builds the release feature set and installs the binary plus a
command link. See [local install](docs/guide/configuration.md#local-install) for
layout, relocation, installer flags, Windows fallback, and uninstall rules, and
[release feature flags](docs/guide/media-capabilities.md#release-feature-flags)
for the `/v1/capabilities` values.

## Scripts

| Script          | What it does                                        |
| --------------- | --------------------------------------------------- |
| `dev`           | Vite + server together                              |
| `build`         | Full build (vendor + web + server)                  |
| `install:local` | Release build + install under XDG data dir + `~/.local/bin` link |
| `check`         | format + lint + typecheck + lint:rs + gpu-slot audit + codegen + hooks + docs gates |
| `test`          | `bun test` + `cargo test` workspace                 |
| `ci`            | `check` + `test` + `build`                          |

Full list with flags lives in [`package.json`](package.json) `scripts`.

## Layout

- `crates/api-types` — shared DTOs only (serde + ts-rs + utoipa derives).
- `crates/sd-core` — `SdPipeline` impl, samplers/schedulers.
- `crates/llm-worker` — llama-cpp sessions, GGUF load, token streaming.
- `crates/audio-worker` — whisper STT, Kokoro TTS, Silero VAD.
- `crates/recognition-worker` — barcode and OCR recognition pipelines.
- `crates/external-tools` — Blender/ComfyUI discovery probes; probed once at startup, `GET /v1/external-tools`.
- `crates/server` — Axum REST+SSE binary, clap CLI, queue, embedded UI.
- `crates/registry` — model-dir scan, weight metadata, SQLite catalog cache.
- `crates/residency` — VRAM ledger + tier presets, `/v1/memory` backing types.
- `packages/web-ui` — tabbed frontend (Vite + TS); `src/gen/` is generated,
  do not hand-edit (`codegen:check` enforces).
- `scripts/` — bun dev/build/codegen helpers run via package scripts.

See [`AGENTS.md`](AGENTS.md) for the authoritative repository contract.

## Docs

- [Configuration](docs/guide/configuration.md) — data dir, model roots, env overrides, auth, logs
- [Operations](docs/guide/operations.md) — jobs, cancel, SSE events, GPU topology, ComfyUI surface
- [Media capabilities](docs/guide/media-capabilities.md) — ControlNet, vision chat, audio formats, release flags, Vulkan caveats
- [Migrating from sd-rip](docs/guide/migrating.md) — moving state to the new layout
- [Docs shipping review](docs/meta/docs-shipping.md) — why plain `.md` is canonical (vs Pages/VitePress)
- [Docs index](docs/README.md) — authority order, structure, sources
- Machine-readable API: live at `GET /v1/openapi.json`, checked in under
  `packages/web-ui/src/gen/` (drift-gated by `bun run codegen:check`); HTML
  type reference via `bun run docs:api` (served at `/api-docs/index.html`);
  Rust API via `bun run doc:rs`.

## License

`AGPL-3.0-or-later` (`LICENSES/AGPL-3.0-or-later.txt`); compliance via
`REUSE.toml` — no per-file license headers.
