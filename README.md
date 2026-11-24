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
  command. If the default LLVM mis-generates (`_IO_FILE` size errors), point
  it at another installed LLVM.
- Multi-GPU Vulkan: sd.cpp uses device 0 unless `GGML_VK_VISIBLE_DEVICES` names
  the dGPU index (e.g. `1`).
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

Release set (loop-lore 1.0.0): `install:local` builds with
`--features release` (ocr-ocrs + audio-vad + audio-cpp; llm-vision,
vulkan, and barcode are default-on), so ONE command yields every
`/v1/capabilities` flag. Dev builds stay lean
(`cargo build -p honkbox-server` skips the heavy audio.cpp CMake build).
Installer flags: `--prefix DIR`, `--bindir DIR` (absolute paths), `--dry-run`,
`--no-build`, `--force`; `bun run uninstall:local` removes the link
(`--purge` also removes the prefix, never the state dir).

## Scripts

| Script          | What it does                                        |
| --------------- | --------------------------------------------------- |
| `dev`           | Vite + server together                              |
| `build`         | Full build (vendor + web + server)                  |
| `install:local` | Release build + install under XDG data dir + `~/.local/bin` link |
| `check`         | format + lint + typecheck + lint:rs + gpu-slot audit + codegen + docs gates |
| `test`          | `bun test` + `cargo test` workspace                 |
| `ci`            | `check` + `test` + `build`                          |

Full list with flags lives in [`package.json`](package.json) `scripts`.

## Layout

- `crates/api-types` — shared DTOs only (serde + ts-rs + utoipa derives).
- `crates/sd-core` — `SdPipeline` impl, samplers/schedulers.
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
- [Media capabilities](docs/guide/media-capabilities.md) — ControlNet, vision chat, audio formats
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
