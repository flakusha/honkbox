# sd-rip

Local-first generative studio in Rust: Stable Diffusion image/video generation plus local LLM chat plus audio (STT/TTS) behind one Axum REST/WS server, Vulkan-first GPU with CPU fallback, and a tabbed web UI served over the same API.

## Prereqs

- Bun >= 1.4.4 (`bun --version`); every JS/TS flow here is bun-only.
- System cargo (see `rust-toolchain.toml`) plus `cargo-watch` for `dev:server` (`cargo install cargo-watch`).
- Model dirs (configurable, defaults): LLM `~/.cache/huggingface`, ComfyUI `~/git-ai/ai-models/Data/Models`.

## Quickstart

```sh
bun install
bun run build
bun run dev        # Vite :5173 + server :8080 together
```

Or run the server directly:

```sh
cargo run -p sd-rip-server -- serve
cargo run -p sd-rip-server -- serve --dev  # serve web-dist from disk
```

## Scripts

| Script | What it does |
|---|---|
| `dev` | Vite + server together |
| `dev:web` | Vite dev server only (:5173) |
| `dev:server` | Server under `cargo watch` (`serve --dev`) |
| `build` | Full build (vendor + web + server) |
| `check` | format + lint + typecheck + lint:rs + codegen:check |
| `format` | `dprint check` |
| `lint` | oxlint over `scripts packages` |
| `typecheck` | `tsc --noEmit` |
| `test` | `bun test` + `cargo test` workspace |
| `lint:rs` | Clippy workspace, warnings denied |
| `deny` | `cargo deny` licenses/bans/advisories |
| `spdx` | `reuse lint` |
| `notice` | Regenerate `NOTICE` via `cargo about` |
| `codegen:check` | Generated TS bindings fresh vs `api-types` |
| `ci` | `check` + `test` + `build` |

## Layout

- `crates/api-types` — shared DTOs only (serde + ts-rs + utoipa derives).
- `crates/sd-core` — `SdPipeline` impl, samplers/schedulers, video job types.
- `crates/llm-worker` — llama-cpp sessions, GGUF load, token streaming.
- `crates/audio-worker` — whisper STT, Kokoro TTS, Silero VAD.
- `crates/residency` — VRAM ledger + tier presets, `/v1/memory` backing types.
- `crates/registry` — SQLite schema, model-dir scan, hf-hub downloads.
- `crates/server` — Axum REST+WS binary, clap CLI, queue, embedded UI, TOML config.
- `packages/web-ui` — tabbed frontend (Vite + TS).
- `scripts/` — bun dev/build/codegen helpers run via package scripts.
- `packages/web-ui/src/gen/` — generated TS from `api-types`; do not hand-edit (`codegen:check` enforces).

## Config

```toml
[models]
llm_dirs = ["~/.cache/huggingface"]
comfyui_dir = "~/git-ai/ai-models/Data/Models"
```

- Env overrides: `SD_RIP_LLM_DIRS` / `SD_RIP_COMFYUI_DIR` (env wins over file).
- Logs: `RUST_LOG` (default `info`); `--log-file <path>` > `SD_RIP_LOG_FILE` > `~/.sd-rip/logs/sd-rip.log` (stderr human-readable, file JSONL).

## Docs

`.tmp/research/Reviewed*.md` are the design record (baseline, crates, model config, logging, licensing); read them before changing architecture.

## License

`AGPL-3.0-or-later` (`LICENSES/AGPL-3.0-or-later.txt`); compliance via `REUSE.toml` — no per-file license headers.
