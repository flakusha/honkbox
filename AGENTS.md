# Repository Guidelines

## Project Overview

`honkbox` — local-first generative studio in Rust: Stable-Diffusion image generation (txt2img today; LLM chat + Whisper STT / Kokoro TTS planned) behind a single Axum server with an embedded Bun/Vite web UI. Vulkan-first with CPU fallback, backed by a vendored `diffusion-rs` (stable-diffusion.cpp FFI). Early-stage: only the SD txt2img path is implemented; LLM/audio workers are doc-stubs. Design record lives in `.tmp/research/Reviewed*.md` — read before architecture changes. License: AGPL-3.0-or-later (REUSE.toml; GPL-family dependencies banned via `deny.toml`).

## Architecture & Data Flow

```
CLI (clap) / HTTP (axum)
  → config::load                    crates/server/src/config.rs   (one data_dir knob: --data-dir > HONKBOX_DATA_DIR > [core] data_dir > $XDG_STATE_HOME/honkbox; config at <data_dir>/honkbox.toml → HONKBOX_* env)
  → registry::scan_roots             crates/registry/src/lib.rs    (model-dir scan; slugs llm:<rel> / comfy:<rel>)
  → residency::global().try_reserve  crates/residency/src/lib.rs   (process-wide single GPU slot, RAII Lease)
  → dyn SdPipeline                   crates/sd-core/src/lib.rs:84  (StubPipeline | SdCppPipeline via diffusion-rs)
  → encode_image → jobs::write_bundle crates/server/src/jobs.rs    (<data_dir>/jobs/<ulid>/, .part + fsync + rename)
```

- **Trait-seam DI is the core pattern**: everything above the pipeline codes against `SdPipeline` (`generate`/`backend_id`); the server swaps constructors at `crates/server/src/main.rs:288-291`. Backend features forward `server/vulkan → sd-core/vulkan`; `experimental-ash-sd` gates the direct-Vulkan skeleton in `crates/ash-backend` (`honkbox-ash-backend`, empty until the server forwards the flag; `generate` still returns an unimplemented error).
- TS types flow one way: `crates/api-types` (serde + ts-rs + utoipa) → generated `packages/web-ui/src/gen/` (never hand-edit); CI gates drift via `bun run codegen:check` and the server `openapi` subcommand.

## Key Directories

| Path | Purpose |
|---|---|
| `crates/server` (`honkbox-server`) | Only binary (`bin` name `honkbox`, shell alias `hbx` documented in README): main.rs (clap subcommands `serve`/`openapi`/`models show`/`generate`), routes.rs, metrics.rs, config.rs, jobs.rs, logging.rs |
| `crates/sd-core` | Diffusion pipeline trait + Stub/SdCpp impls |
| `crates/registry` | Model-dir scanner (std + serde; serde_json only for GGUF/safetensors header metadata; SQLite/hf-hub are planned, NOT built) |
| `crates/residency` | GPU-slot ledger (`LazyLock` global, `Mutex` slot, RAII) |
| `crates/api-types` | DTOs shared Rust↔TS via ts-rs codegen |
| `crates/llm-worker`, `crates/audio-worker` | Zero-dep doc stubs |
| `packages/web-ui` | Bun/Vite frontend; `src/gen/` is generated |
| `scripts/` | Bun helpers: `dev.ts` (Vite :5173 + cargo-watch :8080), `vendor-patches.ts` (fetches `vendor/` from `vendor.manifest.toml` pins, then applies the `vendor_patches/` stack), `codegen-openapi-check.ts` |
| `vendor/` | Fetched scratch (gitignored, never committed) — run `bun scripts/vendor-patches.ts sync` before any cargo command; edit only via `vendor_patches/` + `vendor.manifest.toml` |
| `.tmp/research/Reviewed*.md` | Canonical design docs (~25) |

## Development Commands

```bash
bun run dev            # Vite + cargo-watch server together
bun run check          # dprint + oxlint + tsc + clippy + codegen:check
bun run test           # bun test --isolate && cargo test --workspace --locked
bun run ci             # check + test + build (full local gate)
bun run lint:rs        # cargo clippy --workspace --all-targets -- -D warnings
bun run deny           # cargo deny check licenses bans advisories
bun run spdx           # reuse lint
cargo run -p honkbox-server -- serve --dev   # server alone; openapi/models show subcommands too
```

Linux native deps (CI parity): `cmake clang libvulkan-dev glslang-tools spirv-headers`; `LIBCLANG_PATH` must be exported in every shell.

## Code Conventions & Common Patterns

- **Safety**: workspace-wide `unsafe_code = "forbid"`; clippy `all + pedantic`, warnings denied in CI.
- **Errors**: hand-rolled enums with `Display` + `Error` (e.g. `SdError`, residency `Busy`); CLI paths return `Result<_, String>` + exit(1). No anyhow/thiserror.
- **Async**: tokio only inside `crates/server`; worker crates deliberately sync.
- **State**: `LazyLock` global singletons (residency), axum `with_state(AppConfig)`.
- **Formatting**: dprint (width 100, double quotes) for TS/JSON; `cargo fmt` for Rust; TS strict + `noUncheckedIndexedAccess` + `verbatimModuleSyntax`; oxlint correctness=error.
- **Stub discipline**: unbuilt features (registry SQLite, llm/audio workers, `.plan/`) exist as zero-dep crates/docs — check what's actually implemented before assuming README scope.

## Important Files

- `Cargo.toml` — workspace (edition 2024, resolver 3, `crates/*`), pins `diffusion-rs ="=0.1.20"` via `[patch.crates-io]` → `vendor/`
- `package.json` — all Bun scripts (single entry point for dev/QA commands)
- `rust-toolchain.toml` — Rust 1.99.0 (minimal profile + clippy/rustfmt/rust-analyzer)
- `crates/sd-core/src/lib.rs:84` — `SdPipeline` trait (the seam)
- `crates/server/src/main.rs` — CLI + pipeline construction (line ~288)
- `vendor.manifest.toml` — fetch pins (full-SHA revs) + ordered vendor patch stack
- `.github/workflows/ci.yml` — gate job list; release build requires all gates green

## Runtime/Tooling Preferences

- **Bun ≥1.4.4 only** for JS (no Node/npm/yarn); `bun install --frozen-lockfile` in CI.
- Rust via `rust-toolchain.toml` (rustup-managed); all cargo invocations use `--locked`.
- Pre-commit hooks in `.githooks/` — activate with `git config core.hooksPath .githooks` (dprint fmt + restage, oxlint, `sh -n`).
- Planning tooling (`.plan/` tickets with `TASK|FEAT|BUG-<slug>.md` + `Status/Priority/Epic` headers, giwt worktrees) is **specified but not yet adopted** — see `.tmp/research/ReviewedDocs-Ticketing.md`.

## Testing & QA

- Rust: inline `#[cfg(test)]` modules per crate + integration tests `crates/server/tests/{health,models,metrics,cancel}.rs` (axum `tower::ServiceExt::oneshot`; use `#[path = "../src/routes.rs"]` + `#[path = "../src/metrics.rs"]` since server is bin-only).
- TS: `bun test --isolate --pass-with-no-tests` (no `*.test.ts` files yet).
- Per-crate: `bun run test:core|test:server|test:registry|…` (each `cargo test --locked -p honkbox-<crate>`).
- CI matrix: `test-rs` 3-OS (Linux Vulkan default features; Windows/macOS `--no-default-features` CPU-only) + `lavapipe` CPU-Vulkan smoke + lcov coverage (collected, **no threshold gate**).
- Parity harness: `crates/sd-core/tests/parity.rs` — structural backend agreement (PNG magic + decode, exact dimensions, `MAX_EDGE` cap, stub seed determinism; byte-level cross-backend equality is impossible without GPU determinism and is NOT asserted). Default runs skip the real backend; `bun run test:parity` (`-- --include-ignored`) exercises `SdCppPipeline` and needs `HONKBOX_PARITY_MODEL=path` to a local single-file checkpoint (no model download in CI; ci.yml untouched — the gap stays opt-in locally).
- License gates: `deny.toml` (GPL/LGPL/AGPL incoming denied), `reuse lint`, `cargo about generate about.hbs -o NOTICE`; `doc` job runs `RUSTDOCFLAGS=-D warnings`.
