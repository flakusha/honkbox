# TASK: Make the Ash backend compile-reachable and HTTP-selectable

Status: Done
Priority: P0
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: `serve --sd-backend ash-sd` and CLI `generate` share one compile-reachable Ash constructor; absent-feature builds reject Ash selection and do not advertise it; generated API snapshots match; the epic records crate/CLI/HTTP states separately.
Verification: `cargo check --locked -p honkbox-server --features ash-sd`, `cargo test --locked -p honkbox-server --features ash-sd`, `cargo test --locked -p honkbox-server`, `bun run codegen:check`, `bun run check`; one opted-in real-checkpoint smoke render on one exclusive device.
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §2.1/§3.1, `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md` §2-§5, `FEAT-ash-vulkan-backend.md`, `BUG-ash-sd15-numerics.md`, `TASK-ash-verification-gates.md`.
Resources: owner-controlled merge/promotion; normal server tests for feature/capability truth; one exclusive GPU lease only for the opt-in smoke render.

## Verified master baseline (2026-11-25)

- **[FACT]** `crates/ash-backend` implements reduced SD1.5 generation, SD2 header/spec acceptance, SDXL inspection and shader provenance annotations. The provenance checker exists but is not wired into `bun run spdx`; this is crate implementation, not HTTP availability.
- **[FACT]** `crates/server/src/main.rs:196,225,976-1023` contains CLI `generate`/diagnostic Ash code behind `cfg(feature = "ash-sd")`.
- **[FACT]** `crates/server/Cargo.toml:88-137` declares only empty `experimental-ash-sd = []`; it declares neither `ash-sd` nor a `honkbox-ash-backend` dependency. The CLI cfg path is therefore not manifest-reachable on master.
- **[FACT]** HTTP `serve` calls `backends::make_sd` (`main.rs:677`); `SdBackend` has only `Stub` and `SdCpp` (`backends.rs:101-128`).
- **[FACT]** `routes/capabilities.rs:212` derives `experimental_ash_sd` from the empty feature, so capability output can claim Ash without a compile-reachable pipeline.
- **[UNKNOWN]** No compile, GPU correctness, image-quality, latency or VRAM result was produced by this planning pass.
- **[FACT]** `BUG-ash-sd15-numerics.md` records two current SD1.5 numerical mismatches: interleaved timestep slots and missing SiLU before ResBlock `emb_layers.1`.
- **[FACT]** `TASK-ash-verification-gates.md` records missing package scripts, unwired shader provenance and a stale CI server-feature check.

## Change

1. Forward one real `ash-sd` feature through `honkbox-server`, add the backend dependency, and remove the empty `experimental-ash-sd` feature.
2. Share one constructor between CLI `generate` and HTTP `serve`; extend the existing backend enum rather than growing a second convention. Keep Stub/SdCpp behavior unchanged.
3. Accept `--sd-backend ash-sd` only when that path is compiled. Unsupported builds return a named configuration error before server startup.
4. Derive `experimental_ash_sd` from the real constructor path, not an empty marker. Regenerate `api-types.ts`/OpenAPI only after behavior changes.
5. Land `BUG-ash-sd15-numerics.md` and `TASK-ash-verification-gates.md` first. Add default/feature-on/feature-off selection and capability tests, then run the restored Ash gates. A real checkpoint smoke render remains opt-in evidence; no test downloads or bundles weights.
6. Reconcile `FEAT-ash-vulkan-backend.md` slice history and any branch-only work during the owner-chosen merge. Do not use this ticket to re-file reduced generation, SD2 specs, SDXL inspection, readiness markers, or upscaler spikes.

## Acceptance detail

- [x] Default server build neither accepts `ash-sd` nor advertises it.
- [x] `--features ash-sd` compiles the shared constructor and both CLI/HTTP selection paths.
- [x] HTTP capability output matches actual compiled selection; generated snapshots stay in sync.
- [x] Missing Vulkan capabilities fail by name rather than returning a fake image or silently selecting another backend.
- [x] `BUG-ash-sd15-numerics.md` and `TASK-ash-verification-gates.md` are complete.
- [x] Existing Stub/SdCpp tests pass with and without the feature.
- [ ] Opt-in smoke evidence records checkpoint digest/license, device/driver and output hash; absence of hardware remains an explicit unmeasured result.

## Non-goals

Full SD2/SDXL generation, Ash LLM, new public performance flags, automatic model downloads, replacing the diffusers backend, or claiming parity/throughput without the separate evidence tickets.

**Resolved:** 2026-11-25 registry-driven close: git issue 6de3e1c (registry tip: b6b13e5 Konstantin Fedotov Auto-closed: appended .md marker marks TASK-ASH-BACKEND-PROMOTE done)
