# TASK: Mark Ash family readiness without claiming availability

Status: proposed
Priority: P2
Epic: epic-sd-native-backend
Owner: unassigned
Acceptance: `models inspect-ash` distinguishes accepted headers, subset key audits, device op admission and actual execution scope for SD1.5, SD2 and SDXL without a whole-family availability boolean.
Verification: `cargo test --locked -p honkbox-server --features ash-sd readiness_markers`, `cargo test --locked -p honkbox-server --features ash-sd`, `cargo check --locked -p honkbox-server --features ash-sd`.
Links: `.tmp/research/ash-promotion-refresh-2026-11-25.md` §3.4; `crates/server/src/main.rs:802-925`; `crates/ash-backend/src/checkpoint_family.rs`.
Resources: tiny synthetic safetensors headers and unique temp paths; marker computation is CPU-only/testable without Vulkan; no model download or GPU reservation.

## Verified 2026-11-25

- **[FACT]** `models_inspect_ash` already prints family, inferred stage widths,
  key audits, missing keys and device op admission. This ticket extends that
  command; it does not create another inspector.
- **[FACT]** SD2 detection/header/spec checks landed, but `generate` accepts
  only SD1.5 (`crates/ash-backend/src/sd.rs:316-325`). SD2 currently falls
  through the SD1.5 key/CLIP audit in `models_inspect_ash`.
- **[FACT]** SDXL has a dedicated inspection audit and an early return; no
  SDXL generation path exists.
- **[FACT]** `experimental_ash_sd` capability truth and HTTP selection are
  separate promotion gaps. This ticket must not change them.

## Dependencies

- None for the marker helper and synthetic-header tests.
- `FEAT-ash-sd2-spec` may later supply a real SD2 key manifest; until then the
  marker must say `keys=family-spec-required`, not reuse SD1.5 readiness.
- `TASK-ash-backend-promote` remains the only owner of selectable-backend and
  capability advertisement.

## Change

Add a small pure marker helper in the existing `models inspect-ash` path and
print one stable line per checkpoint before the detailed audit:

| Family | Required marker meanings |
|---|---|
| SD1.5 | `header=accepted`, `keys=<existing partial-audit result>`, `execution=reduced-sd15-path` |
| SD2 | `header=accepted`, `keys=family-spec-required`, `execution=blocked-spec-only` |
| SDXL | `header=accepted`, `keys=<existing inspection-audit result>`, `execution=inspection-only` |

Retain useful shape/key/device details, but label key counts as subset audits.
Do not emit a whole-family `ready=true`; `ops: ready=N/M` remains explicitly a
selected-device admission count, not model readiness.

## Acceptance criteria

- [ ] Synthetic SD1.5, SD2 and SDXL headers produce the exact family and
  execution-scope markers above; wrong-family headers retain named rejection.
- [ ] SD2 never reports SD1.5 CLIP/key readiness or whole-model availability.
- [ ] SDXL inspection success never implies generation support.
- [ ] Marker computation has a CPU-only unit test with no Vulkan device; the
  existing graceful `vulkan: unavailable: ...` diagnostic is preserved.
- [ ] Existing detailed output and `--all-missing` behavior remain useful and
  bounded; no catalog, FeatureFlags, OpenAPI or backend-selection changes.

## Verification

```sh
cargo test --locked -p honkbox-server --features ash-sd readiness_markers
cargo test --locked -p honkbox-server --features ash-sd
cargo check --locked -p honkbox-server --features ash-sd
```

All server-feature commands require the real forwarded feature from
`TASK-ash-backend-promote`; if this ticket lands first, run the marker helper's
crate-level test and leave the server-feature commands to integration. No GPU
or real checkpoint is required.

## Non-goals

Making any family selectable; changing `experimental_ash_sd`; adding a JSON API
or generated snapshot; full SD2/SDXL execution; deriving availability from
header/key counts; inventing a global `ready` score.
