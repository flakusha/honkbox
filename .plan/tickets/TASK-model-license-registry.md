# TASK: Model-license registry + product-license rendering

Status: done (2026-11-20, license-registry wave)
Priority: P1
Epic: epic-binary-formats
Owner: unassigned
Acceptance: every registry model slug carries a machine-readable license record rendered into capabilities, manifests and NOTICE.
Verification: `cargo test --locked -p honkbox-server`, `bun run spdx`, `reuse lint`
Links: `.tmp/research/product-licenses-include-vs-api-2026-11-20.md` §2, `REUSE.toml`, `TASK-model-format-validation.md`
Resources: per-test model dirs; no shared validator cache

## Change

- Record per slug: `{ license, license_url, territory, commercial_terms,
  gated, source_url, sha256, attribution_text }`; weights never committed
  (HF cache / `extra_model_paths` sharing only).
- Render into `/v1/capabilities` + bundle manifests (license ids beside
  existing version/hash fields) + `NOTICE` (extend `about.hbs` with a
  Model-weights section next to Vendored-sources).
- Territory refusal: unlicensed jurisdiction = named job error (gated-
  encoder 412 pattern), never silent skip. Missing token likewise.

## Acceptance detail

- A slug without a license record is unservable (fail-closed).
  loop-lore's `assets-attribution.md` rows join on sha 1:1.
