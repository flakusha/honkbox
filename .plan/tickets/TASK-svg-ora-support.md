# TASK: Safe SVG/ORA read path

Status: done (2026-11-18, honkbox-svg-ora)
Priority: P2
Epic: epic-binary-formats
Owner: honkbox-svg-ora
Acceptance: SVG and ORA inputs are parsed safely, with scripts/external refs rejected and layers capped before any job is admitted.
Verification: `cargo test --locked -p honkbox-server`, `bun run test:server`
Links: `.tmp/research/ReviewedBinaryFormats.md`, `TASK-svg-artifact-validation.md`, `TASK-ora-output.md`
Resources: per-test SVG/ORA fixtures; no shared renderer state

## Change

- Reuse SVG validation rules from `epic-vector-materials`; ORA uses the
  container safety rules.
- Convert SVG to PNG via `resvg` when a raster input is required; keep the raw
  artifact and hash in the bundle.
- Cap layer count and decoded dimensions for ORA.

## What landed

- `crates/server/src/svg.rs` — the safe SVG read path: one streaming
  `quick-xml` pass (no DTD processing, no entity expansion) that rejects,
  by name, a `<script>` element, an `on*` event handler, an external
  reference (`href`/`src`/`data`/`url(...)` in a style attribute or a
  `<style>` element), a DTD or entity declaration, an oversized source, an
  oversized element tree, an oversized viewport and a document that is not
  well-formed (unclosed or unbalanced elements). `rasterize_svg` renders
  the validated document with `resvg` (compiled WITHOUT the `text`
  feature, so no system font is ever loaded and the pixels are
  byte-identical on every machine) into the shared RGBA8 raster.
- `crates/server/src/ora.rs` — the safe ORA read path: the ORA conventions
  are enforced (the first entry is a STORED `mimetype` holding exactly
  `image/openraster`), `stack.xml` is parsed with `quick-xml`, the canvas
  and the layer count are capped, and every layer is checked (path escape,
  missing file, PNG header dimensions against the pixel cap) BEFORE its
  pixels are decoded. `flatten_ora` composites the layers onto the canvas
  in stack order. The archive itself is read through the bounded
  container reader (`TASK-container-safety`), so the entry-count,
  per-entry, total-size and path rules apply for free.
- `crates/server/src/raster.rs` — the shared decoded-raster type
  (`RasterImage`), so the image-ops seam has ONE shape to consume.
- `crates/server/src/formats.rs` — the registry rows for `svg` and `ora`
  are flipped to `image_ops_input: true` and their budgets
  (decoded pixels, container entries, uncompressed bytes, per-entry bytes,
  vector elements) are the limits the read paths enforce — the registry
  stays the single source of truth for what this build accepts.
- `crates/server/src/routes.rs` — the image-ops decode dispatch: the
  sniffed registry row selects the reader (the `image` crate for the flat
  rasters, `svg::rasterize_svg` for a vector, `ora::flatten_ora` for a
  layered document), and the format-specific safety check runs BEFORE the
  job is registered, so a hostile container is a named 400 with NO job
  and NO bundle.
- `crates/server/tests/svg_safety.rs` (16 tests) and
  `crates/server/tests/ora_safety.rs` (13 tests) — the crafted corpora:
  every fixture is built in memory (no files, no temp dirs, no shared
  state), and every rejection asserts the NAMED error variant.
- `crates/server/tests/image_ops.rs` — the integration cases: an SVG and
  an ORA are rasterized/flattened deterministically, and a malicious SVG
  and a malicious ORA are rejected with no job admitted.

## Acceptance detail

A malicious SVG or ORA is rejected before GPU work; a valid one produces a
deterministic preview or layer set. Verified: the SVG corpus rejects the
script, the event handler, the external reference (attribute, style
attribute and `<style>` element), the DTD, the oversized source, the
oversized element tree, the oversized viewport and the malformed document;
the ORA corpus rejects a wrong first entry, a compressed `mimetype`, a
wrong `mimetype` content, a missing/malformed `stack.xml`, a layer that
escapes the archive root, a missing layer, too many layers, a layer that
claims a huge canvas and a huge canvas — each by its own named variant.
The image-ops integration proves the deterministic output (the same bytes
produce the same output digest) and the pre-admission rejection (a 400
with no `X-Job-Id` and no bundle on disk).
