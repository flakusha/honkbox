# TASK: Wire OIIO texture post-process into the server

Status: done (2026-11-20, texture route lane)
Priority: P0
Epic: epic-output-formats
Owner: unassigned
Acceptance: `crates/external-tools` OIIO API has a real server consumer: a postprocess route for baked/generated maps, a capability row, and a named 501 when `oiiotool` is absent.
Verification: `cargo test --locked -p honkbox-server`, `bun run codegen:check`
Links: `TASK-texture-postprocess-oiio.md`, `texture-creation-workflows.md` §6
Resources: per-test tmp dirs; no shared tool process; `oiiotool` absent yields a named 501, never a silent skip

## Change

- Add a postprocess route in the server that drives the already-landed
  `crates/external-tools` OIIO API over baked/generated map bundles (mip chain,
  tiled package, colorspace tags, ORM pack). Reuse that API; do not reimplement
  the argv or sidecar logic in the server.
- Advertise one capability row derived from the live tool probe so clients can
  discover the surface through the capabilities contract. An absent
  `oiiotool`/`maketx` makes the row false and the route returns the named 501.
- Wire the route into the generated OpenAPI surface and job/bundle result so
  post-processed maps reach the export consumer end to end.

## Acceptance detail

- A request against the postprocess route produces the post-processed maps and
  their sidecar annotations in the bundle; the capability row agrees with the
  runtime probe.
- With `oiiotool` unavailable the capability is false and the route fails with
  the named 501 (not a generic 404/500 and not a silent no-op).
- No new Rust dependency; OIIO stays binary-only, matching the Blender probe.
