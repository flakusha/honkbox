# EPIC: Binary formats and containers

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/ReviewedBinaryFormats.md`

## Scope

One format registry, safe artifact transport, and container limits for images,
video frames, SVG/ORA, GLB/splats, and opaque project blobs. Advertise read/write
capabilities from compiled code and configured tools, not from README prose.

## Tickets

- `TASK-format-registry.md` — name/MIME/magic/limits registry (P1).
- `TASK-format-capability-advertisement.md` — expose formats in capabilities (P1).
- `TASK-container-safety.md` — zip/ORA/GLB entry + decompression caps (P1).
- `TASK-binary-artifact-api.md` — chunked upload + Range download (P2).
- `TASK-svg-ora-support.md` — safe SVG/ORA read path (P2).
- `TASK-model-format-validation.md` — GLB/splat validation gate (P2).

## Acceptance

Magic bytes win over extensions; decompression bombs and path traversal are
rejected before extraction; no unbounded base64 body enters a job.

## Verification

`bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run lint:rs`;
malformed fixture corpus generated locally.

## Non-goals

Native `.blend`/`.kra`/`.xcf`/`.sbsar` parsers, arbitrary URL fetch, and new
decoder deps without `deny.toml` approval.
