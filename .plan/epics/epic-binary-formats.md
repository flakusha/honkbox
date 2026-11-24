# EPIC: Binary formats and containers

Status: proposed
Priority: P1
Owner: unassigned
Research: `.tmp/research/media-handling-consolidated-2026-11-24.md`,
`.tmp/research/model-identity-license-download-2026-11-25.md`

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
- `TASK-model-license-registry.md` — license registry + product/manifest rendering (done 2026-11-20).
- `TASK-nc-weight-gate.md` — refuse NC-flagged artifacts + stamp manifests (done 2026-11-20).
- `TASK-model-provenance-contract.md` — header/path/digest identity, persisted confidence/reason, component graph, exact license binding and existing-gate territory/revenue result (P1).
- `TASK-download-options-contract.md` — cross-epic dependency from `epic-external-tools`: full revisions, expected digests, staged-tree probe/commit and secret-safe provider options (P1).

## Acceptance

Magic bytes win over extensions; decompression bombs and path traversal are
rejected before extraction; no unbounded base64 body enters a job. Model identity
is evidence-based and persisted: ambiguous/unknown provenance never becomes a
license or redistribution claim, restricted weights are never auto-fetched or
bundled, and multi-file downloads become catalog-visible only after every digest
and component probe passes. Existing Axum/Salvo auth middleware remains the
server-side boundary: downloads/refresh require admin scope, reads require read,
and rejected credentials/scopes cause no registry or filesystem mutation.

## Verification

`bun run test:server`, `cargo test --locked -p honkbox-server`, `bun run lint:rs`;
malformed fixture corpus generated locally.

## Non-goals

Native `.blend`/`.kra`/`.xcf`/`.sbsar` parsers, arbitrary URL fetch, and new
decoder deps without `deny.toml` approval.
