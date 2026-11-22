# TASK: EXR format support in the artifact registry

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: `.exr` is a named registry format: magic-byte sniff, scanline and
tiled half-float RGB/RGBA read, write only on bake/export paths, and an LDR
preview artifact (`preview`/`map` kinds) for every EXR. License applicability
is re-verified against `deny.toml` before choosing either a Rust crate or the
existing OIIO external-tool route; exactly one route ships. Verification:
`bun run test:registry`, `bun run deny`, `bun run spdx`. Links:
`.tmp/research/media-transport-federation-2026-11-20.md` §2. Resources:
read-only fixture files; no GPU slot for sniffing.

## Change

- Add the EXR registry row and a small fixture-based read test.
- Reuse the artifact preview/map vocabulary; do not add a second preview path.
- Record the chosen route and license verdict in the ticket before code lands.

## Non-goals

EXR encoder for generation output, animation sequences, remote fetch.
