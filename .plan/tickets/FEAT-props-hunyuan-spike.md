# FEAT: Props spike (Hunyuan3D-2.1 shape + paint)

Status: proposed
Priority: P2
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a prepared image produces a PBR-textured GLB through the Blender recipe + OIIO post-process; license decision recorded.
Verification: shape-only run (10 GB) first; `cargo test --locked -p honkbox-server`, `bun run deny`
Links: `.tmp/research/gen-flows-props-characters-locations-2026-11-20.md` §1, `TASK-blender-bake-extended-maps.md`, `TASK-texture-postprocess-oiio.md`
Resources: per-test tmp dirs; weights never committed (HF cache only)

## Change

- License decision FIRST: Tencent community terms (excl. EU/UK/KR, >1M
  MAU re-license, no training other models, "Powered by" notice) — the
  spike is blocked until it lands in `TASK-model-license-registry.md`.
- uv env + per-job run (shape 3.3B → PaintPBR 2B), output into the existing
  `AssetJob` (file-in → remesh → UV → bake → GLB) + OIIO mip/tile/ORM.
- Record shape-only vs full VRAM (official 10/21/29 GB; Turbo 6/16 GB) and
  which config ran; multi-part props stay out (PartCrafter is separate).

## Acceptance detail

- If the license gate fails, this ticket closes and the MIT lane
  (TripoSR/SG + TRELLIS.2) remains the default prop path.
