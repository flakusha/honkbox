# FEAT: HunyuanWorld scene spike (layered mesh world)

Status: proposed
Priority: P3
Epic: epic-3d-capture
Owner: unassigned
Acceptance: not committed — needs the Tencent-license decision, VRAM measurement, and a loop-lore world consumer.
Verification: none until unblocked
Links: `.tmp/research/gen-flows-props-characters-locations-2026-11-20.md` §3, `FEAT-blender-asset-preflight.md`
Resources: none (deferred)

## Change

- `demo_panogen` → `demo_scenegen` (layered mesh export) → Blender
  preflight normalize → GLB + manifest. lite/4090 config measured first.
- Same community-license gate as `FEAT-props-hunyuan-spike.md`.

## Acceptance detail

- Unblock order: license decision → `FEAT-location-panorama.md` proves the
  panorama half → VRAM measurement → this spike. WorldMirror/WorldPlay/
  FlashWorld stay watch-list (licenses unverified).
