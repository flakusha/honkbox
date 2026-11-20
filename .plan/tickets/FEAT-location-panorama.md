# FEAT: Location panorama job (text → equirect sky)

Status: proposed
Priority: P1
Epic: epic-3d-capture
Owner: unassigned
Acceptance: a prompt produces an equirect panorama artifact loop-lore can use as a sky; cheapest location value, one small model.
Verification: fixture prompt run; `cargo test --locked -p honkbox-server`
Links: `.tmp/research/gen-flows-props-characters-locations-2026-11-20.md` §3, `epic-cross-app-integration.md`
Resources: per-test tmp dirs; weights never committed

## Change

- `kind=location-panorama` supervised job: PanoDiT-class text→panorama
  (478 MB) or SD-based 360 pipeline; output equirect + preview + manifest
  (model rev, seed, sha chain).
- License gate first where the model is Tencent-community (territory +
  MAU); MIT panorama alternatives preferred if they verify.
- loop-lore consumes directly as sky/HDRI; no scene graph in v1.

## Acceptance detail

- Seam quality judged by loop-lore sky use, not by metric. Full scenegen
  stays in `FEAT-hunyuanworld-spike.md`.
