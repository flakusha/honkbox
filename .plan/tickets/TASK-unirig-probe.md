# TASK: UniRig probe (ML auto-rig for generated meshes)

Status: proposed
Priority: P2
Epic: epic-blender-assets
Owner: unassigned
Acceptance: a generated prop + a humanoid mesh each get a plausible skeleton + weights report; wrapper-license decision recorded.
Verification: fixture meshes; `cargo test --locked -p honkbox-server`
Links: `.tmp/research/gen-flows-props-characters-locations-2026-11-20.md` §2, `FEAT-blender-rigging.md`
Resources: per-test tmp dirs; no shared rig process

## Change

- Upstream VAST UniRig is MIT; loop-lore's ComfyUI wrapper is GPL-3.0 —
  honkbox-side prefers a direct MIT implementation over the wrapper
  (ComfyUI intended-use carve-out covers driving the node, not vendoring it).
- Probe on two fixtures (prop + humanoid): bone plausibility + zero-
  unweighted-vertices assertion, same honesty bar as `FEAT-blender-rigging`.
- Output feeds the filed rig job (weights + bone/weight report format).

## Acceptance detail

- Base-rig-only honesty inherited: no hero-character or animation-ready
  claim. Generative animation (MDM/MotionGPT) stays a separate watch item.
