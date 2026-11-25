# TASK: Ash second cross-attention growth level (G2)

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: exactly one growth level lands (second cross-attn pair plus mirror up-projection) with per-tensor goldens and a first-divergence reporter naming the tensor on mismatch
Verification: `cargo test --locked -p honkbox-ash-backend second_crossattn`
Links: `.tmp/research/ash-extended-fields-2026-11-25.md` §1, `TASK-ash-block-level-goldens.md` (sibling, not duplicate — this is the first gated growth level under that harness), `crates/ash-backend/src/unet_real.rs`, https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/models/sd.md, https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
Resources: CPU-only fixtures; canonical reference weights are operator-supplied with recorded digest, never downloaded

## Change

- Add exactly one growth level (G2): a second cross-attention block at the same down resolution as the landed first cross-attn (down-block-1), reusing the landed `CrossWeights` shape with the `TimeWeights` MLP unchanged, plus its mirror up-projection.
- Record per-tensor goldens for every new tensor (shape, dtype, sha256 of f32-LE bytes, mean, std, absmax, first-8 of first row, rng_mode=cpu, reference digest, canonical prompt, seed).
- Add a first-divergence reporter: on mismatch the harness names the first diverging tensor, not just the block.
- No change to down/up depth (still 3), VAE, tokenizer, CFG, or DDIM loop. Full down/up depth (up-block resnets, skip concat, remaining attn layers) is G3+ and out of scope.

## Acceptance criteria

- [ ] G2 goldens recorded from the CPU reference for every new tensor; stub-graph goldens still pass.
- [ ] A deliberately perturbed G2 tensor is reported by tensor name as the first divergence.
- [ ] No G3+ depth (resnets, skip concat, further attn layers) lands in this ticket.
- [ ] No download, network, or unrecorded artifact in any test. [UNKNOWN] exact golden values pending operator-supplied reference digest.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license. Reference weights are operator-supplied; any second-family weights stay gated on the operator's own license review.
