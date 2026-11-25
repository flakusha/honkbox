# TASK: Ash TAEHV seam (gated tracking only)

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: family-key selector offers the tiny-autoencoder SD-line vs video-line choice with cross-shape misuse refused before dispatch; second-family weights recorded blocked pending operator license/family decision, with execution out of scope
Verification: `cargo test --locked -p honkbox-ash-backend taehv_seam`
Links: `.tmp/research/ash-extended-fields-2026-11-25.md` §4, `TASK-ash-taesd-decoder.md` (sibling, not duplicate — this adds the family-key seam and blocked marker, not decoder execution), https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/models/taesd.md, https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/backend.md
Resources: CPU-only synthetic shape fixtures; no weights, no GPU, no downloads

## Change

- Add a `vae_decoder` family-key selector with exactly two values (SD-line tiny decoder vs video-line tiny decoder); unknown value → 400.
- Carry per-family weight-shape key maps; cross-shape misuse (SD-line weights presented to the video-line path or vice versa) is refused before dispatch, naming the mismatched family.
- Record second-family weights as blocked (`blocked_weight: true` with license/family-decision reason) so admission refuses them even when the key map parses.

## Acceptance criteria

- [ ] Selector accepts exactly the two family keys and refuses unknown values.
- [ ] Cross-shape misuse is refused before dispatch with the family named.
- [ ] Second-family weights are recorded blocked and refused at admission.
- [ ] Second-family execution is explicitly excluded — no decode path, no parity, no fixtures. [UNKNOWN] unblock criteria pending operator license/family decision.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license. Second-family weights are recorded blocked pending the operator's own license review.
