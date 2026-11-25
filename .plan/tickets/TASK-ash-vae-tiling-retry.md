# TASK: Ash VAE tiling retry on oversized decode

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: oversized VAE decodes retry as half-size spatial tiles with overlap blend, and inputs that fit decode byte-identical to the un-tiled path
Verification: `cargo test --locked -p honkbox-ash-backend vae_tiling_retry`
Links: `.tmp/research/ash-extended-fields-2026-11-25.md` §2, `TASK-ash-taesd-decoder.md` (sibling, not duplicate — tiling is host orchestration around the landed decode, not a new decoder), https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/backend.md, https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/performance.md
Resources: CPU synthetic fixtures only; no GPU, no weights, no downloads

## Change

- Add host-side tiling retry around the landed VAE decode: attempt full-frame decode first; on allocation failure (or pre-estimate exceeding budget) split into half-size spatial tiles, decode each tile, linear-ramp blend the overlap, reassemble.
- Default tile 512px edge with 64px blend overlap; retry mode `auto` (full-frame first, tile on OOM) with `on`/`off` overrides. Tiling applies to VAE decode only; encode stays full-frame.
- Typed OOM error shape on exhausted retry (`needed_bytes`, `free_bytes`, tile size/overlap, `retry_ok: false`); `retry_ok: true` returns image bytes. Never partial-image 200.

## Acceptance criteria

- [ ] Oversized synthetic decode succeeds via tiled retry where full-frame is refused.
- [ ] Fitting input decodes byte-identical with tiling on vs off.
- [ ] No new kernels, no weights, no GPU, no download in this ticket. [UNKNOWN] optimal tile edge/overlap pending measurement on real decodes.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license. This ticket uses synthetic fixtures only; any operator-supplied weights stay gated on the operator's own license review.
