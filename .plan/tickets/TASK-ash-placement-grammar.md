# TASK: Ash per-module placement grammar

Status: proposed
Priority: P3
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: `te=`/`vae=`/`diffusion=`/`all=` placement grammar parses and records the requested/admitted/residency triple, with 409 on unknown module and single-device behavior unchanged without the grammar
Verification: `cargo test --locked -p honkbox-server placement_grammar`
Links: `.tmp/research/ash-extended-fields-2026-11-25.md` §3, `TASK-device-selection-rank-review.md` (sibling, not duplicate — this adds the grammar and recorded triple under that ranking policy), https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/backend.md, https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/development/backend_probe.md
Resources: CPU-only fixtures; no GPU, no weights, no downloads

## Change

- Accept per-module placement assignments (`te=<device>`, `vae=<device>`, `diffusion=<device>`, `all=<device>` shorthand); keys unique, unknown module → 409 with the offending key named.
- Record the triple per job: requested (parsed grammar), admitted (single-device policy grant), residency (existing lease kind vocabulary).
- Multi-device spreads parse but admission pins to one device (records request, admits the diffusion device, spills nothing — single-device invariant holds). Without the grammar, single-device admission is byte-for-byte unchanged.

## Acceptance criteria

- [ ] Grammar assignments parse and the recorded triple round-trips per job.
- [ ] Unknown module returns 409 naming the module.
- [ ] Multi-device request is recorded but admitted to one device with no spill.
- [ ] Requests without the grammar follow the exact pre-existing admission path. [UNKNOWN] capacity-estimate constants pending measurement.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license. No weights involved; any operator-supplied weights stay gated on the operator's own license review.
