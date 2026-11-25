# TASK: Ash diffusion sampler default table

Status: proposed
Priority: P2
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: family→(sampler, scheduler, steps, CFG) default table lands with silent-mismatch refusal; no new schedulers, kernels, or RNG work
Verification: `cargo test --locked -p honkbox-ash-backend sampler_default_table`
Links: `.tmp/research/ash-extended-fields-2026-11-25.md` §5, `TASK-ash-diffusion-scheduler-host.md` (sibling, not duplicate — this adds the recipe table and refusal on top of those host schedulers), https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/sampling.md, https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
Resources: CPU-only synthetic fixtures; no GPU, no weights, no downloads

## Change

- Add a family→(sampler, scheduler, steps, CFG) default table seeded with the canonical SD1.5 row (first-recipe sampler, discrete scheduler, 20 steps, CFG 7.5); future-family rows are placeholders gated on that family's lane existing.
- Refuse silent mismatch by name at validation: any sampler/scheduler pair not yet landed as host math is refused naming the missing kernel — never silently substituted with the default.
- No new schedulers, kernels, or RNG changes; scheduler host math stays owned by the scheduler-host ticket.

## Acceptance criteria

- [ ] SD1.5 family resolves to its default row without caller-supplied sampler args.
- [ ] Unlanded sampler/scheduler names are refused by name, never defaulted silently.
- [ ] No new scheduler, kernel, RNG, or download in this ticket. [UNKNOWN] future-family rows pending those lanes.

Non-goals: `TASK-ash-sampling-contract.md` is an LLM-token sampling contract (temperature/top-p/top-k) and a false cognate — it is explicitly not the diffusion recipe table and is neither closed nor reused here.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license. Table only; any operator-supplied weights stay gated on the operator's own license review.
