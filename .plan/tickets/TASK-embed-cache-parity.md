# TASK: Embed-cache residency parity in the shared readout

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: embed residents appear in the same residency readout with capacity/TTL/pin/bytes/drain fields and follow the same admit-before-load path, without folding `EmbedCache` into `SwapManager`.
Verification: `cargo test --locked -p honkbox-server embed`
Links: `.tmp/research/residency-extended-fields-2026-11-25.md` Candidate 3, `crates/server/src/swap.rs`, `FEAT-residency-control-plane.md`
Resources: per-test tmp dirs; `StubLlm` only; no GPU, no model downloads

## Change

- Extend the embed-cache entry/readout projection to `EmbedResidentInfo` (`slug`, `loaded_at`, `last_use_secs`, `pinned` default false, `in_use` best-effort via `Arc::strong_count`, `weights_bytes`, `measured` false until harness measures) with a `kind: "embed"` discriminator in the same readout shape.
- Route embed loads through the same admit-before-load check (`TASK-llm-load-plan-admission.md`); embed TTL reuses swap `max_ttl` until a dedicated knob exists.
- No shared-manager refactor: `EmbedCache` and `SwapManager` keep separate load paths and locking; equivalent fields plus shared readout shape only.

## Acceptance criteria

- [ ] Embed entries project all 7 fields with the `kind: "embed"` discriminator (additive).
- [ ] Embed loads follow the same admit-before-load path as chat loads.
- [ ] No generic-manager merge, no per-slug embed pin config key, no cross-request KV column.
- [ ] No GPU, checkpoint, download, or new dependency in any test.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.

[UNKNOWN] Whether embed evict-on-conflict needs a reason distinct from the swap `Conflict` reason; provisionally reuse one vocabulary.
