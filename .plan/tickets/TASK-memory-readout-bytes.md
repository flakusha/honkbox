# TASK: Additive memory-readout byte decomposition

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: existing residency routes gain additive per-entry demand fields plus `eviction_reason`, with drain-honesty tests green and `measured: false` until a harness lands.
Verification: `cargo test --locked -p honkbox-server memory`
Links: `.tmp/research/residency-extended-fields-2026-11-25.md` Candidate 5, `crates/api-types/src/lib.rs`, `crates/server/src/routes/runtime.rs`, `FEAT-residency-control-plane.md`
Resources: per-test tmp dirs; `StubLlm` only; no GPU, no model downloads

## Change

- Add additive per-entry DTO fields on existing routes only (`GET /v1/memory`, `GET /running`): `weights_bytes`, `projector_bytes` (null when no projector), `kv_bytes`, `scratch_bytes` (estimate only), `ram_bytes`, `total_bytes` (server-computed sum; clients never re-sum), `measured` (false until harness), `demand_source` (`estimated`/`clamped`/`measured`), `eviction_reason` (last transition; null when none), plus top-level `budget` (`vram_budget_bytes`, `vram_committed_bytes`, `vram_headroom_bytes`, `oom_near_misses`) null when unmanaged.
- `null` means unknown/unapplicable, `0` keeps its existing unmanaged meaning; `idle_secs` and drain-honesty wording stay byte-identical; regenerate bindings.

## Acceptance criteria

- [ ] Exactly these additive fields; no renames, removals, retypes, or new routes.
- [ ] `total_bytes` equals the server-side sum; `null`-vs-`0` semantics tested.
- [ ] Drain-honesty tests stay green ("not listed" is not "drained"); pin/unload outcomes unchanged.
- [ ] `measured` stays false until a measurement harness exists; no per-device, fragmentation, or perf fields.
- [ ] No GPU, checkpoint, download, or new dependency in any test.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.

[UNKNOWN] Safe default byte headroom for mixed LLM/SD/embedding workloads on 8-16 GiB consumer GPUs; the `budget` object reports headroom but sets no policy.
