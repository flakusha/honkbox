# TASK: Per-request LLM keepalive TTL knob

Status: proposed
Priority: P2
Epic: epic-llm-expansion
Owner: unassigned
Acceptance: chat and embeddings routes accept additive `keep_alive`, clamp it through `touch`/`max_ttl`, and project `ttl_secs`/`expires_at`/`requested_keep_alive` with unchanged drain semantics.
Verification: `cargo test --locked -p honkbox-server keep_alive`
Links: `.tmp/research/residency-extended-fields-2026-11-25.md` Candidate 4, `crates/server/src/swap.rs`, `crates/server/src/routes/chat.rs`
Resources: per-test tmp dirs; `StubLlm` only; no GPU, no model downloads

## Change

- Accept additive `keep_alive` (`string | integer | null`) on chat + embeddings routes: seconds number, duration string, `0`/`"0"` unloads after request, `-1`/`null` pins for session, absent keeps entry default; reject garbage with 400, never silently default.
- Clamp per-request values through existing `touch` (`max(remaining, requested)`, capped at `max_ttl`); pinned entries ignore TTL; `keep_alive: 0` removes the map entry subject to in-flight `Arc` drain honesty (never blocks on holders).
- Project additive `ttl_secs`, `expires_at` (projected expiry; null/absent when pinned), and `requested_keep_alive` (last override echoed verbatim, null if none) per resident.

## Acceptance criteria

- [ ] Per-request override extends window within `max_ttl`; `0` evicts after request modulo drain honesty; `-1`/`null` pins for session.
- [ ] `expires_at` reflects the effective window; pinned entries show null/absent.
- [ ] Extend, `0`-evict, and clamp cases covered; garbage input 400s.
- [ ] No new eviction math, no `globalTTL`/`unloadTimeout` knobs, no route additions, drain semantics unchanged.
- [ ] No GPU, checkpoint, download, or new dependency in any test.

Licensing: neural-network weight/tokenizer license and inference-use applicability are the operator's responsibility; honkbox records digest/source and never clears, grants, or assumes a license.

[UNKNOWN] No demonstrated need for a third global TTL knob beyond `llm_ttl` + `max_ttl` + `llm_max_loaded`; not added here.
