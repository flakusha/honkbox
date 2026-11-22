# TASK: Fizgig weight-license and territory gate

Status: proposed
Priority: P2
Epic: epic-binary-formats
Owner: unassigned
Acceptance: each Fizgig family's model weights are checked against the
existing catalog license record before a run starts; non-commercial, gated or
territory-excluded models are refused with a named error instead of being
downloaded or loaded. Fizgig's GPL-3.0 source files stay inside the installed
venv and never enter the honkbox tree or link graph.
Verification: license/fixture tests in `crates/server/tests/licenses.rs`,
`bun run deny`, `bun run spdx`
Links: `product-licenses-include-vs-api-2026-11-20.md` §1-4,
`.tmp/research/fizgig-training-python-sidecar-2026-11-22.md` §4.6,
Fizgig `docs/THIRD_PARTY_NOTICES.md`
Resources: TBD at implementation

## Change

- Add per-family weight terms to the catalog (`commercial`, `territory`,
  `gated`, `license_url`, attribution text), reusing `[licenses]` gates.
- Keep face/SAM3/auto-caption paths disabled by default for commercial builds;
  they pull non-commercial research models and the SAM license.
- Pin the Fizgig rev + `requirements.txt` hash in the install docs; never
  vendor GPL-3.0 `minimax/`, `sam3/` or ROCm detector files into honkbox.
