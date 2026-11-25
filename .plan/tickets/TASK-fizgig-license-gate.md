# TASK: Fizgig weight-license and territory gate

Status: done (2026-11-22)
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
https://github.com/shootthesound/Fizgig/blob/e0514d66bba4a7cc2c03eaf5b16a8e9fe9985d1/docs/THIRD_PARTY_NOTICES.md
Resources: TBD at implementation

## Change

- Add per-family weight terms to the catalog (`commercial`, `territory`,
  `gated`, `license_url`, attribution text), reusing `[licenses]` gates.
- Keep face/SAM3/auto-caption paths disabled by default for commercial builds;
  they pull non-commercial research models and the SAM license.
- Pin the Fizgig rev + `requirements.txt` hash in the install docs; never
  vendor GPL-3.0 `minimax/`, `sam3/` or ROCm detector files into honkbox.

## Landed

`licenses::record_for_weight_file` resolves curated rows by file name;
`validate()` runs the shared `license_gate::check` on dit/VAE/text-encoder
before admission, returning named 412 (`model_license_gated`,
`model_license_territory`, `model_license_non_commercial`) with no lease, job
record or subprocess. Uncurated local files keep the local-only posture.
Pinned tag/commit and `requirements.txt` sha256 recorded in the research note;
no upstream Python files entered the tree or link graph. Evidence: new
fixture assertion in `crates/server/tests/licenses.rs`; fake-CLI refusals in
`--test fizgig`; `bun run deny` and `bun run spdx` both green.
