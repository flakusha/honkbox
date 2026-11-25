# TASK: Lavapipe CPU-Vulkan correctness job plus bless-then-read rule

Status: proposed
Priority: P3
Epic: epic-ash-native-families
Owner: unassigned
Acceptance: CI gains a CPU-executed Vulkan shader job (Lavapipe) proving ash correctness with zero GPU runners; golden updates follow bless-then-read (regenerate, human-read diff, then accept).
Verification: `.github/workflows/ci.yml` loads the new job, `bun run check-links`
Links: `.tmp/research/vulkan-colibri-hyperstream-2026-11-25.md` (colibri Lavapipe-CI pattern), `TASK-ash-block-level-goldens.md`, `TASK-ash-verification-gates.md`
Attribution: CI pattern observed in colibri (Apache-2.0, JustVugg/colibri). Method: white-room, no upstream code copied.
Resources: CI runners only; no GPU, checkpoint, or download in any test.

## Change

- Add a CI job running ash shader/parity tests under Lavapipe (software Vulkan); device absence skips nothing, CPU-executed Vulkan is the assertion path.
- Codify bless-then-read: golden regeneration writes a candidate diff the author must read before acceptance; blind re-bless fails review.
- CI stays build-plus-software-Vulkan; no golden or numerical gate may require a physical GPU.

## Acceptance detail

- [ ] Lavapipe job green on a runner with no GPU; log names the software driver.
- [ ] A golden change without a read diff is rejectable by inspection (candidate artifact preserved).
- [ ] No GPU runner, checkpoint download, or network fetch added to CI.
