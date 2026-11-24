# TASK: Device selection ranking review

**Status:** Not Started
**Priority:** low
**Effort:** Small
**Epic:** epic-full-ai-cycle
**Tags:** ash, vulkan, device
**Caveats:** TOCTOU, security, authorization, parallelism, lock/unlock lifecycle, partial failure, rollback, stale state, data races, idempotency, observability
**Related:** TASK-device-capacity-gates.md

**Summary:**

Re-examine `DeviceType` ranking and the deviceUUID tie-break with observed evidence.

**Context:**

Current ranking lets Virtual outrank Integrated, so a VM passthrough of an iGPU beats a bare-metal iGPU. `crates/ash-backend/src/device.rs` is now on master, but the enumeration-order tie still has no observed latency/VRAM evidence.

**Acceptance Criteria:**

- A decision is recorded citing observed evidence: either ranking changes with updated tests, or the current ranking is documented as correct.
- The deviceUUID stable tie-break is evaluated alongside the P1 instance-extensions work.

**Verification:**

- Research note plus focused device-selection tests.

**Links:**

- `.plan/tickets/TASK-device-capacity-gates.md`
- `crates/ash-backend/src/device.rs`

**Non-goals:**

- No multi-GPU sharding, new backend trait or new `VkError` variant.
