# TASK: Device selection ranking review

Status: proposed
Priority: P3
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: DeviceType ranking decision re-examined with observed evidence; either ranking changes or the decision is documented as correct. deviceUUID stable tie-break evaluated alongside the P1 instance-extensions work.
Verification: research note + tests
Links: `.plan/tickets/TASK-device-capacity-gates.md`
Resources: `crates/ash-backend/src/device.rs`; enumeration-order tie documented at HEAD 5f379b7

## Change

- Current `DeviceType` ranking lets Virtual outrank Integrated: a VM
  passthrough of an iGPU beats a bare-metal iGPU. Re-examine this once real
  backends exist, informed by observed latency/VRAM data rather than device
  type alone.
- Evaluate the deviceUUID stable tie-break for identical GPU models
  (select_device currently resolves ties by enumeration order, documented on
  `select_device`) alongside the P1 instance-extensions work.

## Acceptance detail

A decision is recorded citing observed evidence; it either changes the
ranking (with updated tests) or documents the current ranking as correct.
