# TASK: Device capacity gates in the ledger

Status: done (2026-11-20, ash capacity-gates lane)
Priority: P1
Epic: epic-full-ai-cycle
Owner: unassigned
Acceptance: The future capacity ledger gates kernel dispatch on Vulkan features/limits before model load; dispatch paths cannot exceed reported limits.
Verification: `cargo test --locked -p honkbox-ash-backend --features ash-sd`
Links: `.tmp/research/ash-vulkan-backend-licensing-extensibility-2026-11-22.md`, `.plan/tickets/TASK-resource-ledger-bounded-queue.md`
Resources: PhysicalDeviceInfo limits/features already collected; no new device enumeration

## Change

- `PhysicalDeviceInfo` already collects device `limits` and Vulkan features
  (subgroup size, 8/16-bit storage, shader features) but scores none of them.
- The capacity ledger must consume these as admission gates: check
  feature/limit requirements before any model load is reserved.
- Kernel dispatch paths must be constrained so they cannot exceed the
  reported limits of the selected device.

## Acceptance detail

Admission rejects a model whose feature requirements the selected device
does not report (naming the missing feature), and dispatch never issues a
kernel exceeding a reported limit (workgroup counts, storage ranges).
