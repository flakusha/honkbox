# TASK: Voice cloning consent policy

Status: proposed
Priority: P3
Epic: epic-voice-identity
Owner: unassigned
Acceptance: a written consent/license policy exists and any reference voice without recorded consent is rejected with a named error; no cloning route is added by this ticket.
Verification: documentation review + `bun run spdx`; no code gate
Links: `.tmp/research/ReviewedExpansion-VoiceIdentity.md` §Gate before any feature
Resources: none (policy ticket)

## Change

- Require explicit, recorded consent and operator-supplied reference audio;
  forbid cloning from arbitrary STT input or a fetched URL.
- State that `FEAT-voice-clone-job.md` must carry license/consent metadata in
  the bundle sidecar and a VRAM estimate before it can be promoted.
- Name the rejection error (e.g. `voice_reference_consent_required`).

## Acceptance detail

No impersonation path exists by default; policy is discoverable from docs and
from the named error contract.
