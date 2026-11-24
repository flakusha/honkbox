# EPIC: Video support (gated)

Status: proposed
Priority: P2
Owner: unassigned
Research: `.tmp/research/native-video-backend-refresh-2026-11-25.md`,
`.tmp/research/media-video-normalization-and-repair-2026-11-24.md`,
`.tmp/research/media-masters-audio-video-2026-11-25.md`

## Scope

There is no native video generation pipeline, job class or HTTP route in
honkbox today. The pinned stable-diffusion.cpp revision already exposes native
video generation for MiniMax-H3 and LTX-2; the missing work is the safe Rust
seam, bounded output/resource contract, and one real local clip.

Move from this verified backend gap to a real job class only after the current
pin proves one short clip from a license-cleared local checkpoint. No vendor
upgrade is proposed. `FEAT-video-output.md` stays `postponed` until that proof.

## Tickets

- `FEAT-video-output.md` — parent blocker, **postponed** until one current-pin
  clip has exact family/frame/fps/edge and license evidence.
- `TASK-video-ffi-spike.md` — P2, support probe + params/init/generate/free seam
  and one short clip; no HTTP route.
- `TASK-video-job-class.md` — P2, proposed only after video output exists:
  `JobKind::Video`, caps, exclusive lease, TTL and honest cancellation.
- `FEAT-video-i2v-frames.md` — P2, proposed only after the job/output contract:
  canonical PNG frame sequence + sidecar, no mux.
- `FEAT-video-mux.md` — P3, proposed only after frame output: capability-gated
  derived MP4/WebM; prefer supervised FFmpeg over new native codec linkage.

## Dependency order

1. Select a license-cleared local H3 or LTX component set; record its license,
   territory/revenue gate and hashes.
2. Run `TASK-video-ffi-spike.md` against stable-diffusion.cpp `5dcfa79`.
3. Reopen `FEAT-video-output.md` only after one short local clip succeeds.
4. Add the video job class, then I2V/FL2V frame output, then derived muxing.
5. Add route/DTOs and set `FeatureFlags.video` only after the preceding
   contracts are observable and tested.

## Native versus ComfyUI boundary

Native `sd-core` owns direct H3/LTX generation after the spike. ComfyUI remains
the external workflow path for video-to-audio, Foley, source-audio reuse and
custom-node workflows; those templates do not unblock native video. FFmpeg and
FFprobe remain supervised external media tools, not inference backends.

- `FEAT-comfy-video-audio-templates.md` — proposed external ComfyUI adapter for
  V2A/TV2A; no native video engine.
- `FEAT-video-source-audio-reuse.md` — proposed mix/context reuse of
  engine-provided audio, with provenance.

## Acceptance

No native video route, DTO, job kind or capability advertisement exists before
the spike. After the spike, a video job still requires an owned output contract,
hard resource caps and an exclusive heavy-inference lease; cancellation states
interruptibility only as measured.

## Verification

`bun run test:core`, `bun run test:server`, `bun run lint:rs`, `bun run deny`;
opt-in local checkpoint only, no CI downloads. Structural frame/audio checks
only; one real clip is required to leave the spike, not to run CI.

## Non-goals

Unlimited frames/edge, byte-identical cross-backend output, external
AnimateDiff/SVDProcess adapters (those belong to `epic-external-tools`), a vendor
upgrade, bundling/auto-fetching model weights, and native Foley/remix workflows
before the direct-generation path is proven.
