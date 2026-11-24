# EPIC: Output formats and bounded payloads

Status: in progress
Priority: P1
Owner: unassigned
Research: `.tmp/research/media-masters-audio-video-2026-11-25.md`,
`.tmp/research/media-handling-consolidated-2026-11-24.md`

## Scope

Image/audio container and quality seams, durable master/derived artifact roles,
response compression, zip job bundles, and request-size caps. honkbox owns the
current image RGBA8 and audio PCM16 encode boundaries. Native video generation,
frame ownership and muxing remain separately gated in `epic-video-support`;
external ComfyUI workflows do not change that truth.

## Tickets

- `FEAT-image-output-formats.md` — done: JPEG/lossless WebP, quality semantics and EXR image-ops ingest/preview.
- `FEAT-audio-output-formats.md` — finished: WAV/PCM/FLAC/Opus; MP3 remains blocked.
- `FEAT-media-master-and-derived-artifacts.md` — proposed P1: deterministic per-job `manifest.json`, exact worker master plus derived finals, source/transform/tool/color/audio metadata, no silent correction.
- `TASK-speech-input-cap.md` — cap `SpeechRequest.input` (blocking-grade).
- `FEAT-job-bundle-zip.md` — done: downloadable deterministic flat job ZIP and manifest-listed single-file content.
- `FEAT-response-compression.md` — gzip/brotli layer, SSE excluded (S).
- `TASK-texture-postprocess-oiio.md` — OIIO mip/tile/colorspace/ORM post-process + per-map sidecar (P0).
- `FEAT-comfy-video-audio-templates.md` — proposed external ComfyUI media templates; owned by `epic-comfyui-managed-runtime`, not native video support.

## Acceptance

All format additions keep `codegen:check` green and preserve the reject-not-clamp
policy; no unbounded text → response multiplier remains. Bundle ZIPs pin entry
order and metadata, cap entries/bytes, and never traverse host paths. Durable
media bundles write artifacts before a deterministic manifest, preserve an exact
worker master where the representation permits, label lossy finals truthfully,
and apply no enhancement or normalization unless explicitly requested and
recorded. No default codec/runtime dependency is added.

## Verification

`bun run check`, `bun run test:server`, `bun run lint:rs`, `bun run codegen:check`.

## Non-goals

Native video generation/muxing, audio extraction from video, new codec or
Python/FFmpeg/PyTorch runtime dependencies, re-encoding vendor internals, or
claiming an 8-bit/PCM16/MP4 master is lossless when it is not.
