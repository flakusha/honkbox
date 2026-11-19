# EPIC: Output formats and bounded payloads

Status: in progress
Priority: P1
Owner: main session
Research: `.tmp/research/ReviewedOutput-Formats.md`

## Scope

Image/audio container and quality seams, response compression, zip job bundles,
and request-size caps. The encoder boundary is honkbox-owned: the vendor returns
RGBA pixels and PCM16, so only video is vendor-blocked.

## Tickets

- `FEAT-image-output-formats.md` — JPEG/WebP over HTTP + quality knob (S).
- `FEAT-audio-output-formats.md` — Opus/FLAC; MP3 blocked (S–M).
- `TASK-speech-input-cap.md` — cap `SpeechRequest.input` (blocking-grade).
- `FEAT-job-bundle-zip.md` — downloadable job zip (S, LANDED 2026-11-19: `GET /v1/jobs/{id}/bundle.zip` + allowlisted `GET /v1/jobs/{id}/content/{name}`, deterministic bytes, explicit response caps).
- `FEAT-response-compression.md` — gzip/brotli layer, SSE excluded (S).
- `TASK-texture-postprocess-oiio.md` — OIIO mip/tile/colorspace/ORM post-process + per-map sidecar (P0).

Video work now lives in `epic-video-support`; this epic covers only bytes that
leave already-implemented routes.

## Acceptance

All format additions keep `codegen:check` green and preserve the reject-not-clamp
policy; no unbounded text → response multiplier remains. Bundle ZIPs pin
entry order and metadata, cap entries/bytes, and never traverse host paths.

## Verification

`bun run check`, `bun run test:server`, `bun run lint:rs`, `bun run codegen:check`.

## Non-goals

Changing vendor `SD_WEBP/SD_WEBM` flags for v1; re-encoding vendor internals.
