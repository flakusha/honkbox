# FEAT: reuse engine-provided video audio as mix or context
Status: proposed
Priority: P2
Epic: epic-video-support
Owner: unassigned
Acceptance: a typed template parameter selects `generate`, `mix_original` or `reference_original`; original audio from user-supplied LTX-2/MiniMax-H3 video is used only as a mix layer or short conditioning reference; provenance records the input stream, mode, gains and output artifact; no native decoder, resampler or bundled weights are added.
Verification: `bun run test:server`, `bun run codegen:check`; opt-in fixture with a user-supplied short clip
Links: `.tmp/research/video-audio-reuse-and-download-options-2026-11-24.md`, `video-to-audio-modalities-2026-11-24.md`, `FEAT-comfy-video-audio-templates.md`, `epic-media-timeline.md`
Resources: per-test temp video/audio fixtures; no shared ComfyUI output dir

## Change

- Extend the video-audio template family with an `audio_source` enum: `generate` (current behavior), `mix_original` (core `AudioMerge` + `AudioAdjustVolume`, optional `AudioConcat`), `reference_original` (ControlFoley AC-V2A reference audio, 2-4 s).
- Bounded gains only: `AudioAdjustVolume` dB range −100..100; a named `merge_method` from `add|mean|subtract|multiply`. No ducking/sidechain claim; core does sample-rate matching.
- Identify original audio by explicit input file/stream index from the staged video, never by extension guessing; no HTTP-supplied absolute path reaches ComfyUI.
- Add a provenance sidecar to the job bundle: source video ref, audio stream index, mode, merge method, gains, generated audio artifact and final audio artifact.

## Acceptance detail

- `mix_original` provably contains both the original waveform and the generated track in the fixture; clipping is bounded by the allowed gain range.
- `reference_original` uses only a short reference and does not silently replace the source audio track.
- LTX-2 and MiniMax-H3 weights/audio stay user-supplied; honkbox downloads, mirrors and bundles none of them.
