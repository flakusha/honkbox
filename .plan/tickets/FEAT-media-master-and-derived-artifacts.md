# FEAT: Lossless media masters and derived final artifacts

Status: proposed
Priority: P1
Epic: epic-output-formats
Owner: unassigned
Acceptance: durable image and speech job bundles preserve the exact worker master, name every derived final in a deterministic `manifest.json`, record source hashes and actual transforms/tool/color/audio metadata, and never apply enhancement or normalization implicitly.
Verification: `cargo test --locked -p honkbox-sd-core`, `cargo test --locked -p honkbox-server --test images --test edits --test image_ops --test audio --test job_content`, `bun run codegen:check`, `bun run check`
Links: `.tmp/research/media-masters-audio-video-2026-11-25.md`, `FEAT-image-output-formats.md`, `FEAT-audio-output-formats.md`, `FEAT-job-bundle-zip.md`, `TASK-job-result-artifact-selection-metadata.md`, `FEAT-comfy-video-audio-templates.md`, `epic-audio-streaming.md`
Resources: stub image/TTS backends; per-test temp data dirs and synthetic PCM/RGBA fixtures; CPU blocking pool only; no shared artifact root, model weights, GPU lease or audio device

## Context

[FACT] `sd-core::GeneratedImage` is RGBA8 and `SynthesizedAudio` is mono PCM16.
PNG/JPEG/lossless-WebP and WAV/PCM/FLAC/Opus encoders already exist. Generation
and speech currently retain only the requested final; image ops retain only its
encoded output.

[FACT] `jobs::write_atomic`, job GC, `artifacts::Registry::put_bytes`, the flat
bundle ZIP route and format registry already provide transport and bounds.
Generation/ops bundles do not currently write `manifest.json`, although the
single-file bundle route requires it. `write_ops_bundle` writes `ops.json` before
its outputs despite its artifact-first contract.

[FACT] Content-addressed blobs are role-neutral. The same bytes may be master
and final in one job, or a final in one job and a source in another. Role and
lineage therefore belong in a per-job manifest, not `ArtifactInfo` or a global
format flag.

[INFERENCE] “Lossless master” means lossless relative to the worker's RGBA8 or
PCM16 result. It does not recover model latents, float buffers, scene-linear HDR
or pre-quantization audio. The manifest must say so truthfully.

## Scope

One additive per-job `manifest.json` contract, used by:

- `/v1/images/generations` and `/v1/images/edits`, inline and artifact delivery;
- `/v1/image-ops` when a data dir is configured;
- `/v1/audio/speech` and `/v1/synthesize` when a data dir is configured.

No native video generation, ComfyUI graph, demux/mux, new model family, codec,
Python/PyTorch runtime, FFmpeg process or normalization dependency is added.
Comfy and future native video outputs consume the same manifest later.

## Change

### Shared manifest

Add `honkbox.media-manifest/1` serialization in the existing jobs/bundle layer.
Write it with `write_atomic` **after every referenced file**; never update a
published manifest in place. Use stable ordering and omit volatile fields.

Required fields:

- `artifacts[]`: bundle filename, role (`master` or `derived_final`), media kind,
  format/container, bytes, SHA-256, optional artifact id, `derived_from`, and
  explicit `lossless: bool`.
- `sources[]`: logical source/input kind and exact SHA-256 when bytes exist;
  verified model digest when known; explicit null otherwise.
- `transforms[]`: only operations actually applied, in semantic order, with
  closed parameters and tool name/version.
- `tool`: honkbox version, backend/engine identity and encoder identity already
  available at the route.
- `color`: pixel encoding/alpha handling; transfer function and ICC digest only
  when known. Current RGBA8 is not labeled scene-linear or HDR.
- `audio`: encoding, channels/layout, sample rate, sample count and duration for
  audio artifacts; null for non-audio jobs.
- `measurements[]`: empty unless measured; method/tool version, scope, units and
  value required for any sample peak, true peak or loudness entry.
- `processing`: empty by default. No implicit enhancement, cleanup, gain change,
  limiting, normalization or color conversion.

The manifest is also the existing `job_content` allowlist: every artifact name
must be a safe top-level filename, and `manifest.json` is written last. Reuse the
existing 256-entry/512 MiB ZIP limits, artifact byte cap and job GC; over-cap
output is rejected, never truncated or silently reduced to only the final.

### Image generation and edits

- Extend the existing encode seam to produce the requested final and a PNG master
  from one borrowed RGBA8 buffer. Do not decode the final again or clone the
  full pixel buffer solely to create the master.
- PNG final: one file referenced as both `master` and `derived_final`.
- Lossless WebP final: one file may carry both roles; record the container
  transform truthfully.
- JPEG final: `master_00.png` plus `final_00.jpg`; the final records JPEG quality
  and `derived_from` the master digest.
- Keep existing response bytes, filenames, MIME types, NSFW annotations and
  `final_00.json` fields unchanged. `request.json` continues to carry generation
  provenance; the manifest adds source/output hashes and roles.
- Artifact delivery collects final/master artifact ids before publishing the
  manifest. Emit the existing `job_result` metadata only after the manifest is
  durable; the requested final remains primary. Do not add a second role field
  to the flat event DTO in this ticket.
- Inline bundle writes remain best-effort; artifact delivery remains strict. A
  failed strict bundle/manifest write returns the existing named error and
  exposes no completed bundle.

### Image ops

- Preserve the processed RGBA8 PNG master before delivery encoding, applying the
  same one-file/two-file rules as generation.
- Keep the existing input/output hashes, command hash, tool version and EXR
  `map`/`preview` behavior. An EXR tonemap preview is not an HDR master.
- Fix `write_ops_bundle` ordering: command, all output/master files, `ops.json`,
  run log, then `manifest.json`. A crash may leave orphan artifacts but never a
  manifest claiming missing bytes.
- Keep the current inline `outputs[]` response shape and optional bundle warning.

### Speech and synthesis

- Add a best-effort durable bundle when `data_dir` is configured. Use the job's
  existing ULID as the bundle directory and return an optional `X-Bundle-Id`
  response header on success; preserve `X-Job-Id` and all response bytes/MIME.
- Canonical `master_00.wav` wraps the exact mono PCM16 worker result. WAV final
  shares those bytes; PCM, FLAC and Opus finals are derived. FLAC is lossless
  relative to PCM16; Opus is explicitly lossy.
- Stream the WAV master from PCM16 to its `.part` file and hash while writing;
  do not create a whole extra WAV `Vec` solely for persistence. Reuse the existing
  WAV layout and encoders for final bytes.
- Record exact input-text SHA-256, model/voice selection, requested and actual
  sample rate, backend id, channel layout and sample count. Record a rate transform
  only when the worker actually changed the rate; never infer one from a request
  alone. Unknown model digests stay null.
- Bundle failure logs a warning and omits `X-Bundle-Id`; it does not discard an
  otherwise successful synthesis. No data dir means no bundle and no header, as
  today.
- Document the additive header in the route/OpenAPI description. No binary
  response DTO or API-version change is needed.

### Explicit processing boundary

- Existing typed image-op commands remain the only enhancement operations in
  this ticket; they are explicitly requested and recorded.
- This ticket adds no audio normalization or cleanup switch. A later ticket must
  depend on this manifest and provide an explicit mode/target, measured
  before/after values and bounded CPU work.
- Reject unsupported processing parameters. Never ignore them, auto-normalize,
  auto-denoise, auto-limit, auto-colour-correct or silently resample beyond the
  worker's existing requested-rate behavior.

## Acceptance detail

1. A JPEG generation bundle contains a valid RGBA8 PNG master, the unchanged
   JPEG final, and a manifest whose hashes match the files and whose final points
   to the master. PNG and lossless WebP do not duplicate identical bytes.
2. Generation and edit inline/artifact tests preserve existing response bytes,
   MIME, bundle naming, annotations and primary artifact selection. Artifact ids
   in the manifest resolve to the same content-addressed bytes.
3. Image-op bundles write every declared output before `ops.json` and the
   manifest. Existing resize/crop/rotate/grayscale/format/matting and EXR
   map/preview tests still pass without changed response shapes.
4. Speech WAV/PCM/FLAC/Opus and native synthesize tests retain exact response
   bytes/MIME. Configured bundles contain the PCM16 WAV master, requested final,
   correct mono layout/rate/sample count and optional `X-Bundle-Id`; no data dir
   or failed best-effort write leaves no false bundle header.
5. Manifests contain no ULID, timestamp, elapsed time, host path, ComfyUI path,
   token or raw data-URI payload. Rebuilding identical logical content yields
   identical manifest bytes; ZIP determinism remains intact.
6. Unknown model/source hashes, transfer functions and measurements are explicit
   null/empty values, never invented. No manifest says `normalized`, `denoised`
   or `lossless` without the operation/representation that makes it true.
7. A master/final over the existing artifact or bundle cap fails honestly. No
   output is truncated, silently dropped or replaced by a lower-resolution
   version. Memory tests show no decode-of-final or full RGBA/PCM clone added
   solely for master creation.
8. `GET /v1/jobs/{id}/bundle.zip` includes the manifest deterministically;
   `GET /v1/jobs/{id}/content/{name}` serves manifest-listed master/final files
   and still rejects traversal, symlinks, unlisted names and oversized files.
9. Feature/default builds add no dependency or lockfile entry. Existing WAV,
   FLAC, Opus, PNG, JPEG and WebP paths remain the only default codecs.

## Non-goals

- Native video generation/muxing, audio extraction from video, realtime STT,
  diarization, voice cloning or cloud fallback.
- New codec/runtime dependencies, FFmpeg/OIIO/PyTorch integration, EXR write or
  frame sequences.
- New global artifact roles, a second sidecar convention, raw workflow JSON,
  automatic enhancement/normalization or historical backfill of old bundles.
- Changing response payloads or making best-effort inline persistence fail an
  otherwise successful request.

## Verification

Commands to run after implementation (not run for this ticket):

```sh
cargo test --locked -p honkbox-sd-core
cargo test --locked -p honkbox-server --test images --test edits
cargo test --locked -p honkbox-server --test image_ops --test audio
cargo test --locked -p honkbox-server --test job_content
bun run codegen:check
bun run check
```