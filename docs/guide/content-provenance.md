# Content provenance

Generated and edited PNG, JPEG, and WebP images carry a deterministic
`honkbox.generation/1` record inside the image bytes. Metadata is injected at
the container level; pixels are copied without decoding or re-encoding.

## Record

The record starts from the job bundle reproducibility record and adds:

- `record_type`: `honkbox.generation/1`;
- `honkbox_version`: the server crate version;
- `inputs_sha256`: the canonical generation-input hash.

It includes the prompt, seed, steps, sampler, scheduler, CFG scale, negative
prompt, LoRAs, image size and format, backend/engine identity, model slug and
known model hash, img2img strength and mask hash, checkpoint base model,
recipe, hires-fix settings, and bundle annotations. Unknown model hashes stay
`null`; provenance never fabricates a digest.

## Storage by format

| Format | Record | XMP |
| --- | --- | --- |
| PNG | `iTXt` `honkbox_generation_record` | `iTXt` `XML:com.adobe.xmp` |
| JPEG | APP1 EXIF `UserComment` | APP1 XMP |
| WebP | Extended `VP8X` + `EXIF` chunk | XMP chunk |

JPEG APP1 EXIF payloads and WebP `EXIF` chunks use the standard
`Exif\0\0` prefix. The JSON record and XMP packet contain the same canonical
data. Records are bounded to 1 MiB; JPEG APP1 segment limits still apply.

## Job bundles

An image job bundle contains:

- `final_00.png`, `final_00.jpg`, or `final_00.webp` — the exact bytes returned
  by the API or written by the CLI;
- `final_00.json` — the existing human-readable sidecar;
- `c2pa_manifest.json` — a deterministic, unsigned C2PA-shaped manifest with a
  `c2pa.created` action and a `honkbox.generation` assertion.

The manifest has no timestamp, credential, or signature. It records provenance
for inspection and interchange; it is not a signed authenticity claim. The
artifact and manifest are written before the sidecar, so a crash cannot
make an incomplete bundle look complete. Non-image bundles, such as OCR
ingest, do not receive an image manifest.
