//! Image-generation request input parsing and validation, extracted from
//! `routes.rs`.
//!
//! The router registrations stay in `routes.rs`. Only the request-level
//! helpers (size/format/compression parsing, recipe validation glue, and
//! `LoRA` resolution) live here.

use std::path::PathBuf;

use honkbox_api_types::{ImagesGenerationRequest, ImagesLora, ModelCatalog, ModelKind};

use super::images_recipe::{ImageRecipe, validate_hires_fix, validate_recipe};

/// Validated `POST /v1/images/generations` request: parsed base size,
/// clamped count, the resolved recipe, and the optional hires-fix pass
/// (tuple alias keeps `validate_images_request`'s signature readable — the
/// same pattern as `ResolvedLoras`).
pub(super) type ValidatedImages = (
    u32,
    u32,
    u32,
    ImageRecipe,
    Option<honkbox_sd_core::HiresFix>,
    honkbox_sd_core::OutputFormat,
    Option<u8>,
);

/// Parse the requested output container against the registry's compiled
/// encoder rows (omitted = PNG). Unknown names are a named 400 listing the
/// registry-derived accepted values (same reject-don't-guess policy as
/// `response_format`). The wire name → enum mapping stays in THIS module
/// (the pre-registry call surface — the `#[path]`-included registry module
/// must not link the native backends); alignment with the registry rows is
/// locked by `format_registry_matches_the_compiled_encoders` below.
pub(super) fn parse_output_format(
    raw: &str,
) -> Result<honkbox_sd_core::OutputFormat, (String, &'static str)> {
    let accepted = crate::formats::output_rows()
        .map(|row| format!("{:?}", row.name))
        .collect::<Vec<_>>()
        .join(", ");
    honkbox_sd_core::OutputFormat::from_name(raw).ok_or_else(|| {
        (
            format!("unsupported output_format {raw:?}: accepted values are {accepted}"),
            "unsupported_output_format",
        )
    })
}

/// Parse `output_compression`: an integer JPEG quality in `[0, 100]`.
/// Non-numeric, fractional, negative, or out-of-range values are a named
/// 400 — rejected, never clamped. The value is only applied to JPEG; PNG is
/// lossless and WebP is lossless-only today, so there it is ignored (DTO
/// docs say so). `None` = the sd-core default (quality 90).
pub(super) fn parse_output_compression(
    raw: Option<&serde_json::Value>,
) -> Result<Option<u8>, (String, &'static str)> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    match raw.as_u64() {
        Some(q) if q <= 100 => Ok(Some(u8::try_from(q).expect("q <= 100"))),
        _ => Err((
            format!(
                "output_compression must be an integer in [0, 100]; got {raw}: rejected, not clamped"
            ),
            "invalid_output_compression",
        )),
    }
}

pub(super) fn validate_images_request(
    req: &ImagesGenerationRequest,
) -> Result<ValidatedImages, (String, &'static str)> {
    let (width, height) = crate::jobs::parse_size(req.size.as_deref().unwrap_or("1024x1024"))
        .map_err(|err| (err, "invalid_size"))?;
    if let Some(format) = req
        .response_format
        .as_deref()
        .map(str::trim)
        .filter(|f| *f != "b64_json")
    {
        return Err((
            format!("unsupported response_format {format:?}: only \"b64_json\" is supported"),
            "unsupported_response_format",
        ));
    }
    let n = req.n.unwrap_or(1);
    if n == 0 || n > 4 {
        tracing::warn!(
            requested = n,
            "images n outside [1, 4]; clamping (never rejected)"
        );
    }
    let recipe = validate_recipe(
        req.recipe.as_deref(),
        req.sampler.as_deref(),
        req.scheduler.as_deref(),
        req.cfg_scale,
        req.negative_prompt.as_deref(),
        req.seed,
        req.steps,
    )?;
    let hires = validate_hires_fix(req.hires_fix.as_ref(), (width, height))?;
    // Output container: the registry is the source of truth (omitted =
    // PNG); unknown names are a named 400 listing the registry-derived
    // accepted values (same reject-don't-guess policy as `response_format`).
    let format = parse_output_format(req.output_format.as_deref().unwrap_or("png").trim())?;
    let compression = parse_output_compression(req.output_compression.as_ref())?;
    Ok((
        width,
        height,
        n.clamp(1, 4),
        recipe,
        hires,
        format,
        compression,
    ))
}

/// Accepted `LoRA` multiplier range for the HTTP surface (documented on the
/// DTO); outside it is a 400, never a silent clamp.
pub(super) const IMAGE_LORA_STRENGTH_RANGE: std::ops::RangeInclusive<f32> = -2.0..=2.0;

/// Max `LoRA` adapters per images request (documented on the DTO); more is a
/// 400 — the vendor applies them sequentially, and cost scales linearly.
pub(super) const IMAGE_LORA_MAX: usize = 2;

/// Resolve request `LoRA` references against the catalog (`ModelKind::Lora`
/// entries, alias-or-slug exactly like SD models). Returns the
/// pipeline-ready refs, sidecar records, and advisory warnings. Rejections:
/// more than [`IMAGE_LORA_MAX`] adapters, empty/unknown slug or alias,
/// strength outside [`IMAGE_LORA_STRENGTH_RANGE`], and adapters spanning
/// more than one directory (the vendor applies a single `LoRA` dir per model
/// config).
///
/// License rule on record (ReviewedLicense-AGPL §2b/§2c): honkbox never
/// bundles `LoRA` weights. The curated catalog `license` is surfaced as a
/// response warning when a manifest row matches the local file name; a
/// missing license means all-rights-reserved, i.e. local-only use. A
/// base-model mismatch cannot be detected today (no base tags are recorded
/// on scan entries or `LoRA` files), so it is neither warned about nor
/// rejected — an honest gap, not a silent pass.
/// Resolved `LoRA` request: pipeline-ready refs, sidecar records, and the
/// advisory license warnings (a tuple alias keeps `resolve_loras`'s
/// signature readable).
pub(super) type ResolvedLoras = (
    Vec<honkbox_sd_core::LoraRef>,
    Vec<crate::jobs::LoraRecord>,
    Vec<String>,
);

pub(super) fn resolve_loras(
    loras: &[ImagesLora],
    catalog: &ModelCatalog,
) -> Result<ResolvedLoras, (String, &'static str)> {
    if loras.len() > IMAGE_LORA_MAX {
        return Err((
            format!(
                "loras: at most {IMAGE_LORA_MAX} adapters per request (got {})",
                loras.len()
            ),
            "too_many_loras",
        ));
    }
    let mut refs = Vec::with_capacity(loras.len());
    let mut records = Vec::with_capacity(loras.len());
    let mut warnings = Vec::with_capacity(loras.len());
    for lora in loras {
        let slug = lora.slug.trim();
        if slug.is_empty() {
            return Err(("loras: slug must not be empty".to_owned(), "unknown_lora"));
        }
        let Some(entry) =
            super::resolve_model_entry(slug, catalog).filter(|entry| entry.kind == ModelKind::Lora)
        else {
            return Err((
                format!(
                    "unknown lora {slug:?}: not a LoRA catalog slug or alias (see GET /v1/models)"
                ),
                "unknown_lora",
            ));
        };
        let strength = match lora.strength {
            Some(v) if IMAGE_LORA_STRENGTH_RANGE.contains(&v) && v.is_finite() => v,
            Some(v) => {
                return Err((
                    format!(
                        "lora {slug:?}: strength {v} outside accepted range [-2.0, 2.0]: rejected, not clamped"
                    ),
                    "invalid_lora_strength",
                ));
            }
            None => 1.0,
        };
        // License surfacing: the curated manifest carries a per-file
        // `license`; scan-only entries have none. Join is by file name (the
        // same stable key the catalog build uses).
        let file_name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
        let license = crate::catalog::manifest()
            .iter()
            .find(|m| m.kind == ModelKind::Lora && m.file.rsplit('/').next() == Some(file_name))
            .map(|m| m.license.as_str())
            .filter(|license| !license.is_empty());
        warnings.push(match license {
            Some(license) => format!("lora {slug}: license {license}"),
            None => format!(
                "lora {slug}: no license recorded — treat as all-rights-reserved (local-only)"
            ),
        });
        refs.push(honkbox_sd_core::LoraRef {
            path: PathBuf::from(&entry.path),
            strength,
        });
        records.push(crate::jobs::LoraRecord {
            slug: entry.slug.clone(),
            strength,
        });
    }
    if let Some(first) = refs.first() {
        let dir = first.path.parent();
        if refs.iter().any(|r| r.path.parent() != dir) {
            return Err((
                "loras: all adapters must live in one directory (the vendor applies a single LoRA dir per generation)".to_owned(),
                "mixed_lora_dirs",
            ));
        }
    }
    Ok((refs, records, warnings))
}
