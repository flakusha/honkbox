//! Image generation recipes and validation, extracted from `routes.rs`.
//!
//! Pure helpers: no router registrations live here, so the `OpenAPI`
//! coverage gate in `routes.rs` is unaffected.

use honkbox_api_types::{ImagesHiresFix, ModelCatalog, ModelKind, ModelMetadata};

/// SD-side model resolution for `POST /v1/images/generations` (REQ-10):
/// alias-or-slug over the catalog, restricted to `ModelKind::Sd` entries
/// (the comfy/SD category). Returns the owning registry slug.
pub(super) fn resolve_sd_model<'a>(model: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    super::resolve_model_entry(model, catalog)
        .filter(|e| e.kind == ModelKind::Sd)
        .map(|e| e.slug.as_str())
}

/// Scan-time base-model tag of one catalog entry, for `LoRA` compatibility.
pub(super) fn base_model_of<'a>(slug: &str, catalog: &'a ModelCatalog) -> Option<&'a str> {
    super::resolve_model_entry(slug, catalog).and_then(|entry| match &entry.metadata {
        Some(ModelMetadata::Safetensors(meta)) => meta.base_model.as_deref(),
        _ => None,
    })
}

/// Validated image request: parsed size, clamped count, and the resolved
/// generation recipe (sampler/scheduler/CFG/steps/negative/seed base; the
/// fields a named recipe supplied or a request overrode — omitting every
/// one is byte-identical to the pre-passthrough behavior). The only
/// rejection paths for `POST /v1/images/generations` before the GPU slot;
/// `Err` is (message, error code) for a 400 `OpenAI` envelope. Unlike `n`
/// (clamped with a warn), unknown recipe/sampler/scheduler names,
/// out-of-range `cfg_scale`/`steps`, and invalid `hires_fix` values are
/// hard 400s — silently substituting a recipe the caller did not ask for
/// would be dishonest about what was rendered.
pub(super) struct ImageRecipe {
    pub(super) sampler: honkbox_sd_core::Sampler,
    pub(super) scheduler: honkbox_sd_core::Scheduler,
    pub(super) cfg_scale: f32,
    pub(super) negative_prompt: Option<String>,
    pub(super) seed_base: u64,
    /// Diffusion steps actually used (server default, recipe default, or
    /// explicit override — see `validate_recipe`'s precedence docs).
    pub(super) steps: u32,
    /// Named curated recipe applied, if any (`None` = field-by-field or
    /// pure defaults). Recorded verbatim in the job bundle.
    pub(super) name: Option<&'static str>,
}

/// CFG guidance bounds for the HTTP surface (documented on the DTO).
pub(super) const IMAGE_CFG_SCALE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=30.0;
/// Accepted diffusion-steps range for the HTTP surface (documented on the
/// DTO); outside it is a 400, never a clamp.
pub(super) const IMAGE_STEPS_RANGE: std::ops::RangeInclusive<u32> = 1..=100;
/// Hires-fix scale bounds for the HTTP surface (documented on the DTO).
pub(super) const IMAGE_HIRES_SCALE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=4.0;

/// One curated generation recipe: a sampler/scheduler/cfg/steps bundle for
/// a checkpoint family. Values mirror the vendored stable-diffusion.cpp
/// presets/community standards, constrained to what sd-core's
/// `Sampler`/`Scheduler` enums support — nothing here can fail to parse at
/// generation time.
pub(super) struct ImageRecipeSpec {
    pub(super) name: &'static str,
    pub(super) sampler: honkbox_sd_core::Sampler,
    pub(super) scheduler: honkbox_sd_core::Scheduler,
    pub(super) cfg_scale: f32,
    pub(super) steps: u32,
}

/// The curated recipe set (`POST /v1/images/generations` + `/edits`):
/// - `sd15`: SD 1.x/2.x — the vendor SD1.5/SD2.1 presets leave everything
///   at the sd.cpp defaults (`euler_a`, discrete schedule, CFG 7, 20 steps).
/// - `sdxl`: SDXL base + finetunes — the community-standard
///   dpm++2m/karras combo at 25 steps.
/// - `sd35`: SD3.5 — the vendor `stable_diffusion_3_5_large` preset
///   (euler, CFG 4.5, 28 steps).
/// - `turbo`: turbo-distilled (SD/SDXL turbo) — the vendor turbo presets
///   set CFG 1.0 at 4 steps, sampler/scheduler left at defaults.
/// - `lcm`: LCM-distilled / LCM-LoRA — the vendor `lcm_lora_sdxl` modifier
///   (lcm sampler, CFG 2.0, 8 steps).
/// - `flux-dev` / `flux-schnell`: FLUX.1 — the vendor flux presets (euler,
///   CFG 1.0 — distilled guidance handles steering, 28 / 4 steps).
pub(super) const IMAGE_RECIPES: &[ImageRecipeSpec] = &[
    ImageRecipeSpec {
        name: "sd15",
        sampler: honkbox_sd_core::Sampler::EulerA,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 7.0,
        steps: 20,
    },
    ImageRecipeSpec {
        name: "sdxl",
        sampler: honkbox_sd_core::Sampler::Dpmpp2m,
        scheduler: honkbox_sd_core::Scheduler::Karras,
        cfg_scale: 7.0,
        steps: 25,
    },
    ImageRecipeSpec {
        name: "sd35",
        sampler: honkbox_sd_core::Sampler::Euler,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 4.5,
        steps: 28,
    },
    ImageRecipeSpec {
        name: "turbo",
        sampler: honkbox_sd_core::Sampler::EulerA,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 1.0,
        steps: 4,
    },
    ImageRecipeSpec {
        name: "lcm",
        sampler: honkbox_sd_core::Sampler::Lcm,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 2.0,
        steps: 8,
    },
    ImageRecipeSpec {
        name: "flux-dev",
        sampler: honkbox_sd_core::Sampler::Euler,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 1.0,
        steps: 28,
    },
    ImageRecipeSpec {
        name: "flux-schnell",
        sampler: honkbox_sd_core::Sampler::Euler,
        scheduler: honkbox_sd_core::Scheduler::Discrete,
        cfg_scale: 1.0,
        steps: 4,
    },
];

/// Every accepted recipe wire name, in table order (for 400 messages).
pub(super) fn image_recipe_names() -> Vec<&'static str> {
    IMAGE_RECIPES.iter().map(|r| r.name).collect()
}

/// Look up one curated recipe by exact wire name.
pub(super) fn resolve_recipe(name: &str) -> Option<&'static ImageRecipeSpec> {
    IMAGE_RECIPES.iter().find(|r| r.name == name)
}

/// Shared recipe validation for both images routes: `generations` passes
/// JSON DTO fields straight through; `edits` passes multipart-parsed
/// strings. `Err` is (message, error code) for a 400 `OpenAI` envelope.
/// Unknown recipe/sampler/scheduler names and out-of-range `cfg_scale`/
/// `steps` are hard 400s — silently substituting a recipe the caller did
/// not ask for would be dishonest about what was rendered.
///
/// Precedence per field: an explicit request field wins over the named
/// recipe's default, which wins over the server default — so `recipe:
/// "sdxl"` plus `sampler: "euler"` renders euler with the recipe's
/// remaining (scheduler/cfg/steps) defaults.
pub(super) fn validate_recipe(
    recipe: Option<&str>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    cfg_scale: Option<f32>,
    negative_prompt: Option<&str>,
    seed: Option<u64>,
    steps: Option<u32>,
) -> Result<ImageRecipe, (String, &'static str)> {
    let recipe_name = recipe.map(str::trim).filter(|s| !s.is_empty());
    let spec = match recipe_name {
        Some(name) => Some(resolve_recipe(name).ok_or_else(|| {
            (
                format!(
                    "unknown recipe {name:?}: accepted values are {}",
                    image_recipe_names().join(", ")
                ),
                "unknown_recipe",
            )
        })?),
        None => None,
    };
    let sampler_name = sampler.map(str::trim).filter(|s| !s.is_empty());
    let sampler = match sampler_name {
        Some(name) => honkbox_sd_core::Sampler::parse(name).ok_or_else(|| {
            (
                format!(
                    "unknown sampler {name:?}: accepted values are {}",
                    honkbox_sd_core::SAMPLER_NAMES.join(", ")
                ),
                "unknown_sampler",
            )
        })?,
        None => spec.map_or(honkbox_sd_core::Sampler::Euler, |s| s.sampler),
    };
    let scheduler_name = scheduler.map(str::trim).filter(|s| !s.is_empty());
    let scheduler = match scheduler_name {
        Some(name) => honkbox_sd_core::Scheduler::parse(name).ok_or_else(|| {
            (
                format!(
                    "unknown scheduler {name:?}: accepted values are {}",
                    honkbox_sd_core::SCHEDULER_NAMES.join(", ")
                ),
                "unknown_scheduler",
            )
        })?,
        // The fixed recipe's recorded `normal` name IS the vendor discrete
        // schedule; wire spellings follow sd.cpp (`discrete`).
        None => spec.map_or(honkbox_sd_core::Scheduler::Discrete, |s| s.scheduler),
    };
    let cfg_scale = match cfg_scale {
        Some(v) if IMAGE_CFG_SCALE_RANGE.contains(&v) && v.is_finite() => v,
        Some(v) => {
            return Err((
                format!("cfg_scale {v} outside accepted range [1.0, 30.0]: rejected, not clamped"),
                "invalid_cfg_scale",
            ));
        }
        None => spec.map_or(7.0, |s| s.cfg_scale),
    };
    let steps = match steps {
        Some(v) if IMAGE_STEPS_RANGE.contains(&v) => v,
        Some(v) => {
            return Err((
                format!("steps {v} outside accepted range [1, 100]: rejected, not clamped"),
                "invalid_steps",
            ));
        }
        None => spec.map_or(super::IMAGE_STEPS, |s| s.steps),
    };
    let negative_prompt = negative_prompt
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Ok(ImageRecipe {
        sampler,
        scheduler,
        cfg_scale,
        negative_prompt,
        seed_base: seed.unwrap_or(super::IMAGE_SEED_BASE),
        steps,
        name: spec.map(|s| s.name),
    })
}

/// Validate the optional hires-fix pass against the parsed base size:
/// per-field range checks (`scale`, `denoise`), `target` size parsing, and
/// the shared [`honkbox_sd_core::MAX_EDGE`] output cap (the final PNG lands
/// at the hires target, so it must stay under the cap too).
pub(super) fn validate_hires_fix(
    hires: Option<&ImagesHiresFix>,
    base: (u32, u32),
) -> Result<Option<honkbox_sd_core::HiresFix>, (String, &'static str)> {
    let Some(hires) = hires else {
        return Ok(None);
    };
    let scale = match hires.scale {
        Some(v) if IMAGE_HIRES_SCALE_RANGE.contains(&v) && v.is_finite() => v,
        Some(v) => {
            return Err((
                format!(
                    "hires_fix.scale {v} outside accepted range [1.0, 4.0]: rejected, not clamped"
                ),
                "invalid_hires_scale",
            ));
        }
        None => 2.0,
    };
    let target = match hires
        .target
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        Some(raw) => Some(
            crate::jobs::parse_size(raw)
                .map_err(|err| (format!("hires_fix.target: {err}"), "invalid_hires_target"))?,
        ),
        None => None,
    };
    let denoising_strength = match hires.denoise {
        Some(v) if v.is_finite() && v > 0.0 && v <= 1.0 => v,
        Some(v) => {
            return Err((
                format!(
                    "hires_fix.denoise {v} outside accepted range (0, 1]: rejected, not clamped"
                ),
                "invalid_hires_denoise",
            ));
        }
        None => 0.7,
    };
    let fix = honkbox_sd_core::HiresFix {
        scale,
        target,
        denoising_strength,
    };
    let (target_width, target_height) = honkbox_sd_core::hires_target(base, &fix);
    let edge = base.0.max(base.1).max(target_width).max(target_height);
    if edge > honkbox_sd_core::MAX_EDGE {
        return Err((
            format!(
                "hires_fix output {target_width}x{target_height} exceeds the {}px output cap",
                honkbox_sd_core::MAX_EDGE
            ),
            "invalid_hires_target",
        ));
    }
    Ok(Some(fix))
}
