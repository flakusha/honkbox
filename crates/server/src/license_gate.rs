//! Serving gates for curated model licenses (TASK-model-license-registry).
//!
//! One refusal shape, mapped to the OpenAI error envelope by the route
//! (412 + named code). The gate runs before GPU admission and before any
//! bundle write, so a refusal leaves no job, no lease and no partial bundle.
//!
//! Fail-closed scope: a curated row with no license is refused. An
//! undeclared local file stays local-only (served with the existing warning)
//! — making every scan hit unservable would break local LoRA/checkpoint use.

use std::fmt;

use axum::http::StatusCode;
use honkbox_api_types::ModelCatalog;

use crate::catalog::licenses::{self, LicenseRecord};
use crate::config;

/// Named 412 refusal. `Display` prefixes the wire code; [`Self::message`]
/// carries the human text that names the slug and the term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LicenseRefusal {
    MissingRecord,
    Territory {
        slug: String,
        territory: String,
    },
    Gated {
        slug: String,
        terms_url: Option<String>,
        token_missing: bool,
    },
    NonCommercial {
        slug: String,
        terms_url: Option<String>,
    },
}

impl LicenseRefusal {
    /// Wire code for the OpenAI-style error envelope.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::MissingRecord => "model_license_missing",
            Self::Territory { .. } => "model_license_territory",
            Self::Gated { .. } => "model_license_gated",
            Self::NonCommercial { .. } => "model_license_non_commercial",
        }
    }

    /// Every license refusal is a failed precondition: the deployment has not
    /// declared the territory/token/commercial posture the weights require.
    pub(crate) fn status(&self) -> StatusCode {
        StatusCode::PRECONDITION_FAILED
    }

    pub(crate) fn message(&self) -> String {
        match self {
            Self::MissingRecord => {
                "model has no license record; serving refused (fail-closed)".to_owned()
            }
            Self::Territory { slug, territory } => format!(
                "model {slug:?}: territory restriction {territory:?} refuses serving from the declared jurisdiction"
            ),
            Self::Gated {
                slug,
                terms_url,
                token_missing,
            } => format!(
                "model {slug:?}: gated weights require an accepted HF token (token_missing={token_missing}); terms: {}",
                terms_text(terms_url)
            ),
            Self::NonCommercial { slug, terms_url } => format!(
                "model {slug:?}: non-commercial weights cannot serve a commercial pipeline; terms: {}",
                terms_text(terms_url)
            ),
        }
    }
}

impl fmt::Display for LicenseRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

fn terms_text(url: &Option<String>) -> &str {
    url.as_deref()
        .filter(|url| !url.trim().is_empty())
        .unwrap_or("the model's source terms")
}

/// Declared territory is outside the recorded exclusion set. `unknown` (the
/// default) always refuses; an unrecognized restriction also fails closed.
/// ponytail: exact region-code match only; map real jurisdictions to codes
/// when request geography is available.
fn territory_allowed(restriction: &str, declared: &str) -> bool {
    let declared = declared.trim();
    if declared.is_empty() || declared.eq_ignore_ascii_case("unknown") {
        return false;
    }
    let Some(excluded) = restriction.trim().strip_prefix("excludes-") else {
        return false;
    };
    !excluded
        .split('-')
        .any(|code| code.eq_ignore_ascii_case(declared))
}

/// A scan hit with no curated row: local-only, never bundled. Serving is not
/// refused — the route already warns that no license was recorded.
fn local_only_record() -> LicenseRecord {
    LicenseRecord {
        license: "all-rights-reserved".to_owned(),
        license_url: None,
        territory: None,
        commercial_terms: "all-rights-reserved".to_owned(),
        gated: false,
        source_url: None,
        sha256: None,
        attribution_text: None,
    }
}

/// Refusal logic over an already-resolved record (the test seam: keeps the
/// gate branches independent of the shipped manifest).
pub(crate) fn check(
    slug: &str,
    record: Option<LicenseRecord>,
    config: &config::Config,
) -> Result<LicenseRecord, LicenseRefusal> {
    let Some(record) = record else {
        return Ok(local_only_record());
    };
    if record.license.trim().is_empty() {
        return Err(LicenseRefusal::MissingRecord);
    }
    if let Some(territory) = record.territory.as_deref().filter(|t| !t.trim().is_empty())
        && !territory_allowed(territory, &config.territory)
    {
        return Err(LicenseRefusal::Territory {
            slug: slug.to_owned(),
            territory: territory.to_owned(),
        });
    }
    if record.gated {
        // No HF token plumbing exists yet; the gate reports the requirement
        // instead of silently downloading or serving gated weights.
        return Err(LicenseRefusal::Gated {
            slug: slug.to_owned(),
            terms_url: record
                .license_url
                .clone()
                .or_else(|| record.source_url.clone()),
            token_missing: true,
        });
    }
    if config.commercial
        && record
            .commercial_terms
            .eq_ignore_ascii_case("non-commercial")
    {
        return Err(LicenseRefusal::NonCommercial {
            slug: slug.to_owned(),
            terms_url: record
                .license_url
                .clone()
                .or_else(|| record.source_url.clone()),
        });
    }
    Ok(record)
}

/// Resolve `slug` against the live catalog, then apply the license policy.
pub(crate) fn gate(
    slug: &str,
    catalog: &ModelCatalog,
    config: &config::Config,
) -> Result<LicenseRecord, LicenseRefusal> {
    check(slug, licenses::record_for_slug(slug, catalog), config)
}
