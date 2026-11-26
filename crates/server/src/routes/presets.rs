//! Read-only prompt/style preset routes and generation composition.
//!
//! The manifest is operator-owned JSON beside the model roots. These
//! handlers never write it, never fetch anything, and never touch the GPU
//! slot: they parse a small local document and compose its text into the
//! existing generation request shape.

use axum::{
    Json,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::ImagesGenerationRequest;
use honkbox_registry::{
    Preset, PresetLoadError, PresetManifest, empty_preset_manifest, load_presets,
};
use serde::Serialize;

use super::AppConfig;

/// `GET /v1/presets` envelope. The schema version is the parsed manifest's
/// version so a client can detect a stale operator file.
#[derive(Debug, Serialize)]
pub struct PresetListResponse {
    /// Schema version of the parsed operator manifest.
    pub schema_version: u32,
    /// Presets in stable id order.
    pub presets: Vec<Preset>,
}

/// `GET /v1/presets/{id}` envelope.
#[derive(Debug, Serialize)]
pub struct PresetDetailResponse {
    /// Schema version of the parsed operator manifest.
    pub schema_version: u32,
    /// The requested preset.
    pub preset: Preset,
}

/// Read the operator manifest, or return the honest empty document when no
/// data dir or manifest is configured.
fn manifest(config: &AppConfig) -> Result<PresetManifest, PresetLoadError> {
    match config.data_dir.as_deref() {
        Some(dir) => load_presets(dir).map(|loaded| loaded.unwrap_or_else(empty_preset_manifest)),
        None => Ok(empty_preset_manifest()),
    }
}

fn manifest_error_response(err: PresetLoadError) -> Response {
    tracing::error!(%err, "preset manifest rejected");
    super::openai_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "preset manifest is invalid or unreadable".to_owned(),
        Some("preset_manifest_invalid"),
    )
}

/// `GET /v1/presets`: all operator presets in stable id order. Empty list
/// when no manifest is configured; a malformed manifest is a named 500,
/// never a silent empty list.
#[utoipa::path(
    get,
    path = "/v1/presets",
    responses(
        (status = 200, description = "Versioned prompt/style presets", body = String),
        (status = 500, description = "Manifest invalid or unreadable", body = String)
    )
)]
pub(crate) async fn get_presets(State(config): State<AppConfig>) -> Response {
    match manifest(&config) {
        Ok(loaded) => Json(PresetListResponse {
            schema_version: loaded.schema_version,
            presets: loaded.presets,
        })
        .into_response(),
        Err(err) => manifest_error_response(err),
    }
}

/// `GET /v1/presets/{id}`: one preset by its stable id. An unknown id is a
/// named 400 so a client can distinguish a stale picker entry from a server
/// failure.
#[utoipa::path(
    get,
    path = "/v1/presets/{id}",
    params(("id" = String, Path, description = "Stable preset id")),
    responses(
        (status = 200, description = "One versioned preset", body = String),
        (status = 400, description = "Unknown preset id", body = String),
        (status = 500, description = "Manifest invalid or unreadable", body = String)
    )
)]
pub(crate) async fn get_preset(
    State(config): State<AppConfig>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let loaded = match manifest(&config) {
        Ok(loaded) => loaded,
        Err(err) => return manifest_error_response(err),
    };
    let Some(preset) = loaded.presets.iter().find(|preset| preset.id == id) else {
        return super::openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("unknown preset id {id:?}"),
            Some("unknown_preset"),
        );
    };
    Json(PresetDetailResponse {
        schema_version: loaded.schema_version,
        preset: preset.clone(),
    })
    .into_response()
}

/// Compose a named preset into the existing request DTO before validation.
/// `X-Honkbox-Preset` is a routing header (H1): prompt text stays in the
/// body. No new client format and no GPU slot are involved.
pub(super) fn apply_preset(
    config: &AppConfig,
    preset_id: Option<&str>,
    req: &mut ImagesGenerationRequest,
) -> Result<(), Response> {
    let Some(id) = preset_id else {
        return Ok(());
    };
    let loaded = manifest(config).map_err(manifest_error_response)?;
    let Some(preset) = loaded.presets.iter().find(|preset| preset.id == id) else {
        return Err(super::openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("unknown preset id {id:?}"),
            Some("unknown_preset"),
        ));
    };
    compose(req, preset);
    Ok(())
}

/// Append preset text deterministically: caller text first, preset text
/// second; explicit caller text is never discarded.
fn compose(req: &mut ImagesGenerationRequest, preset: &Preset) {
    let prompt = req.prompt.trim();
    if let Some(style) = preset
        .style_prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        req.prompt = if prompt.is_empty() {
            style.to_owned()
        } else {
            format!("{prompt}, {style}")
        };
    } else {
        req.prompt = prompt.to_owned();
    }
    if let Some(negative) = preset
        .negative_prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let existing = req
            .negative_prompt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        req.negative_prompt = Some(match existing {
            Some(existing) => format!("{existing}, {negative}"),
            None => negative.to_owned(),
        });
    }
}

#[cfg(test)]
mod tests {
    use honkbox_registry::PRESET_SCHEMA_VERSION as SCHEMA;

    use super::*;

    fn request(prompt: &str, negative: Option<&str>) -> ImagesGenerationRequest {
        let mut body = serde_json::json!({ "prompt": prompt });
        if let Some(negative) = negative {
            body["negative_prompt"] = serde_json::Value::String(negative.to_owned());
        }
        serde_json::from_value(body).expect("request fixture")
    }

    #[test]
    fn compose_appends_without_dropping_caller_text() {
        let preset = Preset {
            id: "clean".to_owned(),
            version: SCHEMA,
            title: None,
            negative_prompt: Some(" blur ".to_owned()),
            style_prompt: Some("cinematic ".to_owned()),
            tags: Vec::new(),
        };
        let mut req = request("a fox", Some("jpeg artifacts"));
        compose(&mut req, &preset);
        assert_eq!(req.prompt, "a fox, cinematic");
        assert_eq!(req.negative_prompt.as_deref(), Some("jpeg artifacts, blur"));
        assert_eq!(req.size, None);
    }

    #[test]
    fn compose_fills_empty_fields_without_duplication() {
        let preset = Preset {
            id: "only-negative".to_owned(),
            version: 1,
            title: None,
            negative_prompt: Some("blurry".to_owned()),
            style_prompt: Some("  ".to_owned()),
            tags: Vec::new(),
        };
        let mut req = request(" a fox ", Some("  "));
        compose(&mut req, &preset);
        assert_eq!(req.prompt, "a fox");
        assert_eq!(req.negative_prompt.as_deref(), Some("blurry"));
    }
}
