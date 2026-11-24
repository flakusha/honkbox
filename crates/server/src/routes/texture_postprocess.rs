//! `POST /v1/texture-postprocess` — OIIO texture post-process (`TASK-texture-postprocess-wiring`).
//!
//! Drives the `honkbox_external_tools` OIIO API; argv, colorspace rules and
//! the sidecar are not reimplemented here. The toolchain is probed once by
//! `crate::texture_postprocess`; an absent binary is a named 501
//! (`texture_postprocess_unavailable`) before any process starts.
//!
//! CPU-only: `oiiotool`/`maketx` are external binaries, not GPU inference —
//! no residency lease or `ResourceKind` admission, only `RUN_TIMEOUT`.
//! Outputs land in the job bundle that `job_content.rs` serves, so the
//! sidecar and every `.tx` reach the export consumer.

use std::path::Path;

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_external_tools::{
    ColorSpace, InputMap, MapRole, PostprocessError, PostprocessJob, PostprocessManifest,
    run_texture_postprocess,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::{AppConfig, openai_error_response};

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub(crate) struct TexturePostprocessMap {
    pub role: String,
    /// Content digest of an already-uploaded artifact — never a host path.
    pub artifact_id: String,
    pub colorspace: Option<String>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub(crate) struct TexturePostprocessRequest {
    pub name: String,
    pub maps: Vec<TexturePostprocessMap>,
    pub tile_size: u32,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TexturePostprocessArtifact {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
    pub role: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TexturePostprocessResult {
    pub bundle: String,
    pub manifest: serde_json::Value,
    pub artifacts: Vec<String>,
}

/// The named 501 body: the tool's own verdict, never a binary path.
fn unavailable() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        crate::texture_postprocess::unavailable_error(),
        Some("texture_postprocess_unavailable"),
    )
}

/// Split from the handler so the exact 501 contract is testable without the
/// process-wide frozen probe.
fn gate(available: bool) -> Option<Response> {
    if available { None } else { Some(unavailable()) }
}

/// `POST /v1/texture-postprocess`: post-process a PBR map set and publish the
/// sidecar and `.tx` maps as a job bundle. The gate runs before the body is
/// validated or any process starts.
#[utoipa::path(
    post,
    path = "/v1/texture-postprocess",
    request_body = TexturePostprocessRequest,
    responses(
        (status = 200, description = "Sidecar and bundle carrying the .tx maps", body = TexturePostprocessResult),
        (status = 400, description = "Invalid job", body = honkbox_api_types::OpenAiError),
        (status = 500, description = "Tool, bundle write or I/O failure", body = honkbox_api_types::OpenAiError),
        (status = 501, description = "OIIO tools absent or broken", body = honkbox_api_types::OpenAiError),
        (status = 503, description = "No data dir: job bundles disabled", body = honkbox_api_types::OpenAiError)
    )
)]
#[allow(clippy::too_many_lines)] // one validation + staging + sidecar flow by design
pub(crate) async fn post_texture_postprocess(
    State(config): State<AppConfig>,
    body: Result<Json<TexturePostprocessRequest>, JsonRejection>,
) -> Response {
    if let Some(response) = gate(crate::texture_postprocess::capability().available) {
        return response;
    }
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "invalid texture post-process request: {}",
                    rejection.body_text()
                ),
                Some("invalid_texture_postprocess_request"),
            );
        }
    };
    let Some(data_dir) = config.data_dir.clone() else {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "no data dir: the job bundles are disabled".to_string(),
            Some("texture_postprocess_storage_unavailable"),
        );
    };
    let bundle_id = ulid::Ulid::new().to_string();
    let bundle = data_dir.join("jobs").join(&bundle_id);
    // Inputs resolve only through the artifact store; outputs always land in
    // this server-owned bundle. No client string becomes a host path.
    let job = match build_job(&req, &config.artifacts, &bundle) {
        Ok(job) => job,
        Err(message) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                message,
                Some("invalid_texture_postprocess_request"),
            );
        }
    };
    if let Err(err) = std::fs::create_dir_all(&bundle) {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create the job bundle: {err}"),
            Some("texture_postprocess_bundle_failed"),
        );
    }

    let (oiiotool, maketx) = crate::texture_postprocess::binaries();
    let run = tokio::task::spawn_blocking(move || {
        run_texture_postprocess(
            &job,
            &oiiotool,
            &maketx,
            crate::texture_postprocess::RUN_TIMEOUT,
        )
    });
    let manifest = match run.await {
        Ok(Ok(manifest)) => manifest,
        // The binaries can disappear between the frozen probe and the run.
        Ok(Err(PostprocessError::Unavailable(_))) => return unavailable(),
        Ok(Err(PostprocessError::InvalidJob(reason))) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid texture post-process job: {reason}"),
                Some("invalid_texture_postprocess_request"),
            );
        }
        Ok(Err(PostprocessError::InvalidMapSet(reason))) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid PBR map set: {reason}"),
                Some("pbr_map_set_mismatch"),
            );
        }
        Ok(Err(err)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("texture post-process failed: {err}"),
                Some("texture_postprocess_failed"),
            );
        }
        Err(err) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("texture post-process task failed: {err}"),
                Some("texture_postprocess_failed"),
            );
        }
    };
    let (value, artifacts) = match publish_bundle(&bundle, &manifest) {
        Ok(pair) => pair,
        Err(message) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                message,
                Some("texture_postprocess_bundle_failed"),
            );
        }
    };
    (
        StatusCode::OK,
        Json(TexturePostprocessResult {
            bundle: bundle_id,
            manifest: value,
            artifacts,
        }),
    )
        .into_response()
}

/// Overwrite the toolchain sidecar in the bundle with the same data plus the
/// `artifacts[]` list the bundle-content route reads. One file serves as both
/// the sidecar and the bundle index; no host path enters it.
fn publish_bundle(
    bundle: &Path,
    manifest: &PostprocessManifest,
) -> Result<(serde_json::Value, Vec<String>), String> {
    let mut value = serde_json::to_value(manifest)
        .map_err(|err| format!("cannot encode the texture post-process manifest: {err}"))?;
    let mut artifacts =
        Vec::with_capacity(manifest.maps.len() + usize::from(manifest.orm_pack.is_some()));
    let mut names = Vec::with_capacity(artifacts.capacity());
    for map in &manifest.maps {
        let name = file_name(&map.path)?;
        artifacts.push(TexturePostprocessArtifact {
            name: name.to_string(),
            bytes: map.bytes,
            sha256: map.sha256.clone(),
            role: Some(map.role.to_string()),
        });
        names.push(name.to_string());
    }
    if let Some(orm) = &manifest.orm_pack {
        let name = file_name(&orm.path)?;
        artifacts.push(TexturePostprocessArtifact {
            name: name.to_string(),
            bytes: orm.bytes,
            sha256: orm.sha256.clone(),
            role: None,
        });
        names.push(name.to_string());
    }
    // The toolchain sidecar carries host paths (binary, input blobs, bundle
    // output). Publish only path basenames: the sidecar stays a useful
    // audit trail without leaking the host layout through the API.
    if let Some(object) = value.as_object_mut() {
        object.remove("job");
        if let Some(maps) = object.get_mut("maps").and_then(|v| v.as_array_mut()) {
            for map in maps {
                if let Some(map) = map.as_object_mut() {
                    redact_path(map, "path")?;
                }
            }
        }
        if let Some(orm) = object.get_mut("orm_pack").and_then(|v| v.as_object_mut()) {
            redact_path(orm, "path")?;
            redact_path(orm, "packed_png")?;
        }
        if let Some(commands) = object.get_mut("commands").and_then(|v| v.as_array_mut()) {
            for command in commands.iter_mut() {
                if let Some(args) = command.as_array_mut() {
                    for arg in args.iter_mut() {
                        if let Some(raw) = arg.as_str()
                            && Path::new(raw).is_absolute()
                        {
                            *arg =
                                serde_json::Value::String(file_name(Path::new(raw))?.to_string());
                        }
                    }
                }
            }
        }
    }
    value["artifacts"] = serde_json::to_value(artifacts)
        .map_err(|err| format!("cannot encode the bundle artifact list: {err}"))?;
    let bytes = serde_json::to_vec_pretty(&value)
        .map_err(|err| format!("cannot encode the bundle manifest: {err}"))?;
    crate::jobs::write_atomic(bundle, "manifest.json", &bytes)?;
    Ok((value, names))
}

/// Replace a published host path with its file name.
fn redact_path(
    object: &mut serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<(), String> {
    let Some(raw) = object.get(field).and_then(serde_json::Value::as_str) else {
        return Ok(());
    };
    let name = file_name(Path::new(raw))?.to_owned();
    object.insert(field.to_owned(), serde_json::Value::String(name));
    Ok(())
}

/// A bundle entry must be a plain file name, never a path.
fn file_name(path: &Path) -> Result<&str, String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "a post-processed map has no valid file name".to_string())
}

/// Validate the request and turn it into the toolchain's job shape. Every
/// input is an artifact digest, resolved by the store; a client never names
/// a host path and the output dir is always the server-owned job bundle.
fn build_job(
    req: &TexturePostprocessRequest,
    artifacts: &crate::artifacts::Registry,
    bundle: &Path,
) -> Result<PostprocessJob, String> {
    if req.name.is_empty()
        || req.name.len() > 64
        || !req
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("name must be 1..=64 characters of [A-Za-z0-9_-]".to_string());
    }
    if req.maps.is_empty() {
        return Err("no maps to post-process".to_string());
    }
    if req.tile_size == 0 || !req.tile_size.is_multiple_of(16) {
        return Err("tile_size must be a non-zero multiple of 16".to_string());
    }
    let mut maps = Vec::with_capacity(req.maps.len());
    for map in &req.maps {
        let role = map_role(&map.role)?;
        let colorspace = match &map.colorspace {
            Some(claim) => Some(color_space(claim)?),
            None => None,
        };
        let id = map.artifact_id.to_ascii_lowercase();
        if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("map.artifact_id must be a 64-character sha256 digest".to_string());
        }
        let (path, _) = artifacts
            .stored_blob(&id)
            .ok_or_else(|| format!("map artifact {id} is not stored"))?;
        maps.push(InputMap {
            role,
            path,
            colorspace,
        });
    }
    Ok(PostprocessJob {
        name: req.name.clone(),
        maps,
        output_dir: bundle.to_path_buf(),
        tile_size: req.tile_size,
    })
}

/// The role token → the toolchain's role enum.
fn map_role(raw: &str) -> Result<MapRole, String> {
    Ok(match raw {
        "albedo" => MapRole::Albedo,
        "normal" => MapRole::Normal,
        "rough" => MapRole::Rough,
        "metal" => MapRole::Metal,
        "ao" => MapRole::Ao,
        "height" => MapRole::Height,
        "orm" => MapRole::Orm,
        other => return Err(format!("unknown map role: {other}")),
    })
}

/// The colorspace token → the toolchain's colorspace enum.
fn color_space(raw: &str) -> Result<ColorSpace, String> {
    Ok(match raw {
        "srgb" => ColorSpace::Srgb,
        "linear" => ColorSpace::Linear,
        other => return Err(format!("unknown colorspace: {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unavailable_toolchain_is_a_named_501() {
        // The probe is a frozen LazyLock and this host may have the real
        // binaries, so the unavailable state is injected at the gate — the
        // exact branch the handler takes when capability().available is
        // false — instead of racing a process-global env override.
        assert!(gate(true).is_none());
        let response = gate(false).expect("an unavailable toolchain must be gated");
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read the 501 body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON error body");
        assert_eq!(body["error"]["code"], "texture_postprocess_unavailable");
    }

    #[test]
    fn published_manifest_never_carries_host_paths() {
        use honkbox_external_tools::{OrmPack, OutputMap};

        let dir = std::env::temp_dir().join(format!("honkbox-redact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("bundle dir");
        let albedo = dir.join("albedo.tx");
        let manifest = PostprocessManifest {
            manifest: 1,
            oiiotool_version: "2.5".to_owned(),
            maketx_version: "7.1".to_owned(),
            job: PostprocessJob {
                name: "set".to_owned(),
                maps: Vec::new(),
                output_dir: dir.clone(),
                tile_size: 256,
            },
            commands: vec![vec![
                "/usr/bin/oiiotool".to_owned(),
                albedo.to_string_lossy().into_owned(),
            ]],
            maps: vec![OutputMap {
                role: MapRole::Albedo,
                path: albedo,
                sha256: "hash".to_owned(),
                width: 8,
                height: 8,
                colorspace: ColorSpace::Srgb,
                mip_levels: 4,
                bytes: 1,
            }],
            orm_pack: Some(OrmPack {
                path: dir.join("orm.tx"),
                packed_png: dir.join("orm.png"),
                sha256: "hash".to_owned(),
                width: 8,
                height: 8,
                mip_levels: 4,
                bytes: 1,
                sources: [MapRole::Ao, MapRole::Rough, MapRole::Metal],
            }),
            warnings: Vec::new(),
            missing_required_maps: Vec::new(),
            bit_depth_unvalidated: Vec::new(),
        };

        let (value, _) = publish_bundle(&dir, &manifest).expect("published manifest");
        let host = dir.to_string_lossy();
        assert!(!value.to_string().contains(host.as_ref()), "{value}");
        assert_eq!(value["maps"][0]["path"], "albedo.tx");
        assert_eq!(value["orm_pack"]["path"], "orm.tx");
        assert_eq!(value["orm_pack"]["packed_png"], "orm.png");
        std::fs::remove_dir_all(&dir).ok();
    }
}
