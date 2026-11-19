//! `POST /v1/jobs/blender` — the typed Blender asset-job route
//! (FEAT-blender-job-route).
//!
//! The route admits a job ASYNCHRONOUSLY: it validates the typed request,
//! resolves the mesh from the content-addressed artifact store, checks the
//! engine contract against the probed Blender, takes the admission lease
//! and answers `202` with the registry id. The work runs in a supervised
//! Blender process; the lifecycle streams through `GET /v1/jobs` and
//! `GET /v1/events`.
//!
//! The route never accepts a filesystem path: the mesh (and every texture)
//! is resolved from the artifact store by id, and an artifact id IS the
//! content digest — a wrong digest is rejected before any process starts.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{BlenderJobAccepted, BlenderJobRequest, BlenderStage, OpenAiError};
use honkbox_external_tools::{AssetStage, MeshFormat, recipe_sha256};

use super::AppConfig;
use super::blender_stages::{map_stage, mesh_format};
use super::chat_prep::{admit_then_register, with_job_id};
use super::external_tools::blender_engine;
use super::openai_constraints::openai_error_response;

/// The largest mesh the route hands to Blender (bytes).
pub(super) const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;

/// The largest artifact set a job may produce (bytes).
const MAX_OUTPUT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The job's wall-clock bounds (seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 600;
const MAX_TIMEOUT_SECS: u64 = 3600;

/// The default artifact name prefix.
const DEFAULT_PREFIX: &str = "output";

/// A named rejection: status, message, error code.
pub(super) type Rejection = (StatusCode, String, &'static str);

/// Everything one accepted job needs: the recipe's stages (with every
/// artifact path resolved) and the path-free echo the manifest records.
#[derive(Debug)]
pub(super) struct JobPlan {
    /// The recipe's stages, in execution order.
    pub stages: Vec<AssetStage>,
    /// The typed stage echo (no paths) for the bundle manifest.
    pub echo: Vec<BlenderStage>,
    /// The input mesh on disk (resolved from the artifact store).
    pub input: PathBuf,
    /// The input mesh's format.
    pub format: MeshFormat,
    /// The input artifact id (the content digest).
    pub input_id: String,
    /// The artifact name prefix.
    pub prefix: String,
    /// The seed recorded in the manifest.
    pub seed: u64,
    /// The wall-clock bound.
    pub timeout: Duration,
    /// The caller's byte cap for the produced artifacts.
    pub max_output_bytes: u64,
    /// Shared registry used to promote committed outputs by content digest.
    pub artifacts: Arc<crate::artifacts::Registry>,
    /// The registered job id, filled from the guard after admission.
    pub job_id: String,
}

/// `POST /v1/jobs/blender` (FEAT-blender-job-route): admit a typed Blender
/// asset job. The route validates the request, resolves the mesh from the
/// artifact store, checks the engine contract against the probed Blender
/// and takes the admission lease; the work itself runs in a supervised
/// Blender process (`routes/blender_exec.rs`) and the outcome lands in the
/// job bundle (`<data_dir>/jobs/{bundle}/`).
#[utoipa::path(
    post,
    path = "/v1/jobs/blender",
    request_body = BlenderJobRequest,
    responses(
        (status = 202, description = "Job admitted; the bundle id names the job directory", body = BlenderJobAccepted),
        (status = 400, description = "Malformed request, unsupported manifest or output format, out-of-range stage params, or an input digest that does not match the artifact id", body = OpenAiError),
        (status = 404, description = "The input mesh (or a texture) is not in the artifact store", body = OpenAiError),
        (status = 409, description = "The engine contract does not match the installed Blender or recipe", body = OpenAiError),
        (status = 413, description = "The input mesh is over the byte cap", body = OpenAiError),
        (status = 429, description = "Admission busy (the ledger queue)", body = OpenAiError),
        (status = 501, description = "No Blender engine is installed, or the host cannot supervise a process group", body = OpenAiError),
        (status = 503, description = "No data dir: the artifact store and the job bundles are disabled", body = OpenAiError)
    )
)]
pub(super) async fn post_blender_job(
    State(config): State<AppConfig>,
    body: Result<Json<BlenderJobRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid blender job request: {}", rejection.body_text()),
                Some("blender_request_invalid"),
            );
        }
    };
    let Some(data_dir) = config.data_dir.clone() else {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "no data dir: the artifact store and the job bundles are disabled".to_string(),
            Some("blender_storage_unavailable"),
        );
    };
    let mut plan = match plan_job(&req, config.artifacts.clone()) {
        Ok(plan) => plan,
        Err(rejection) => return rejection_response(rejection),
    };
    let engine = match engine_gate(&req) {
        Ok(engine) => engine,
        Err(response) => return response,
    };
    if !process_groups_supported() {
        return openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "blender jobs need a unix host: the process supervisor tears down the process group"
                .to_string(),
            Some("blender_unsupported_platform"),
        );
    }
    // Admission first, registration second (the shared seam): a rejected
    // request never enters the registry.
    let (lease, guard) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Blender).await,
        crate::jobs_registry::JobKind::Blender,
        None,
        true,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => {
            return openai_error_response(status, message, Some("blender_busy"));
        }
    };
    let job_id = guard.id().to_string();
    plan.job_id.clone_from(&job_id);
    let bundle_id = ulid::Ulid::new().to_string();
    let bundle = data_dir.join("jobs").join(&bundle_id);
    if let Err(err) = std::fs::create_dir_all(&bundle) {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create the job bundle: {err}"),
            Some("blender_bundle_failed"),
        );
    }
    let token = guard.token().clone();
    super::blender_exec::start(plan, bundle, engine.binary, token, lease, guard);
    let accepted = BlenderJobAccepted {
        job_id: job_id.clone(),
        status: "queued".to_string(),
        bundle: Some(bundle_id),
    };
    with_job_id(
        (StatusCode::ACCEPTED, Json(accepted)).into_response(),
        &job_id,
    )
}

/// The engine gate: no Blender installed is a 501, a version or a recipe
/// digest the caller did not expect is a 409. Both are checked BEFORE the
/// process starts.
#[allow(clippy::result_large_err)] // axum Response is a fixed-size request-path type
fn engine_gate(req: &BlenderJobRequest) -> Result<super::external_tools::BlenderEngine, Response> {
    let Some(engine) = blender_engine() else {
        return Err(openai_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "no blender engine is installed".to_string(),
            Some("blender_unavailable"),
        ));
    };
    if engine.version != req.engine.blender_version {
        return Err(openai_error_response(
            StatusCode::CONFLICT,
            format!(
                "engine.blender_version {:?} does not match the installed blender {:?}",
                req.engine.blender_version, engine.version
            ),
            Some("blender_version_mismatch"),
        ));
    }
    // The recipe pin is optional: a caller that pins it gets a named 409
    // when the recipe changed, a caller that omits it runs the server's
    // current recipe. The pin is the sha256 of the recipe BYTES
    // (`recipe_sha256`) — a content digest. A digest of a script path (as
    // loop-lore's `scriptDigest` computes) can never match and earns the
    // same 409; callers without the byte digest simply omit the field.
    let recipe = recipe_sha256();
    if let Some(expected) = req.engine.script_version.as_deref()
        && expected != recipe
    {
        return Err(openai_error_response(
            StatusCode::CONFLICT,
            format!("engine.script_version does not match the installed recipe ({recipe})"),
            Some("blender_script_version_mismatch"),
        ));
    }
    Ok(engine)
}

/// Whether this host can supervise a Blender process group: the process
/// supervisor tears the whole GROUP down, and process groups are a unix
/// mechanism.
fn process_groups_supported() -> bool {
    cfg!(unix)
}

/// Validate the typed request and resolve every artifact it names.
/// `Err` is the named rejection (400 / 404 / 413).
fn plan_job(
    req: &BlenderJobRequest,
    artifacts: Arc<crate::artifacts::Registry>,
) -> Result<JobPlan, Rejection> {
    if req.manifest != 1 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported manifest version {}; this server speaks 1",
                req.manifest
            ),
            "blender_manifest_version",
        ));
    }
    if req.output.format != "glb" {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported output format {:?}; this server writes glb",
                req.output.format
            ),
            "blender_output_format",
        ));
    }
    if req.output.max_bytes == 0 || req.output.max_bytes > MAX_OUTPUT_BYTES {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("output.max_bytes must be 1..={MAX_OUTPUT_BYTES} bytes"),
            "blender_output_bytes",
        ));
    }
    if req.stages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "stages must not be empty".to_string(),
            "blender_stages_empty",
        ));
    }
    // An artifact id IS its content digest: a caller naming another digest
    // asks for bytes the store does not hold.
    if !req.input.sha256.eq_ignore_ascii_case(&req.input.asset_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            "input.sha256 must equal input.asset_id: an artifact id is its content digest"
                .to_string(),
            "blender_input_sha256_mismatch",
        ));
    }
    let params = req.params.as_ref();
    let timeout_secs = params
        .and_then(|params| params.timeout_secs)
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    if timeout_secs == 0 || timeout_secs > MAX_TIMEOUT_SECS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("params.timeout_secs must be 1..={MAX_TIMEOUT_SECS} seconds"),
            "blender_timeout",
        ));
    }
    let prefix = params
        .and_then(|params| params.prefix.as_deref())
        .unwrap_or(DEFAULT_PREFIX);
    if !is_safe_prefix(prefix) {
        return Err((
            StatusCode::BAD_REQUEST,
            "params.prefix must be 1..=64 characters of [A-Za-z0-9_-]".to_string(),
            "blender_prefix",
        ));
    }
    let (input, input_bytes) = artifacts.stored_blob(&req.input.asset_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("input artifact {:?} is not stored", req.input.asset_id),
            "blender_input_not_found",
        )
    })?;
    if input_bytes > MAX_INPUT_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("the input mesh is {input_bytes} bytes; the cap is {MAX_INPUT_BYTES}"),
            "blender_input_too_large",
        ));
    }
    let mut stages = Vec::with_capacity(req.stages.len());
    for stage in &req.stages {
        stages.push(map_stage(stage, &artifacts)?);
    }
    Ok(JobPlan {
        stages,
        echo: req.stages.clone(),
        input,
        format: mesh_format(req.input.format),
        input_id: req.input.asset_id.to_ascii_lowercase(),
        prefix: prefix.to_string(),
        seed: params.map_or(0, |params| params.seed),
        timeout: Duration::from_secs(timeout_secs),
        max_output_bytes: req.output.max_bytes,
        artifacts,
        job_id: String::new(), // replaced with the guard's id at admission
    })
}

/// Whether an artifact prefix is a safe file name.
fn is_safe_prefix(prefix: &str) -> bool {
    !prefix.is_empty()
        && prefix.len() <= 64
        && prefix
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The named rejection as an error response.
fn rejection_response(rejection: Rejection) -> Response {
    let (status, message, code) = rejection;
    openai_error_response(status, message, Some(code))
}

#[cfg(test)]
mod tests {
    use honkbox_api_types::{
        BlenderEngine, BlenderJobInput, BlenderJobParams, BlenderMeshFormat, BlenderOutput,
    };

    use super::*;

    /// A request with the shape the route accepts. The mesh is NOT stored,
    /// so every rejection below is one the route answers BEFORE it looks
    /// at the store.
    fn request() -> BlenderJobRequest {
        BlenderJobRequest {
            input: BlenderJobInput {
                asset_id: "a".repeat(64),
                format: BlenderMeshFormat::Glb,
                sha256: "a".repeat(64),
            },
            stages: vec![BlenderStage::ExportGlb],
            params: None,
            engine: BlenderEngine {
                blender_version: "4.5.15".to_owned(),
                script_version: None,
            },
            output: BlenderOutput {
                format: "glb".to_owned(),
                max_bytes: 1024,
            },
            manifest: 1,
        }
    }

    /// An empty store: nothing is stored, so every lookup misses.
    fn store() -> crate::artifacts::Registry {
        crate::artifacts::Registry::new(None, 1024)
    }

    /// The code of the named rejection the request earns.
    fn code(req: &BlenderJobRequest) -> &'static str {
        plan_job(req, Arc::new(store())).unwrap_err().2
    }

    #[test]
    fn the_request_shape_is_checked_before_the_store() {
        let mut req = request();
        req.manifest = 2;
        assert_eq!(code(&req), "blender_manifest_version");
        let mut req = request();
        req.output.format = "obj".to_owned();
        assert_eq!(code(&req), "blender_output_format");
        let mut req = request();
        req.output.max_bytes = 0;
        assert_eq!(code(&req), "blender_output_bytes");
        let mut req = request();
        req.stages.clear();
        assert_eq!(code(&req), "blender_stages_empty");
        let mut req = request();
        req.input.sha256 = "c".repeat(64);
        assert_eq!(code(&req), "blender_input_sha256_mismatch");
        let mut req = request();
        req.params = Some(BlenderJobParams {
            seed: 0,
            prefix: Some("../escape".to_owned()),
            timeout_secs: Some(0),
        });
        assert_eq!(code(&req), "blender_timeout");
    }

    #[test]
    fn an_unstored_mesh_is_a_404() {
        assert_eq!(code(&request()), "blender_input_not_found");
    }
}
