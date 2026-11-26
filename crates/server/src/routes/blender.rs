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

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    BlenderJobAccepted, BlenderJobRequest, BlenderLimitsCapability, BlenderStage, OpenAiError,
};
use honkbox_external_tools::{AssetStage, MeshFormat, recipe_sha256};
use tokio::sync::OnceCell;

use crate::jobs::determinism::canonical_hash;

use super::AppConfig;
use super::blender_stages::{map_stage, mesh_format};
use super::chat_prep::{admit_then_register, with_job_id};
use super::external_tools::blender_engine;
use super::middleware::consumer_id;
use super::openai_constraints::openai_error_response;

/// The largest mesh the route hands to Blender (bytes).
pub const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;

/// The largest artifact set a job may produce (bytes).
pub const MAX_OUTPUT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The job's wall-clock bounds (seconds).
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;
pub const MAX_TIMEOUT_SECS: u64 = 3600;

/// The largest accepted `Idempotency-Key` header (characters).
const IDEMPOTENCY_KEY_MAX_CHARS: usize = 128;
/// Remembered submissions before oldest-first eviction.
const BLENDER_DEDUP_MAX_ENTRIES: usize = 256;

/// The enforced Blender job bounds, as advertised in
/// `GET /v1/capabilities`. The capability document reads THIS, so the
/// advertised bounds cannot drift from the ones the route enforces.
pub fn limits() -> BlenderLimitsCapability {
    BlenderLimitsCapability {
        max_input_bytes: MAX_INPUT_BYTES,
        max_output_bytes: MAX_OUTPUT_BYTES,
        default_timeout_secs: DEFAULT_TIMEOUT_SECS,
        max_timeout_secs: MAX_TIMEOUT_SECS,
    }
}

/// The default artifact name prefix.
const DEFAULT_PREFIX: &str = "output";

/// A named rejection: status, message, error code.
pub(super) type Rejection = (StatusCode, String, &'static str);

/// A settled Blender submission. The outcome is owned, not an axum `Response`:
/// a process-global dedup entry must be `Sync`, and `Body` is not. Replays
/// render a fresh response from the stored outcome.
struct BlenderSettled {
    succeeded: bool,
    outcome: SettledOutcome,
}

enum SettledOutcome {
    Accepted(BlenderJobAccepted),
    Rejected {
        status: StatusCode,
        message: String,
        code: Option<&'static str>,
    },
}

impl BlenderSettled {
    fn response(&self) -> Response {
        match &self.outcome {
            SettledOutcome::Accepted(accepted) => with_job_id(
                (StatusCode::ACCEPTED, Json(accepted.clone())).into_response(),
                &accepted.job_id,
            ),
            SettledOutcome::Rejected {
                status,
                message,
                code,
            } => openai_error_response(*status, message.clone(), *code),
        }
    }
}

impl From<Result<BlenderJobAccepted, Rejection>> for BlenderSettled {
    fn from(result: Result<BlenderJobAccepted, Rejection>) -> Self {
        match result {
            Ok(accepted) => Self {
                succeeded: true,
                outcome: SettledOutcome::Accepted(accepted),
            },
            Err((status, message, code)) => Self {
                succeeded: false,
                outcome: SettledOutcome::Rejected {
                    status,
                    message,
                    code: Some(code),
                },
            },
        }
    }
}

/// One remembered submission. Concurrent same-key callers await the same
/// `OnceCell`, so the first submission executes exactly once.
struct BlenderDedupEntry {
    command_hash: String,
    outcome: OnceCell<BlenderSettled>,
}

/// Process-local Blender idempotency ledger: (consumer, `Idempotency-Key`) to
/// entry, bounded at 256 entries with oldest-first eviction.
#[derive(Default)]
struct BlenderDedup {
    map: HashMap<(String, String), Arc<BlenderDedupEntry>>,
    order: VecDeque<(String, String)>,
}

/// Outcome of [`BlenderDedup::reserve`].
enum BlenderReservation {
    /// Fresh key: this caller executes and settles the entry.
    Execute(Arc<BlenderDedupEntry>),
    /// Same consumer + key + command hash: replay the first response.
    Replay(Arc<BlenderDedupEntry>),
    /// Same consumer + key with a different command hash.
    Conflict,
}

impl BlenderDedup {
    fn reserve(&mut self, consumer: &str, key: &str, command_hash: String) -> BlenderReservation {
        let key = (consumer.to_owned(), key.to_owned());
        if let Some(entry) = self.map.get(&key) {
            return if entry.command_hash == command_hash {
                BlenderReservation::Replay(Arc::clone(entry))
            } else {
                BlenderReservation::Conflict
            };
        }
        let entry = Arc::new(BlenderDedupEntry {
            command_hash,
            outcome: OnceCell::new(),
        });
        self.map.insert(key.clone(), Arc::clone(&entry));
        self.order.push_back(key);
        self.trim();
        BlenderReservation::Execute(entry)
    }

    /// Forget a failed attempt so a genuine retry re-executes.
    fn forget(&mut self, consumer: &str, key: &str) {
        let key = (consumer.to_owned(), key.to_owned());
        self.map.remove(&key);
        self.order.retain(|old| old != &key);
    }

    /// Evict oldest entries past the cap. The bound is soft while entries
    /// are in flight: an unsettled entry (a reserved key whose job is still
    /// being admitted) is rotated to the back instead of evicted, so a
    /// caller replaying the same Idempotency-Key can never re-execute a
    /// submission that is in flight. Only when every remaining entry is in
    /// flight does the ledger keep them all and exceed the cap. Keys absent
    /// from `map` are stale `order` leftovers and are dropped.
    fn trim(&mut self) {
        let mut examined = 0usize;
        while self.map.len() > BLENDER_DEDUP_MAX_ENTRIES && examined < self.order.len() {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            let unsettled = self
                .map
                .get(&oldest)
                .is_some_and(|entry| entry.outcome.get().is_none());
            if unsettled {
                self.order.push_back(oldest);
                examined += 1;
            } else {
                self.map.remove(&oldest);
            }
        }
    }
}

/// The Blender idempotency ledger (process-global, like the job registry).
static BLENDER_DEDUP: LazyLock<Mutex<BlenderDedup>> =
    LazyLock::new(|| Mutex::new(BlenderDedup::default()));

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
pub(crate) async fn post_blender_job(
    State(config): State<AppConfig>,
    headers: HeaderMap,
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
    let idempotency_key = match headers.get("Idempotency-Key") {
        Some(value) => match value
            .to_str()
            .ok()
            .filter(|key| (1..=IDEMPOTENCY_KEY_MAX_CHARS).contains(&key.chars().count()))
        {
            Some(key) => Some(key.to_owned()),
            None => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Idempotency-Key must be 1..={IDEMPOTENCY_KEY_MAX_CHARS} characters"),
                    Some("blender_idempotency_key_invalid"),
                );
            }
        },
        None => None,
    };
    if let Some(key) = idempotency_key {
        let consumer = consumer_id(&headers);
        let command_hash =
            canonical_hash(&serde_json::to_value(&req).unwrap_or(serde_json::Value::Null));
        let reserved = {
            // std mutex, no await inside: reserve-or-conflict is decided here.
            let mut dedup = BLENDER_DEDUP
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            dedup.reserve(&consumer, &key, command_hash)
        };
        let (entry, is_replay) = match reserved {
            BlenderReservation::Conflict => {
                return openai_error_response(
                    StatusCode::CONFLICT,
                    format!(
                        "Idempotency-Key {:?} was already used with a different request — retries must resend the identical request; use a fresh key for different work",
                        key
                    ),
                    Some("blender_idempotency_conflict"),
                );
            }
            BlenderReservation::Execute(entry) => (entry, false),
            BlenderReservation::Replay(entry) => (entry, true),
        };
        // Single flight: every same-key caller awaits the one execution and
        // replays its exact response.
        let settled = entry
            .outcome
            .get_or_init(|| {
                let cfg = config.clone();
                let req = req.clone();
                async move { BlenderSettled::from(execute_blender_job(cfg, req).await) }
            })
            .await;
        if !is_replay && !settled.succeeded {
            // A failed attempt is forgotten: a genuine retry can run.
            let mut dedup = BLENDER_DEDUP
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            dedup.forget(&consumer, &key);
        }
        return settled.response();
    }
    BlenderSettled::from(execute_blender_job(config, req).await).response()
}

/// Validate and execute one Blender job. Shared by fresh submissions and the
/// single-flight replay path; conversion stamps the response envelope once.
async fn execute_blender_job(
    config: AppConfig,
    req: BlenderJobRequest,
) -> Result<BlenderJobAccepted, Rejection> {
    let Some(data_dir) = config.data_dir.clone() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "no data dir: the artifact store and the job bundles are disabled".to_string(),
            "blender_storage_unavailable",
        ));
    };
    let mut plan = match plan_job(&req, config.artifacts.clone()) {
        Ok(plan) => plan,
        Err(rejection) => return Err(rejection),
    };
    let engine = match engine_gate(&req) {
        Ok(engine) => engine,
        Err(rejection) => return Err(rejection),
    };
    if !process_groups_supported() {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "blender jobs need a unix host: the process supervisor tears down the process group"
                .to_string(),
            "blender_unsupported_platform",
        ));
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
            return Err((status, message, "blender_busy"));
        }
    };
    let job_id = guard.id().to_string();
    plan.job_id.clone_from(&job_id);
    let bundle_id = ulid::Ulid::new().to_string();
    let bundle = data_dir.join("jobs").join(&bundle_id);
    if let Err(err) = std::fs::create_dir_all(&bundle) {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create the job bundle: {err}"),
            "blender_bundle_failed",
        ));
    }
    let token = guard.token().clone();
    super::blender_exec::start(plan, bundle, engine.binary, token, lease, guard);
    Ok(BlenderJobAccepted {
        job_id,
        status: "queued".to_string(),
        bundle: Some(bundle_id),
    })
}

/// The engine gate: no Blender installed is a 501, a version or a recipe
/// digest the caller did not expect is a 409. Both are checked BEFORE the
/// process starts.
fn engine_gate(req: &BlenderJobRequest) -> Result<super::external_tools::BlenderEngine, Rejection> {
    let Some(engine) = blender_engine() else {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "no blender engine is installed".to_string(),
            "blender_unavailable",
        ));
    };
    if engine.version != req.engine.blender_version {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "engine.blender_version {:?} does not match the installed blender {:?}",
                req.engine.blender_version, engine.version
            ),
            "blender_version_mismatch",
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
        return Err((
            StatusCode::CONFLICT,
            format!("engine.script_version does not match the installed recipe ({recipe})"),
            "blender_script_version_mismatch",
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

    fn accepted(job_id: &str) -> BlenderJobAccepted {
        BlenderJobAccepted {
            job_id: job_id.to_owned(),
            status: "queued".to_owned(),
            bundle: Some("bundle".to_owned()),
        }
    }

    fn settled(accepted: BlenderJobAccepted) -> BlenderSettled {
        BlenderSettled::from(Ok::<_, Rejection>(accepted))
    }

    #[test]
    fn ledger_replays_the_first_response() {
        let mut ledger = BlenderDedup::default();
        let entry = match ledger.reserve("consumer", "key", "hash".to_owned()) {
            BlenderReservation::Execute(entry) => entry,
            _ => panic!("a fresh key must execute"),
        };
        assert!(entry.outcome.set(settled(accepted("job-1"))).is_ok());
        match ledger.reserve("consumer", "key", "hash".to_owned()) {
            BlenderReservation::Replay(entry) => {
                let replayed = entry.outcome.get().expect("settled");
                assert!(replayed.succeeded);
                let response = replayed.response();
                assert_eq!(response.status(), StatusCode::ACCEPTED);
                assert_eq!(
                    response
                        .headers()
                        .get("X-Job-Id")
                        .unwrap()
                        .to_str()
                        .unwrap(),
                    "job-1"
                );
            }
            _ => panic!("same consumer, key and command must replay"),
        }
    }

    #[test]
    fn ledger_names_a_changed_command_as_a_conflict() {
        let mut ledger = BlenderDedup::default();
        assert!(matches!(
            ledger.reserve("consumer", "key", "first".to_owned()),
            BlenderReservation::Execute(_)
        ));
        assert!(matches!(
            ledger.reserve("consumer", "key", "second".to_owned()),
            BlenderReservation::Conflict
        ));
        assert!(matches!(
            ledger.reserve("other", "key", "second".to_owned()),
            BlenderReservation::Execute(_)
        ));
    }

    #[test]
    fn ledger_forgets_a_failed_attempt() {
        let mut ledger = BlenderDedup::default();
        let entry = match ledger.reserve("consumer", "key", "hash".to_owned()) {
            BlenderReservation::Execute(entry) => entry,
            _ => panic!("a fresh key must execute"),
        };
        let failed = BlenderSettled {
            succeeded: false,
            outcome: SettledOutcome::Rejected {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: "boom".to_owned(),
                code: None,
            },
        };
        assert!(!failed.succeeded);
        assert!(entry.outcome.set(failed).is_ok());
        ledger.forget("consumer", "key");
        match ledger.reserve("consumer", "key", "hash".to_owned()) {
            BlenderReservation::Execute(fresh) => {
                assert!(fresh.outcome.get().is_none());
            }
            _ => panic!("a failed attempt must be retriable"),
        }
    }

    /// Every outcome is settled before the cap overflows, so plain
    /// oldest-first eviction applies: `key-0` is gone after 257 keys.
    #[test]
    fn ledger_evicts_oldest_at_256_entries() {
        let mut ledger = BlenderDedup::default();
        for i in 0..=BLENDER_DEDUP_MAX_ENTRIES {
            let BlenderReservation::Execute(entry) =
                ledger.reserve("consumer", &format!("key-{i}"), "hash".to_owned())
            else {
                panic!("a fresh key must execute");
            };
            assert!(
                entry
                    .outcome
                    .set(settled(accepted(&format!("job-{i}"))))
                    .is_ok()
            );
        }
        assert_eq!(ledger.map.len(), BLENDER_DEDUP_MAX_ENTRIES);
        assert!(matches!(
            ledger.reserve("consumer", "key-0", "hash".to_owned()),
            BlenderReservation::Execute(_)
        ));
    }

    /// An in-flight entry is never evicted: with `key-0` still unsettled,
    /// the cap overflow takes the oldest SETTLED entry (`key-1`) instead,
    /// `key-0` survives, and replaying its key replays rather than
    /// re-executes a submission that is in flight.
    #[test]
    fn ledger_keeps_unsettled_entries_and_evicts_the_next_settled_one() {
        let mut ledger = BlenderDedup::default();
        let BlenderReservation::Execute(in_flight) =
            ledger.reserve("consumer", "key-0", "hash".to_owned())
        else {
            panic!("a fresh key must execute");
        };
        assert!(in_flight.outcome.get().is_none());
        for i in 1..=BLENDER_DEDUP_MAX_ENTRIES {
            let BlenderReservation::Execute(entry) =
                ledger.reserve("consumer", &format!("key-{i}"), "hash".to_owned())
            else {
                panic!("a fresh key must execute");
            };
            assert!(
                entry
                    .outcome
                    .set(settled(accepted(&format!("job-{i}"))))
                    .is_ok()
            );
        }
        let key0 = ("consumer".to_owned(), "key-0".to_owned());
        let key1 = ("consumer".to_owned(), "key-1".to_owned());
        assert!(ledger.map.contains_key(&key0), "in-flight key survives");
        assert!(
            !ledger.map.contains_key(&key1),
            "oldest settled key evicted"
        );
        assert!(matches!(
            ledger.reserve("consumer", "key-0", "hash".to_owned()),
            BlenderReservation::Replay(_)
        ));
    }

    /// N concurrent callers of one key share one entry: the first caller
    /// holds `Execute`, and the outcome initializes exactly once no matter
    /// how the runtime interleaves the tasks. Local ledger only — never the
    /// process-global static.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ledger_single_flights_concurrent_reserves() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut ledger = BlenderDedup::default();
        let BlenderReservation::Execute(entry) =
            ledger.reserve("consumer", "key", "hash".to_owned())
        else {
            panic!("a fresh key must execute");
        };
        let runs = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let entry = Arc::clone(&entry);
            let runs = Arc::clone(&runs);
            tasks.push(tokio::spawn(async move {
                entry
                    .outcome
                    .get_or_init(|| async {
                        runs.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        settled(accepted("once"))
                    })
                    .await
                    .response()
                    .status()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.expect("task joins"), StatusCode::ACCEPTED);
        }
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }
}
