//! Bounded, scope-aware mesh reroute handler (TASK-reroute-policy).
//!
//! `POST /v1/mesh/reroute` accepts one chat turn, selects at most
//! `ReroutePolicy::attempt_budget` providers, and forwards through
//! [`MeshProxy::forward`]. The client keeps the whole header trust boundary:
//! routing hints (`x-mesh-scope`, `x-mesh-target`, `x-mesh-exclude`) are read
//! here and never travel upstream, as are all caller credentials.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, Method},
    response::{IntoResponse, Response},
};
use honkbox_api_types::OpenAiError;

use crate::config::mesh::{
    Cooldowns, DEFAULT_ATTEMPT_BUDGET, DEFAULT_COOLDOWN, MAX_ATTEMPT_BUDGET, MESH_EXCLUDE_HEADER,
    MESH_SCOPE_HEADER, MESH_TARGET_HEADER, MeshConfig, MeshError, MeshMetricKind, MeshProxy,
    RerouteHints, model_group, parse_name_list, parse_scope, record_mesh_metric, select_upstreams,
};

/// Route path the orchestrator mounts this handler on.
pub(crate) const MESH_REROUTE_PATH: &str = "/v1/mesh/reroute";

/// Resolved `[mesh]` config installed once at startup. `AppConfig` is the
/// per-route state and is deliberately rebuilt in tests, so the operator's
/// mesh config lives here instead of widening that struct.
static RUNTIME_CONFIG: OnceLock<MeshConfig> = OnceLock::new();

/// Install the startup config before the router is built. Later calls are
/// no-ops (the router is built once).
#[allow(dead_code)] // only main.rs installs it; #[path] test harnesses omit main
pub(crate) fn install_mesh_config(config: MeshConfig) {
    let _ = RUNTIME_CONFIG.set(config);
}

/// Runtime config for the mounted handler; default-off when unset (tests).
pub(super) fn runtime_mesh_config() -> &'static MeshConfig {
    RUNTIME_CONFIG.get_or_init(MeshConfig::default)
}

/// Mesh-compatible endpoint on each upstream. A private request is routed to
/// a private provider (by its `@private` label); the forwarded path stays the
/// same so the upstream's trust boundary is unchanged.
const FORWARD_PATH: &str = "/v1/chat/completions";

/// Router-wide body ceiling; the body is read once for routing hints and the
/// same bytes are forwarded unchanged on every attempt.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Explicit, bounded retry policy. `attempt_budget` is the TOTAL number of
/// forwards (first try plus retries); 1 means no retries at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReroutePolicy {
    pub attempt_budget: usize,
    pub cooldown: Duration,
}

impl Default for ReroutePolicy {
    fn default() -> Self {
        Self {
            attempt_budget: DEFAULT_ATTEMPT_BUDGET,
            cooldown: DEFAULT_COOLDOWN,
        }
    }
}

/// Handler state: the immutable proxy plus per-provider cooldowns. Clone-safe
/// for `State<Arc<_>>`; the cooldown mutex is never held across an await.
#[derive(Clone)]
pub(crate) struct RerouteState {
    proxy: MeshProxy,
    enabled: bool,
    policy: ReroutePolicy,
    cooldowns: Arc<Cooldowns>,
}

impl RerouteState {
    /// Build from the resolved `[mesh]` config with the default policy.
    pub(crate) fn new(config: &MeshConfig) -> Result<Self, reqwest::Error> {
        Self::with_policy(config, ReroutePolicy::default())
    }

    /// Build with explicit bounds (also used by the cooldown contract tests).
    pub(crate) fn with_policy(
        config: &MeshConfig,
        policy: ReroutePolicy,
    ) -> Result<Self, reqwest::Error> {
        let policy = ReroutePolicy {
            attempt_budget: policy.attempt_budget.clamp(1, MAX_ATTEMPT_BUDGET),
            ..policy
        };
        Ok(Self {
            proxy: MeshProxy::new(config)?,
            enabled: config.enabled,
            policy,
            cooldowns: Arc::new(Cooldowns::new()),
        })
    }
}

/// Reject a header supplied more than once: a duplicated pin is ambiguous,
/// and silently taking the first value would be a routing decision made by
/// an attacker-controlled framing.
fn single_header<'a>(
    headers: &'a HeaderMap,
    name: &'static str,
) -> Result<Option<&'a HeaderValue>, MeshError> {
    let mut values = headers.get_all(name).iter();
    let first = values.next();
    if values.next().is_some() {
        return Err(MeshError::InvalidHeader(name));
    }
    Ok(first)
}

/// House envelope: `{"error":{message,type,param,code}}` with a named code.
fn error_response(err: MeshError) -> Response {
    let status = err.response_status();
    let code = err.code().to_owned();
    let message = match err {
        // reqwest transport text may contain a URL; never echo it.
        MeshError::UpstreamUnreachable { name, .. } => {
            format!("mesh upstream {name:?} is unavailable")
        }
        other => other.to_string(),
    };
    (
        status,
        Json(OpenAiError::new(status.as_u16(), message, Some(code))),
    )
        .into_response()
}

/// Every failure goes through one place so the metric `kind` stays bounded.
fn fail(err: MeshError) -> Response {
    match &err {
        MeshError::InvalidHeader(_) | MeshError::InvalidBody | MeshError::TargetConflict(_) => {
            record_mesh_metric(MeshMetricKind::Rejected);
        }
        MeshError::PrivateScopeBlocked => record_mesh_metric(MeshMetricKind::PrivateBlocked),
        _ => {}
    }
    error_response(err)
}

/// `POST /v1/mesh/reroute` — one chat turn, bounded scope-aware fallback.
///
/// Malformed hints are a named 400, conflicting/unknown targets a named 409,
/// and an unavailable provider or exhausted budget a named 502. A successful
/// response is the upstream's streamed body with the response allowlist only.
#[utoipa::path(
    post,
    path = "/v1/mesh/reroute",
    responses(
        (status = 200, description = "Relayed upstream response"),
        (status = 400, description = "Malformed routing hint", body = OpenAiError),
        (status = 409, description = "Conflicting mesh target", body = OpenAiError),
        (status = 502, description = "Upstream unavailable or budget exhausted", body = OpenAiError)
    ),
    tag = "mesh"
)]
pub(crate) async fn post_mesh_reroute(
    State(state): State<Arc<RerouteState>>,
    request: Request,
) -> Response {
    if !state.enabled {
        return fail(MeshError::Disabled);
    }
    let (parts, body) = request.into_parts();
    let headers = &parts.headers;
    let scope_header = match single_header(headers, MESH_SCOPE_HEADER) {
        Ok(value) => value,
        Err(err) => return fail(err),
    };
    let targets = match single_header(headers, MESH_TARGET_HEADER)
        .and_then(|value| parse_name_list(value, MESH_TARGET_HEADER))
    {
        Ok(names) => names,
        Err(err) => return fail(err),
    };
    let exclude = match single_header(headers, MESH_EXCLUDE_HEADER)
        .and_then(|value| parse_name_list(value, MESH_EXCLUDE_HEADER))
    {
        Ok(names) => names,
        Err(err) => return fail(err),
    };
    let Ok(body) = axum::body::to_bytes(body, MAX_BODY_BYTES).await else {
        return fail(MeshError::InvalidBody);
    };
    let model = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| value.get("model")?.as_str().map(str::to_owned));
    let scope = match parse_scope(scope_header, parts.uri.path(), model.as_deref()) {
        Ok(scope) => scope,
        Err(err) => return fail(err),
    };
    let hints = RerouteHints {
        scope,
        group: model.as_deref().map(model_group).map(str::to_owned),
        targets,
        exclude,
    };

    let now = Instant::now();
    let plan = match select_upstreams(
        state.proxy.upstreams(),
        &hints,
        |name| {
            let cooling = state.cooldowns.is_cooling(name, now);
            if cooling {
                record_mesh_metric(MeshMetricKind::Cooldown);
            }
            cooling
        },
        state.policy.attempt_budget,
    ) {
        Ok(plan) => plan,
        Err(err) => return fail(err),
    };

    let mut attempts = 0;
    for upstream in plan {
        attempts += 1;
        if let Ok(forwarded) = state
            .proxy
            .forward(upstream, Method::POST, FORWARD_PATH, headers, body.clone())
            .await
        {
            state.cooldowns.clear(&upstream.name);
            record_mesh_metric(MeshMetricKind::Selected);
            if attempts > 1 {
                record_mesh_metric(MeshMetricKind::Retry);
            }
            let status = forwarded.status;
            let mut builder = Response::builder().status(status);
            for (name, value) in &forwarded.headers {
                builder = builder.header(name, value);
            }
            return builder
                .body(forwarded.body)
                .unwrap_or_else(|_| fail(MeshError::InvalidBody));
        }
        // Only transport failures reach here (body failures are stream
        // items): no retry of a partially streamed response.
        state
            .cooldowns
            .record_failure(&upstream.name, state.policy.cooldown, Instant::now());
        if attempts > 1 {
            record_mesh_metric(MeshMetricKind::Retry);
        }
    }
    record_mesh_metric(MeshMetricKind::Exhausted);
    fail(MeshError::RetryBudgetExhausted { attempts })
}
