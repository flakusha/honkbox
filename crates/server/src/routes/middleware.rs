use std::time::Instant;

use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::AppConfig;
use super::runtime_config;

/// Bearer auth middleware (REQ-7). `route_layer` runs ONLY for matched
/// routes, so the UI/static fallback (and its assets) is structurally
/// exempt — the web UI keeps working without a key. `GET /health` and
/// `GET /metrics` are exempt too (probes/scrapers). When
/// `[server] auth = "bearer"`, every other route requires
/// `Authorization: Bearer <key>`; the key VALUE is read from the
/// `[auth] api_key_env` env var AT REQUEST TIME (`std::env` is
/// process-global, safe in async; rotation works without a restart) and is
/// never logged, never in metrics, never echoed in responses. Missing/
/// unset env is fail-closed: every bearer request gets 401.
///
/// Opt-in per-consumer scopes (TASK-per-consumer-api-scopes): when
/// `runtime_config::scope_facts()` names key env vars with grants, the key's
/// route family is checked here and a miss is a named 403. An empty scope map
/// keeps the legacy single-key path byte-for-byte. Signed content requests
/// bypass the header check because their HMAC is verified in `job_content`.
pub(super) async fn require_bearer(
    State(cfg): State<AppConfig>,
    req: Request,
    next: Next,
) -> Response {
    if cfg.auth.mode != crate::config::AuthMode::Bearer
        || is_probe_exempt(req.method(), req.uri().path())
        || is_signed_content_request(&req)
    {
        return next.run(req).await;
    }
    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .map(str::trim)
        .filter(|key| !key.is_empty());
    // Scoped consumers are keyed by env NAME; only the VALUE is compared, in
    // constant time, against the presented bearer key. No scope config means
    // the legacy single-key check below runs exactly as before.
    let (scopes_enabled, scope_map) = runtime_config::scope_facts();
    if scopes_enabled && !scope_map.is_empty() {
        let required = required_scope(req.method(), req.uri().path());
        if let Some(allowed) = scoped_key_verdict(provided, &scope_map, required) {
            if !allowed {
                return insufficient_scope_response(required);
            }
            // Minting a signed content URL is a write that hands back a READ
            // credential; a write-only consumer must not be able to mint one.
            if mints_read_credential(req.method(), req.uri().path())
                && !matches!(scoped_key_verdict(provided, &scope_map, "read"), Some(true))
            {
                return insufficient_scope_response("read");
            }
            return next.run(req).await;
        }
    }
    let expected = env_var(&cfg.auth.api_key_env).filter(|key| !key.is_empty());
    let ok = matches!((expected, provided), (Some(e), Some(p)) if constant_time_eq(e.as_bytes(), p.as_bytes()));
    if ok {
        return next.run(req).await;
    }
    unauthorized_response()
}

/// Keyless routes: the liveness/metrics probes, GET only (so a scraper
/// cannot write anything and probes stay cheap).
pub(super) fn is_probe_exempt(method: &axum::http::Method, path: &str) -> bool {
    method == axum::http::Method::GET && (path == "/health" || path == "/metrics")
}

/// Signed content requests carry their credential in additive query params.
/// Presence alone grants passage through this layer: `job_content` verifies
/// the HMAC and expiry before any file read, so a forged set still gets the
/// named `invalid_signed_url` 401. Only the two content routes qualify.
fn is_signed_content_request(req: &Request) -> bool {
    if req.method() != axum::http::Method::GET {
        return false;
    }
    let path = req.uri().path();
    if !(path.contains("/content/") || path.ends_with("/bundle.zip")) {
        return false;
    }
    // `scope` is optional in the signature (it defaults to `read`), so only
    // the two non-optional params gate this exemption. Parse pairs rather
    // than searching raw substrings: `junk=expires=0` has no `expires` key,
    // and a substring check would skip bearer for a default (unsigned) query.
    req.uri().query().is_some_and(|query| {
        let value = |key: &str| {
            query.split('&').find_map(|pair| {
                let (name, value) = pair.split_once('=')?;
                (name == key && !value.is_empty()).then_some(value)
            })
        };
        value("expires").is_some() && value("sig").is_some()
    })
}

/// `Some(allowed)` when the presented key belongs to a configured scoped env
/// name; `None` when it is not a scoped consumer (legacy or unknown key).
fn scoped_key_verdict(
    key: Option<&str>,
    scope_map: &[(String, Vec<String>)],
    required: &str,
) -> Option<bool> {
    let key = key?;
    for (env_name, scopes) in scope_map {
        let expected = env_var(env_name).filter(|value| !value.is_empty());
        if matches!((expected, key), (Some(e), p) if constant_time_eq(e.as_bytes(), p.as_bytes())) {
            return Some(scopes.iter().any(|scope| scope.as_str() == required));
        }
    }
    None
}

/// `POST /v1/content-urls` mints a signed READ credential, so a scoped
/// consumer needs both `write` (the verb) and `read` (what it hands back).
fn mints_read_credential(method: &axum::http::Method, path: &str) -> bool {
    method == axum::http::Method::POST && path == "/v1/content-urls"
}

/// Route-family → required scope. Admin families are checked first for every
/// verb, then all non-GET verbs are writes, then GET is a read. Family-level
/// only: no per-route special cases.
fn required_scope(method: &axum::http::Method, path: &str) -> &'static str {
    if path.starts_with("/v1/downloads")
        || path.starts_with("/v1/external-tools")
        || path == "/v1/models/refresh"
        || path.starts_with("/api/config")
        || path.starts_with("/v1/config")
    {
        return "admin";
    }
    if method != axum::http::Method::GET {
        return "write";
    }
    "read"
}

/// Named 403 carrying the required scope; no key value is echoed.
fn insufficient_scope_response(required: &str) -> Response {
    super::openai_constraints::openai_error_response(
        StatusCode::FORBIDDEN,
        format!("Missing required scope: {required}"),
        Some("insufficient_scope"),
    )
}

/// Env read for the bearer key VALUE (REQ-7). Production: `std::env`, per
/// request, so rotation needs no restart. Test builds (this module is
/// `#[path]`-included into integration tests that run with
/// `forbid(unsafe_code)`): [`set_env_var`] injects overrides instead of
/// mutating the process-global env; the real env is still consulted first
/// when the name is not overridden.
#[cfg(not(test))]
pub(super) fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[cfg(test)]
pub(super) static ENV_OVERRIDES: std::sync::Mutex<
    Option<std::collections::HashMap<String, String>>,
> = std::sync::Mutex::new(None);

/// Test-only env override (`Some` sets, `None` clears); see [`env_var`].
/// Only the `#[path]`-included integration tests call it, so the unit-test
/// build of this module sees it as dead code.
#[cfg(test)]
#[allow(dead_code)]
pub fn set_env_var(name: &str, value: Option<&str>) {
    let mut slot = ENV_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let map = slot.get_or_insert_with(Default::default);
    match value {
        Some(v) => {
            map.insert(name.to_owned(), v.to_owned());
        }
        None => {
            map.remove(name);
        }
    }
}

#[cfg(test)]
pub(super) fn env_var(name: &str) -> Option<String> {
    let slot = ENV_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    slot.as_ref()
        .and_then(|map| map.get(name).cloned())
        .or_else(|| std::env::var(name).ok())
}

/// Hand-rolled constant-time byte comparison (REQ-7: no new dependency).
/// Double-pass over the longer slice with zero-fill for the shorter one, so
/// the work done does not depend on where (or whether) the inputs differ;
/// the length XOR folds any length mismatch into the same accumulator.
pub(super) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut acc = (a.len() ^ b.len()) as u64;
    for i in 0..a.len().max(b.len()) {
        let x = u64::from(a.get(i).copied().unwrap_or(0));
        let y = u64::from(b.get(i).copied().unwrap_or(0));
        acc |= x ^ y;
    }
    acc == 0
}

/// 401 with the OpenAI-shaped error envelope. Built inline via `json!` —
/// `OpenAiError::new` maps non-400/404/429/5xx classes to `server_error`,
/// and auth needs `authentication_error` + `invalid_api_key`; api-types is
/// shared DTO territory we do not extend here.
pub(super) fn unauthorized_response() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "Invalid API key provided.",
            "type": "authentication_error",
            "param": null,
            "code": "invalid_api_key"
        }
    });
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

/// Per-request metrics: one route/code counter + one duration observation
/// (see `metrics`); no per-request allocation. Every 429 without an
/// explicit `Retry-After` (the GPU-slot busy contract, shared by the native
/// and `OpenAI` chat routes) carries a 1s hint — additive on the native
/// route, whose body/status do not change. The admission queue's own
/// rejections rely on this default (the queue has no better estimate than
/// the operator's retry cadence), so an explicit header is never clobbered.
pub(super) async fn track_metrics(req: Request, next: Next) -> Response {
    let route = crate::metrics::route_label(req.uri().path());
    let started = Instant::now();
    let mut response = next.run(req).await;
    if response.status() == StatusCode::TOO_MANY_REQUESTS
        && !response.headers().contains_key(header::RETRY_AFTER)
    {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            crate::admission::retry_after_header_value(),
        );
    }
    crate::metrics::record_request(route, response.status().as_u16(), started.elapsed());
    response
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Method;

    use super::*;

    #[test]
    fn signed_exemption_parses_query_pairs_not_substrings() {
        let request = |uri: &str| {
            Request::builder()
                .method(Method::GET)
                .uri(uri)
                .body(Body::empty())
                .expect("request")
        };
        assert!(!is_signed_content_request(&request(
            "/v1/jobs/01J/content/out.png?junk=expires=0&junk2=sig=0"
        )));
        assert!(is_signed_content_request(&request(
            "/v1/jobs/01J/content/out.png?expires=1&sig=deadbeef"
        )));
    }

    #[test]
    fn minting_a_read_credential_requires_read_scope() {
        assert!(mints_read_credential(&Method::POST, "/v1/content-urls"));
        assert!(!mints_read_credential(&Method::GET, "/v1/content-urls"));
        assert!(!mints_read_credential(&Method::POST, "/v1/jobs"));
    }
}
