use std::time::Instant;

use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::AppConfig;

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
pub(super) async fn require_bearer(
    State(cfg): State<AppConfig>,
    req: Request,
    next: Next,
) -> Response {
    if cfg.auth.mode != crate::config::AuthMode::Bearer
        || is_probe_exempt(req.method(), req.uri().path())
    {
        return next.run(req).await;
    }
    let expected = env_var(&cfg.auth.api_key_env).filter(|key| !key.is_empty());
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
