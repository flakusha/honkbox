//! Salvo port of the axum API middleware stack.
//!
//! `api_hoops` returns the hoops in axum's layering order: metrics outermost,
//! compression inside it, bearer auth innermost. Attach them to API routers
//! only; the UI/static fallback must stay outside the hooped router, exactly
//! as axum's `route_layer` left the fallback untouched.
//!
//! All decision logic (probe/signed exemptions, scopes, key comparison, env
//! reads, error envelopes, compression constants) is reused from
//! `routes::middleware` / the `routes` re-exports — this module only adapts
//! it to the Salvo handler signature.

use std::sync::Arc;
use std::time::Instant;

use salvo::compression::{Compression, CompressionLevel};
use salvo::http::{Mime, StatusCode, header};
use salvo::prelude::*;

use crate::routes::AppConfig;
use crate::routes::middleware::{
    constant_time_eq, env_var, insufficient_scope_response, is_probe_exempt,
    is_signed_content_request, mints_read_credential, required_scope, scoped_key_verdict,
    unauthorized_response,
};
use crate::routes::runtime_config;
use crate::routes::COMPRESSION_MIN_BYTES;

/// Hoops wrapping API routes, in axum's outside-in layer order.
///
/// The caller attaches these to the API router rather than the root router so
/// the UI/static fallback (and its assets) stays bearer-free.
#[must_use]
pub fn api_hoops(config: Arc<AppConfig>) -> Vec<handler::ArcHandler> {
    vec![
        TrackMetrics.arc(),
        compression().arc(),
        RequireBearer { config }.arc(),
    ]
}

/// Bearer auth hoop: the exact `require_bearer` contract, expressed as a
/// Salvo handler. Rejections set the same OpenAI-shaped response and stop the
/// rest of the chain; allows simply return so the flow proceeds.
struct RequireBearer {
    config: Arc<AppConfig>,
}

#[async_trait]
impl Handler for RequireBearer {
    async fn handle(
        &self,
        req: &mut Request,
        _depot: &mut Depot,
        res: &mut Response,
        ctrl: &mut FlowCtrl,
    ) {
        let method = req.method();
        let path = req.uri().path();
        if self.config.auth.mode != crate::config::AuthMode::Bearer
            || is_probe_exempt(method, path)
            || is_signed_content_request(&salvo_to_axum_request(req))
        {
            return;
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

        let (scopes_enabled, scope_map) = runtime_config::scope_facts();
        if scopes_enabled && !scope_map.is_empty() {
            let required = required_scope(method, path);
            if let Some(allowed) = scoped_key_verdict(provided, &scope_map, required) {
                if !allowed {
                    crate::salvo_support::respond(res, insufficient_scope_response(required)).await;
                    ctrl.skip_rest();
                    return;
                }
                if mints_read_credential(method, path)
                    && !matches!(scoped_key_verdict(provided, &scope_map, "read"), Some(true))
                {
                    crate::salvo_support::respond(res, insufficient_scope_response("read")).await;
                    ctrl.skip_rest();
                    return;
                }
                return;
            }
        }

        let expected = env_var(&self.config.auth.api_key_env).filter(|key| !key.is_empty());
        let ok = matches!((expected, provided), (Some(e), Some(p)) if constant_time_eq(e.as_bytes(), p.as_bytes()));
        if ok {
            return;
        }
        crate::salvo_support::respond(res, unauthorized_response()).await;
        ctrl.skip_rest();
    }
}

/// Reuse the axum helper's query-pair parsing without duplicating it. Both
/// frameworks resolve to the same `http` 1.x `Method`/`Uri` types.
fn salvo_to_axum_request(req: &Request) -> axum::http::Request<axum::body::Body> {
    let mut out = axum::http::Request::new(axum::body::Body::empty());
    *out.method_mut() = req.method().clone();
    *out.uri_mut() = req.uri().clone();
    out
}

// Error envelopes are rendered by `crate::salvo_support::respond`.
/// Metrics hoop: one route/code counter and one duration observation, plus
/// the 429 `Retry-After` default. Mirrors `track_metrics` exactly; runs
/// outermost so the recorded response is the final (compressed) one.
struct TrackMetrics;

#[async_trait]
impl Handler for TrackMetrics {
    async fn handle(
        &self,
        req: &mut Request,
        depot: &mut Depot,
        res: &mut Response,
        ctrl: &mut FlowCtrl,
    ) {
        let route = crate::metrics::route_label(req.uri().path());
        let started = Instant::now();
        ctrl.call_next(req, depot, res).await;
        if res.status_code == Some(StatusCode::TOO_MANY_REQUESTS)
            && !res.headers().contains_key(header::RETRY_AFTER)
        {
            res.headers_mut().insert(
                header::RETRY_AFTER,
                crate::admission::retry_after_header_value(),
            );
        }
        crate::metrics::record_request(
            route,
            res.status_code.unwrap_or(StatusCode::OK).as_u16(),
            started.elapsed(),
        );
    }
}

/// Compression hoop: gzip/brotli only, 1 KiB floor, and an explicit MIME
/// allow-list that excludes gRPC, `image/*` (except SVG, matching
/// `DefaultPredicate`'s exception), and `text/event-stream`. `Content-Range`
/// and 206 are skipped by `Compression` itself.
fn compression() -> Compression {
    // ponytail: explicit MIME list; exotic text/* and +json subtypes stay plain.
    let compressible = [
        "text/plain",
        "text/html",
        "text/css",
        "text/javascript",
        "text/xml",
        "text/csv",
        "application/json",
        "application/javascript",
        "application/xml",
        "application/rss+xml",
        "application/wasm",
        "application/octet-stream",
        "image/svg+xml",
    ];
    let types: Vec<Mime> = compressible
        .iter()
        .map(|mime| mime.parse().expect("static MIME type"))
        .collect();
    Compression::new()
        .disable_all()
        .enable_gzip(CompressionLevel::Default)
        .enable_brotli(CompressionLevel::Default)
        .min_length(usize::from(COMPRESSION_MIN_BYTES))
        .content_types(&types)
}
