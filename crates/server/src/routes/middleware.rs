use std::{collections::HashMap, time::Instant};

use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use jsonwebtoken::{Algorithm, DecodingKey, Validation};

use super::AppConfig;
use super::image_ops_dedup::sha256_hex;
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
#[allow(dead_code)] // axum middleware fixture; salvo has its own hoops
pub(crate) async fn require_bearer(
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
    // JWT arm (additive): off unless `[auth] jwt_*` configures material, so
    // the env-key path above runs byte-for-byte when JWT is unconfigured.
    // Fail-closed: missing material, bad signature, expired token, or a
    // wrong iss/aud all fall through to the 401 below — never trusted.
    match jwt_verdict(provided, &cfg.auth.jwt, req.method(), req.uri().path()).await {
        JwtVerdict::Pass => return next.run(req).await,
        JwtVerdict::Forbidden(scope) => return insufficient_scope_response(scope),
        JwtVerdict::Reject => {}
    }
    unauthorized_response()
}

/// JWT arm outcome: `Pass` runs the request, `Forbidden` answers the named
/// 403, `Reject` falls through to the 401 (arm off or token untrusted).
#[derive(Debug, PartialEq, Eq)]
enum JwtVerdict {
    Pass,
    Forbidden(&'static str),
    Reject,
}

/// JWT bearer attempt (see [`verify_jwt`]). `None` token or unconfigured
/// material is `Reject`, not an error — the env-key path already ran.
async fn jwt_verdict(
    provided: Option<&str>,
    jwt: &crate::config::JwtConfig,
    method: &axum::http::Method,
    path: &str,
) -> JwtVerdict {
    if !jwt.enabled() {
        return JwtVerdict::Reject;
    }
    let claims = match provided {
        Some(token) => match verify_jwt(token, jwt).await {
            Some(claims) => claims,
            None => return JwtVerdict::Reject,
        },
        None => return JwtVerdict::Reject,
    };
    let required = required_scope(method, path);
    if !jwt_scope_grants(&claims, required) {
        return JwtVerdict::Forbidden(required);
    }
    // Minting a signed content URL hands back a READ credential (same rule
    // as the scoped env-key path above).
    if mints_read_credential(method, path) && !jwt_scope_grants(&claims, "read") {
        return JwtVerdict::Forbidden("read");
    }
    JwtVerdict::Pass
}

/// Keyless routes: the liveness/metrics probes, GET only (so a scraper
/// cannot write anything and probes stay cheap).
pub(crate) fn is_probe_exempt(method: &axum::http::Method, path: &str) -> bool {
    method == axum::http::Method::GET && (path == "/health" || path == "/metrics")
}

/// Signed content requests carry their credential in additive query params.
/// Presence alone grants passage through this layer: `job_content` verifies
/// the HMAC and expiry before any file read, so a forged set still gets the
/// named `invalid_signed_url` 401. Only the two content routes qualify.
pub(crate) fn is_signed_content_request(req: &Request) -> bool {
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

/// One lookup in the scope layer: `(env NAME, allowed)` for a matching key.
/// The name is safe to hand out; the key value never leaves this function.
fn scoped_key_match<'a>(
    key: Option<&str>,
    scope_map: &'a [(String, Vec<String>)],
    required: &str,
) -> Option<(&'a str, bool)> {
    let key = key?;
    for (env_name, scopes) in scope_map {
        let expected = env_var(env_name).filter(|value| !value.is_empty());
        if matches!((expected, key), (Some(e), p) if constant_time_eq(e.as_bytes(), p.as_bytes())) {
            return Some((
                env_name,
                scopes.iter().any(|scope| scope.as_str() == required),
            ));
        }
    }
    None
}

/// `Some(allowed)` when the presented key belongs to a configured scoped env
/// name; `None` when it is not a scoped consumer (legacy or unknown key).
pub(crate) fn scoped_key_verdict(
    key: Option<&str>,
    scope_map: &[(String, Vec<String>)],
    required: &str,
) -> Option<bool> {
    scoped_key_match(key, scope_map, required).map(|(_, allowed)| allowed)
}

/// Stable idempotency consumer for `POST /v1/image-ops`: the matched scope's
/// env NAME when scopes are configured (never a credential), otherwise the
/// sha256 of the presented bearer key. Both the axum and Salvo stacks call
/// this one handler, so the namespace is identical on both. No header at
/// all (auth-off dev/test) shares a single anonymous consumer.
pub(crate) fn consumer_id(headers: &HeaderMap) -> String {
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            value
                .strip_prefix("Bearer ")
                .or_else(|| value.strip_prefix("bearer "))
        })
        .map(str::trim)
        .filter(|key| !key.is_empty());
    let (enabled, scope_map) = runtime_config::scope_facts();
    if enabled
        && !scope_map.is_empty()
        && let Some((env_name, _)) = scoped_key_match(presented, &scope_map, "write")
    {
        return env_name.to_owned();
    }
    presented.map_or_else(
        || sha256_hex(b"anonymous"),
        |key| sha256_hex(key.as_bytes()),
    )
}

/// `POST /v1/content-urls` mints a signed READ credential, so a scoped
/// consumer needs both `write` (the verb) and `read` (what it hands back).
pub(crate) fn mints_read_credential(method: &axum::http::Method, path: &str) -> bool {
    method == axum::http::Method::POST && path == "/v1/content-urls"
}

/// Route-family → required scope. Admin families are checked first for every
/// verb, then all non-GET verbs are writes, then GET is a read. Family-level
/// only: no per-route special cases.
pub(crate) fn required_scope(method: &axum::http::Method, path: &str) -> &'static str {
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
pub(crate) fn insufficient_scope_response(required: &str) -> Response {
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
pub(crate) fn env_var(name: &str) -> Option<String> {
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
pub(crate) fn env_var(name: &str) -> Option<String> {
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
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
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
pub(crate) fn unauthorized_response() -> Response {
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

/// JWT claims the bearer arm enforces. `exp` is required by [`Validation`].
/// (missing/expired = reject); `iss`/`aud` pin to the configured expectations
/// when set. Scopes ride the `scope` (space-delimited, `OAuth2` convention)
/// and/or `scp` (string or array) claims; a token carrying neither is
/// unconstrained (like the legacy key), a token carrying either must name
/// the route family's required scope or the arm answers the named 403.
#[derive(Debug, serde::Deserialize)]
struct JwtClaims {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    scp: Option<serde_json::Value>,
}

/// Verify a bearer token against `[auth] jwt_*`: signature + expiry +
/// issuer + audience. `Some(claims)` on full pass, `None` on ANY failure
/// (missing material, bad signature, expired token, wrong iss/aud,
/// unreachable JWKS) — fail-closed; the caller answers 401. The token VALUE
/// is never logged, never in metrics, never echoed. HS256 re-reads its
/// secret per request and JWKS refreshes through a TTL cache, so rotation
/// needs no restart either way.
async fn verify_jwt(token: &str, jwt: &crate::config::JwtConfig) -> Option<JwtClaims> {
    let validation = jwt_validation(jwt);
    if let Some(claims) = verify_hs256(token, jwt, &validation) {
        return Some(claims);
    }
    verify_rs256(token, jwt, &validation).await
}

/// Shared claim rules: expiry required, `iss`/`aud` pinned when configured,
/// drift leeway from `[auth] jwt_clock_skew_secs`.
fn jwt_validation(jwt: &crate::config::JwtConfig) -> Validation {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.leeway = jwt.clock_skew_secs;
    if let Some(issuer) = jwt.issuer.as_deref() {
        validation.set_issuer(&[issuer]);
    }
    if let Some(audience) = jwt.audience.as_deref() {
        validation.set_audience(&[audience]);
    }
    validation
}

/// HS256 attempt: secret re-read per request (rotation without a restart);
/// missing/empty secret is a quiet miss so the JWKS arm still runs.
fn verify_hs256(
    token: &str,
    jwt: &crate::config::JwtConfig,
    validation: &Validation,
) -> Option<JwtClaims> {
    let name = jwt.secret_env.as_deref()?;
    let secret = env_var(name).filter(|s| !s.is_empty())?;
    let key = DecodingKey::from_secret(secret.as_bytes());
    jsonwebtoken::decode::<JwtClaims>(token, &key, validation)
        .ok()
        .map(|data| data.claims)
}

/// RS256 attempt: `kid`-selected key from the TTL-cached JWKS.
async fn verify_rs256(
    token: &str,
    jwt: &crate::config::JwtConfig,
    hs: &Validation,
) -> Option<JwtClaims> {
    let url = jwt.jwks_url.as_deref()?;
    let kid = jsonwebtoken::decode_header(token).ok()?.kid?;
    let mut rs = hs.clone();
    rs.algorithms = vec![Algorithm::RS256];
    let key = jwks_key(url, &kid).await?;
    jsonwebtoken::decode::<JwtClaims>(token, &key, &rs)
        .ok()
        .map(|data| data.claims)
}

/// Scope check for a verified token: every `scope`/`scp` entry is granted;
/// an empty grant set means the token states no restriction (allow).
fn jwt_scope_grants(claims: &JwtClaims, required: &str) -> bool {
    let mut granted: Vec<&str> = Vec::new();
    if let Some(scope) = claims.scope.as_deref() {
        granted.extend(scope.split_whitespace());
    }
    match &claims.scp {
        Some(serde_json::Value::String(s)) => granted.extend(s.split_whitespace()),
        Some(serde_json::Value::Array(items)) => {
            granted.extend(items.iter().filter_map(|v| v.as_str()));
        }
        _ => {}
    }
    granted.is_empty() || granted.contains(&required)
}

/// JWKS key cache: `kid` → decoding key, pinned to the URL it came from.
/// Stale entries and `kid` misses trigger one refresh before giving up, so
/// rotation lands without a restart; a failed refresh is fail-closed
/// (reject) rather than served stale.
#[derive(Default)]
struct JwksCache {
    url: Option<String>,
    fetched_at: Option<Instant>,
    keys: HashMap<String, DecodingKey>,
}

/// JWKS entries outlive this long before a request forces a refresh.
const JWKS_TTL: std::time::Duration = std::time::Duration::from_secs(300);

static JWKS: std::sync::OnceLock<tokio::sync::RwLock<JwksCache>> = std::sync::OnceLock::new();

/// `kid`-selected RS256 key from the issuer JWKS, TTL-cached. `None` on any
/// fetch/parse/miss failure — fail-closed upstream.
async fn jwks_key(url: &str, kid: &str) -> Option<DecodingKey> {
    let slot = JWKS.get_or_init(|| tokio::sync::RwLock::new(JwksCache::default()));
    {
        let cache = slot.read().await;
        let fresh = cache.fetched_at.is_some_and(|at| at.elapsed() < JWKS_TTL);
        if cache.url.as_deref() == Some(url)
            && fresh
            && let Some(key) = cache.keys.get(kid)
        {
            return Some(key.clone());
        }
    }
    let fresh = fetch_jwks(url).await?;
    let mut cache = slot.write().await;
    cache.url = Some(url.to_owned());
    cache.fetched_at = Some(Instant::now());
    cache.keys = fresh;
    cache.keys.get(kid).cloned()
}

/// Fetch + parse an issuer JWKS document, keeping RSA keys only.
async fn fetch_jwks(url: &str) -> Option<HashMap<String, DecodingKey>> {
    let doc: JwksDocument = reqwest::get(url).await.ok()?.json().await.ok()?;
    let mut keys = HashMap::new();
    for key in doc.keys {
        if key.kty.as_deref() != Some("RSA") {
            continue;
        }
        let (Some(kid), Some(n), Some(e)) = (key.kid, key.n, key.e) else {
            continue;
        };
        if let Ok(key) = DecodingKey::from_rsa_components(&n, &e) {
            keys.insert(kid, key);
        }
    }
    Some(keys)
}

/// JWKS document shape: only the RSA fields the arm needs.
#[derive(Debug, serde::Deserialize)]
struct JwksDocument {
    #[serde(default)]
    keys: Vec<JwkEntry>,
}

/// One JWK: `kid`-addressed RSA components (all optional — incomplete
/// entries are skipped, never trusted).
#[derive(Debug, serde::Deserialize)]
struct JwkEntry {
    #[serde(default)]
    kid: Option<String>,
    #[serde(default)]
    kty: Option<String>,
    #[serde(default)]
    n: Option<String>,
    #[serde(default)]
    e: Option<String>,
}

/// Per-request metrics: one route/code counter + one duration observation
/// (see `metrics`); no per-request allocation. Every 429 without an
/// explicit `Retry-After` (the GPU-slot busy contract, shared by the native
/// and `OpenAI` chat routes) carries a 1s hint — additive on the native
/// route, whose body/status do not change. The admission queue's own
/// rejections rely on this default (the queue has no better estimate than
/// the operator's retry cadence), so an explicit header is never clobbered.
#[allow(dead_code)] // axum middleware fixture; salvo has its own hoops
pub(crate) async fn track_metrics(req: Request, next: Next) -> Response {
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
    /// HS256 helper: config pinning `secret_env` to a per-test env NAME
    /// (parallel-safe: `ENV_OVERRIDES` is keyed by name) plus optional
    /// `iss`/`aud` pins.
    fn hs_config(
        name: &str,
        issuer: Option<&str>,
        audience: Option<&str>,
    ) -> crate::config::JwtConfig {
        crate::config::JwtConfig {
            secret_env: Some(name.to_owned()),
            jwks_url: None,
            issuer: issuer.map(str::to_owned),
            audience: audience.map(str::to_owned),
            clock_skew_secs: 60,
        }
    }

    fn epoch_now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs()
    }

    /// Mint an HS256 token with `jsonwebtoken::encode` (already a server
    /// dependency — no new dev-deps). Claims ride a JSON value because
    /// `JwtClaims` is decode-only.
    fn mint_hs256(secret: &str, claims: &serde_json::Value) -> String {
        jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("mint")
    }

    #[test]
    fn hs256_valid_token_passes() {
        let name = "HONKBOX_TEST_JWT_MW_VALID";
        set_env_var(name, Some("topsekrit"));
        let jwt = hs_config(name, Some("honkbox"), Some("api"));
        let token = mint_hs256(
            "topsekrit",
            &serde_json::json!({"iss": "honkbox", "aud": "api", "exp": epoch_now() + 3600}),
        );
        assert!(verify_hs256(&token, &jwt, &jwt_validation(&jwt)).is_some());
        set_env_var(name, None);
    }

    #[test]
    fn hs256_bad_signature_expired_wrong_aud_and_missing_secret_reject() {
        let name = "HONKBOX_TEST_JWT_MW_REJECT";
        set_env_var(name, Some("topsekrit"));
        let jwt = hs_config(name, Some("honkbox"), Some("api"));
        let validation = jwt_validation(&jwt);
        let good = mint_hs256(
            "topsekrit",
            &serde_json::json!({"iss": "honkbox", "aud": "api", "exp": epoch_now() + 3600}),
        );
        // Bad signature: same shape, wrong key.
        let forged = mint_hs256(
            "wrongkey",
            &serde_json::json!({"iss": "honkbox", "aud": "api", "exp": epoch_now() + 3600}),
        );
        assert!(verify_hs256(&forged, &jwt, &validation).is_none());
        // Tampered body: flip the last signature char.
        let mut tampered = good.clone();
        let last = tampered.pop().expect("token");
        tampered.push(if last == 'A' { 'B' } else { 'A' });
        assert!(verify_hs256(&tampered, &jwt, &validation).is_none());
        // Expired beyond leeway.
        let stale = mint_hs256(
            "topsekrit",
            &serde_json::json!({"iss": "honkbox", "aud": "api", "exp": epoch_now() - 3600}),
        );
        assert!(verify_hs256(&stale, &jwt, &validation).is_none());
        // Wrong audience.
        let alien = mint_hs256(
            "topsekrit",
            &serde_json::json!({"iss": "honkbox", "aud": "someone-else", "exp": epoch_now() + 3600}),
        );
        assert!(verify_hs256(&alien, &jwt, &validation).is_none());
        // Garbage is not a token.
        assert!(verify_hs256("not.a.token", &jwt, &validation).is_none());
        // Missing secret material is a quiet miss (the RS256 arm runs next).
        set_env_var(name, None);
        assert!(verify_hs256(&good, &jwt, &validation).is_none());
    }

    #[test]
    fn jwt_validation_pins_issuer_audience_and_leeway() {
        let jwt = hs_config("HONKBOX_TEST_JWT_MW_UNUSED", Some("honkbox"), Some("api"));
        let validation = jwt_validation(&jwt);
        assert_eq!(validation.leeway, 60);
        assert!(
            validation
                .iss
                .as_ref()
                .is_some_and(|iss| iss.contains("honkbox"))
        );
        assert!(
            validation
                .aud
                .as_ref()
                .is_some_and(|aud| aud.contains("api"))
        );
        let open = crate::config::JwtConfig::default();
        let validation = jwt_validation(&open);
        assert!(validation.iss.is_none(), "unset issuer is unchecked");
        assert!(validation.aud.is_none(), "unset audience is unchecked");
    }

    #[test]
    fn jwt_scope_grants_covers_scope_and_scp_forms() {
        let claims = |body: serde_json::Value| -> JwtClaims {
            serde_json::from_value(body).expect("claims decode")
        };
        // No restriction stated: allow (the legacy-key path).
        assert!(jwt_scope_grants(&claims(serde_json::json!({})), "read"));
        assert!(jwt_scope_grants(
            &claims(serde_json::json!({"scope": ""})),
            "read"
        ));
        // `scope` is space-delimited (OAuth2 convention).
        let scoped = claims(serde_json::json!({"scope": "read write"}));
        assert!(jwt_scope_grants(&scoped, "read"));
        assert!(jwt_scope_grants(&scoped, "write"));
        assert!(!jwt_scope_grants(&scoped, "admin"));
        // `scp` as a space-delimited string.
        let scp_str = claims(serde_json::json!({"scp": "read write"}));
        assert!(jwt_scope_grants(&scp_str, "write"));
        assert!(!jwt_scope_grants(&scp_str, "admin"));
        // `scp` as an array; non-strings are ignored, never granted.
        let scp_arr = claims(serde_json::json!({"scp": ["read", 42, "write"]}));
        assert!(jwt_scope_grants(&scp_arr, "read"));
        assert!(!jwt_scope_grants(&scp_arr, "admin"));
        // A token carrying either claim names the required family or fails.
        let both = claims(serde_json::json!({"scope": "read", "scp": ["write"]}));
        assert!(jwt_scope_grants(&both, "write"));
        assert!(!jwt_scope_grants(&both, "admin"));
    }

    #[test]
    fn jwt_arm_enabled_only_with_material() {
        assert!(!crate::config::JwtConfig::default().enabled());
        assert!(hs_config("HONKBOX_TEST_JWT_MW_ENABLE", None, None).enabled());
        let jwks_only = crate::config::JwtConfig {
            jwks_url: Some("https://issuer.example.com/.well-known/jwks.json".to_owned()),
            ..crate::config::JwtConfig::default()
        };
        assert!(jwks_only.enabled());
    }
}
