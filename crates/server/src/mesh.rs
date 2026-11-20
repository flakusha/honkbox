//! Mesh upstream proxy client (TASK-mesh-client-spike + TASK-proxy-stream-safety).
//!
//! The foundation the reroute/discovery tickets build on: an opt-in,
//! config-bounded list of mesh-compatible upstreams and one HTTP client that
//! forwards a framed request per turn, relays responses (SSE included)
//! without buffering, and enforces the header trust boundary below.
//!
//! # The upstream hop is an explicit trust boundary
//!
//! - The ONLY credential ever sent upstream is the per-upstream bearer token
//!   read from honkbox config (`[[mesh.upstreams]] bearer_token`), applied at
//!   this client layer. Caller-supplied credentials (`authorization`, `cookie`,
//!   `x-api-key`) and any `x-honkbox-*` header are NEVER forwarded — the
//!   outbound request is built fresh from [`FORWARDED_REQUEST_HEADERS`], never
//!   by copying the inbound map; the pin tests in `tests/mesh_proxy.rs` prove
//!   the stripped set against a mock upstream.
//! - `x-request-id` is always regenerated locally; client values never travel.
//! - The documented trust assumption: the configured upstream is a trusted
//!   peer over the network path the operator chose. If a deployment's
//!   upstreams are not on a trusted path, isolating that path is operator
//!   configuration (this module performs no transport-level isolation).
//! - Unpredictable job/prompt/request ids are NOT authorization here: this
//!   client makes no identity decisions, and the bearer token is the only
//!   secret material it ever emits.
//! - Bearer tokens are never logged, never in metrics, never in error
//!   strings, and the config `Debug` output redacts them.
//!
//! # Stream rules (TASK-proxy-stream-safety)
//!
//! One framed upstream connection per forwarded request (`Connection: close`
//! per request plus an idle-pool-free client — no v1 multiplexing), redirects
//! disabled (a redirect is an upstream-controlled trust-boundary escape
//! hatch), response bodies streamed through under a byte cap, and cancel
//! propagation: the caller dropping the relayed body aborts the upstream
//! request (dropping the `reqwest` response body closes the connection, and
//! the empty idle pool means nothing lingers).

#![allow(dead_code)] // TASK-reroute-policy mounts the routes that consume this client; the spike ships the foundation + tests only.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use reqwest::Url;
use tokio_stream::StreamExt as _;

/// Routing hints the reroute layer reads BEFORE forwarding. They are absent
/// from [`FORWARDED_REQUEST_HEADERS`], so the client strips them on every hop.
pub(crate) const MESH_SCOPE_HEADER: &str = "x-mesh-scope";
pub(crate) const MESH_TARGET_HEADER: &str = "x-mesh-target";
pub(crate) const MESH_EXCLUDE_HEADER: &str = "x-mesh-exclude";

/// Scope markers (`@private` in a model/upstream name, `x-mesh-scope:
/// private`, `/private/v1` in a path). Any one of them makes the request
/// private; private requests never fall back to public providers.
pub(crate) const PRIVATE_SCOPE_MARKER: &str = "private";
pub(crate) const PRIVATE_PATH_PREFIX: &str = "/private/v1";

/// "Retry budget" is the TOTAL number of forwards one request may make,
/// including the first try. The cap keeps a routing loop from multiplying
/// traffic even if an operator tunes the policy upward.
pub(crate) const DEFAULT_ATTEMPT_BUDGET: usize = 3;
pub(crate) const MAX_ATTEMPT_BUDGET: usize = 4;
pub(crate) const DEFAULT_COOLDOWN: Duration = Duration::from_secs(30);

/// Request headers forwarded to an upstream, by exact (lowercase) name — the
/// whole allowlist. Values are copied from the inbound request; everything
/// else, including every credential header and every `x-honkbox-*` header, is
/// stripped by construction (the outbound request is built fresh from this
/// table, never by copying the inbound map). `user-agent` and `x-request-id`
/// are NOT taken from here: they are always rewritten locally.
///
/// `accept-encoding` is deliberately excluded: this client is built without
/// reqwest's decompression features, so a compressed upstream body would
/// arrive undecodable with `content-encoding` stripped by the response
/// allowlist.
pub(crate) const FORWARDED_REQUEST_HEADERS: &[&str] = &["accept"];

/// Response headers relayed to the honkbox caller, by exact name. Everything
/// else — response `authorization`/`set-cookie`, upstream `x-request-id`
/// echoes, and every hop-by-hop header — is stripped. `retry-after` is not in
/// this table because it is not relayed raw: it is parsed into
/// [`UpstreamResponse::retry_after`] so honkbox's own error envelopes can map
/// it.
pub(crate) const FORWARDED_RESPONSE_HEADERS: &[&str] = &["content-type"];

/// Hop-by-hop headers (RFC 7230 §6.1 plus the `proxy-*` family), stripped in
/// BOTH directions per the ticket. Stripping is by construction on each side
/// (fresh allowlist-built request; allowlist-filtered response); this table
/// exists so the pin tests can prove no allowlist entry ever collides with it
/// and so a future route layer has the named set to assert against.
pub(crate) const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Credential headers the ticket names explicitly. They are stripped by the
/// allowlist construction above (they are not in
/// [`FORWARDED_REQUEST_HEADERS`]); the set is named so the injection
/// regression test and any future reviewer can pin exactly what must never
/// appear on an upstream-bound request.
pub(crate) const STRIPPED_CREDENTIAL_HEADERS: &[&str] =
    &["authorization", "cookie", "x-api-key", "x-forwarded-for"];

/// `User-Agent` sent upstream: identifies honkbox + version, never the
/// caller's value.
const USER_AGENT: &str = concat!("honkbox/", env!("CARGO_PKG_VERSION"));

/// Upstream connect timeout. Separate from the read timeout so a hung peer
/// fails fast while a slow SSE stream stays alive.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Upstream read timeout between body chunks (a chat/SSE stream may pause;
/// a dead one must not).
const READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Default cap on ONE relayed response body (64 MiB, the router's own body
/// ceiling): a runaway upstream cannot balloon memory. `[mesh]` may override.
pub const DEFAULT_MAX_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;

/// Default `[mesh] discovery_ttl_secs`: how long a discovery fact stays fresh.
pub const DEFAULT_DISCOVERY_TTL_SECS: u64 = 300;
/// Default `[mesh] discovery_cap`: bounded number of discovered peers.
pub const DEFAULT_DISCOVERY_CAP: usize = 256;

/// Why a mesh forward failed; named codes follow the house envelope
/// conventions (`error.code`). Route-layer rendering (into the OpenAI-style
/// envelope) lands with the reroute ticket; `Display`/`std::error::Error` are
/// implemented so the stream items and that future `IntoResponse` share this
/// enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MeshError {
    /// The upstream connection/request itself failed.
    UpstreamUnreachable {
        /// Upstream name (config label, no secrets).
        name: String,
        /// Underlying transport message (reqwest never embeds the bearer).
        message: String,
    },
    /// Reading the response body failed mid-stream.
    UpstreamRead(String),
    /// The relayed body crossed the configured byte cap mid-stream —
    /// rejected, never silently truncated.
    StreamLimitExceeded(u64),
    /// A routing/scope header failed syntax validation (400). Named header,
    /// never the raw value: no URL/token is echoed.
    InvalidHeader(&'static str),
    /// The request body could not be read for routing (400).
    InvalidBody,
    /// Targets conflict with the configured set or with each other (409).
    TargetConflict(String),
    /// A pinned provider exists but is cooling down (502): never reroute.
    TargetUnavailable,
    /// No provider is available for this request (502).
    NoUpstream,
    /// Private scope has no private provider (502): fail closed.
    PrivateScopeBlocked,
    /// The explicit attempt budget was consumed without success (502).
    RetryBudgetExhausted {
        /// Number of forwards actually made.
        attempts: usize,
    },
    /// Mesh is disabled in config (502): no silent local fallback.
    Disabled,
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UpstreamUnreachable { name, message } => {
                write!(f, "mesh upstream {name:?} unreachable: {message}")
            }
            Self::UpstreamRead(message) => write!(f, "mesh upstream body read failed: {message}"),
            Self::StreamLimitExceeded(limit) => {
                write!(f, "mesh upstream body exceeded the {limit} byte cap")
            }
            Self::InvalidHeader(name) => write!(f, "invalid {name} header"),
            Self::InvalidBody => write!(f, "mesh request body is unreadable"),
            Self::TargetConflict(name) => write!(f, "mesh target {name:?} conflicts"),
            Self::TargetUnavailable => write!(f, "pinned mesh target is unavailable"),
            Self::NoUpstream => write!(f, "no mesh upstream is available"),
            Self::PrivateScopeBlocked => {
                write!(f, "private scope has no private mesh upstream")
            }
            Self::RetryBudgetExhausted { attempts } => {
                write!(f, "mesh retry budget exhausted after {attempts} attempts")
            }
            Self::Disabled => write!(f, "mesh is disabled"),
        }
    }
}

impl std::error::Error for MeshError {}

impl MeshError {
    /// Named envelope code for the `error.code` field.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::UpstreamUnreachable { .. } | Self::UpstreamRead(_) => "upstream_unavailable",
            Self::StreamLimitExceeded(_) => "upstream_stream_limit",
            Self::InvalidHeader(MESH_TARGET_HEADER) => "mesh_invalid_target",
            Self::InvalidHeader(MESH_EXCLUDE_HEADER) => "mesh_invalid_exclude",
            Self::InvalidHeader(MESH_SCOPE_HEADER) => "mesh_invalid_scope",
            Self::InvalidHeader(_) => "mesh_invalid_header",
            Self::InvalidBody => "mesh_invalid_body",
            Self::TargetConflict(_) => "mesh_target_conflict",
            Self::TargetUnavailable => "mesh_target_unavailable",
            Self::NoUpstream => "mesh_no_upstream",
            Self::PrivateScopeBlocked => "mesh_private_scope_blocked",
            Self::RetryBudgetExhausted { .. } => "mesh_retry_budget_exhausted",
            Self::Disabled => "mesh_disabled",
        }
    }

    /// HTTP status for transport failures (keeps the client's 502 pin).
    pub(crate) fn status() -> StatusCode {
        StatusCode::BAD_GATEWAY
    }

    /// Status for the house envelope: bad routing hints are 400, conflicting
    /// targets 409, everything else is an upstream-class 502.
    pub(crate) fn response_status(&self) -> StatusCode {
        match self {
            Self::InvalidHeader(_) | Self::InvalidBody => StatusCode::BAD_REQUEST,
            Self::TargetConflict(_) => StatusCode::CONFLICT,
            _ => Self::status(),
        }
    }
}

/// One `[[mesh.upstreams]]` entry. `Debug` is HAND-WRITTEN: `LoadedConfig`
/// derives `Debug` and the bearer token must never appear in it (or in any
/// log/panic dump that prints config).
#[derive(Clone)]
pub struct MeshUpstream {
    /// Config label (`name`); used to select an upstream and in error text.
    pub name: String,
    /// Upstream origin (`base_url`); paths are joined onto it per request.
    pub base_url: Url,
    /// Optional bearer token applied at this client layer. Stored, redacted
    /// in `Debug`, never logged or echoed.
    pub bearer_token: Option<String>,
}

impl std::fmt::Debug for MeshUpstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshUpstream")
            .field("name", &self.name)
            .field("base_url", &self.base_url.as_str())
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// `[mesh]` config (TASK-mesh-client-spike): opt-in (`enabled` defaults to
/// false) and bounded to the operator's static list — there is no discovery
/// in this ticket. Resolved from [`FileMesh`] via [`MeshConfig::resolve`].
#[derive(Debug, Clone, Default)]
pub struct MeshConfig {
    /// Mesh forwarding off unless the operator flips this on.
    pub enabled: bool,
    /// The bounded upstream list; empty by default.
    pub upstreams: Vec<MeshUpstream>,
    /// Per-response body cap in bytes (`max_response_bytes`), default
    /// [`DEFAULT_MAX_RESPONSE_BYTES`]; enforced mid-stream.
    pub max_response_bytes: u64,
    /// `[mesh] discovery_ttl_secs`: discovery freshness window, default 300.
    pub discovery_ttl_secs: u64,
    /// `[mesh] discovery_cap`: bounded discovered peers, default 256 (min 1).
    pub discovery_cap: usize,
}

/// `[[mesh.upstreams]]` as written: raw strings; [`MeshConfig::resolve`]
/// validates and parses.
#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct FileMeshUpstream {
    /// Config label; trimmed, must be non-empty and unique.
    #[serde(default)]
    name: Option<String>,
    /// Upstream origin URL; must parse as absolute http(s).
    #[serde(default)]
    base_url: Option<String>,
    /// Optional bearer token (config secret; redacted everywhere).
    #[serde(default)]
    bearer_token: Option<String>,
}

/// `[mesh]` table as written.
#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct FileMesh {
    /// `enabled`; absent = off.
    #[serde(default)]
    enabled: Option<bool>,
    /// `[[mesh.upstreams]]` entries; absent = none.
    #[serde(default)]
    upstreams: Option<Vec<FileMeshUpstream>>,
    /// `max_response_bytes`; absent = the default cap.
    #[serde(default)]
    max_response_bytes: Option<u64>,
    /// `discovery_ttl_secs`; absent = [`DEFAULT_DISCOVERY_TTL_SECS`].
    #[serde(default)]
    discovery_ttl_secs: Option<u64>,
    /// `discovery_cap`; absent = [`DEFAULT_DISCOVERY_CAP`], min 1.
    #[serde(default)]
    discovery_cap: Option<usize>,
}

/// Resolve `[mesh] discovery_cap`: default 256; `0` is a named config error
/// (a zero cap would silently drop every discovery hit). Pure.
fn resolve_discovery_cap(raw: Option<usize>) -> Result<usize, String> {
    match raw.unwrap_or(DEFAULT_DISCOVERY_CAP) {
        0 => Err("[mesh] discovery_cap must be at least 1".to_owned()),
        cap => Ok(cap),
    }
}

impl MeshConfig {
    /// Resolve the `[mesh]` file stage: validate the upstream list (labels,
    /// URL parse, uniqueness) — a typo must be a loud config error, never a
    /// silently skipped upstream. `enabled = true` with zero upstreams is an
    /// error; upstreams configured while disabled are allowed (the operator
    /// can flip `enabled` without rewriting the list) and warn once.
    ///
    /// # Errors
    /// Named config-stage strings per the house `load` convention (bad label,
    /// unparseable/non-http `base_url`, duplicate name, enabled-but-empty).
    pub(crate) fn resolve(file: &FileMesh) -> Result<Self, String> {
        let enabled = file.enabled.unwrap_or(false);
        let discovery_cap = resolve_discovery_cap(file.discovery_cap)?;
        let discovery_ttl_secs = file
            .discovery_ttl_secs
            .unwrap_or(DEFAULT_DISCOVERY_TTL_SECS);
        let mut upstreams = Vec::new();
        for (index, entry) in file.upstreams.iter().flatten().enumerate() {
            let name = entry
                .name
                .as_deref()
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| format!("[[mesh.upstreams]] #{index} is missing `name`"))?;
            if upstreams
                .iter()
                .any(|upstream: &MeshUpstream| upstream.name == name)
            {
                return Err(format!("[[mesh.upstreams]] declares {name:?} twice"));
            }
            let raw = entry.base_url.as_deref().map(str::trim).unwrap_or_default();
            let base_url = Url::parse(raw)
                .map_err(|e| format!("[[mesh.upstreams]] {name:?} base_url {raw:?}: {e}"))?;
            if base_url.scheme() != "http" && base_url.scheme() != "https" {
                return Err(format!(
                    "[[mesh.upstreams]] {name:?} base_url must be http(s), got {:?}",
                    base_url.scheme()
                ));
            }
            upstreams.push(MeshUpstream {
                name: name.to_owned(),
                base_url,
                bearer_token: entry.bearer_token.clone(),
            });
        }
        if enabled && upstreams.is_empty() {
            return Err(
                "[mesh] enabled = true but no [[mesh.upstreams]] are configured".to_owned(),
            );
        }
        if !enabled && !upstreams.is_empty() {
            tracing::warn!("[mesh] upstreams are configured but enabled = false; mesh stays off");
        }
        Ok(Self {
            enabled,
            upstreams,
            max_response_bytes: file
                .max_response_bytes
                .filter(|cap| *cap > 0)
                .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES),
            discovery_ttl_secs,
            discovery_cap,
        })
    }
}

/// The mesh proxy client: one shared `reqwest::Client` (bounded timeouts, no
/// redirects, no connection reuse) plus the validated upstream list.
#[derive(Clone)]
pub struct MeshProxy {
    client: reqwest::Client,
    upstreams: Vec<MeshUpstream>,
    /// Per-response body cap.
    max_response_bytes: u64,
}

impl MeshProxy {
    /// Build a client for a resolved config.
    ///
    /// # Errors
    /// Only if `reqwest` cannot build its client (TLS backend failure).
    pub fn new(config: &MeshConfig) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            // TASK-proxy-stream-safety: no shared upstream multiplexing in
            // v1 — every request owns one framed connection (`Connection:
            // close` is also set per request) and nothing sits idle in a pool.
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(0)
            .user_agent(USER_AGENT)
            .build()?;
        Ok(Self {
            client,
            upstreams: config.upstreams.clone(),
            max_response_bytes: config.max_response_bytes,
        })
    }

    /// Resolve an upstream by config label.
    pub(crate) fn upstream(&self, name: &str) -> Option<&MeshUpstream> {
        self.upstreams.iter().find(|u| u.name == name)
    }

    /// The configured list, for the reroute policy's pure selector. The
    /// list is immutable after construction and ordered by config.
    pub(crate) fn upstreams(&self) -> &[MeshUpstream] {
        &self.upstreams
    }

    /// Forward ONE framed request to the named upstream and return the
    /// relayed response. The body streams through under the byte cap;
    /// dropping the returned body aborts the upstream connection (cancel
    /// propagation). Headers follow the module trust boundary exactly.
    ///
    /// # Errors
    /// [`MeshError::UpstreamUnreachable`] for a non-path target, an origin
    /// mismatch, or a transport failure; body-read failures and the size cap
    /// surface as stream items, not here.
    pub(crate) async fn forward(
        &self,
        upstream: &MeshUpstream,
        method: Method,
        target: &str,
        inbound_headers: &HeaderMap,
        body: Bytes,
    ) -> Result<UpstreamResponse, MeshError> {
        let request_id = new_request_id();
        // Trust boundary: `target` must be a plain absolute-path reference.
        // An absolute URL, a network-path reference (`//host`), or a
        // backslash variant can replace the configured authority — and this
        // is exactly where the per-upstream bearer gets attached. Reject
        // every non-path form before the URL is even parsed.
        if !target.starts_with('/')
            || target.starts_with("//")
            || target.as_bytes().get(1).is_some_and(|b| *b == b'\\')
        {
            return Err(MeshError::UpstreamUnreachable {
                name: upstream.name.clone(),
                message: format!("target {target:?} is not a path-only reference"),
            });
        }
        let url = upstream
            .base_url
            .join(target)
            .map_err(|_| MeshError::UpstreamUnreachable {
                name: upstream.name.clone(),
                message: format!("target path {target:?} does not join the configured base_url"),
            })?;
        // Belt and suspenders: URL normalization (e.g. a stripped newline
        // before a backslash) can still turn a path into a foreign authority.
        // Assert the origin here, before the bearer is attached, and fail
        // closed with the named unreachable error.
        if url.scheme() != upstream.base_url.scheme()
            || url.host_str() != upstream.base_url.host_str()
            || url.port_or_known_default() != upstream.base_url.port_or_known_default()
        {
            return Err(MeshError::UpstreamUnreachable {
                name: upstream.name.clone(),
                message: format!("target {target:?} resolves outside the configured origin"),
            });
        }
        let mut request = self
            .client
            .request(method, url)
            .headers(build_request_headers(inbound_headers, &request_id))
            .header(axum::http::header::CONNECTION, "close");
        // The ONLY credential that ever travels upstream: the operator's
        // per-upstream bearer, applied here at the client layer. Never
        // logged, never echoed.
        if let Some(token) = upstream.bearer_token.as_deref().filter(|t| !t.is_empty()) {
            request = request.bearer_auth(token);
        }
        if !body.is_empty() {
            request = request.body(body);
        }
        let response = request.send().await.map_err(|e| {
            MeshError::UpstreamUnreachable {
                name: upstream.name.clone(),
                // reqwest error text covers the transport cause; it does
                // not include request headers (and the bearer was applied
                // by value, never stored on the client).
                message: e.to_string(),
            }
        })?;
        let status = response.status();
        let headers = filter_response_headers(response.headers());
        let retry_after = parse_retry_after(response.headers());
        let stream = CappedStream::new(
            response
                .bytes_stream()
                .map(|item| item.map_err(|e| MeshError::UpstreamRead(e.to_string()))),
            self.max_response_bytes,
        );
        Ok(UpstreamResponse {
            status,
            headers,
            retry_after,
            body: Body::from_stream(stream),
        })
    }
}

/// A relayed upstream response: status, the allowlisted header set,
/// `retry-after` mapped out of the raw headers, and the streaming body.
pub(crate) struct UpstreamResponse {
    /// Upstream status, relayed as-is (route layers map error classes).
    pub status: StatusCode,
    /// [`FORWARDED_RESPONSE_HEADERS`] only; everything else stripped.
    pub headers: HeaderMap,
    /// Parsed `retry-after` (delta-seconds), for honkbox's own envelopes.
    pub retry_after: Option<Duration>,
    /// Streaming body (SSE frames included) under the byte cap.
    pub body: Body,
}

/// Regenerate the `x-request-id` for an upstream hop: a fresh ULID, never a
/// client-supplied value.
fn new_request_id() -> String {
    ulid::Ulid::new().to_string()
}

/// Build the outbound header set from the static allowlist. PURE — the pin
/// tests call this directly. Copies `accept` VALUE from the inbound request,
/// rewrites `user-agent` and `x-request-id`, and lets
/// the caller set `connection: close` (one framed connection per request).
/// Every other inbound header — `accept-encoding`, credentials,
/// `x-honkbox-*`, `x-forwarded-for`, hop-by-hop — never reaches the returned
/// map because the map is built fresh from [`FORWARDED_REQUEST_HEADERS`],
/// not copied.
pub(crate) fn build_request_headers(inbound: &HeaderMap, request_id: &str) -> HeaderMap {
    let mut out = HeaderMap::new();
    for name in FORWARDED_REQUEST_HEADERS {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        if let Some(value) = inbound.get(&name) {
            out.insert(name, value.clone());
        }
    }
    out.insert(
        axum::http::header::USER_AGENT,
        HeaderValue::from_static(USER_AGENT),
    );
    if let Ok(value) = HeaderValue::from_str(request_id) {
        out.insert("x-request-id", value);
    }
    out
}

/// Filter an upstream response down to [`FORWARDED_RESPONSE_HEADERS`]. PURE.
/// Hop-by-hop headers and `x-request-id` echoes are stripped by construction
/// (not in the table).
pub(crate) fn filter_response_headers(upstream: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for name in FORWARDED_RESPONSE_HEADERS {
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        if let Some(value) = upstream.get(&name) {
            out.insert(name, value.clone());
        }
    }
    out
}

/// Parse `retry-after` (delta-seconds form) from an upstream response for
/// mapping into honkbox's own error envelopes. `None` when absent, invalid,
/// or an HTTP-date form (only seconds are mapped).
fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(axum::http::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Request scope — a closed two-value vocabulary. Private is sticky: any
/// private marker forces it, and nothing widens it back to public.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeshScope {
    Public,
    Private,
}

/// Closed outcome vocabulary for reroute metrics. The label is the only
/// value that reaches a metric — never a name, URL, path, model or token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeshMetricKind {
    Selected,
    Retry,
    Cooldown,
    Rejected,
    PrivateBlocked,
    Exhausted,
}

impl MeshMetricKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::Retry => "retry",
            Self::Cooldown => "cooldown",
            Self::Rejected => "rejected",
            Self::PrivateBlocked => "private_blocked",
            Self::Exhausted => "exhausted",
        }
    }
}

/// The complete label set, fixed at compile time (six bounded series).
pub(crate) const MESH_METRIC_KINDS: [&str; 6] = [
    "selected",
    "retry",
    "cooldown",
    "rejected",
    "private_blocked",
    "exhausted",
];
static MESH_METRIC_COUNTS: [AtomicU64; MESH_METRIC_KINDS.len()] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Record one bounded `kind` outcome. No dynamic label can reach this grid.
pub(crate) fn record_mesh_metric(kind: MeshMetricKind) {
    MESH_METRIC_COUNTS[kind as usize].fetch_add(1, Ordering::Relaxed);
}

/// Counter readout for the metric contract tests.
pub(crate) fn mesh_metric_count(kind: MeshMetricKind) -> u64 {
    MESH_METRIC_COUNTS[kind as usize].load(Ordering::Relaxed)
}

/// Validated, fully resolved inputs to provider selection. Pure data: no
/// headers, client or clock cross this boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RerouteHints {
    pub scope: MeshScope,
    pub group: Option<String>,
    pub targets: Vec<String>,
    pub exclude: Vec<String>,
}

/// `@private` / `private:` tag detection on a name segment. Byte-prefix
/// comparison avoids a lowercase allocation and rejects non-ASCII mimics.
fn private_tag(part: &str) -> bool {
    part.as_bytes()
        .get(..PRIVATE_SCOPE_MARKER.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(PRIVATE_SCOPE_MARKER.as_bytes()))
}

/// A model id `group@private` is a private-scope marker.
pub(crate) fn model_is_private(model: &str) -> bool {
    model.split('@').skip(1).any(private_tag)
}

/// Model group is the part before `@` (`chat@private` -> `chat`).
pub(crate) fn model_group(model: &str) -> &str {
    model
        .split_once('@')
        .map_or(model, |(group, _)| group)
        .trim()
}

/// Upstreams are tagged private by a `@private*` name segment, a `private:`
/// name prefix, or a `/private/v1` base path. No config changes needed: the
/// operator's existing label is the scope declaration.
pub(crate) fn upstream_is_private(upstream: &MeshUpstream) -> bool {
    upstream.name.split('@').skip(1).any(private_tag)
        || upstream.name.split(':').next().is_some_and(private_tag)
        || upstream.base_url.path().starts_with(PRIVATE_PATH_PREFIX)
}

/// Upstream group is the part of its name before `@`.
pub(crate) fn upstream_group(upstream: &MeshUpstream) -> &str {
    upstream
        .name
        .split_once('@')
        .map_or(upstream.name.as_str(), |(group, _)| group)
}

/// Resolve scope from the explicit header, the request path and the model.
/// Any private marker wins; an unknown header value is a named 400, never a
/// silent public default.
pub(crate) fn parse_scope(
    value: Option<&HeaderValue>,
    path: &str,
    model: Option<&str>,
) -> Result<MeshScope, MeshError> {
    let header_scope = match value {
        None => MeshScope::Public,
        Some(value) => match value.to_str().ok().map(str::trim) {
            Some(raw) if raw.eq_ignore_ascii_case(PRIVATE_SCOPE_MARKER) => MeshScope::Private,
            Some(raw) if raw.eq_ignore_ascii_case("public") => MeshScope::Public,
            _ => return Err(MeshError::InvalidHeader(MESH_SCOPE_HEADER)),
        },
    };
    if header_scope == MeshScope::Private
        || path.starts_with(PRIVATE_PATH_PREFIX)
        || model.is_some_and(model_is_private)
    {
        Ok(MeshScope::Private)
    } else {
        Ok(MeshScope::Public)
    }
}

/// Parse a comma-separated provider-name header. Strict syntax: no empty
/// entries, whitespace, path separators or oversized labels. The RAW value is
/// never echoed into the named error (it could carry a token).
pub(crate) fn parse_name_list(
    value: Option<&HeaderValue>,
    header: &'static str,
) -> Result<Vec<String>, MeshError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let invalid = || MeshError::InvalidHeader(header);
    let raw = value.to_str().map_err(|_| invalid())?;
    if raw.trim().is_empty() {
        return Err(invalid());
    }
    let mut names = Vec::new();
    for part in raw.split(',') {
        let name = part.trim();
        if name.is_empty()
            || name.len() > 128
            || name.chars().any(char::is_whitespace)
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'@' | b':'))
        {
            return Err(invalid());
        }
        names.push(name.to_owned());
    }
    Ok(names)
}

/// Per-provider failure cooldown. Keys are configured upstream names only,
/// so the map is bounded by config size; every method is deterministic given
/// an injected `Instant` (no hidden clock in the policy helpers).
#[derive(Debug, Default)]
pub(crate) struct Cooldowns {
    until: Mutex<HashMap<String, Instant>>,
}

/// A pathological config value must not pin a provider for days.
const MAX_COOLDOWN: Duration = Duration::from_secs(3600);

impl Cooldowns {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn is_cooling(&self, name: &str, now: Instant) -> bool {
        self.until
            .lock()
            .ok()
            .and_then(|until| until.get(name).copied())
            .is_some_and(|until| until > now)
    }

    pub(crate) fn record_failure(&self, name: &str, ttl: Duration, now: Instant) {
        if let Ok(mut until) = self.until.lock() {
            let until_at = now.checked_add(ttl.min(MAX_COOLDOWN)).unwrap_or(now);
            until.insert(name.to_owned(), until_at);
        }
    }

    pub(crate) fn clear(&self, name: &str) {
        if let Ok(mut until) = self.until.lock() {
            until.remove(name);
        }
    }
}

/// Pure provider selection. Given the hints, a cooling predicate and a clock
/// snapshot it returns at most `budget` upstreams in fallback order, or a
/// named error. No network, no side effects. Pinned targets are never
/// widened: a cooling pin fails closed instead of silently rerouting.
pub(crate) fn select_upstreams<'a>(
    upstreams: &'a [MeshUpstream],
    hints: &RerouteHints,
    cooling: impl Fn(&str) -> bool,
    budget: usize,
) -> Result<Vec<&'a MeshUpstream>, MeshError> {
    let budget = budget.clamp(1, MAX_ATTEMPT_BUDGET);
    for name in hints.targets.iter().chain(hints.exclude.iter()) {
        if !upstreams.iter().any(|u| &u.name == name) {
            return Err(MeshError::TargetConflict(name.clone()));
        }
    }
    if hints
        .targets
        .iter()
        .enumerate()
        .any(|(index, name)| hints.targets[..index].contains(name))
    {
        let name = hints
            .targets
            .last()
            .expect("duplicate implies entries")
            .clone();
        return Err(MeshError::TargetConflict(name));
    }
    if let Some(name) = hints
        .targets
        .iter()
        .find(|name| hints.exclude.contains(*name))
    {
        return Err(MeshError::TargetConflict(name.clone()));
    }

    let pinned = !hints.targets.is_empty();
    // A private request may never be pinned to a public provider: fail closed
    // instead of letting a caller-supplied target widen the scope.
    if pinned
        && hints.scope == MeshScope::Private
        && hints.targets.iter().any(|name| {
            upstreams
                .iter()
                .find(|upstream| &upstream.name == name)
                .is_none_or(|upstream| !upstream_is_private(upstream))
        })
    {
        return Err(MeshError::PrivateScopeBlocked);
    }
    let mut candidates: Vec<&MeshUpstream> = if pinned {
        if hints.targets.iter().any(|name| cooling(name)) {
            return Err(MeshError::TargetUnavailable);
        }
        hints
            .targets
            .iter()
            .filter_map(|name| upstreams.iter().find(|u| &u.name == name))
            .collect()
    } else {
        upstreams
            .iter()
            .filter(|u| !hints.exclude.contains(&u.name))
            // Scope is a strict partition: private requests see private
            // providers only, public requests see public ones only.
            .filter(|u| upstream_is_private(u) == (hints.scope == MeshScope::Private))
            .filter(|u| !cooling(&u.name))
            .collect()
    };
    if candidates.is_empty() {
        return Err(if hints.scope == MeshScope::Private {
            MeshError::PrivateScopeBlocked
        } else {
            MeshError::NoUpstream
        });
    }
    if !pinned && let Some(group) = hints.group.as_deref() {
        // Stable sort: matching model group first, config order within
        // a group. This is the documented fallback order.
        candidates.sort_by_key(|u| std::cmp::Reverse(upstream_group(u) == group));
    }
    candidates.truncate(budget);
    Ok(candidates)
}

/// A stream wrapper enforcing the response byte cap MID-STREAM: once the
/// cap is crossed the stream yields [`MeshError::StreamLimitExceeded`] and
/// terminates — rejected, never silently truncated.
struct CappedStream<S> {
    inner: S,
    /// The configured cap (kept for the error message).
    limit: u64,
    /// Bytes still allowed.
    remaining: u64,
    /// Set after the cap error fires; the stream then ends.
    done: bool,
}

impl<S> CappedStream<S> {
    fn new(inner: S, limit: u64) -> Self {
        Self {
            inner,
            limit,
            remaining: limit,
            done: false,
        }
    }
}

impl<S> tokio_stream::Stream for CappedStream<S>
where
    S: tokio_stream::Stream<Item = Result<Bytes, MeshError>> + Unpin,
{
    type Item = Result<Bytes, MeshError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.done {
            return std::task::Poll::Ready(None);
        }
        match std::pin::Pin::new(&mut self.inner).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                let len = u64::try_from(chunk.len()).unwrap_or(u64::MAX);
                if len > self.remaining {
                    self.done = true;
                    return std::task::Poll::Ready(Some(Err(MeshError::StreamLimitExceeded(
                        self.limit,
                    ))));
                }
                self.remaining -= len;
                std::task::Poll::Ready(Some(Ok(chunk)))
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_cap_defaults_and_rejects_zero() {
        assert_eq!(
            resolve_discovery_cap(None).expect("default"),
            DEFAULT_DISCOVERY_CAP
        );
        assert_eq!(resolve_discovery_cap(Some(7)).expect("override"), 7);
        assert_eq!(
            resolve_discovery_cap(Some(0)).unwrap_err(),
            "[mesh] discovery_cap must be at least 1"
        );
    }

    /// The const tables are exactly what the ticket pins: the request
    /// allowlist, the response allowlist, and the hop-by-hop set.
    #[test]
    fn header_tables_pin_exact_sets() {
        assert_eq!(
            FORWARDED_REQUEST_HEADERS,
            ["accept"],
            "request allowlist is exactly {{accept}}; accept-encoding is excluded (no reqwest decoder), user-agent/x-request-id are rewritten, not forwarded"
        );
        assert_eq!(
            FORWARDED_RESPONSE_HEADERS,
            ["content-type"],
            "response allowlist is exactly {{content-type}}; x-request-id echoes stripped, retry-after mapped not relayed"
        );
        assert_eq!(
            HOP_BY_HOP_HEADERS,
            [
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "proxy-connection",
                "te",
                "trailer",
                "transfer-encoding",
                "upgrade",
            ],
            "hop-by-hop set per RFC 7230 §6.1 + proxy-*"
        );
        assert_eq!(
            STRIPPED_CREDENTIAL_HEADERS,
            ["authorization", "cookie", "x-api-key", "x-forwarded-for"],
        );
    }

    /// No allowlist entry may ever collide with a hop-by-hop header (they are
    /// stripped in BOTH directions; an overlap would be a contract bug).
    #[test]
    fn allowlists_never_overlap_hop_by_hop() {
        for name in FORWARDED_REQUEST_HEADERS
            .iter()
            .chain(FORWARDED_RESPONSE_HEADERS.iter())
        {
            assert!(
                !HOP_BY_HOP_HEADERS.contains(name),
                "allowlist entry {name} collides with the hop-by-hop set"
            );
        }
        // Credential headers are never forwarded either.
        for name in STRIPPED_CREDENTIAL_HEADERS {
            assert!(!FORWARDED_REQUEST_HEADERS.contains(name));
            assert!(!FORWARDED_RESPONSE_HEADERS.contains(name));
        }
    }

    /// Pure builder: only allowlisted + rewritten headers survive.
    #[test]
    fn build_request_headers_strips_everything_else() {
        let mut inbound = HeaderMap::new();
        inbound.insert("accept", HeaderValue::from_static("text/event-stream"));
        inbound.insert("accept-encoding", HeaderValue::from_static("gzip"));
        inbound.insert(
            "authorization",
            HeaderValue::from_static("Bearer client-secret"),
        );
        inbound.insert("cookie", HeaderValue::from_static("session=steal"));
        inbound.insert("x-api-key", HeaderValue::from_static("sk-steal"));
        inbound.insert("x-forwarded-for", HeaderValue::from_static("10.9.8.7"));
        inbound.insert("x-honkbox-internal", HeaderValue::from_static("nope"));
        inbound.insert("x-request-id", HeaderValue::from_static("client-id"));
        inbound.insert("connection", HeaderValue::from_static("keep-alive"));

        let out = build_request_headers(&inbound, "fresh-id");
        assert_eq!(out.get("accept").unwrap(), "text/event-stream");
        assert!(
            out.get("accept-encoding").is_none(),
            "accept-encoding is never forwarded: reqwest here cannot decode a compressed body"
        );
        assert_eq!(
            out.get("user-agent").unwrap(),
            concat!("honkbox/", env!("CARGO_PKG_VERSION")),
            "user-agent rewritten"
        );
        assert_eq!(out.get("x-request-id").unwrap(), "fresh-id", "regenerated");
        for name in [
            "accept-encoding",
            "authorization",
            "cookie",
            "x-api-key",
            "x-forwarded-for",
            "x-honkbox-internal",
            "connection",
        ] {
            assert!(
                out.get(name).is_none(),
                "credential/forbidden header {name} must never reach the outbound request"
            );
        }
    }

    /// An empty inbound request still yields the rewritten identity headers.
    #[test]
    fn build_request_headers_minimal() {
        let out = build_request_headers(&HeaderMap::new(), "id-1");
        assert_eq!(out.len(), 2);
        assert_eq!(out.get("x-request-id").unwrap(), "id-1");
        assert!(out.get("user-agent").is_some());
    }

    /// Response filter: only content-type survives; x-request-id echoes and
    /// retry-after (mapped elsewhere) are not relayed.
    #[test]
    fn filter_response_headers_allowlist_only() {
        let mut upstream = HeaderMap::new();
        upstream.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        upstream.insert("x-request-id", HeaderValue::from_static("upstream-echo"));
        upstream.insert("retry-after", HeaderValue::from_static("7"));
        upstream.insert("set-cookie", HeaderValue::from_static("a=b"));
        upstream.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        upstream.insert("server", HeaderValue::from_static("mesh"));

        let out = filter_response_headers(&upstream);
        assert_eq!(out.len(), 1);
        assert_eq!(out.get("content-type").unwrap(), "text/event-stream");
    }

    /// `retry-after` parses as delta-seconds only; absent/invalid → None.
    #[test]
    fn retry_after_mapped_not_relayed() {
        let mut headers = HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
        headers.insert("retry-after", HeaderValue::from_static("12"));
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(12)));
        headers.insert("retry-after", HeaderValue::from_static("soon"));
        assert_eq!(parse_retry_after(&headers), None);
        headers.insert("retry-after", HeaderValue::from_static(" 30 "));
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(30)));
    }

    /// Config resolve: loud errors for typos, enabled-without-upstreams, and
    /// duplicate labels; disabled-with-upstreams is allowed with a warning.
    #[test]
    fn mesh_config_resolution_rules() {
        let upstream = |name: &str, url: &str| FileMeshUpstream {
            name: Some(name.to_owned()),
            base_url: Some(url.to_owned()),
            bearer_token: Some("secret".to_owned()),
        };
        let mesh = FileMesh {
            enabled: Some(true),
            upstreams: Some(vec![upstream("peer-a", "http://10.0.0.9:8080")]),
            max_response_bytes: None,
            discovery_ttl_secs: Some(DEFAULT_DISCOVERY_TTL_SECS),
            discovery_cap: Some(DEFAULT_DISCOVERY_CAP),
        };
        let resolved = MeshConfig::resolve(&mesh).unwrap();
        assert!(resolved.enabled);
        assert_eq!(resolved.upstreams.len(), 1);
        assert_eq!(
            resolved.max_response_bytes, DEFAULT_MAX_RESPONSE_BYTES,
            "absent cap keeps the default"
        );

        // Default is OFF.
        let off = MeshConfig::resolve(&FileMesh::default()).unwrap();
        assert!(!off.enabled);
        assert!(off.upstreams.is_empty());

        // enabled + zero upstreams is a config error.
        let empty = FileMesh {
            enabled: Some(true),
            upstreams: None,
            max_response_bytes: None,
            discovery_ttl_secs: Some(DEFAULT_DISCOVERY_TTL_SECS),
            discovery_cap: Some(DEFAULT_DISCOVERY_CAP),
        };
        assert!(MeshConfig::resolve(&empty).is_err());

        // Duplicate labels are a config error.
        let dup = FileMesh {
            enabled: Some(true),
            upstreams: Some(vec![
                upstream("a", "http://a:1"),
                upstream("a", "http://a:2"),
            ]),
            max_response_bytes: None,
            discovery_ttl_secs: Some(DEFAULT_DISCOVERY_TTL_SECS),
            discovery_cap: Some(DEFAULT_DISCOVERY_CAP),
        };
        assert!(MeshConfig::resolve(&dup).is_err());

        // Bad scheme / unparseable URL is a config error.
        for bad_url in ["ftp://a:1", "not a url", ""] {
            let bad = FileMesh {
                enabled: Some(true),
                upstreams: Some(vec![upstream("a", bad_url)]),
                max_response_bytes: None,
                discovery_ttl_secs: Some(DEFAULT_DISCOVERY_TTL_SECS),
                discovery_cap: Some(DEFAULT_DISCOVERY_CAP),
            };
            assert!(MeshConfig::resolve(&bad).is_err(), "url {bad_url:?}");
        }

        // Upstreams while disabled is allowed (warn + keep off).
        let disabled = FileMesh {
            enabled: Some(false),
            upstreams: Some(vec![upstream("peer-a", "http://10.0.0.9:8080")]),
            max_response_bytes: None,
            discovery_ttl_secs: Some(DEFAULT_DISCOVERY_TTL_SECS),
            discovery_cap: Some(DEFAULT_DISCOVERY_CAP),
        };
        let cfg = MeshConfig::resolve(&disabled).unwrap();
        assert!(!cfg.enabled);
        assert_eq!(cfg.upstreams.len(), 1);
    }

    /// The config Debug output (which `LoadedConfig` derives) never shows the
    /// bearer token — the redaction contract in one assertion.
    #[test]
    fn debug_redacts_bearer_token() {
        let cfg = MeshConfig::resolve(&FileMesh {
            enabled: Some(true),
            upstreams: Some(vec![FileMeshUpstream {
                name: Some("peer".to_owned()),
                base_url: Some("https://peer.internal:9000".to_owned()),
                bearer_token: Some("super-secret-value".to_owned()),
            }]),
            max_response_bytes: None,
            discovery_ttl_secs: Some(DEFAULT_DISCOVERY_TTL_SECS),
            discovery_cap: Some(DEFAULT_DISCOVERY_CAP),
        })
        .unwrap();
        let rendered = format!("{cfg:?}");
        assert!(!rendered.contains("super-secret-value"));
        assert!(rendered.contains("<redacted>"));
    }

    /// One configured provider for the policy tests. Names are the scope /
    /// group declaration, so no new config field is needed.
    fn peer(name: &str) -> MeshUpstream {
        MeshUpstream {
            name: name.to_owned(),
            base_url: Url::parse("http://peer.example").expect("valid base"),
            bearer_token: None,
        }
    }

    fn hints(scope: MeshScope, group: Option<&str>) -> RerouteHints {
        RerouteHints {
            scope,
            group: group.map(str::to_owned),
            targets: Vec::new(),
            exclude: Vec::new(),
        }
    }

    #[test]
    fn scope_markers_are_sticky_and_unknown_is_named() {
        assert_eq!(
            parse_scope(None, "/v1/chat", None).unwrap(),
            MeshScope::Public
        );
        assert_eq!(
            parse_scope(Some(&HeaderValue::from_static("private")), "/v1/chat", None).unwrap(),
            MeshScope::Private
        );
        assert_eq!(
            parse_scope(None, "/private/v1/chat", None).unwrap(),
            MeshScope::Private
        );
        assert_eq!(
            parse_scope(None, "/v1/chat", Some("chat@private")).unwrap(),
            MeshScope::Private
        );
        // A public header cannot widen a request that carries a private marker.
        assert_eq!(
            parse_scope(
                Some(&HeaderValue::from_static("public")),
                "/private/v1/chat",
                Some("chat@private")
            )
            .unwrap(),
            MeshScope::Private
        );
        let bad =
            parse_scope(Some(&HeaderValue::from_static("bogus")), "/v1/chat", None).unwrap_err();
        assert_eq!(bad.code(), "mesh_invalid_scope");
        assert_eq!(bad.response_status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn provider_name_lists_are_strict_without_echoing_values() {
        assert_eq!(
            parse_name_list(None, MESH_TARGET_HEADER).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            parse_name_list(
                Some(&HeaderValue::from_static(" chat@one , peer-two ")),
                MESH_TARGET_HEADER
            )
            .unwrap(),
            ["chat@one", "peer-two"]
        );
        for raw in ["a,,b", "a b", "", "http://x", ",a"] {
            let err = parse_name_list(
                Some(&HeaderValue::from_str(raw).unwrap()),
                MESH_TARGET_HEADER,
            )
            .unwrap_err();
            assert_eq!(err.code(), "mesh_invalid_target", "raw {raw:?}");
            assert_eq!(err.response_status(), StatusCode::BAD_REQUEST);
            if !raw.is_empty() {
                assert!(!err.to_string().contains(raw), "raw value never echoed");
            }
        }
        let err = parse_name_list(Some(&HeaderValue::from_static("a,,b")), MESH_EXCLUDE_HEADER)
            .unwrap_err();
        assert_eq!(err.code(), "mesh_invalid_exclude");
    }

    #[test]
    fn scope_and_group_derived_from_existing_config_labels() {
        assert!(upstream_is_private(&peer("chat@private")));
        assert!(upstream_is_private(&peer("chat@private-2")));
        assert!(upstream_is_private(&peer("private:chat")));
        assert!(!upstream_is_private(&peer("chat@peer-a")));
        let mut path_private = peer("peer");
        path_private.base_url = Url::parse("http://peer.example/private/v1").unwrap();
        assert!(upstream_is_private(&path_private));
        assert_eq!(upstream_group(&peer("chat@peer-a")), "chat");
        assert_eq!(model_group("chat@private"), "chat");
        assert!(model_is_private("chat@private"));
        assert!(!model_is_private("chat@peer-a"));
    }

    #[test]
    fn selection_orders_groups_filters_scope_and_caps_budget() {
        let upstreams = vec![
            peer("other@public-a"),
            peer("chat@public-b"),
            peer("chat@private"),
        ];
        let public = hints(MeshScope::Public, Some("chat"));
        let plan = select_upstreams(&upstreams, &public, |_| false, 2).unwrap();
        assert_eq!(
            plan.iter().map(|u| u.name.as_str()).collect::<Vec<_>>(),
            ["chat@public-b", "other@public-a"],
            "matching model group first, config order within a group"
        );
        assert!(plan.len() <= 2, "budget is a hard cap");

        let private = hints(MeshScope::Private, None);
        let plan =
            select_upstreams(&upstreams, &private, |_| false, DEFAULT_ATTEMPT_BUDGET).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].name, "chat@private");
        assert!(plan.iter().all(|u| upstream_is_private(u)));

        let mut excluded = hints(MeshScope::Public, None);
        excluded.exclude.push("chat@public-b".to_owned());
        let plan = select_upstreams(&upstreams, &excluded, |_| false, 4).unwrap();
        assert_eq!(plan.len(), 1, "excluded provider never selected");
        assert_eq!(plan[0].name, "other@public-a");

        let cooling = |name: &str| name == "chat@private";
        let err = select_upstreams(&upstreams, &private, cooling, 3).unwrap_err();
        assert_eq!(err.code(), "mesh_private_scope_blocked");
        assert_eq!(err.response_status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn pinned_targets_fail_closed_on_conflict_or_cooldown() {
        let upstreams = vec![peer("chat@public-a"), peer("chat@private")];
        let mut pinned = hints(MeshScope::Public, Some("chat"));
        pinned.targets.push("ghost".to_owned());
        let err = select_upstreams(&upstreams, &pinned, |_| false, 3).unwrap_err();
        assert_eq!(err.code(), "mesh_target_conflict");
        assert_eq!(err.response_status(), StatusCode::CONFLICT);

        pinned.targets = vec!["chat@public-a".to_owned()];
        pinned.exclude = vec!["chat@public-a".to_owned()];
        assert_eq!(
            select_upstreams(&upstreams, &pinned, |_| false, 3)
                .unwrap_err()
                .code(),
            "mesh_target_conflict"
        );

        pinned.exclude.clear();
        pinned.targets.push("chat@public-a".to_owned());
        assert_eq!(
            select_upstreams(&upstreams, &pinned, |_| false, 3)
                .unwrap_err()
                .code(),
            "mesh_target_conflict"
        );

        pinned.targets.pop();
        let cooling = |name: &str| name == "chat@public-a";
        let err = select_upstreams(&upstreams, &pinned, cooling, 3).unwrap_err();
        assert_eq!(err.code(), "mesh_target_unavailable");
        assert_eq!(err.response_status(), StatusCode::BAD_GATEWAY);

        let plan = select_upstreams(&upstreams, &pinned, |_| false, 1).unwrap();
        assert_eq!(plan.len(), 1, "pins keep declaration order");
        assert_eq!(plan[0].name, "chat@public-a");
    }

    #[test]
    fn cooldown_is_clock_injected_bounded_and_clearable() {
        let now = Instant::now();
        let cooldowns = Cooldowns::new();
        assert!(!cooldowns.is_cooling("peer", now));
        cooldowns.record_failure("peer", Duration::from_secs(5), now);
        assert!(cooldowns.is_cooling("peer", now));
        assert!(cooldowns.is_cooling("peer", now + Duration::from_secs(4)));
        assert!(!cooldowns.is_cooling("peer", now + Duration::from_secs(6)));
        cooldowns.clear("peer");
        assert!(!cooldowns.is_cooling("peer", now));
        // Even an absurd config ttl cannot pin a provider for long.
        cooldowns.record_failure("peer", Duration::from_secs(u64::MAX), now);
        assert!(!cooldowns.is_cooling("peer", now + Duration::from_secs(3601)));
    }

    #[test]
    fn metric_labels_are_a_closed_bounded_set() {
        let kinds = [
            MeshMetricKind::Selected,
            MeshMetricKind::Retry,
            MeshMetricKind::Cooldown,
            MeshMetricKind::Rejected,
            MeshMetricKind::PrivateBlocked,
            MeshMetricKind::Exhausted,
        ];
        for (index, kind) in kinds.into_iter().enumerate() {
            assert_eq!(kind.label(), MESH_METRIC_KINDS[index]);
        }
        let before = mesh_metric_count(MeshMetricKind::Selected);
        record_mesh_metric(MeshMetricKind::Selected);
        assert_eq!(mesh_metric_count(MeshMetricKind::Selected), before + 1);
    }
}
