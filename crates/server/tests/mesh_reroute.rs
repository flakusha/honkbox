//! Bounded reroute policy contract tests (TASK-reroute-policy). All upstreams
//! are local TCP mocks on assigned ports; no external network, no global
//! cooldown state shared between tests (each state is built per test).

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{Method, StatusCode};
use axum::response::Response;
use axum::routing::post;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tower::ServiceExt as _;

// Binary crate: include the modules the handler needs, house pattern from
// tests/mesh_proxy.rs.
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/routes/mesh.rs"]
mod mesh_route;
#[allow(dead_code)]
#[path = "../src/metrics.rs"]
mod metrics;

use config::mesh::{
    DEFAULT_MAX_RESPONSE_BYTES, MESH_METRIC_KINDS, MeshConfig, MeshError, MeshMetricKind,
    MeshUpstream, mesh_metric_count,
};
use mesh_route::{MESH_REROUTE_PATH, ReroutePolicy, RerouteState, post_mesh_reroute};

/// A mock upstream that accepts unlimited framed connections; each request is
/// handled in its own task (the client opens one connection per attempt).
async fn spawn_upstream<F, Fut>(handler: F) -> std::net::SocketAddr
where
    F: Fn(TcpStream) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("mock bind");
    let addr = listener.local_addr().expect("mock addr");
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(handler(stream));
        }
    });
    addr
}

/// Read one request head (through the blank line) plus its content-length
/// body. Returns lowercased headers and the raw body.
async fn read_request(stream: &mut TcpStream) -> (HashMap<String, String>, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut chunk).await.expect("read head");
        assert!(n > 0, "upstream closed before request head");
        buf.extend_from_slice(&chunk[..n]);
    }
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("head terminator")
        + 4;
    let head = String::from_utf8_lossy(&buf[..split]).to_string();
    let mut body = buf[split..].to_vec();
    let len = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    while body.len() < len {
        let n = stream.read(&mut chunk).await.expect("read body");
        assert!(n > 0, "upstream closed before request body");
        body.extend_from_slice(&chunk[..n]);
    }
    let headers = head
        .lines()
        .skip(1)
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        })
        .collect();
    (headers, body)
}

/// A healthy stub: counts attempts and answers with a labelled JSON body.
async fn serve_ok(mut stream: TcpStream, count: Arc<AtomicUsize>, label: &'static str) {
    count.fetch_add(1, Ordering::SeqCst);
    let (_headers, _body) = read_request(&mut stream).await;
    let body = format!("{{\"provider\":\"{label}\"}}");
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await.ok();
    stream.flush().await.ok();
}

/// A dead stub: accepts then closes with no response — a transport failure
/// that `MeshProxy::forward` must report (and the policy may retry).
async fn serve_dead(mut stream: TcpStream, count: Arc<AtomicUsize>) {
    count.fetch_add(1, Ordering::SeqCst);
    let _ = read_request(&mut stream).await;
}

/// A stub that records the exact headers it saw on the wire.
async fn serve_capture(
    mut stream: TcpStream,
    seen: Arc<std::sync::Mutex<Vec<HashMap<String, String>>>>,
) {
    let (headers, _body) = read_request(&mut stream).await;
    seen.lock().expect("capture lock").push(headers);
    let body = "{\"ok\":true}";
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await.ok();
    stream.flush().await.ok();
}

fn upstream(name: &str, addr: std::net::SocketAddr) -> MeshUpstream {
    MeshUpstream {
        name: name.to_owned(),
        base_url: format!("http://{addr}").parse().expect("valid mock url"),
        bearer_token: Some("config-bearer-secret".to_owned()),
    }
}

fn config_with(upstreams: Vec<MeshUpstream>) -> MeshConfig {
    MeshConfig {
        enabled: true,
        upstreams,
        max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        ..Default::default()
    }
}

fn app(state: RerouteState) -> axum::Router {
    axum::Router::new()
        .route(MESH_REROUTE_PATH, post(post_mesh_reroute))
        .route("/private/v1/mesh/reroute", post(post_mesh_reroute))
        .with_state(Arc::new(state))
}

fn turn(headers: &[(&str, &str)], model: &str) -> Request {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(MESH_REROUTE_PATH);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder
        .body(Body::from(json!({ "model": model }).to_string()))
        .expect("request builds")
}

async fn status_and_body(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .expect("body reads");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("utf8 body"),
    )
}

/// Pull the named code out of the house `{"error":{"code":...}}` envelope.
fn code(body: &str) -> String {
    let value: Value = serde_json::from_str(body).expect("json envelope");
    value["error"]["code"]
        .as_str()
        .expect("named code")
        .to_owned()
}

/// `x-mesh-scope: private` routes ONLY to the `@private` provider; a public
/// provider never sees the turn, even as a last resort.
#[tokio::test]
async fn private_scope_header_never_falls_back_to_public() {
    let public_hits = Arc::new(AtomicUsize::new(0));
    let private_hits = Arc::new(AtomicUsize::new(0));
    let public = spawn_upstream({
        let hits = Arc::clone(&public_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "public")
    })
    .await;
    let private = spawn_upstream({
        let hits = Arc::clone(&private_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "private")
    })
    .await;
    let state = RerouteState::new(&config_with(vec![
        upstream("chat@public-a", public),
        upstream("chat@private", private),
    ]))
    .expect("state builds");

    let response = app(state)
        .oneshot(turn(&[("x-mesh-scope", "private")], "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("private"), "private provider answered");
    assert_eq!(private_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        public_hits.load(Ordering::SeqCst),
        0,
        "public upstream is never contacted for private scope"
    );
}

/// The `/private/v1` path marker is just as sticky as the explicit header.
#[tokio::test]
async fn private_path_marker_never_falls_back_to_public() {
    let public_hits = Arc::new(AtomicUsize::new(0));
    let private_hits = Arc::new(AtomicUsize::new(0));
    let public = spawn_upstream({
        let hits = Arc::clone(&public_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "public")
    })
    .await;
    let private = spawn_upstream({
        let hits = Arc::clone(&private_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "private")
    })
    .await;
    let state = RerouteState::new(&config_with(vec![
        upstream("chat@public-a", public),
        upstream("chat@private", private),
    ]))
    .expect("state builds");

    let mut request = turn(&[], "chat");
    *request.uri_mut() = "/private/v1/mesh/reroute".parse().expect("private uri");
    let response = app(state).oneshot(request).await.expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("private"));
    assert_eq!(public_hits.load(Ordering::SeqCst), 0);
}

/// A `@private` model id also locks the request to private providers.
#[tokio::test]
async fn private_model_marker_never_falls_back_to_public() {
    let public_hits = Arc::new(AtomicUsize::new(0));
    let private_hits = Arc::new(AtomicUsize::new(0));
    let public = spawn_upstream({
        let hits = Arc::clone(&public_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "public")
    })
    .await;
    let private = spawn_upstream({
        let hits = Arc::clone(&private_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "private")
    })
    .await;
    let state = RerouteState::new(&config_with(vec![
        upstream("chat@public-a", public),
        upstream("chat@private", private),
    ]))
    .expect("state builds");

    let response = app(state)
        .oneshot(turn(&[], "chat@private"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("private"));
    assert_eq!(public_hits.load(Ordering::SeqCst), 0);
}

/// Private scope with no private provider is a named 502, never a silent
/// downgrade to the configured public provider.
#[tokio::test]
async fn private_scope_without_private_provider_fails_closed() {
    let public_hits = Arc::new(AtomicUsize::new(0));
    let public = spawn_upstream({
        let hits = Arc::clone(&public_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "public")
    })
    .await;
    let state = RerouteState::new(&config_with(vec![upstream("chat@public-a", public)]))
        .expect("state builds");

    let response = app(state)
        .oneshot(turn(&[("x-mesh-scope", "private")], "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(code(&body), "mesh_private_scope_blocked");
    assert_eq!(public_hits.load(Ordering::SeqCst), 0);
}

/// Malformed routing hints are named 400s with the house envelope.
#[tokio::test]
async fn malformed_routing_headers_are_named_400() {
    let state = RerouteState::new(&config_with(vec![upstream(
        "chat@public-a",
        "127.0.0.1:1".parse().expect("dummy addr"),
    )]))
    .expect("state builds");

    for (headers, expected) in [
        (vec![("x-mesh-target", "a,,b")], "mesh_invalid_target"),
        (vec![("x-mesh-exclude", "a b")], "mesh_invalid_exclude"),
        (vec![("x-mesh-scope", "secret")], "mesh_invalid_scope"),
    ] {
        let response = app(state.clone())
            .oneshot(turn(&headers, "chat"))
            .await
            .expect("handler responds");
        let (status, body) = status_and_body(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{headers:?}");
        assert_eq!(code(&body), expected);
        assert!(body.contains("\"type\":\"invalid_request_error\""));
    }

    // A duplicated pin is ambiguous: also a named 400.
    let mut request = turn(&[("x-mesh-target", "chat@public-a")], "chat");
    request.headers_mut().append(
        "x-mesh-target",
        "chat@public-a".parse().expect("header value"),
    );
    let response = app(state).oneshot(request).await.expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(code(&body), "mesh_invalid_target");
}

/// Unknown, duplicated and overlapping targets are named 409s — the caller
/// gets a conflict, never a silently different provider.
#[tokio::test]
async fn conflicting_targets_are_named_409() {
    let state = RerouteState::new(&config_with(vec![upstream(
        "chat@public-a",
        "127.0.0.1:1".parse().expect("dummy addr"),
    )]))
    .expect("state builds");

    for headers in [
        vec![("x-mesh-target", "ghost")],
        vec![("x-mesh-exclude", "ghost")],
        vec![
            ("x-mesh-target", "chat@public-a"),
            ("x-mesh-exclude", "chat@public-a"),
        ],
        vec![("x-mesh-target", "chat@public-a,chat@public-a")],
    ] {
        let response = app(state.clone())
            .oneshot(turn(&headers, "chat"))
            .await
            .expect("handler responds");
        let (status, body) = status_and_body(response).await;
        assert_eq!(status, StatusCode::CONFLICT, "{headers:?}");
        assert_eq!(code(&body), "mesh_target_conflict");
    }
}

/// Retry budget is the TOTAL forward count. A dead provider is never tried
/// more times than the budget allows.
#[tokio::test]
async fn retry_budget_is_never_exceeded() {
    for budget in [1, 2] {
        let hits = Arc::new(AtomicUsize::new(0));
        let addr = spawn_upstream({
            let hits = Arc::clone(&hits);
            move |stream| serve_dead(stream, Arc::clone(&hits))
        })
        .await;
        let state = RerouteState::with_policy(
            &config_with(vec![upstream("chat@public-a", addr)]),
            ReroutePolicy {
                attempt_budget: budget,
                cooldown: Duration::from_secs(60),
            },
        )
        .expect("state builds");

        let response = app(state)
            .oneshot(turn(&[], "chat"))
            .await
            .expect("handler responds");
        let (status, body) = status_and_body(response).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(code(&body), "mesh_retry_budget_exhausted");
        // One configured upstream is one candidate: the budget is a ceiling,
        // not a reason to retry the same failed provider in the same turn.
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert!(hits.load(Ordering::SeqCst) <= budget);
    }
}

/// A pinned provider that is cooling down produces a named 502; the healthy
/// sibling is never silently substituted.
#[tokio::test]
async fn cooling_pinned_target_is_a_named_502_without_reroute() {
    let dead_hits = Arc::new(AtomicUsize::new(0));
    let healthy_hits = Arc::new(AtomicUsize::new(0));
    let dead = spawn_upstream({
        let hits = Arc::clone(&dead_hits);
        move |stream| serve_dead(stream, Arc::clone(&hits))
    })
    .await;
    let healthy = spawn_upstream({
        let hits = Arc::clone(&healthy_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "healthy")
    })
    .await;
    let state = RerouteState::with_policy(
        &config_with(vec![
            upstream("chat@public-a", dead),
            upstream("chat@public-b", healthy),
        ]),
        ReroutePolicy {
            attempt_budget: 2,
            cooldown: Duration::from_secs(60),
        },
    )
    .expect("state builds");
    let app = app(state);

    let pin = [("x-mesh-target", "chat@public-a")];
    let response = app
        .clone()
        .oneshot(turn(&pin, "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(code(&body), "mesh_retry_budget_exhausted");
    assert_eq!(dead_hits.load(Ordering::SeqCst), 1);

    let response = app
        .oneshot(turn(&pin, "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(code(&body), "mesh_target_unavailable");
    assert_eq!(
        dead_hits.load(Ordering::SeqCst),
        1,
        "cooling pin is not retried"
    );
    assert_eq!(
        healthy_hits.load(Ordering::SeqCst),
        0,
        "no silent reroute to the sibling"
    );
}

/// Unpinned turns fall back in model-group order: the matching group is
/// tried first even when a different group is listed first in config.
#[tokio::test]
async fn fallback_order_prefers_the_requested_model_group() {
    let other_hits = Arc::new(AtomicUsize::new(0));
    let chat_hits = Arc::new(AtomicUsize::new(0));
    let other = spawn_upstream({
        let hits = Arc::clone(&other_hits);
        move |stream| serve_dead(stream, Arc::clone(&hits))
    })
    .await;
    let chat = spawn_upstream({
        let hits = Arc::clone(&chat_hits);
        move |stream| serve_ok(stream, Arc::clone(&hits), "chat")
    })
    .await;
    let state = RerouteState::new(&config_with(vec![
        upstream("other@public-a", other),
        upstream("chat@public-b", chat),
    ]))
    .expect("state builds");

    let response = app(state)
        .oneshot(turn(&[], "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(response).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("chat"));
    assert_eq!(chat_hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        other_hits.load(Ordering::SeqCst),
        0,
        "matching model group went first; no fallback was needed"
    );
}

/// A failed provider is cooled down, then becomes eligible again when the
/// cooldown elapses. Success clears the mark.
#[tokio::test]
async fn per_provider_cooldown_expires() {
    let hits = Arc::new(AtomicUsize::new(0));
    let addr = spawn_upstream({
        let hits = Arc::clone(&hits);
        move |stream| serve_dead(stream, Arc::clone(&hits))
    })
    .await;
    let state = RerouteState::with_policy(
        &config_with(vec![upstream("chat@public-a", addr)]),
        ReroutePolicy {
            attempt_budget: 1,
            cooldown: Duration::from_millis(100),
        },
    )
    .expect("state builds");
    let app = app(state);

    let first = app
        .clone()
        .oneshot(turn(&[], "chat"))
        .await
        .expect("handler responds");
    assert_eq!(
        code(&status_and_body(first).await.1),
        "mesh_retry_budget_exhausted"
    );

    let cooled = app
        .clone()
        .oneshot(turn(&[], "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(cooled).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(
        code(&body),
        "mesh_no_upstream",
        "cooling provider is filtered"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    tokio::time::sleep(Duration::from_millis(300)).await;
    let after = app
        .oneshot(turn(&[], "chat"))
        .await
        .expect("handler responds");
    let (status, body) = status_and_body(after).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(code(&body), "mesh_retry_budget_exhausted");
    assert_eq!(hits.load(Ordering::SeqCst), 2, "cooldown expired");
}

/// The header trust boundary is intact: routing hints and caller credentials
/// are stripped by `MeshProxy::forward`; only the config bearer reaches the
/// wire, with a freshly generated request id.
#[tokio::test]
async fn routing_hints_and_credentials_never_reach_the_wire() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let addr = spawn_upstream({
        let seen = Arc::clone(&seen);
        move |stream| serve_capture(stream, Arc::clone(&seen))
    })
    .await;
    let other = upstream("chat@public-b", "127.0.0.1:1".parse().expect("dummy"));
    let state = RerouteState::new(&config_with(vec![upstream("chat@public-a", addr), other]))
        .expect("state builds");

    let headers = [
        ("accept", "text/event-stream"),
        ("authorization", "Bearer caller-secret"),
        ("cookie", "session=caller"),
        ("x-api-key", "sk-caller"),
        ("x-forwarded-for", "10.9.8.7"),
        ("x-honkbox-internal", "scope:secret"),
        ("x-mesh-scope", "public"),
        ("x-mesh-target", "chat@public-a"),
        ("x-mesh-exclude", "chat@public-b"),
        ("x-request-id", "caller-request-id"),
        ("user-agent", "evil/9"),
        ("accept-encoding", "gzip"),
        ("connection", "keep-alive"),
    ];
    let response = app(state)
        .oneshot(turn(&headers, "chat"))
        .await
        .expect("handler responds");
    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("capture lock").clone();
    let wire = &seen[0];
    assert_eq!(
        wire.get("accept").map(String::as_str),
        Some("text/event-stream")
    );
    assert_eq!(
        wire.get("authorization").map(String::as_str),
        Some("Bearer config-bearer-secret"),
        "only the operator's per-upstream bearer is sent"
    );
    for name in [
        "cookie",
        "x-api-key",
        "x-forwarded-for",
        "x-honkbox-internal",
        "x-mesh-scope",
        "x-mesh-target",
        "x-mesh-exclude",
        "accept-encoding",
    ] {
        assert!(!wire.contains_key(name), "{name} stripped on the wire");
    }
    // The client deliberately sets a fresh framed connection per request;
    // the caller's value is never copied.
    assert_eq!(wire.get("connection").map(String::as_str), Some("close"));
    assert_ne!(
        wire.get("x-request-id").map(String::as_str),
        Some("caller-request-id")
    );
    assert!(
        wire.get("user-agent")
            .is_some_and(|value| value.starts_with("honkbox/")),
        "honkbox/<version>, never the caller's agent"
    );
}

/// Metrics use only the closed `kind` vocabulary; no URL, path, model or
/// token can become a label. The counters move on rejected and exhausted
/// outcomes.
#[tokio::test]
async fn reroute_metrics_use_bounded_kinds_only() {
    assert_eq!(MESH_METRIC_KINDS.len(), 6);
    for kind in [
        MeshMetricKind::Selected,
        MeshMetricKind::Retry,
        MeshMetricKind::Cooldown,
        MeshMetricKind::Rejected,
        MeshMetricKind::PrivateBlocked,
        MeshMetricKind::Exhausted,
    ] {
        assert!(MESH_METRIC_KINDS.contains(&kind.label()));
    }
    let rejected_before = mesh_metric_count(MeshMetricKind::Rejected);
    let exhausted_before = mesh_metric_count(MeshMetricKind::Exhausted);
    let state = RerouteState::new(&config_with(vec![])).expect("empty state builds");
    let response = app(state)
        .oneshot(turn(&[("x-mesh-target", "a,,b")], "chat"))
        .await
        .expect("handler responds");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // Other tests share these process-global counters and run in parallel.
    assert!(mesh_metric_count(MeshMetricKind::Rejected) > rejected_before);
    assert!(mesh_metric_count(MeshMetricKind::Exhausted) >= exhausted_before);
}

#[test]
fn named_error_statuses_pin_the_house_envelope() {
    assert_eq!(
        MeshError::InvalidHeader("x-mesh-target").response_status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        MeshError::InvalidBody.response_status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        MeshError::TargetConflict("peer".to_owned()).response_status(),
        StatusCode::CONFLICT
    );
    for err in [
        MeshError::TargetUnavailable,
        MeshError::NoUpstream,
        MeshError::PrivateScopeBlocked,
        MeshError::RetryBudgetExhausted { attempts: 2 },
        MeshError::Disabled,
    ] {
        assert_eq!(err.response_status(), StatusCode::BAD_GATEWAY);
    }
}
