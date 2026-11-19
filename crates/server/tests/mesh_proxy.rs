//! Mesh proxy stream-safety contract tests (TASK-mesh-client-spike +
//! TASK-proxy-stream-safety). Every interaction runs against a local
//! `tokio` TCP mock upstream — no external network. Covers: the header
//! trust boundary (injection regression pin), SSE unbuffered pass-through,
//! cancel propagation to the upstream, the mid-stream size cap, and the
//! no-lease rule for remote-only turns.

#![allow(clippy::too_many_lines)] // one mock harness per test, by design

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method};
use axum::response::Response;
use axum::routing::post;
use serde_json::json;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_stream::StreamExt as _;
use tower::ServiceExt as _;

// No lib target on this binary crate, so include the modules directly
// (house pattern: per-mod allow(dead_code); this harness does not mount the
// full app, only the config/mesh/jobs slice it exercises).
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;

use config::mesh::{MeshConfig, MeshError, MeshProxy, MeshUpstream};

/// Upstream mock: listen, hand the FIRST accepted connection to `handler`.
async fn spawn_upstream<F, Fut>(handler: F) -> std::net::SocketAddr
where
    F: FnOnce(TcpListener, TcpStream) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("mock upstream bind");
    let addr = listener.local_addr().expect("mock upstream addr");
    // Accept in the background: awaiting accept here would deadlock (the
    // caller connects only after this fn returns the address).
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("mock upstream accept");
        handler(listener, stream).await;
    });
    addr
}

/// Read the HTTP request head (through `\r\n\r\n`) plus a content-length
/// body. Returns (head bytes, body bytes).
async fn read_request(stream: &mut TcpStream) -> (Vec<u8>, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut chunk).await.expect("read request head");
        assert!(n > 0, "upstream closed before request head completed");
        buf.extend_from_slice(&chunk[..n]);
    }
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("head terminator")
        + 4;
    let mut head = buf[..split].to_vec();
    let mut body = buf[split..].to_vec();
    // Content-length body (the mock never speaks chunked requests).
    let len = String::from_utf8_lossy(&head)
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    while body.len() < len {
        let n = stream.read(&mut chunk).await.expect("read request body");
        assert!(n > 0, "upstream closed before request body completed");
        body.extend_from_slice(&chunk[..n]);
    }
    head.truncate(split - 4);
    (head, body)
}

/// Parse a request head into a lowercase-name -> value map (last value
/// wins; the mock's assertions never rely on duplicates).
fn parse_head(head: &[u8]) -> HashMap<String, String> {
    String::from_utf8_lossy(head)
        .lines()
        .skip(1)
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        })
        .collect()
}

/// A proxy pointed at `addr` with the given per-response cap.
fn proxy_at(addr: std::net::SocketAddr, max_response_bytes: u64) -> MeshProxy {
    let config = MeshConfig {
        enabled: true,
        upstreams: vec![MeshUpstream {
            name: "peer".to_owned(),
            base_url: format!("http://{addr}").parse().expect("mock base url"),
            bearer_token: Some("config-bearer-secret".to_owned()),
        }],
        max_response_bytes,
    };
    MeshProxy::new(&config).expect("mesh client builds")
}

/// Inbound request headers carrying every credential the ticket forbids,
/// plus the allowlisted ones and the headers that must be rewritten.
fn poisoned_inbound_headers() -> HeaderMap {
    let mut inbound = HeaderMap::new();
    inbound.insert("accept", HeaderValue::from_static("text/event-stream"));
    inbound.insert("accept-encoding", HeaderValue::from_static("gzip"));
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer caller-secret"),
    );
    inbound.insert("cookie", HeaderValue::from_static("session=caller"));
    inbound.insert("x-api-key", HeaderValue::from_static("sk-caller"));
    inbound.insert("x-forwarded-for", HeaderValue::from_static("10.9.8.7"));
    inbound.insert(
        "x-honkbox-internal",
        HeaderValue::from_static("scope:secret"),
    );
    inbound.insert(
        "x-request-id",
        HeaderValue::from_static("caller-request-id"),
    );
    inbound.insert("user-agent", HeaderValue::from_static("evil-client/9"));
    inbound.insert(
        "proxy-authorization",
        HeaderValue::from_static("Basic proxy"),
    );
    inbound.insert("te", HeaderValue::from_static("trailers"));
    inbound
}

/// Inbound body for the chat turn.
fn chat_body() -> axum::body::Bytes {
    json!({"model": "remote-model", "stream": true})
        .to_string()
        .into()
}

/// Read the relayed body to a string (bounded).
async fn body_text(body: Body) -> String {
    let bytes = axum::body::to_bytes(body, 8 * 1024 * 1024)
        .await
        .expect("body reads under the test cap");
    String::from_utf8(bytes.to_vec()).expect("utf8 body")
}

/// HEADER PASSING (trust boundary): the upstream-bound request carries ONLY
/// the allowlisted + rewritten headers. `authorization` from the caller and
/// the malicious `x-forwarded-for` must NOT arrive; the ONLY credential on
/// the wire is the per-upstream config bearer.
#[tokio::test]
async fn header_allowlist_upstream_wire_pin() {
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (head, _body) = read_request(&mut stream).await;
        let headers = parse_head(&head);
        let echoed = serde_json::to_string(
            &headers.iter().map(|(k, v)| vec![k.clone(), v.clone()]).collect::<Vec<_>>(),
        )
        .expect("serialize headers");
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            echoed.len(),
            echoed
        );
        stream.write_all(response.as_bytes()).await.ok();
        stream.flush().await.ok();
    })
    .await;
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let response = proxy
        .forward(
            upstream,
            Method::POST,
            "/mesh/echo",
            &poisoned_inbound_headers(),
            chat_body(),
        )
        .await
        .expect("forward succeeds");
    assert_eq!(response.status, 200);

    let wire: Vec<Vec<String>> =
        serde_json::from_str(&body_text(response.body).await).expect("echoed header map");
    let wire: HashMap<String, String> = wire
        .into_iter()
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect();

    // Allowlisted values pass through unchanged.
    assert_eq!(
        wire.get("accept").map(String::as_str),
        Some("text/event-stream")
    );
    assert!(
        !wire.contains_key("accept-encoding"),
        "accept-encoding is never forwarded: this client cannot decode a compressed response"
    );
    // Rewritten identity headers.
    assert_eq!(
        wire.get("user-agent").map(String::as_str),
        Some(concat!("honkbox/", env!("CARGO_PKG_VERSION"))),
        "user-agent identifies honkbox, never the caller's value"
    );
    assert_ne!(
        wire.get("x-request-id").map(String::as_str),
        Some("caller-request-id"),
        "an x-request-id arrives, but never the client's value"
    );
    // The one credential on the wire: the CONFIG bearer, nothing else.
    assert_eq!(
        wire.get("authorization").map(String::as_str),
        Some("Bearer config-bearer-secret"),
        "the only credential upstream is the per-upstream config bearer"
    );
    // Everything else is stripped — credentials, spoofing vectors,
    // hop-by-hop, the x-honkbox-* namespace.
    for forbidden in [
        "accept-encoding",
        "cookie",
        "x-api-key",
        "x-forwarded-for",
        "x-honkbox-internal",
        "proxy-authorization",
        "te",
    ] {
        assert!(
            !wire.contains_key(forbidden),
            "header {forbidden} must never reach the upstream (saw {:?})",
            wire.get(forbidden)
        );
    }
}

/// RESPONSE headers: only the allowlist is relayed; the upstream
/// `x-request-id` echo is rejected, `retry-after` is mapped (not relayed
/// raw), `set-cookie` never leaves the boundary.
#[tokio::test]
async fn response_headers_allowlisted_and_retry_after_mapped() {
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (_head, _body) = read_request(&mut stream).await;
        let response = concat!(
            "HTTP/1.1 429 Too Many Requests\r\n",
            "content-type: application/json\r\n",
            "x-request-id: upstream-echo-id\r\n",
            "retry-after: 7\r\n",
            "set-cookie: session=pwn\r\n",
            "connection: keep-alive\r\n\r\n",
            "{\"error\":{}}"
        );
        stream.write_all(response.as_bytes()).await.ok();
        stream.flush().await.ok();
    })
    .await;
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let response = proxy
        .forward(
            upstream,
            Method::POST,
            "/mesh/echo",
            &HeaderMap::new(),
            chat_body(),
        )
        .await
        .expect("forward succeeds");
    assert_eq!(response.status, 429);
    assert_eq!(response.headers.len(), 1, "only content-type is relayed");
    assert_eq!(
        response.headers.get("content-type").unwrap(),
        "application/json"
    );
    assert!(
        response.headers.get("x-request-id").is_none(),
        "upstream x-request-id echoes are stripped, never relayed"
    );
    assert!(
        response.headers.get("set-cookie").is_none(),
        "response credentials never leave the boundary"
    );
    assert_eq!(
        response.retry_after,
        Some(Duration::from_secs(7)),
        "retry-after is mapped into honkbox's envelope vocabulary"
    );
}

/// Route-shaped relay (what the reroute ticket mounts): an axum handler
/// that forwards through `MeshProxy` and returns the streamed body.
async fn relay(State(proxy): State<Arc<MeshProxy>>, request: Request) -> Response {
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let headers = request.headers().clone();
    let body = axum::body::to_bytes(request.into_body(), 1024 * 1024)
        .await
        .expect("request body");
    let forwarded = proxy
        .forward(upstream, Method::POST, "/mesh/echo", &headers, body)
        .await
        .expect("forward succeeds");
    let mut builder = axum::http::Response::builder().status(forwarded.status);
    for (name, value) in &forwarded.headers {
        builder = builder.header(name, value);
    }
    builder.body(forwarded.body).expect("static relay response")
}

/// SSE pass-through: a frame arrives at the caller WHILE the upstream is
/// still holding the stream open (before the gated second frame is even
/// written) — the same unbuffered proof style as the compression suite. If
/// the relay buffered the whole body, the gated second frame would never be
/// sent and this test would time out.
#[tokio::test]
async fn sse_frames_arrive_incrementally_through_relay() {
    let (frame1_seen_tx, frame1_seen_rx) = oneshot::channel::<()>();
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (_head, _body) = read_request(&mut stream).await;
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
            )
            .await
            .ok();
        stream
            .write_all(b"data: {\"i\":1}\n\ndata: {\"i\":2}\n\n")
            .await
            .ok();
        stream.flush().await.ok();
        // Frame 1 must be CONSUMED by the caller before frame 2 is written.
        if tokio::time::timeout(Duration::from_secs(10), frame1_seen_rx)
            .await
            .is_ok()
        {
            stream.write_all(b"data: {\"i\":3}\n\n").await.ok();
            stream.flush().await.ok();
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    })
    .await;
    let proxy = Arc::new(proxy_at(addr, 8 * 1024 * 1024));
    let app = axum::Router::new()
        .route("/_mesh_relay", post(relay))
        .with_state(proxy);

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/_mesh_relay")
                .header("accept", "text/event-stream")
                .body(Body::from(chat_body()))
                .expect("request"),
        )
        .await
        .expect("relay responds");
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream",
        "SSE content type passes through the relay"
    );

    let mut stream = response.into_body().into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("frame 1 arrives while the upstream is still open — NOT buffered")
        .expect("frame 1 item")
        .expect("frame 1 ok");
    assert!(
        String::from_utf8_lossy(&first).contains("{\"i\":2}"),
        "frame 1 chunk arrives incrementally (got {first:?})"
    );
    // Frame 1 is consumed; the upstream is now free to send frame 2.
    frame1_seen_tx
        .send(())
        .expect("upstream still holds the gate");
    let second = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("frame 2 arrives")
        .expect("frame 2 item")
        .expect("frame 2 ok");
    assert!(
        String::from_utf8_lossy(&second).contains("{\"i\":3}"),
        "frame 2 chunk arrives after frame 1 was consumed (got {second:?})"
    );
}

/// Cancel propagation: the caller dropping the relayed body aborts the
/// upstream connection — the mock observes EOF/reset on its read end within
/// a bounded window; no orphan stream remains.
#[tokio::test]
async fn dropping_relayed_body_aborts_upstream() {
    let (upstream_closed_tx, upstream_closed_rx) = oneshot::channel::<&'static str>();
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (_head, _body) = read_request(&mut stream).await;
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\ndata: {\"i\":1}\n\n",
            )
            .await
            .ok();
        stream.flush().await.ok();
        // Hold the stream open; any read failure/EOF now means the caller
        // dropped the body and honkbox aborted the connection.
        let mut sink = [0_u8; 64];
        let outcome = loop {
            match stream.read(&mut sink).await {
                Ok(0) => break "eof",
                Ok(_) => {} // ignore any late client bytes
                Err(_) => break "reset",
            }
        };
        upstream_closed_tx.send(outcome).ok();
    })
    .await;
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let response = proxy
        .forward(
            upstream,
            Method::POST,
            "/mesh/echo",
            &HeaderMap::new(),
            chat_body(),
        )
        .await
        .expect("forward succeeds");
    let mut stream = response.body.into_data_stream();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("frame 1 arrives")
        .expect("frame 1 item")
        .expect("frame 1 ok");
    assert!(String::from_utf8_lossy(&first).contains("{\"i\":1}"));
    // The honkbox side cancels: drop the relayed body (what a cancelled
    // axum job / client disconnect does to the stream).
    drop(stream);

    let outcome = tokio::time::timeout(Duration::from_secs(5), upstream_closed_rx)
        .await
        .expect("upstream observes the abort in bounded time — no orphan stream")
        .expect("mock still holds the channel");
    assert!(
        matches!(outcome, "eof" | "reset"),
        "upstream connection was aborted, got {outcome:?}"
    );
}

/// Stream safety: the response byte cap trips MID-STREAM with the named
/// `upstream_stream_limit` error — rejected, never silently truncated.
#[tokio::test]
async fn size_cap_rejects_mid_stream() {
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (_head, _body) = read_request(&mut stream).await;
        let payload = "x".repeat(256);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\n\r\n{payload}",
            payload.len()
        );
        stream.write_all(response.as_bytes()).await.ok();
        stream.flush().await.ok();
        // Hold the connection open; the cap must fire without waiting for
        // the upstream to finish.
        tokio::time::sleep(Duration::from_secs(5)).await;
    })
    .await;
    // 8 bytes of cap against a 256-byte body.
    let proxy = proxy_at(addr, 8);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let response = proxy
        .forward(
            upstream,
            Method::POST,
            "/mesh/echo",
            &HeaderMap::new(),
            chat_body(),
        )
        .await
        .expect("forward itself succeeds; the cap is a STREAM error");
    let mut stream = response.body.into_data_stream();
    // Some prefix may arrive before the cap error (chunking is hyper's).
    let mut saw_error = false;
    for _ in 0..16 {
        match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
            Ok(None) | Err(_) => break,
            Ok(Some(Err(err))) => {
                // The axum error renders the named
                // `MeshError::StreamLimitExceeded` text.
                saw_error = err.to_string().contains("byte cap");
                assert!(saw_error, "cap error names the limit, got: {err}");
                break;
            }
            Ok(Some(Ok(_))) => {}
        }
    }
    assert!(
        saw_error,
        "the mid-stream cap must surface as a named error"
    );
}

/// No lease for a remote-only turn (TASK-proxy-stream-safety): a pure proxy
/// pass-through never touches the GPU residency ledger — the running gauge
/// and admission counter are unchanged across a complete proxied turn.
#[tokio::test]
async fn remote_turn_holds_no_residency_lease() {
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (_head, _body) = read_request(&mut stream).await;
        let body = br#"{"choices":[{"message":{"role":"assistant"}}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        stream.write_all(response.as_bytes()).await.ok();
        stream.flush().await.ok();
    })
    .await;
    let ledger = honkbox_residency::global();
    let before = ledger.stats(honkbox_residency::ResourceKind::Llm);
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let response = proxy
        .forward(
            upstream,
            Method::POST,
            "/mesh/echo",
            &HeaderMap::new(),
            chat_body(),
        )
        .await
        .expect("forward succeeds");
    let text = body_text(response.body).await;
    assert!(
        text.contains("assistant"),
        "a complete remote turn finished"
    );
    let after = ledger.stats(honkbox_residency::ResourceKind::Llm);
    assert_eq!(
        after.running, before.running,
        "no residency lease is held during (or after) a proxied turn"
    );
    assert_eq!(
        after.admissions, before.admissions,
        "the proxy path never acquires the GPU slot"
    );
}

/// A disabled mesh config builds an empty proxy: `upstream()` resolves
/// nothing (the reroute layer surfaces the named disabled case — no silent
/// fallback). Also pins `MeshError`'s named codes.
#[tokio::test]
async fn disabled_mesh_resolves_no_upstream_and_errors_have_codes() {
    let config = MeshConfig::default();
    assert!(!config.enabled);
    let proxy = MeshProxy::new(&config).expect("empty proxy builds");
    assert!(proxy.upstream("peer").is_none());

    let transport = MeshError::UpstreamUnreachable {
        name: "peer".to_owned(),
        message: "connect refused".to_owned(),
    };
    assert_eq!(transport.code(), "upstream_unavailable");
    assert_eq!(
        config::mesh::MeshError::status(),
        axum::http::StatusCode::BAD_GATEWAY
    );
    let cap = MeshError::StreamLimitExceeded(64);
    assert_eq!(cap.code(), "upstream_stream_limit");
    assert_eq!(MeshError::status(), axum::http::StatusCode::BAD_GATEWAY);
    assert_eq!(
        transport.to_string(),
        "mesh upstream \"peer\" unreachable: connect refused"
    );
}

/// The mocked upstream sees a fresh, unpredictable `x-request-id` on every
/// hop — never a client value, never reused.
#[tokio::test]
async fn request_ids_are_regenerated_per_hop() {
    let ids = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ids_for_handler = Arc::clone(&ids);
    let addr = spawn_upstream(move |listener: TcpListener, first: TcpStream| {
        let ids = ids_for_handler;
        async move {
            // `Connection: close` per request (the stream-safety rule), so
            // each hop arrives on its OWN connection: handle the first, then
            // accept one more.
            let mut conn = first;
            for _ in 0..2 {
                let (head, _body) = read_request(&mut conn).await;
                let headers = parse_head(&head);
                ids.lock()
                    .expect("ids lock")
                    .push(headers.get("x-request-id").cloned());
                let response = "HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}";
                conn.write_all(response.as_bytes()).await.ok();
                conn.flush().await.ok();
                // `Connection: close` semantics: the next hop arrives on a
                // fresh connection.
                drop(conn);
                let Ok((next, _)) = listener.accept().await else {
                    return;
                };
                conn = next;
            }
        }
    })
    .await;
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    for _ in 0..2 {
        proxy
            .forward(
                upstream,
                Method::POST,
                "/mesh/echo",
                &HeaderMap::new(),
                chat_body(),
            )
            .await
            .expect("forward succeeds");
    }
    let ids = ids.lock().expect("ids lock").clone();
    assert_eq!(ids.len(), 2, "both hops observed an id");
    let first = ids[0].as_deref().expect("id 1 present");
    let second = ids[1].as_deref().expect("id 2 present");
    assert_ne!(first, second, "ids are regenerated per hop");
    assert_eq!(first.len(), 26, "ULID canonical width");
    assert!(
        first.chars().all(|c| c.is_ascii_alphanumeric()),
        "header-safe id"
    );
}

/// PATH-ONLY TARGETS: a plain absolute-path reference joins onto the
/// configured origin and the upstream sees that exact path.
#[tokio::test]
async fn path_only_target_stays_on_configured_origin() {
    let addr = spawn_upstream(|_listener, mut stream| async move {
        let (head, _body) = read_request(&mut stream).await;
        let request_line = String::from_utf8_lossy(&head)
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            request_line.len(),
            request_line
        );
        stream.write_all(response.as_bytes()).await.ok();
        stream.flush().await.ok();
    })
    .await;
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");
    let response = proxy
        .forward(
            upstream,
            Method::POST,
            "/mesh/echo?x=1",
            &HeaderMap::new(),
            chat_body(),
        )
        .await
        .expect("path-only target forwards");
    assert_eq!(response.status, 200);
    let line = body_text(response.body).await;
    assert!(
        line.starts_with("POST /mesh/echo?x=1 HTTP/1.1"),
        "path-only target must join the configured origin, got {line:?}"
    );
}

/// ORIGIN ESCAPE: absolute URLs, network-path (`//host`) references, and
/// backslash/newline tricks must be rejected BEFORE any request is sent —
/// this is where the config bearer is attached, so it can never reach a host
/// other than the one configured.
#[tokio::test]
async fn hostile_targets_are_rejected_before_any_request() {
    let saw_request = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&saw_request);
    let addr = spawn_upstream(move |_listener, mut stream| async move {
        flag.store(true, Ordering::SeqCst);
        let _ = read_request(&mut stream).await;
    })
    .await;
    let proxy = proxy_at(addr, 8 * 1024 * 1024);
    let upstream = proxy.upstream("peer").expect("peer resolves");

    for target in [
        "http://evil.example/mesh/echo", // absolute URL
        "https://evil.example/x",        // absolute URL, other scheme
        "//evil.example/x",              // network-path reference
        "///evil.example/x",             // network-path reference
        "\\evil.example/x",              // backslash authority trick
        "/\\evil.example/x",             // backslash after the leading slash
        "/\n\\evil.example/x",           // newline stripped, then `\` becomes `/`
        "http:/\\evil.example/x",        // scheme-ish form with no leading slash
    ] {
        let Err(err) = proxy
            .forward(
                upstream,
                Method::POST,
                target,
                &HeaderMap::new(),
                chat_body(),
            )
            .await
        else {
            panic!("hostile target {target:?} must be rejected")
        };
        assert!(
            matches!(err, MeshError::UpstreamUnreachable { .. }),
            "target {target:?} must fail as UpstreamUnreachable, got {err:?}"
        );
    }
    assert!(
        !saw_request.load(Ordering::SeqCst),
        "no hostile target may reach ANY upstream: the config bearer must never leave the configured origin"
    );
}
