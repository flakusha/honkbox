//! Salvo spike: `GET /api/version` over a real, per-test port-0 listener.

// No lib target on this binary crate, so include the spike module directly.
#[path = "../src/salvo_version.rs"]
mod salvo_version;

#[path = "support/listener.rs"]
mod listener;

use listener::TestServer;
use serde_json::Value;

#[tokio::test]
async fn version_route_over_real_listener() {
    let server = TestServer::start(salvo_version::router()).await;
    let response = server.get("/api/version").await;
    assert_eq!(response.status, reqwest::StatusCode::OK);
    assert!(response.headers.contains_key("link"));
    let body: Value = response.json().unwrap();
    assert_eq!(body["name"], "honkbox");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    server.shutdown().await;
}
