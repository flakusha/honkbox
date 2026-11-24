//! Salvo spike: `GET /api/version` over a real, per-test port-0 listener.

// No lib target on this binary crate, so include the spike module directly.
#[path = "../src/salvo_version.rs"]
mod salvo_version;

use salvo::Server;
use salvo::conn::Listener as _;
use salvo::conn::tcp::TcpListener;
use salvo::server::ServerHandle;
use serde_json::Value;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::task::JoinHandle;

/// One OS-assigned port per test; graceful stop with a bounded wait, forceful abort on `Drop`.
struct TestServer {
    addr: SocketAddr,
    handle: ServerHandle,
    task: JoinHandle<std::io::Result<()>>,
}

impl TestServer {
    async fn start() -> Self {
        let acceptor = TcpListener::new("127.0.0.1:0").bind().await;
        let addr = acceptor.local_addr().unwrap();
        let server = Server::new(acceptor);
        let handle = server.handle();
        let task = tokio::spawn(server.try_serve(salvo_version::router()));
        Self { addr, handle, task }
    }

    async fn shutdown(mut self) {
        self.handle.stop_graceful(None);
        let _ = tokio::time::timeout(Duration::from_secs(2), &mut self.task).await;
        self.task.abort();
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.stop_forceful();
        self.task.abort();
    }
}

#[tokio::test]
async fn version_route_over_real_listener() {
    let server = TestServer::start().await;
    let response = reqwest::get(format!("http://{}/api/version", server.addr))
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert!(response.headers().contains_key("link"));
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["name"], "honkbox");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    server.shutdown().await;
}
