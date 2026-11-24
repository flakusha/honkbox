//! Self-contained integration-test helper for salvo route tests.
//!
//! Resource contract: each `TestServer::start` binds one OS-assigned port
//! (`127.0.0.1:0`), owns one `ServerHandle` and one spawned serve task, and
//! shuts them down deterministically via `shutdown` (graceful stop, bounded
//! 2 s join wait, then abort). `Drop` is a fallback only: forceful stop plus
//! task abort. No globals or shared ports; instances are independent and safe
//! under the default parallel test runner.

// Included via #[path] by multiple test binaries; each uses a different subset.
#![allow(dead_code)]

use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use salvo::Server;
use salvo::conn::Listener as _;
use salvo::conn::tcp::TcpListener;
use salvo::server::ServerHandle;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Owned response bytes, kept raw (no decompression) and copied out of the
/// `reqwest` body without adding a direct `bytes` dependency.
pub type Bytes = Vec<u8>;

/// One OS-assigned port per instance; graceful stop with a bounded wait,
/// forceful abort on `Drop`.
pub struct TestServer {
    pub addr: SocketAddr,
    handle: ServerHandle,
    task: JoinHandle<std::io::Result<()>>,
}

impl TestServer {
    pub async fn start(router: salvo::Router) -> Self {
        let acceptor = TcpListener::new("127.0.0.1:0").bind().await;
        let addr = acceptor.local_addr().unwrap();
        let server = Server::new(acceptor);
        let handle = server.handle();
        let task = tokio::spawn(server.try_serve(router));
        Self { addr, handle, task }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }

    /// Auto-decompression disabled so compression assertions see wire bytes.
    pub fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .build()
            .unwrap()
    }

    pub async fn shutdown(mut self) {
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

/// Status, headers and raw body of a completed request.
pub struct ResponseEnvelope {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl ResponseEnvelope {
    pub fn json<T>(&self) -> serde_json::Result<T>
    where
        T: for<'de> salvo::serde::Deserialize<'de>,
    {
        serde_json::from_slice(&self.body)
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

impl TestServer {
    async fn envelope(&self, response: reqwest::Response) -> ResponseEnvelope {
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.bytes().await.unwrap().to_vec();
        ResponseEnvelope {
            status,
            headers,
            body,
        }
    }

    pub async fn get(&self, path: &str) -> ResponseEnvelope {
        let response = Self::client().get(self.url(path)).send().await.unwrap();
        self.envelope(response).await
    }

    pub async fn post_json(&self, path: &str, body: &serde_json::Value) -> ResponseEnvelope {
        let response = Self::client()
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .unwrap();
        self.envelope(response).await
    }

    pub async fn request(&self, req: reqwest::RequestBuilder) -> ResponseEnvelope {
        let response = req.send().await.unwrap();
        self.envelope(response).await
    }
}
