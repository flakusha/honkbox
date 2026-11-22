//! `llama-server` attach probe (TASK-llama-server-attach).
//!
//! Discovers a user-installed `llama-server` on `PATH` (or at the path named by
//! `HONKBOX_LLAMA_SERVER`) and, when an endpoint is configured, proves an
//! `OpenAI`-compatible server is actually answering there. Nothing here starts,
//! stops, or restarts a process: the operator owns the app's lifecycle in this
//! slice. A passing version probe alone claims no serving capability.
//!
//! The endpoint probe uses the registry's shared async client, so it can run
//! alongside the other startup probes without blocking a runtime worker.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{Capability, Status, ToolId, Unavailable, probe_version};

/// Environment variable naming the operator's binary (an absolute path or a
/// name resolved on `PATH`).
pub const LLAMA_SERVER_ENV: &str = "HONKBOX_LLAMA_SERVER";

/// Binary used when [`LLAMA_SERVER_ENV`] is unset.
pub const DEFAULT_LLAMA_SERVER: &str = "llama-server";

/// The configured binary. Never a value from an HTTP request.
#[must_use]
pub fn binary() -> PathBuf {
    std::env::var_os(LLAMA_SERVER_ENV)
        .map_or_else(|| PathBuf::from(DEFAULT_LLAMA_SERVER), PathBuf::from)
}

/// Trim an operator-supplied base URL so `format!("{base}/health")` is stable.
fn base_url(url: &str) -> &str {
    url.trim().trim_end_matches('/')
}

/// Prove the executable can answer a version request. No server capability is
/// claimed until [`probe_endpoint`] proves an endpoint is answering too.
pub(crate) fn probe(
    binary: &Path,
    timeout: Duration,
) -> Result<(String, Vec<Capability>), Unavailable> {
    let version = probe_version(binary, timeout)?;
    Ok((version, Vec::new()))
}

/// Prove an attached server is ready (`/health` 200) and exposes the
/// `OpenAI`-compatible model list (`/v1/models` with a `data` array).
pub(crate) async fn probe_endpoint(
    client: &reqwest::Client,
    url: &str,
) -> Result<Vec<Capability>, Unavailable> {
    let base = base_url(url);

    let health = format!("{base}/health");
    let response = client
        .get(&health)
        .send()
        .await
        .map_err(|err| Unavailable::ProbeFailed(format!("{health}: {err}")))?;
    if !response.status().is_success() {
        return Err(Unavailable::ProbeFailed(format!(
            "{health}: HTTP {}",
            response.status()
        )));
    }

    let models = format!("{base}/v1/models");
    let response = client
        .get(&models)
        .send()
        .await
        .map_err(|err| Unavailable::ProbeFailed(format!("{models}: {err}")))?;
    if !response.status().is_success() {
        return Err(Unavailable::ProbeFailed(format!(
            "{models}: HTTP {}",
            response.status()
        )));
    }
    let body = response
        .json::<serde_json::Value>()
        .await
        .map_err(|err| Unavailable::ProbeFailed(format!("{models}: invalid JSON body ({err})")))?;
    if !body.get("data").is_some_and(serde_json::Value::is_array) {
        return Err(Unavailable::ProbeFailed(format!(
            "{models}: no models array in the payload"
        )));
    }

    Ok(vec![Capability::OpenAiCompat])
}

/// Combine the binary and endpoint probes into one cacheable status.
pub(crate) fn status(
    version: Result<(String, Vec<Capability>), Unavailable>,
    endpoint: Option<Result<Vec<Capability>, Unavailable>>,
) -> Status {
    let (version, mut capabilities) = match version {
        Ok(ok) => ok,
        Err(unavailable) => {
            return Status {
                id: ToolId::LlamaServer,
                version: None,
                capabilities: Vec::new(),
                devices: Vec::new(),
                unavailable: Some(unavailable),
            };
        }
    };
    match endpoint {
        Some(Ok(mut proved)) => capabilities.append(&mut proved),
        Some(Err(unavailable)) => {
            return Status {
                id: ToolId::LlamaServer,
                version: None,
                capabilities: Vec::new(),
                devices: Vec::new(),
                unavailable: Some(unavailable),
            };
        }
        None => {}
    }
    Status {
        id: ToolId::LlamaServer,
        version: Some(version),
        capabilities,
        devices: Vec::new(),
        unavailable: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{ErrorKind, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::Instant;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "honkbox-llama-server-{}-{}-{tag}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[cfg(unix)]
    fn fake_binary(tag: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;

        let staged = scratch(tag).join("llama-server.staged");
        let path = staged.with_file_name("llama-server");
        std::fs::write(&staged, script).expect("fake binary");
        let mut perms = std::fs::metadata(&staged).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&staged, perms).expect("chmod");
        std::fs::rename(staged, &path).expect("rename");
        path
    }

    #[cfg(unix)]
    #[test]
    fn a_versioned_binary_claims_no_endpoint_capability() {
        let bin = fake_binary("versioned", "#!/bin/sh\necho 'llama-server 1.2.3'\n");
        let staged = probe(&bin, Duration::from_secs(2)).expect("probe");
        assert_eq!(staged.0, "1.2.3");
        assert_eq!(staged.1, Vec::new());
        let status = status(Ok(staged), None);
        assert!(status.is_available(), "{status:?}");
        assert_eq!(status.capabilities, Vec::new());
    }

    #[test]
    fn a_missing_binary_is_a_named_not_installed_status() {
        let bin = scratch("missing").join("no-such-llama-server");
        let status = status(probe(&bin, Duration::from_secs(2)), None);
        assert_eq!(status.id, ToolId::LlamaServer);
        assert_eq!(status.unavailable, Some(Unavailable::NotInstalled));
        assert!(status.version.is_none());
    }

    #[test]
    fn an_unreachable_endpoint_fails_the_status_even_with_a_version() {
        let status = status(
            Ok(("1.2.3".to_owned(), Vec::new())),
            Some(Err(Unavailable::ProbeFailed("endpoint down".to_owned()))),
        );
        assert!(!status.is_available());
        assert_eq!(status.version, None);
        assert_eq!(status.capabilities, Vec::new());
    }

    #[test]
    fn base_urls_are_normalized() {
        assert_eq!(
            base_url("  http://127.0.0.1:9931/  "),
            "http://127.0.0.1:9931"
        );
    }

    /// Two fixed responses, one per probe request, on an ephemeral loopback
    /// port. No framework, no fixture server shared with other tests.
    fn fake_endpoint(
        health_status: u16,
        health: &str,
        models: &str,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let health = health.to_owned();
        let models = models.to_owned();
        let handle = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_millis(500);
            for _ in 0..2 {
                // A negative probe may stop after the first request; the
                // helper must never hang waiting for an accept that won't come.
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(accepted) => break accepted,
                        Err(err) if err.kind() == ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                return;
                            }
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => return,
                    }
                };
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request);
                let line = String::from_utf8_lossy(&request);
                let path = line.split_whitespace().nth(1).unwrap_or("");
                let models_request = path.starts_with("/v1/models");
                let payload = if models_request { &models } else { &health };
                let status = if models_request {
                    "200 OK".to_owned()
                } else {
                    format!("{health_status} NOT READY")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (url, handle)
    }

    #[tokio::test]
    async fn a_ready_endpoint_proves_the_openai_compatible_capability() {
        let (url, server) = fake_endpoint(
            200,
            r#"{"status":"ok"}"#,
            r#"{"object":"list","data":[{"id":"local"}]}"#,
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("client");
        let capabilities = probe_endpoint(&client, &url).await.expect("endpoint");
        assert!(capabilities.contains(&Capability::OpenAiCompat));
        server.join().expect("fake server");
    }

    #[tokio::test]
    async fn a_loading_endpoint_claims_no_capability() {
        let (url, server) = fake_endpoint(
            503,
            r#"{"error":{"message":"Loading model"}}"#,
            r#"{"object":"list","data":[]}"#,
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("client");
        let err = probe_endpoint(&client, &url)
            .await
            .expect_err("503 is not ready");
        assert!(matches!(err, Unavailable::ProbeFailed(_)), "{err:?}");
        server.join().expect("fake server");
    }
}
