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

/// Largest `/v1/models` body the probe will buffer before parsing JSON.
const MAX_MODELS_BYTES: usize = 1024 * 1024;

/// Validate an operator-supplied base URL and return a concatenation-safe
/// base: credential-free loopback HTTP, optional path prefix, no query or
/// fragment — the same rules the attached client enforces.
fn validated_base(url: &str) -> Result<String, Unavailable> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|err| Unavailable::ProbeFailed(format!("invalid llama-server URL: {err}")))?;
    let loopback = matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
    if parsed.scheme() != "http"
        || !loopback
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(Unavailable::ProbeFailed(
            "llama-server URL must be credential-free loopback HTTP without query or fragment"
                .to_owned(),
        ));
    }
    Ok(parsed.to_string().trim_end_matches('/').to_owned())
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
    let base = validated_base(url)?;

    let health = format!("{base}/health");
    let response = client
        .get(&health)
        .send()
        .await
        .map_err(|err| Unavailable::ProbeFailed(format!("{health}: {err}")))?;
    if response.status().as_u16() != 200 {
        return Err(Unavailable::ProbeFailed(format!(
            "{health}: HTTP {}",
            response.status()
        )));
    }

    let models = format!("{base}/v1/models");
    let mut response = client
        .get(&models)
        .send()
        .await
        .map_err(|err| Unavailable::ProbeFailed(format!("{models}: {err}")))?;
    if response.status().as_u16() != 200 {
        return Err(Unavailable::ProbeFailed(format!(
            "{models}: HTTP {}",
            response.status()
        )));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| Unavailable::ProbeFailed(format!("{models}: {err}")))?
    {
        if body.len().saturating_add(chunk.len()) > MAX_MODELS_BYTES {
            return Err(Unavailable::ProbeFailed(format!(
                "{models}: response body exceeds {MAX_MODELS_BYTES} bytes"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    let body: serde_json::Value = serde_json::from_slice(&body)
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
    fn validated_base_accepts_loopback_urls_with_a_path_prefix() {
        assert_eq!(
            validated_base("http://127.0.0.1:9931/api/").expect("loopback"),
            "http://127.0.0.1:9931/api"
        );
        assert_eq!(
            validated_base("http://localhost:9931").expect("localhost"),
            "http://localhost:9931"
        );
        assert_eq!(
            validated_base("http://[::1]:9931/api").expect("ipv6"),
            "http://[::1]:9931/api"
        );
    }

    #[test]
    fn validated_base_rejects_unattachable_urls() {
        for url in [
            "http://192.168.1.5:8080",
            "http://user:pass@127.0.0.1:8080",
            "https://127.0.0.1:8080",
            "http://127.0.0.1:8080/?x=1",
            "http://127.0.0.1:8080/#frag",
        ] {
            assert!(validated_base(url).is_err(), "{url}");
        }
    }

    /// Two fixed responses, one per probe request, on an ephemeral loopback
    /// port. No framework, no fixture server shared with other tests.
    fn fake_endpoint(
        health_status: u16,
        health: &str,
        models: &str,
        location: Option<&str>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let health = health.to_owned();
        let models = models.to_owned();
        let location = location.map_or_else(String::new, |url| format!("Location: {url}\r\n"));
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
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{location}Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
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
            None,
        );
        let client = crate::probe_client(Duration::from_secs(2));
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
            None,
        );
        let client = crate::probe_client(Duration::from_secs(2));
        let err = probe_endpoint(&client, &url)
            .await
            .expect_err("503 is not ready");
        assert!(matches!(err, Unavailable::ProbeFailed(_)), "{err:?}");
        server.join().expect("fake server");
    }

    #[tokio::test]
    async fn only_http_200_health_counts_as_ready() {
        for status in [201, 204] {
            let (url, server) = fake_endpoint(
                status,
                r#"{"status":"ok"}"#,
                r#"{"object":"list","data":[]}"#,
                None,
            );
            let client = crate::probe_client(Duration::from_secs(2));
            let err = probe_endpoint(&client, &url)
                .await
                .expect_err("only 200 is ready");
            assert!(err.to_string().contains(&format!("HTTP {status}")), "{err}");
            server.join().expect("fake server");
        }
    }

    #[tokio::test]
    async fn a_local_redirect_cannot_fake_readiness() {
        let models = r#"{"object":"list","data":[{"id":"local"}]}"#;
        let (target, target_server) = fake_endpoint(200, r#"{"status":"ok"}"#, models, None);
        let location = format!("{target}/health");

        // Control: a redirect-following client would be fooled by this server.
        let (source, source_server) = fake_endpoint(302, "{}", models, Some(&location));
        let following = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("client");
        probe_endpoint(&following, &source)
            .await
            .expect("control client follows the local redirect");
        source_server.join().expect("source server");

        let (source, source_server) = fake_endpoint(302, "{}", models, Some(&location));
        let err = probe_endpoint(&crate::probe_client(Duration::from_secs(2)), &source)
            .await
            .expect_err("no-follow client rejects the redirect");
        assert!(err.to_string().contains("HTTP 302"), "{err}");
        source_server.join().expect("source server");
        target_server.join().expect("target server");
    }

    #[tokio::test]
    async fn an_oversized_models_body_is_rejected_before_parsing() {
        let huge = "x".repeat(MAX_MODELS_BYTES + 1);
        let (url, server) = fake_endpoint(200, r#"{"status":"ok"}"#, &huge, None);
        let client = crate::probe_client(Duration::from_secs(5));
        let err = probe_endpoint(&client, &url)
            .await
            .expect_err("oversized body");
        assert!(err.to_string().contains("exceeds"), "{err}");
        server.join().expect("fake server");
    }
}
