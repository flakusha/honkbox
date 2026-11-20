//! Managed downloads (NEW-4 phase A): supervisor state machine over a local
//! HTTP fixture (std `TcpListener`; no external network in tests), resume
//! 206/200 branches, checksum enforcement, cancel, single-flight join,
//! `already_present`, the 2-slot cap, and the `/v1/downloads` route flow.

// No lib target on this binary crate, so include the modules directly.
#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "../src/artifacts.rs"]
mod artifacts;
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/container.rs"]
mod container;
#[path = "../src/downloads.rs"]
mod downloads;
#[allow(dead_code)]
#[path = "../src/formats.rs"]
mod formats;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/jobs_registry.rs"]
mod jobs_registry;
#[path = "../src/metrics.rs"]
mod metrics;
#[allow(dead_code)]
#[path = "../src/ora.rs"]
mod ora;
#[allow(dead_code)]
#[path = "../src/raster.rs"]
mod raster;
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;
// Included only so `routes.rs`'s `crate::swap` paths resolve in this test
// binary; the swap module itself is not exercised here.
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::StatusCode;
use honkbox_api_types::{DownloadKind, DownloadRequest, DownloadStatus};
use routes::{AppConfig, AuthConfig};
use sha2::{Digest, Sha256};
use tower::ServiceExt as _;

// ---------------------------------------------------------------- fixture

/// One parsed request (lowercased-name headers; the fixture handler only
/// inspects headers).
struct MockRequest {
    /// Header `(name, value)` pairs, names lowercased.
    headers: Vec<(String, String)>,
}

impl MockRequest {
    /// First header value matching `name` (names stored lowercased).
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Canned response for one request; the body is written in `chunk_bytes`
/// pieces with `gap_ms` between pieces (a drip keeps a download in-flight
/// long enough to observe queued/running/cancelled states).
struct MockResponse {
    /// HTTP status line code.
    status: u16,
    /// Extra headers (e.g. `ETag`, `Accept-Ranges`, `Content-Range`).
    headers: Vec<(String, String)>,
    /// Body bytes (Content-Length declared up front, close after).
    body: Vec<u8>,
    /// Piece size for the drip writer.
    chunk_bytes: usize,
    /// Sleep between pieces.
    gap_ms: u64,
}

impl MockResponse {
    /// Whole body in one write.
    fn once(status: u16, headers: Vec<(String, String)>, body: Vec<u8>) -> Self {
        let chunk_bytes = body.len().max(1);
        Self {
            status,
            headers,
            body,
            chunk_bytes,
            gap_ms: 0,
        }
    }

    /// Body dripped in `chunk_bytes` pieces, `gap_ms` apart.
    fn drip(
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
        chunk_bytes: usize,
        gap_ms: u64,
    ) -> Self {
        Self {
            status,
            headers,
            body,
            chunk_bytes,
            gap_ms,
        }
    }
}

/// Local HTTP server answering every connection from `handler`.
struct MockServer {
    /// Base URL (`http://127.0.0.1:PORT`).
    base: String,
}

impl MockServer {
    /// Bind an ephemeral port and spawn per-connection threads.
    fn new(handler: impl Fn(&MockRequest) -> MockResponse + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let base = format!("http://{}", listener.local_addr().expect("mock addr"));
        let handler: Arc<dyn Fn(&MockRequest) -> MockResponse + Send + Sync> = Arc::new(handler);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let handler = Arc::clone(&handler);
                std::thread::spawn(move || serve_connection(stream, &handler));
            }
        });
        Self { base }
    }

    /// `base/path` convenience for source URLs.
    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base, path.trim_start_matches('/'))
    }
}

/// Read one request, answer, close (`Connection: close` keeps the fixture
/// single-request-per-connection; reqwest reconnects transparently).
fn serve_connection(
    mut stream: TcpStream,
    handler: &Arc<dyn Fn(&MockRequest) -> MockResponse + Send + Sync>,
) {
    let Some(request) = read_request(&mut stream) else {
        return;
    };
    let response = handler(&request);
    let mut head = format!(
        "HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status,
        response.body.len()
    );
    for (name, value) in &response.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    for piece in response.body.chunks(response.chunk_bytes.max(1)) {
        if stream.write_all(piece).is_err() {
            return; // client gone (cancel path)
        }
        if response.gap_ms > 0 {
            std::thread::sleep(Duration::from_millis(response.gap_ms));
        }
    }
}

/// Accumulate bytes until the blank line; parse request line + headers
/// (fixture GETs/DELETEs carry no body). `None` on a closed/short read.
fn read_request(stream: &mut TcpStream) -> Option<MockRequest> {
    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match stream.read(&mut byte) {
            Ok(0) | Err(_) => return None,
            Ok(_) => {
                raw.push(byte[0]);
                if raw.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
        }
        if raw.len() > 64 * 1024 {
            return None;
        }
    }
    let mut lines = raw.split(|b| *b == b'\n');
    let request_line = String::from_utf8_lossy(lines.next()?);
    if request_line.split_whitespace().count() < 2 {
        return None;
    }
    let headers = lines
        .map(|line| String::from_utf8_lossy(line).trim().to_owned())
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        })
        .collect();
    Some(MockRequest { headers })
}

// ---------------------------------------------------------------- helpers

/// Unique per-test temp dir under the system temp dir.
fn temp_root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "honkbox-dl-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp root");
    dir
}

/// Supervisor rooted at `root/models`.
fn manager(root: &Path) -> Arc<downloads::DownloadManager> {
    Arc::new(
        downloads::DownloadManager::new_for_fixture(Some(root.join("models")))
            .expect("download manager"),
    )
}

/// sha256 of `data` as lowercase hex (mirrors the checksum format).
fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Seed a `.part` + `.part.meta.json` pair for a resume test.
fn seed_part(target: &Path, prefix: &[u8], url: &str, etag: Option<&str>) {
    use std::fmt::Write as _;
    std::fs::create_dir_all(target.parent().expect("parent")).expect("seed dir");
    std::fs::write(with_part(target), prefix).expect("seed .part");
    let mut meta = String::new();
    let _ = write!(meta, "{{\"url\":{url:?}");
    if let Some(etag) = etag {
        let _ = write!(meta, ",\"etag\":{etag:?}");
    }
    let _ = write!(meta, ",\"bytes_total\":{}", prefix.len() + 8);
    meta.push('}');
    std::fs::write(with_meta(target), meta).expect("seed meta");
}

/// `<target>.part` sibling path.
fn with_part(target: &Path) -> PathBuf {
    sibling(target, ".part")
}

/// `<target>.part.meta.json` sibling path.
fn with_meta(target: &Path) -> PathBuf {
    sibling(target, ".part.meta.json")
}

/// `<target><suffix>` sibling (mirrors the module's discipline).
fn sibling(target: &Path, suffix: &str) -> PathBuf {
    let mut os = target.as_os_str().to_os_string();
    os.push(suffix);
    PathBuf::from(os)
}

/// Poll until the download reaches a `want`ed status (20 ms steps).
async fn wait_state(
    manager: &downloads::DownloadManager,
    id: &str,
    want: impl Fn(&DownloadStatus) -> bool,
) -> honkbox_api_types::DownloadState {
    for _ in 0..500 {
        if let Some(state) = manager.state(id)
            && want(&state.state)
        {
            return state;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "download {id} never reached the expected state; last = {:?}",
        manager.state(id)
    );
}

/// Poll until `want` holds on the state snapshot (progress assertions on
/// non-terminal states).
async fn wait_progress(
    manager: &downloads::DownloadManager,
    id: &str,
    want: impl Fn(&honkbox_api_types::DownloadState) -> bool,
) -> honkbox_api_types::DownloadState {
    for _ in 0..500 {
        if let Some(state) = manager.state(id)
            && want(&state)
        {
            return state;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "download {id} never showed the expected progress; last = {:?}",
        manager.state(id)
    );
}

/// Direct-URL request for `url`, kind `llm`.
fn llm_request(url: String) -> DownloadRequest {
    DownloadRequest {
        repo: None,
        file: None,
        url: Some(url),
        kind: DownloadKind::Llm,
    }
}

/// Target path the supervisor derives for a kind + file name.
fn target_for(root: &Path, kind: DownloadKind, name: &str) -> PathBuf {
    let models = root.join("models");
    match kind {
        DownloadKind::Llm => models.join("llm").join(name),
        DownloadKind::Sd => models.join("sd").join("checkpoints").join(name),
        DownloadKind::Stt => models.join("stt").join(name),
        DownloadKind::Tts => models.join("tts").join(name),
    }
}

// ------------------------------------------------------------ unit tests

/// Cold download: accepted queued → done, bytes on disk, sidecars gone,
/// rescan signal fired, bytes metric counted.
#[tokio::test]
async fn cold_download_completes_and_fires_rescan() {
    let root = temp_root("cold");
    let body = b"cold-download-body".to_vec();
    let payload = body.clone();
    let server = MockServer::new(move |_| {
        MockResponse::once(
            200,
            vec![
                ("ETag".to_owned(), "\"etag-cold\"".to_owned()),
                ("Accept-Ranges".to_owned(), "bytes".to_owned()),
            ],
            payload.clone(),
        )
    });
    let mgr = manager(&root);
    let rescan = mgr.rescan_signal();
    let accepted = mgr
        .enqueue(llm_request(server.url("cold.gguf")))
        .await
        .expect("enqueue");
    assert_eq!(accepted.state, DownloadStatus::Queued);
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(state.bytes_done, body.len() as u64);
    assert_eq!(state.bytes_total, Some(body.len() as u64));
    assert!(state.resumable, "etag + accept-ranges ⇒ resumable");
    assert!(!state.already_present);
    let target = target_for(&root, DownloadKind::Llm, "cold.gguf");
    assert_eq!(std::fs::read(&target).expect("final file"), body);
    assert!(!with_part(&target).exists(), "part consumed by rename");
    assert!(!with_meta(&target).exists(), "meta removed on commit");
    // Rescan signal stored a permit for the (already fired) commit.
    tokio::time::timeout(Duration::from_secs(1), rescan.notified())
        .await
        .expect("rescan signal fired");
    let expo = downloads::exposition();
    assert!(expo.contains("honkbox_download_bytes_total{kind=\"llm\"}"));
    std::fs::remove_dir_all(&root).ok();
}

/// Resume: seeded `.part` + meta ⇒ ranged GET with `Range`/`If-Range`; the
/// 206 suffix appends and the file assembles whole.
#[tokio::test]
async fn resume_206_appends_prefix() {
    let root = temp_root("resume206");
    let prefix = b"PREFIX".to_vec();
    let suffix = b"SUFFIX12".to_vec();
    let total = prefix.len() + suffix.len();
    let payload = suffix.clone();
    let server = MockServer::new(move |req| {
        assert_eq!(req.header("range"), Some("bytes=6-"));
        assert_eq!(req.header("if-range"), Some("\"etag-206\""));
        MockResponse::once(
            206,
            vec![
                ("ETag".to_owned(), "\"etag-206\"".to_owned()),
                ("Accept-Ranges".to_owned(), "bytes".to_owned()),
                ("Content-Range".to_owned(), format!("bytes 6-13/{total}")),
            ],
            payload.clone(),
        )
    });
    let url = server.url("resume.gguf");
    let target = target_for(&root, DownloadKind::Llm, "resume.gguf");
    seed_part(&target, &prefix, &url, Some("\"etag-206\""));
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(state.bytes_done, total as u64);
    assert_eq!(state.bytes_total, Some(total as u64));
    let mut whole = prefix;
    whole.extend_from_slice(&suffix);
    assert_eq!(std::fs::read(&target).expect("final file"), whole);
    assert!(!with_meta(&target).exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Resume against a server that ignores the range (200): truncate +
/// restart from scratch — the final file is the fresh body only.
#[tokio::test]
async fn resume_200_restarts_from_scratch() {
    let root = temp_root("resume200");
    let body = b"fresh-full-body".to_vec();
    let payload = body.clone();
    let server = MockServer::new(move |req| {
        // The ranged request WAS sent; the server just ignores it.
        assert!(req.header("range").is_some());
        MockResponse::once(
            200,
            vec![
                ("ETag".to_owned(), "\"etag-new\"".to_owned()),
                ("Accept-Ranges".to_owned(), "bytes".to_owned()),
            ],
            payload.clone(),
        )
    });
    let url = server.url("restart.gguf");
    let target = target_for(&root, DownloadKind::Llm, "restart.gguf");
    seed_part(&target, b"stale-bytes", &url, Some("\"etag-new\""));
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(state.bytes_done, body.len() as u64);
    assert_eq!(std::fs::read(&target).expect("final file"), body);
    std::fs::remove_dir_all(&root).ok();
}

/// A `.part` without a usable meta sidecar cannot be validated: cold
/// restart over it, and without validator headers the DTO advertises
/// `resumable: false`.
#[tokio::test]
async fn resume_without_meta_cold_restarts_unresumable() {
    let root = temp_root("resume-nometa");
    let body = b"no-meta-body".to_vec();
    let payload = body.clone();
    let server = MockServer::new(move |_| MockResponse::once(200, Vec::new(), payload.clone()));
    let url = server.url("nometa.gguf");
    let target = target_for(&root, DownloadKind::Llm, "nometa.gguf");
    std::fs::create_dir_all(target.parent().expect("dir")).expect("dir");
    std::fs::write(with_part(&target), b"orphan-part").expect("seed orphan .part");
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(std::fs::read(&target).expect("final file"), body);
    assert!(!state.resumable, "no etag/accept-ranges ⇒ resumable: false");
    std::fs::remove_dir_all(&root).ok();
}

/// Resume is gated on a saved strong `ETag`: a missing or weak validator must
/// cold-restart even when a non-empty `.part` exists.
#[tokio::test]
async fn resume_requires_strong_saved_etag() {
    let root = temp_root("resume-etag-gate");
    let body = b"fresh-whole-body".to_vec();
    let payload = body.clone();
    let server = MockServer::new(move |req| {
        assert_eq!(
            req.header("range"),
            None,
            "no strong saved etag ⇒ no Range request"
        );
        MockResponse::once(200, Vec::new(), payload.clone())
    });
    let no_etag = target_for(&root, DownloadKind::Llm, "gate-noetag.gguf");
    let weak_etag = target_for(&root, DownloadKind::Llm, "gate-weak.gguf");
    let no_url = server.url("gate-noetag.gguf");
    let weak_url = server.url("gate-weak.gguf");
    seed_part(&no_etag, b"stale-prefix", &no_url, None);
    seed_part(
        &weak_etag,
        b"stale-prefix",
        &weak_url,
        Some("W/\"weak-etag\""),
    );
    let mgr = manager(&root);
    for (target, url) in [(&no_etag, no_url), (&weak_etag, weak_url)] {
        let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
        let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
        assert_eq!(state.bytes_done, body.len() as u64);
        assert_eq!(std::fs::read(target).expect("final file"), body);
    }
    std::fs::remove_dir_all(&root).ok();
}

/// A 206 whose `ETag` differs from the saved one is unusable: the poisoned
/// sidecars are deleted and the file is fetched cold once.
#[tokio::test]
async fn resume_206_etag_mismatch_retries_cold() {
    let root = temp_root("resume-etag-mismatch");
    let prefix = b"OLD-PREFIX".to_vec();
    let full = b"brand-new-full-body".to_vec();
    let calls = Arc::new(Mutex::new(0usize));
    let seen = Arc::clone(&calls);
    let full_payload = full.clone();
    let server = MockServer::new(move |req| {
        let mut calls = seen.lock().expect("calls lock");
        *calls += 1;
        if *calls == 1 {
            assert_eq!(req.header("range"), Some("bytes=10-"));
            assert_eq!(req.header("if-range"), Some("\"old-etag\""));
            MockResponse::once(
                206,
                vec![
                    ("ETag".to_owned(), "\"new-etag\"".to_owned()),
                    ("Accept-Ranges".to_owned(), "bytes".to_owned()),
                    ("Content-Range".to_owned(), "bytes 10-28/29".to_owned()),
                ],
                b"new-tail-123456789".to_vec(),
            )
        } else {
            assert_eq!(req.header("range"), None, "retry must be cold");
            MockResponse::once(
                200,
                vec![
                    ("ETag".to_owned(), "\"new-etag\"".to_owned()),
                    ("Accept-Ranges".to_owned(), "bytes".to_owned()),
                ],
                full_payload.clone(),
            )
        }
    });
    let url = server.url("mismatch.gguf");
    let target = target_for(&root, DownloadKind::Llm, "mismatch.gguf");
    seed_part(&target, &prefix, &url, Some("\"old-etag\""));
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(
        *calls.lock().expect("calls lock"),
        2,
        "one ranged + one cold GET"
    );
    assert_eq!(state.bytes_done, full.len() as u64);
    assert_eq!(std::fs::read(&target).expect("final file"), full);
    assert!(!with_part(&target).exists());
    assert!(!with_meta(&target).exists());
    std::fs::remove_dir_all(&root).ok();
}

/// A 416 on a ranged request must not leave the Range in place: drop the
/// sidecars and retry cold.
#[tokio::test]
async fn resume_416_self_heals_cold() {
    let root = temp_root("resume-416");
    let full = b"cold-after-416".to_vec();
    let calls = Arc::new(Mutex::new(0usize));
    let seen = Arc::clone(&calls);
    let full_payload = full.clone();
    let server = MockServer::new(move |req| {
        let mut calls = seen.lock().expect("calls lock");
        *calls += 1;
        if *calls == 1 {
            assert_eq!(req.header("range"), Some("bytes=5-"));
            assert_eq!(req.header("if-range"), Some("\"old-etag\""));
            MockResponse::once(416, Vec::new(), Vec::new())
        } else {
            assert_eq!(req.header("range"), None, "retry must be cold");
            MockResponse::once(200, Vec::new(), full_payload.clone())
        }
    });
    let url = server.url("gone.gguf");
    let target = target_for(&root, DownloadKind::Llm, "gone.gguf");
    seed_part(&target, b"OLD-5", &url, Some("\"old-etag\""));
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(*calls.lock().expect("calls lock"), 2);
    assert_eq!(state.bytes_done, full.len() as u64);
    assert_eq!(std::fs::read(&target).expect("final file"), full);
    assert!(!with_part(&target).exists());
    assert!(!with_meta(&target).exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Self-healing a bad Range must not drop the checksum saved in the sidecar:
/// the cold retry still rejects a body that does not match it.
#[tokio::test]
async fn resume_self_heal_keeps_saved_checksum() {
    let root = temp_root("resume-sha-heal");
    let calls = Arc::new(Mutex::new(0usize));
    let seen = Arc::clone(&calls);
    let server = MockServer::new(move |req| {
        let mut calls = seen.lock().expect("calls lock");
        *calls += 1;
        if *calls == 1 {
            assert!(req.header("range").is_some());
            MockResponse::once(416, Vec::new(), Vec::new())
        } else {
            assert_eq!(req.header("range"), None, "retry must be cold");
            MockResponse::once(200, Vec::new(), b"wrong-body".to_vec())
        }
    });
    let url = server.url("sha-heal.gguf");
    let target = target_for(&root, DownloadKind::Llm, "sha-heal.gguf");
    let meta = format!(
        "{{\"url\":{url:?},\"etag\":\"\\\"old-etag\\\"\",\"bytes_total\":10,\"sha256\":{:?}}}",
        sha256_hex(b"expected-body")
    );
    std::fs::create_dir_all(target.parent().expect("dir")).expect("dir");
    std::fs::write(with_part(&target), b"OLD-PREFIX").expect("seed .part");
    std::fs::write(with_meta(&target), meta).expect("seed meta");
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Failed).await;
    assert!(
        state
            .error
            .as_deref()
            .is_some_and(|e| e.contains("checksum"))
    );
    assert_eq!(*calls.lock().expect("calls lock"), 2);
    assert!(!with_part(&target).exists());
    assert!(!with_meta(&target).exists());
    assert!(!target.exists());
    std::fs::remove_dir_all(&root).ok();
}

/// A 206 starting at the wrong offset is just as unusable as a 416: self-heal
/// with one cold retry instead of retrying the same bad Range forever.
#[tokio::test]
async fn resume_206_wrong_offset_retries_cold() {
    let root = temp_root("resume-offset");
    let full = b"fresh-after-bad-range".to_vec();
    let calls = Arc::new(Mutex::new(0usize));
    let seen = Arc::clone(&calls);
    let full_payload = full.clone();
    let server = MockServer::new(move |req| {
        let mut calls = seen.lock().expect("calls lock");
        *calls += 1;
        if *calls == 1 {
            MockResponse::once(
                206,
                vec![
                    ("ETag".to_owned(), "\"same-etag\"".to_owned()),
                    ("Accept-Ranges".to_owned(), "bytes".to_owned()),
                    ("Content-Range".to_owned(), "bytes 0-21/22".to_owned()),
                ],
                full_payload.clone(),
            )
        } else {
            assert_eq!(req.header("range"), None, "retry must be cold");
            MockResponse::once(200, Vec::new(), full_payload.clone())
        }
    });
    let url = server.url("offset.gguf");
    let target = target_for(&root, DownloadKind::Llm, "offset.gguf");
    seed_part(&target, b"PREFIX", &url, Some("\"same-etag\""));
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(*calls.lock().expect("calls lock"), 2);
    assert_eq!(std::fs::read(&target).expect("final file"), full);
    assert!(!with_part(&target).exists());
    std::fs::remove_dir_all(&root).ok();
}

/// `Content-Range: bytes N-M/*` leaves the total unknown but the range valid:
/// the suffix still appends and the DTO keeps `bytes_total: None`.
#[tokio::test]
async fn resume_206_unknown_total_appends() {
    let root = temp_root("resume-unknown-total");
    let prefix = b"PREFIX".to_vec();
    let server = MockServer::new(move |req| {
        assert_eq!(req.header("range"), Some("bytes=6-"));
        MockResponse::once(
            206,
            vec![
                ("ETag".to_owned(), "\"etag-unknown\"".to_owned()),
                ("Accept-Ranges".to_owned(), "bytes".to_owned()),
                ("Content-Range".to_owned(), "bytes 6-13/*".to_owned()),
            ],
            b"SUFFIX12".to_vec(),
        )
    });
    let url = server.url("unknown-total.gguf");
    let target = target_for(&root, DownloadKind::Llm, "unknown-total.gguf");
    std::fs::create_dir_all(target.parent().expect("dir")).expect("dir");
    std::fs::write(with_part(&target), &prefix).expect("seed .part");
    let meta = format!("{{\"url\":{url:?},\"etag\":\"\\\"etag-unknown\\\"\"}}");
    std::fs::write(with_meta(&target), meta).expect("seed meta");
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert_eq!(state.bytes_total, None, "unknown total stays unknown");
    let mut whole = prefix;
    whole.extend_from_slice(b"SUFFIX12");
    assert_eq!(std::fs::read(&target).expect("final file"), whole);
    std::fs::remove_dir_all(&root).ok();
}

/// Checksum enforced when provided: a matching body completes.
#[tokio::test]
async fn checksum_ok_completes() {
    let root = temp_root("sha-ok");
    let body = b"checked-body".to_vec();
    let expected = sha256_hex(&body);
    let server = MockServer::new(move |_| MockResponse::once(200, Vec::new(), body.clone()));
    let mgr = manager(&root);
    let accepted = mgr
        .enqueue_with_sha(
            DownloadKind::Llm,
            server.url("checked.gguf"),
            Some(expected),
        )
        .expect("enqueue with sha");
    wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    std::fs::remove_dir_all(&root).ok();
}

/// Checksum mismatch: the download fails, the sidecars are deleted (no
/// bogus bytes left for a future resume), and the failure metric counts.
#[tokio::test]
async fn checksum_mismatch_deletes_sidecars() {
    let root = temp_root("sha-bad");
    let body = b"real-body".to_vec();
    let wrong = sha256_hex(b"not-the-body");
    let server = MockServer::new(move |_| MockResponse::once(200, Vec::new(), body.clone()));
    let mgr = manager(&root);
    let accepted = mgr
        .enqueue_with_sha(DownloadKind::Llm, server.url("mismatch.gguf"), Some(wrong))
        .expect("enqueue with sha");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Failed).await;
    assert!(
        state
            .error
            .as_deref()
            .is_some_and(|e| e.contains("checksum"))
    );
    let target = target_for(&root, DownloadKind::Llm, "mismatch.gguf");
    assert!(!with_part(&target).exists(), "mismatch deletes .part");
    assert!(!with_meta(&target).exists(), "mismatch deletes meta");
    assert!(!target.exists());
    assert!(downloads::exposition().contains("honkbox_download_failures_total{kind=\"llm\"}"));
    std::fs::remove_dir_all(&root).ok();
}

/// Resume with checksum: the persisted sidecar sha applies to the resumed
/// fetch; a mismatched body fails exactly like a cold fetch.
#[tokio::test]
async fn resume_with_checksum_mismatch_fails() {
    let root = temp_root("sha-resume");
    let prefix = b"PRE".to_vec();
    let body = b"wrong-suffix".to_vec();
    let server = MockServer::new(move |_| {
        MockResponse::once(
            206,
            vec![
                ("ETag".to_owned(), "\"etag-sha\"".to_owned()),
                ("Accept-Ranges".to_owned(), "bytes".to_owned()),
                ("Content-Range".to_owned(), "bytes 3-14/15".to_owned()),
            ],
            body.clone(),
        )
    });
    let url = server.url("sha-resume.gguf");
    let target = target_for(&root, DownloadKind::Llm, "sha-resume.gguf");
    let mut meta = String::new();
    let _ = write!(
        meta,
        "{{\"url\":{url:?},\"etag\":\"\\\"etag-sha\\\"\",\"bytes_total\":15,\"sha256\":\"{}\"}}",
        sha256_hex(b"not-this")
    );
    std::fs::create_dir_all(target.parent().expect("dir")).expect("dir");
    std::fs::write(with_part(&target), &prefix).expect("seed .part");
    std::fs::write(with_meta(&target), meta).expect("seed meta");
    let mgr = manager(&root);
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Failed).await;
    assert!(
        state
            .error
            .as_deref()
            .is_some_and(|e| e.contains("checksum"))
    );
    assert!(!with_part(&target).exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Cancel mid-stream: DELETE triggers the token, the task stops at the
/// next chunk boundary, deletes sidecars, and lands in `cancelled`.
#[tokio::test]
async fn cancel_mid_stream_deletes_sidecars() {
    let root = temp_root("cancel");
    let body = vec![0xAB; 16 * 1024];
    let server =
        MockServer::new(move |_| MockResponse::drip(200, Vec::new(), body.clone(), 1024, 50));
    let mgr = manager(&root);
    let accepted = mgr
        .enqueue(llm_request(server.url("cancel.gguf")))
        .await
        .expect("enqueue");
    let target = target_for(&root, DownloadKind::Llm, "cancel.gguf");
    // Wait until at least one chunk landed, then cancel.
    wait_progress(&mgr, &accepted.id, |state| state.bytes_done >= 1024).await;
    mgr.cancel(&accepted.id).expect("cancel");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Cancelled).await;
    assert_eq!(state.state, DownloadStatus::Cancelled);
    // Cleanup is task-side (the handler only marks cancelled): wait for the
    // sidecar deletion at the task's chunk boundary.
    for _ in 0..250 {
        if !with_part(&target).exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(!with_part(&target).exists(), "cancel deletes .part");
    assert!(!with_meta(&target).exists());
    assert!(!target.exists());
    // Terminal: a second cancel conflicts.
    assert_eq!(
        mgr.cancel(&accepted.id),
        Err(downloads::CancelError::Terminal)
    );
    std::fs::remove_dir_all(&root).ok();
}

/// Single-flight: a second pull of the same file while the first is
/// running joins it (same id) instead of racing.
#[tokio::test]
async fn single_flight_joins_in_flight() {
    let root = temp_root("single-flight");
    let body = vec![0xCD; 8 * 1024];
    let server =
        MockServer::new(move |_| MockResponse::drip(200, Vec::new(), body.clone(), 512, 40));
    let mgr = manager(&root);
    let url = server.url("join.gguf");
    let first = mgr.enqueue(llm_request(url.clone())).await.expect("first");
    let second = mgr.enqueue(llm_request(url)).await.expect("second");
    assert_eq!(first.id, second.id, "join returns the in-flight id");
    wait_state(&mgr, &first.id, |s| s == &DownloadStatus::Done).await;
    std::fs::remove_dir_all(&root).ok();
}

/// A target file that already exists resolves as `done` + `already_present`
/// without fetching anything.
#[tokio::test]
async fn already_present_short_circuits() {
    let root = temp_root("present");
    let target = target_for(&root, DownloadKind::Llm, "present.gguf");
    std::fs::create_dir_all(target.parent().expect("dir")).expect("dir");
    std::fs::write(&target, b"already-here").expect("seed final");
    let hits = Arc::new(Mutex::new(0usize));
    let hit = Arc::clone(&hits);
    let server = MockServer::new(move |_| {
        *hit.lock().expect("hit") += 1;
        MockResponse::once(200, Vec::new(), b"should-not-fetch".to_vec())
    });
    let mgr = manager(&root);
    let accepted = mgr
        .enqueue(llm_request(server.url("present.gguf")))
        .await
        .expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Done).await;
    assert!(state.already_present);
    assert_eq!(state.bytes_done, 0);
    assert_eq!(*hits.lock().expect("hit"), 0, "no bytes fetched");
    assert_eq!(
        std::fs::read(&target).expect("final untouched"),
        b"already-here"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// Concurrency cap: three pulls, the third stays queued while the first
/// two hold the slots; all three eventually finish.
#[tokio::test]
async fn concurrency_cap_two_slots() {
    let root = temp_root("cap");
    let body = vec![0xEF; 4 * 1024];
    let server =
        MockServer::new(move |_| MockResponse::drip(200, Vec::new(), body.clone(), 256, 60));
    let mgr = manager(&root);
    let mut ids = Vec::new();
    for i in 0..3 {
        let accepted = mgr
            .enqueue(llm_request(server.url(&format!("cap{i}.gguf"))))
            .await
            .expect("enqueue");
        ids.push(accepted.id);
    }
    // Stable while bodies drip: the third stays queued.
    let third = wait_progress(&mgr, &ids[2], |state| state.state == DownloadStatus::Queued).await;
    assert_eq!(third.state, DownloadStatus::Queued);
    for id in &ids {
        wait_state(&mgr, id, |s| s == &DownloadStatus::Done).await;
    }
    std::fs::remove_dir_all(&root).ok();
}

/// A server error status fails the download; nothing commits.
#[tokio::test]
async fn server_error_status_fails_without_commit() {
    let root = temp_root("failed");
    let server =
        MockServer::new(move |_| MockResponse::once(503, Vec::new(), b"overloaded".to_vec()));
    let mgr = manager(&root);
    let url = server.url("failing.gguf");
    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| s == &DownloadStatus::Failed).await;
    assert!(state.error.as_deref().is_some_and(|e| e.contains("503")));
    let target = target_for(&root, DownloadKind::Llm, "failing.gguf");
    assert!(!target.exists());
    assert!(!with_part(&target).exists());
    std::fs::remove_dir_all(&root).ok();
}

/// Repo-form validation: bad repo ids, unusable names, ambiguous or missing
/// sources are rejected without any network activity.
#[tokio::test]
async fn rejects_bad_repo_and_names() {
    let root = temp_root("reject");
    let mgr = manager(&root);
    let bad_repo = DownloadRequest {
        repo: Some("no-slash".to_owned()),
        file: Some("model.gguf".to_owned()),
        url: None,
        kind: DownloadKind::Llm,
    };
    assert!(matches!(
        mgr.enqueue(bad_repo).await,
        Err(downloads::EnqueueError::BadRepo(_))
    ));
    let bad_file = DownloadRequest {
        repo: Some("owner/repo".to_owned()),
        file: Some(String::new()),
        url: None,
        kind: DownloadKind::Llm,
    };
    assert!(matches!(
        mgr.enqueue(bad_file).await,
        Err(downloads::EnqueueError::BadFileName(_))
    ));
    let both = DownloadRequest {
        repo: Some("owner/repo".to_owned()),
        file: Some("model.gguf".to_owned()),
        url: Some("http://127.0.0.1/x".to_owned()),
        kind: DownloadKind::Llm,
    };
    assert_eq!(
        mgr.enqueue(both).await,
        Err(downloads::EnqueueError::InvalidSource)
    );
    let neither = DownloadRequest {
        repo: None,
        file: None,
        url: None,
        kind: DownloadKind::Tts,
    };
    assert_eq!(
        mgr.enqueue(neither).await,
        Err(downloads::EnqueueError::InvalidSource)
    );
    std::fs::remove_dir_all(&root).ok();
}

/// Startup sweep contract: stale `.part` files are counted, not deleted.
#[test]
fn count_stale_parts_counts_not_deletes() {
    let root = temp_root("stale");
    let dir = root.join("models/llm");
    std::fs::create_dir_all(&dir).expect("dir");
    std::fs::write(dir.join("a.gguf.part"), b"x").expect("part");
    std::fs::write(dir.join("a.gguf.part.meta.json"), b"{}").expect("meta");
    std::fs::write(dir.join("b.gguf"), b"y").expect("final");
    assert_eq!(downloads::count_stale_parts(Some(&root)), 1);
    assert!(dir.join("a.gguf.part").exists(), "kept for resume");
    assert_eq!(downloads::count_stale_parts(None), 0);
    std::fs::remove_dir_all(&root).ok();
}

/// A strict supervisor refuses non-public destinations at enqueue time —
/// before any request can leave the process (SSRF regression).
#[tokio::test]
async fn unsafe_urls_are_refused_before_http() {
    let root = temp_root("unsafe-url");
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let server = MockServer::new(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        MockResponse::once(200, Vec::new(), b"data".to_vec())
    });
    let strict = Arc::new(
        downloads::DownloadManager::new(Some(root.join("models"))).expect("strict manager"),
    );

    let cases = [
        server.url("loopback.gguf"), // 127.0.0.1 literal
        "http://169.254.169.254/latest/meta-data/iam.gguf".to_owned(), // link-local metadata
        "http://10.1.2.3/private.gguf".to_owned(), // RFC1918 private
        "http://[::1]/v6loop.gguf".to_owned(), // IPv6 loopback
        "http://[fd00:ec2::254]/imds.gguf".to_owned(), // IPv6 ULA metadata
        "http://100.100.100.200/cgnat.gguf".to_owned(), // CGNAT / Alibaba metadata
        "http://localhost:9/localhost.gguf".to_owned(), // hostname resolving to loopback
        "ftp://example.com/file.gguf".to_owned(), // wrong scheme
        "not a url".to_owned(),      // unparseable
    ];
    for url in cases {
        assert_eq!(
            strict.enqueue(llm_request(url)).await.unwrap_err(),
            downloads::EnqueueError::UnsafeUrl,
        );
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0, "no HTTP before validation");
    std::fs::remove_dir_all(&root).ok();
}

/// Redirects are not followed: a 302 stays a terminal failure and the
/// fixture sees exactly one request (reqwest's default follows redirects).
#[tokio::test]
async fn redirects_are_not_followed() {
    let root = temp_root("no-redirect");
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let server = MockServer::new(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        MockResponse::once(
            302,
            vec![("Location".to_owned(), "/target.gguf".to_owned())],
            Vec::new(),
        )
    });
    let mgr = manager(&root);
    let accepted = mgr
        .enqueue(llm_request(server.url("origin.gguf")))
        .await
        .expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| {
        matches!(s, DownloadStatus::Failed | DownloadStatus::Done)
    })
    .await;
    assert_eq!(
        state.state,
        DownloadStatus::Failed,
        "302 must not be followed"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1, "no redirect request");
    std::fs::remove_dir_all(&root).ok();
}

/// A failed fetch returns a named cause, never the absolute `.part` path.
#[tokio::test]
async fn failure_dto_has_no_absolute_path() {
    let root = temp_root("no-path-leak");
    let server = MockServer::new(|_| MockResponse::once(200, Vec::new(), b"data".to_vec()));
    let mgr = manager(&root);
    let url = server.url("blocked.gguf");
    // A directory at the `.part` path makes `open_part` fail deterministically.
    let target = target_for(&root, DownloadKind::Llm, "blocked.gguf");
    std::fs::create_dir_all(with_part(&target)).expect("blocked part dir");

    let accepted = mgr.enqueue(llm_request(url)).await.expect("enqueue");
    let state = wait_state(&mgr, &accepted.id, |s| *s == DownloadStatus::Failed).await;
    let cause = state.error.expect("failure cause");
    assert_eq!(cause, "part_open_failed");
    assert!(!cause.contains('/'), "host path leaked to client: {cause}");
    assert!(!cause.contains(root.to_string_lossy().as_ref()));
    std::fs::remove_dir_all(&root).ok();
}

// ----------------------------------------------------------- route tests

/// POST a JSON body at `/v1/downloads` (oneshot helper).
async fn post_json(
    app: &axum::Router,
    body: &serde_json::Value,
) -> axum::http::Response<axum::body::Body> {
    app.clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/downloads")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response")
}

/// GET a URI via oneshot (helper for the polling loop).
async fn get_uri(app: &axum::Router, uri: String) -> axum::http::Response<axum::body::Body> {
    app.clone()
        .oneshot(
            axum::http::Request::builder()
                .uri(uri)
                .body(axum::body::Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
}

/// Full app flow over the routes: 202 acceptance, poll to done, list,
/// unknown-id 404, cancel-after-done 409, bad payloads 400.
#[tokio::test]
async fn downloads_route_flow() {
    let root = temp_root("routes");
    let body = b"route-body".to_vec();
    let payload = body.clone();
    let server = MockServer::new(move |_| MockResponse::once(200, Vec::new(), payload.clone()));
    let app = routes::app(app_config(&root, manager(&root)));

    // Bad payload: no source at all → 400.
    let response = post_json(&app, &serde_json::json!({"kind": "llm"})).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Happy path: POST → 202 with an id, then poll GET until done.
    let response = post_json(
        &app,
        &serde_json::json!({"url": server.url("route.gguf"), "kind": "llm"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("body");
    let accepted: honkbox_api_types::DownloadAccepted =
        serde_json::from_slice(&bytes).expect("accepted dto");
    let id = accepted.id;

    let mut done = None;
    for _ in 0..250 {
        let response = get_uri(&app, format!("/v1/downloads/{id}")).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("body");
        let state: honkbox_api_types::DownloadState =
            serde_json::from_slice(&bytes).expect("state dto");
        if state.state == DownloadStatus::Done {
            done = Some(state);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let state = done.expect("download done via route polling");
    assert_eq!(state.bytes_done, body.len() as u64);
    assert!(
        state
            .target_path
            .as_deref()
            .is_some_and(|p| p.contains("models/llm"))
    );

    // List contains the download.
    let response = get_uri(&app, "/v1/downloads".to_owned()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .expect("body");
    let list: honkbox_api_types::DownloadList = serde_json::from_slice(&bytes).expect("list");
    assert!(list.downloads.iter().any(|d| d.id == id));

    // Cancel after done → 409; unknown id → 404 (DELETE and GET).
    let delete = |uri: String, app: axum::Router| async move {
        app.oneshot(
            axum::http::Request::builder()
                .method("DELETE")
                .uri(uri)
                .body(axum::body::Body::empty())
                .expect("request"),
        )
        .await
        .expect("response")
    };
    assert_eq!(
        delete(format!("/v1/downloads/{id}"), app.clone())
            .await
            .status(),
        StatusCode::CONFLICT
    );
    assert_eq!(
        delete("/v1/downloads/dl_unknown".to_owned(), app.clone())
            .await
            .status(),
        StatusCode::NOT_FOUND
    );
    let response = get_uri(&app, "/v1/downloads/dl_unknown".to_owned()).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    std::fs::remove_dir_all(&root).ok();
}

/// `GET /v1/catalog`: curated manifest merged with the app's snapshot and
/// the configured budget — present entry gets a green pill, absent entry
/// has no `local`, envelope shape matches.
#[tokio::test]
async fn catalog_route_merges_snapshot_and_budget() {
    let root = temp_root("catalog");
    let mut config = app_config(&root, manager(&root));
    config.vram_budget = Some(2_000_000);
    let snapshot = honkbox_api_types::ModelCatalog {
        object: "list".to_owned(),
        data: vec![honkbox_api_types::ModelCatalogEntry {
            id: "llm:models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_owned(),
            slug: "llm:models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_owned(),
            kind: honkbox_api_types::ModelKind::Llm,
            owned_by: "local".to_owned(),
            context_length: None,
            max_output: None,
            thinking: false,
            tool_calling: false,
            modalities: vec!["text".to_owned()],
            modality: honkbox_api_types::ModelModalities::empty(),
            capabilities: honkbox_api_types::ModelCapabilities::for_kind(
                honkbox_api_types::ModelKind::Llm,
            ),
            category: "root".to_owned(),
            path: root
                .join("models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
                .display()
                .to_string(),
            bytes: 1_000_000, // est 1_350_000 <= budget 2_000_000 → ok
            metadata: None,
        }],
    };
    *config.catalog.write().expect("catalog lock") = std::sync::Arc::new(snapshot);
    let app = routes::app(config);

    let response = get_uri(&app, "/v1/catalog".to_owned()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .expect("body");
    let list: honkbox_api_types::CatalogList = serde_json::from_slice(&bytes).expect("list dto");
    assert_eq!(list.object, "list");
    assert_ne!(list.data.len(), 0);
    let llama = list
        .data
        .iter()
        .find(|entry| entry.id == "llama-3.2-1b-q4")
        .expect("curated llama entry");
    assert_eq!(llama.repo, "bartowski/Llama-3.2-1B-Instruct-GGUF");
    assert_eq!(llama.verified_date, "2026-11");
    let local = llama.local.as_ref().expect("present in the snapshot");
    assert_eq!(local.bytes, 1_000_000);
    assert_eq!(local.est_vram_bytes, Some(1_350_000));
    assert_eq!(local.pill, Some(honkbox_api_types::Pill::Ok));
    // Absent file: manifest metadata only, no local/pill.
    let flux = list
        .data
        .iter()
        .find(|entry| entry.id == "flux1-schnell-bf16")
        .expect("curated flux entry");
    assert!(flux.local.is_none());
}

/// POST without a managed models root maps the enqueue error to 500.
#[tokio::test]
async fn post_without_models_root_is_500() {
    let root = temp_root("noroot");
    let app = routes::app(app_config(
        &root,
        Arc::new(downloads::DownloadManager::new(None).expect("manager")),
    ));
    let response = post_json(
        &app,
        &serde_json::json!({"url": "http://127.0.0.1:9/x.gguf", "kind": "llm"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    std::fs::remove_dir_all(&root).ok();
}

/// Release-shaped `AppConfig` (empty catalog, stub pipelines) with the
/// given download supervisor.
fn app_config(root: &Path, mgr: Arc<downloads::DownloadManager>) -> AppConfig {
    AppConfig {
        auth: AuthConfig::default(),
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(std::sync::RwLock::new(Arc::new(
            honkbox_api_types::ModelCatalog {
                object: "list".to_owned(),
                data: Vec::new(),
            },
        ))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
            ocr_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(honkbox_llm_worker::StubLlm::default()),
        stt: Arc::new(honkbox_audio_worker::StubStt),
        tts: Arc::new(honkbox_audio_worker::StubTts),
        sd: std::sync::Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: Some(root.to_path_buf()),
        artifacts: std::sync::Arc::new(crate::artifacts::Registry::new(
            Some(root.to_path_buf()).clone(),
            crate::config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: mgr,
    }
}
