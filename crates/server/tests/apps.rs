#![cfg(unix)]

//! Managed app lifecycle contract: config-only launch modes, version probe,
//! allowlisted argv, readiness transitions, process-group teardown with no
//! orphan, and bearer-gated `/comfyui` honesty.

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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[path = "../src/swap.rs"]
mod swap;
#[allow(dead_code)]
#[path = "../src/texture_postprocess.rs"]
mod texture_postprocess;

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use axum::{
    body::to_bytes,
    http::{Request, StatusCode},
};
use tower::ServiceExt as _;

use config::{ManagedAppLaunch, ManagedAppsConfig};
use honkbox_external_tools::{
    Config as ToolsConfig, Registry as ToolsRegistry, Status as ToolStatus, ToolId,
};
use routes::{AppConfig, AuthConfig, AuthMode, ManagedAppManager, app, set_env_var};
use routes::{set_manager_for_tests, set_registry_for_tests};

static SEQ: AtomicUsize = AtomicUsize::new(0);
/// The residency ledger has one slot per kind; lease-holding tests serialize.
static LEASE_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
/// The manager and external-tools slots are global; tests that swap them serialize.
static SLOT_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Unique temp dir per test; nothing under a fixed port or shared data dir.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "honkbox-apps-{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A fake PATH binary: `--version` probe, argv log, optional helper child.
fn fake_binary(path: &Path, argv_log: &Path, helper_pid: Option<&Path>) {
    let helper = helper_pid.map_or_else(String::new, |p| {
        format!("sleep 300 & echo $! > '{}'\n", p.display())
    });
    std::fs::write(
        path,
        format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'llama-server 9.9.9'; exit 0; fi\nprintf '%s\\n' \"$@\" > '{}'\necho fixture-stderr >&2\n{helper}while :; do sleep 1; done\n",
            argv_log.display()
        ),
    )
    .unwrap();
    set_mode(path);
}

fn set_mode(path: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Counted loopback responder serving ONE route; other paths are 404. The
/// first `loading` requests to that route are 503, later ones 200. No real app
/// or GPU is involved.
fn health_server(port: u16, loading: usize, ready_path: &str) {
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
    let prefix = format!("GET {ready_path} ");
    std::thread::spawn(move || {
        let seen = AtomicUsize::new(0);
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0_u8; 256];
            let read = stream.read(&mut buf).unwrap_or(0);
            let served = String::from_utf8_lossy(&buf[..read]).starts_with(&prefix);
            let count = seen.fetch_add(1, Ordering::Relaxed);
            let (status, body) = if !served {
                ("404 Not Found", "404")
            } else if count < loading {
                ("503 Service Unavailable", "503")
            } else {
                ("200 OK", "200")
            };
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(reply.as_bytes());
        }
    });
}

fn free_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap().port()
}

fn llama_config(binary: &Path, port: u16) -> ManagedAppsConfig {
    let mut cfg = ManagedAppsConfig::default();
    cfg.llama_server.enabled = true;
    cfg.llama_server.binary = Some(binary.to_path_buf());
    cfg.llama_server.model = Some(PathBuf::from("/models/fixture.gguf"));
    cfg.llama_server.port = Some(port);
    cfg.llama_server.health_timeout_secs = 15;
    cfg
}

fn comfy_config(
    launch: ManagedAppLaunch,
    root: &Path,
    script: Option<&Path>,
    port: u16,
) -> ManagedAppsConfig {
    let mut cfg = ManagedAppsConfig::default();
    cfg.comfyui.enabled = true;
    cfg.comfyui.launch = launch;
    cfg.comfyui.installation = Some(root.to_path_buf());
    cfg.comfyui.script = script.map(Path::to_path_buf);
    cfg.comfyui.port = Some(port);
    cfg.comfyui.health_timeout_secs = 15;
    cfg
}

async fn wait_for(manager: &ManagedAppManager, id: &str, state: routes::AppState) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if manager
            .status(id)
            .is_some_and(|status| status.state == state)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("app {id} never reached {}", state.as_str());
}

fn alive(pid: u32) -> bool {
    std::fs::metadata(format!("/proc/{pid}")).is_ok()
}

fn gone_within(pid: u32) -> bool {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if !alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    !alive(pid)
}

#[tokio::test]
async fn llama_version_probe_argv_and_group_teardown() {
    let _lease_serial = LEASE_LOCK.lock().await;
    let dir = scratch("llama");
    let binary = dir.join("fake-llama-server");
    let argv = dir.join("argv");
    let helper_pid = dir.join("helper.pid");
    fake_binary(&binary, &argv, Some(&helper_pid));
    let port = free_port();
    health_server(port, 0, "/health");
    let manager = Arc::new(
        ManagedAppManager::new(
            llama_config(&binary, port),
            Some(dir.clone()),
            "http://127.0.0.1:8188",
        )
        .unwrap(),
    );
    manager.start("llama_server").await.unwrap();
    wait_for(&manager, "llama_server", routes::AppState::Ready).await;
    let status = manager.status("llama_server").unwrap();
    assert_eq!(status.version.as_deref(), Some("9.9.9"));
    assert!(status.endpoint.is_some());
    let captured = std::fs::read_to_string(&argv).unwrap();
    for flag in [
        "--model",
        "--host",
        "--port",
        "--api-key",
        "--cors-origins",
        "localhost",
        "--no-webui",
    ] {
        assert!(captured.contains(flag), "missing {flag} in {captured}");
    }
    assert!(
        !captured.contains("--tools")
            && !captured.contains("--agent")
            && !captured.contains("--mcp")
    );
    let helper = std::fs::read_to_string(&helper_pid)
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();
    assert!(alive(helper));
    manager.stop("llama_server").await.unwrap();
    assert_eq!(
        manager.status("llama_server").unwrap().state,
        routes::AppState::Stopped
    );
    assert!(!manager.ready("llama_server"));
    assert!(gone_within(helper), "helper survived group teardown");
    manager.shutdown().await;
}

#[tokio::test]
async fn crashed_app_is_failed_with_stderr_tail_and_no_endpoint() {
    let _lease_serial = LEASE_LOCK.lock().await;
    let dir = scratch("crash");
    let binary = dir.join("fake-llama-server");
    let argv = dir.join("argv");
    std::fs::write(
        &binary,
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'llama-server 9.9.9'; exit 0; fi\necho crash-marker >&2\nexit 1\n",
    )
    .unwrap();
    set_mode(&binary);
    let port = free_port();
    let manager = Arc::new(
        ManagedAppManager::new(
            llama_config(&binary, port),
            Some(dir),
            "http://127.0.0.1:8188",
        )
        .unwrap(),
    );
    manager.start("llama_server").await.unwrap();
    wait_for(&manager, "llama_server", routes::AppState::Failed).await;
    let status = manager.status("llama_server").unwrap();
    assert!(
        status
            .error
            .is_some_and(|error| error.contains("crash-marker"))
    );
    assert!(status.endpoint.is_none());
    assert!(!manager.ready("llama_server"));
    let _ = argv;
    manager.shutdown().await;
}

#[tokio::test]
async fn comfyui_installation_and_script_modes_use_allowlisted_argv() {
    let _lease_serial = LEASE_LOCK.lock().await;
    for (tag, launch, script_name) in [
        ("install", ManagedAppLaunch::Installation, None),
        ("script", ManagedAppLaunch::Script, Some("start_local.py")),
    ] {
        let dir = scratch(tag);
        let root = dir.join("comfyui");
        let venv = root.join(".venv/bin");
        std::fs::create_dir_all(&venv).unwrap();
        std::fs::write(root.join("main.py"), "# fixture\n").unwrap();
        std::fs::write(
            root.join("comfyui_version.py"),
            "__version__ = \"0.46.0\"\n",
        )
        .unwrap();
        let python = venv.join("python");
        let argv = dir.join("argv");
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'Python 3.12.0'; exit 0; fi\nprintf '%s\\n' \"$@\" > '{}'\nwhile :; do sleep 1; done\n",
                argv.display()
            ),
        )
        .unwrap();
        set_mode(&python);
        let script = script_name.map(|name| {
            let path = root.join("scripts").join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "# fixture launcher\n").unwrap();
            path
        });
        let port = free_port();
        // ComfyUI registers /queue, not /health: a probe of /health gets 404.
        health_server(port, 1, "/queue"); // one 503 then 200: countdown restarts
        let manager = Arc::new(
            ManagedAppManager::new(
                comfy_config(launch, &root, script.as_deref(), port),
                Some(dir),
                "http://127.0.0.1:8188",
            )
            .unwrap(),
        );
        manager.start("comfyui").await.unwrap();
        wait_for(&manager, "comfyui", routes::AppState::Ready).await;
        let status = manager.status("comfyui").unwrap();
        assert_eq!(status.version.as_deref(), Some("0.46.0"));
        let captured = std::fs::read_to_string(&argv).unwrap();
        assert!(captured.contains("--listen"), "{captured}");
        assert!(captured.contains("--port"), "{captured}");
        assert!(captured.contains("--disable-auto-launch"), "{captured}");
        if launch == ManagedAppLaunch::Installation {
            assert!(captured.contains("main.py"), "{captured}");
        } else {
            assert!(captured.contains("start_local.py"), "{captured}");
        }
        assert!(!captured.contains("--tools") && !captured.contains("--agent"));
        manager.stop("comfyui").await.unwrap();
        assert!(!manager.ready("comfyui"));
        manager.shutdown().await;
    }
}

fn test_app_config() -> AppConfig {
    AppConfig {
        recognition: Arc::new(honkbox_recognition_worker::StubRecognition),
        auth: AuthConfig {
            mode: AuthMode::Bearer,
            api_key_env: "HONKBOX_TEST_APPS_KEY".to_owned(),
        },
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(honkbox_api_types::ModelCatalog {
            object: "list".to_owned(),
            data: Vec::new(),
        }))),
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
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: Duration::from_hours(24),
        data_dir: None,
        artifacts: Arc::new(artifacts::Registry::new(
            None,
            config::DEFAULT_ARTIFACT_MAX_BYTES,
        )),
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).unwrap()),
    }
}

#[tokio::test]
async fn admin_routes_are_bearer_gated_and_comfyui_stays_honest_when_down() {
    const KEY_ENV: &str = "HONKBOX_TEST_APPS_KEY";
    let _slot_serial = SLOT_LOCK.lock().await;
    set_env_var(KEY_ENV, Some("test-apps-key"));
    let manager = Arc::new(
        ManagedAppManager::new(ManagedAppsConfig::default(), None, "http://127.0.0.1:8188")
            .unwrap(),
    );
    set_manager_for_tests(Some(Arc::clone(&manager)));
    let router = app(test_app_config());
    let get_apps = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/apps")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_apps.status(), StatusCode::UNAUTHORIZED);

    let auth = "Bearer test-apps-key";
    let start = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/apps/comfyui/start")
                .header("Authorization", auth)
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(start.status(), StatusCode::NOT_IMPLEMENTED);

    let status = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/comfyui/status")
                .header("Authorization", auth)
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::NOT_IMPLEMENTED);
    let body = to_bytes(status.into_body(), 4096).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("comfyui unavailable"));

    let list = router
        .oneshot(
            Request::builder()
                .uri("/v1/apps")
                .header("Authorization", auth)
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list.status(), StatusCode::OK);
    let body = to_bytes(list.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["apps"].as_array().unwrap().len(), 2);

    set_manager_for_tests(None);
    set_env_var(KEY_ENV, None);
}

#[tokio::test]
async fn disabled_app_start_is_a_named_501_even_with_bearer() {
    let manager = Arc::new(
        ManagedAppManager::new(ManagedAppsConfig::default(), None, "http://127.0.0.1:8188")
            .unwrap(),
    );
    let error = manager.start("comfyui").await.unwrap_err();
    assert_eq!(error.status, StatusCode::NOT_IMPLEMENTED);
    assert!(error.message.contains("not enabled"));
}

/// Every typed read route must fail closed without a reachable upstream.
async fn assert_comfyui_501(router: &axum::Router) {
    for uri in [
        "/comfyui/status",
        "/comfyui/queue",
        "/comfyui/history/abc",
        "/comfyui/view?filename=out.png",
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .header("Authorization", "Bearer test-apps-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{uri}");
    }
}

#[tokio::test]
async fn external_tools_row_without_manager_never_claims_comfyui() {
    const KEY_ENV: &str = "HONKBOX_TEST_APPS_KEY";
    let _slot_serial = SLOT_LOCK.lock().await;
    set_env_var(KEY_ENV, Some("test-apps-key"));
    // An available external-tools row with no installed manager must not
    // advertise a read surface whose handlers have no upstream to call.
    let registry = ToolsRegistry::from_statuses(
        ToolsConfig::default(),
        vec![ToolStatus {
            id: ToolId::ComfyUi,
            version: Some("1.0.0".to_owned()),
            capabilities: Vec::new(),
            devices: Vec::new(),
            unavailable: None,
        }],
    );
    set_registry_for_tests(Some(registry)).await;
    set_manager_for_tests(None);
    let router = app(test_app_config());
    assert_comfyui_501(&router).await;
    set_registry_for_tests(None).await;
    set_env_var(KEY_ENV, None);
}

#[tokio::test]
async fn detached_comfyui_status_never_claims_live_availability() {
    const KEY_ENV: &str = "HONKBOX_TEST_APPS_KEY";
    let _slot_serial = SLOT_LOCK.lock().await;
    set_env_var(KEY_ENV, Some("test-apps-key"));
    let registry = ToolsRegistry::from_statuses(
        ToolsConfig::default(),
        vec![ToolStatus {
            id: ToolId::ComfyUi,
            version: Some("1.0.0".to_owned()),
            capabilities: Vec::new(),
            devices: Vec::new(),
            unavailable: None,
        }],
    );
    set_registry_for_tests(Some(registry)).await;
    let manager = Arc::new(
        ManagedAppManager::new(ManagedAppsConfig::default(), None, "http://127.0.0.1:8188")
            .unwrap(),
    );
    set_manager_for_tests(Some(Arc::clone(&manager)));
    let router = app(test_app_config());
    let response = router
        .oneshot(
            Request::builder()
                .uri("/comfyui/status")
                .header("Authorization", "Bearer test-apps-key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["available"], false);
    assert_eq!(json["state"], "detached-unverified");
    set_manager_for_tests(None);
    set_registry_for_tests(None).await;
    set_env_var(KEY_ENV, None);
}

#[tokio::test]
async fn configured_but_stopped_comfyui_app_is_named_501() {
    const KEY_ENV: &str = "HONKBOX_TEST_APPS_KEY";
    let _slot_serial = SLOT_LOCK.lock().await;
    set_env_var(KEY_ENV, Some("test-apps-key"));
    let dir = scratch("stopped-comfy");
    let file = config::FileApps {
        comfyui: Some(config::FileApp {
            enabled: Some(true),
            launch: Some("installation".into()),
            installation: Some(dir.to_string_lossy().into_owned()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cfg = config::managed_apps_config(file).unwrap();
    let manager =
        Arc::new(ManagedAppManager::new(cfg, Some(dir), "http://127.0.0.1:8188").unwrap());
    set_manager_for_tests(Some(manager));
    let router = app(test_app_config());
    assert_comfyui_501(&router).await;
    set_manager_for_tests(None);
    set_env_var(KEY_ENV, None);
}
