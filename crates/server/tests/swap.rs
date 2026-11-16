//! LLM swap manager (P1 of the llama-swap parity plan): load-on-demand
//! caching, TTL + LRU eviction, per-request `ttl_seconds` (NEW-8: clamp
//! bounds, pin precedence, use-then-unload), HTTP error mapping, and the
//! untouched stub path. Eviction is exercised through the exposed
//! `evict_idle` hook with short TTLs — no sleeps on reaper periods, so the
//! tests stay deterministic.

// No lib target on this binary crate, so include the modules directly.
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/downloads.rs"]
mod downloads;
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;
#[path = "../src/swap.rs"]
mod swap;

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::{
    ChatRequest as WorkerChatRequest, ChatResponse, LlmError, LlmPipeline, StubLlm,
};
use tower::ServiceExt as _;

use routes::{AppConfig, app};
use swap::{LlmLoader, SwapError, SwapManager, SwapSettings};

/// Serializes the residency-slot tests: they share the process-global GPU
/// slot, and cargo runs test fns in parallel threads. Async mutex so the
/// guard may be held across `.await`.
static SLOT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Construction/drop counters shared by the pipelines a test builds:
/// `loads` counts constructions, `live` the pipelines not yet dropped
/// (cache entries + in-flight holders).
#[derive(Debug, Default)]
struct Probe {
    loads: AtomicU64,
    live: AtomicU64,
}

/// Test double: canned completions; `Drop` decrements `live` — that drop is
/// the observable "model memory freed".
#[derive(Debug)]
struct CountingLlm {
    probe: Arc<Probe>,
}

impl Drop for CountingLlm {
    fn drop(&mut self) {
        self.probe.live.fetch_sub(1, Ordering::Relaxed);
    }
}

fn canned_response() -> ChatResponse {
    ChatResponse {
        backend_id: "counting-llm".to_owned(),
        text: "hello swap".to_owned(),
        tokens: 2,
        millis: 0,
    }
}

impl LlmPipeline for CountingLlm {
    fn complete(&self, _req: &WorkerChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(canned_response())
    }

    fn complete_stream(
        &self,
        _req: &WorkerChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        sink("hello ");
        sink("swap");
        Ok(canned_response())
    }

    fn backend_id(&self) -> &'static str {
        "counting-llm"
    }
}

/// Constructor seam: counts constructions, fails for `broken` slugs without
/// touching the counters (the cache must stay consistent across failures).
fn counting_loader(probe: &Arc<Probe>, broken: &[&str]) -> LlmLoader {
    let broken: HashSet<String> = broken.iter().map(|slug| (*slug).to_owned()).collect();
    let probe = Arc::clone(probe);
    Arc::new(move |slug: &str| {
        if broken.contains(slug) {
            return Err(SwapError::Load(format!(
                "intentional load failure for {slug:?}"
            )));
        }
        probe.loads.fetch_add(1, Ordering::Relaxed);
        probe.live.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(CountingLlm {
            probe: Arc::clone(&probe),
        }) as Arc<dyn LlmPipeline>)
    })
}

/// Swap settings from the pieces the tests vary; unpinned/unbounded except
/// where a test needs the cap or a pin.
fn settings(ttl: Duration, max_loaded: usize, max_ttl: Duration, pinned: &[&str]) -> SwapSettings {
    SwapSettings {
        llm_ttl: ttl,
        llm_max_loaded: max_loaded,
        max_ttl,
        pinned: pinned.iter().map(|slug| (*slug).to_owned()).collect(),
    }
}

fn manager_with(swap_settings: SwapSettings, loader: LlmLoader) -> Arc<SwapManager> {
    Arc::new(SwapManager::new(swap_settings, loader))
}

fn manager(ttl: Duration, max_loaded: usize, loader: LlmLoader) -> Arc<SwapManager> {
    manager_with(
        settings(ttl, max_loaded, Duration::from_hours(24), &[]),
        loader,
    )
}

/// Catalog with fake paths — the counting loader never stats them.
fn catalog_with(entries: &[(&str, ModelKind)]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries
            .iter()
            .map(|(id, kind)| ModelCatalogEntry {
                id: (*id).to_owned(),
                kind: *kind,
                category: "root".to_owned(),
                path: "/dev/null-fake-weights".to_owned(),
                bytes: 1,
                metadata: None,
            })
            .collect(),
    }
}

fn swap_app(
    catalog: ModelCatalog,
    loader: LlmLoader,
    ttl: Duration,
    max_loaded: usize,
    max_ttl: Duration,
) -> Router {
    app(AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        // The static pipeline is inert while a swap manager is present —
        // exactly the wiring `main.rs` produces.
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        swap: Some(manager_with(
            settings(ttl, max_loaded, max_ttl, &[]),
            loader,
        )),
        swap_max_ttl: max_ttl,
        data_dir: None,
        vram_budget: None,
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    })
}

/// `POST /v1/chat` with `model` + optional `ttl_seconds`; status + body.
async fn chat(app: Router, model: &str, ttl_seconds: Option<u64>) -> (StatusCode, String) {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}]
    });
    if let Some(ttl) = ttl_seconds {
        body["ttl_seconds"] = serde_json::json!(ttl);
    }
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// Load-on-demand caches: the second request for the same slug reuses the
/// constructed pipeline end to end through the route.
#[tokio::test]
async fn load_on_demand_caches_across_requests() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let app = swap_app(
        catalog_with(&[("llm:fake.gguf", ModelKind::Llm)]),
        counting_loader(&probe, &[]),
        Duration::from_secs(3600),
        2,
        Duration::from_hours(24),
    );
    let (status, _) = chat(app.clone(), "llm:fake.gguf", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = chat(app, "llm:fake.gguf", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("counting-llm"), "swap-served reply: {body}");
    assert_eq!(
        probe.loads.load(Ordering::Relaxed),
        1,
        "second request must reuse the cached pipeline"
    );
}

/// TTL eviction frees the idle pipeline (the drop is the free) and the next
/// request reloads on demand.
#[tokio::test]
async fn ttl_eviction_frees_idle_pipeline() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_millis(150), 2, counting_loader(&probe, &[]));
    let pipe = mgr.get("llm:a.gguf", None).await.unwrap();
    assert_eq!(probe.live.load(Ordering::Relaxed), 1);
    drop(pipe);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(mgr.evict_idle(), 1, "TTL-expired entry evicted");
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        0,
        "cache-held pipeline freed by the eviction"
    );
    let pipe = mgr.get("llm:a.gguf", None).await.unwrap();
    assert_eq!(
        probe.loads.load(Ordering::Relaxed),
        2,
        "reload after eviction"
    );
    drop(pipe);
    std::thread::sleep(Duration::from_millis(300));
    mgr.evict_idle();
    assert_eq!(probe.live.load(Ordering::Relaxed), 0);
}

/// LRU beyond `max_loaded = 1`: capacity is enforced before the incoming
/// load, so the replacement swaps the resident out (classic swap). The
/// test handle keeping the evicted pipeline's `Arc` proves in-flight
/// safety — the pipeline lives until its last holder drops.
#[tokio::test]
async fn lru_evicts_beyond_max_loaded() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_secs(3600), 1, counting_loader(&probe, &[]));
    let a = mgr.get("llm:a.gguf", None).await.unwrap();
    let b = mgr.get("llm:b.gguf", None).await.unwrap();
    assert_eq!(probe.loads.load(Ordering::Relaxed), 2);
    drop(a);
    drop(b);
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        1,
        "only the replacement (b) stays resident"
    );
    let a2 = mgr.get("llm:a.gguf", None).await.unwrap();
    assert_eq!(
        probe.loads.load(Ordering::Relaxed),
        3,
        "a was swapped out and must reload"
    );
    drop(a2);
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        1,
        "the incoming load swapped b out again"
    );
}

/// Per-request `ttl_seconds = 0` on a fresh load = use-then-unload (Ollama
/// semantics): the entry is evictable at the very next sweep despite a long
/// configured TTL.
#[tokio::test]
async fn ttl_zero_unloads_at_next_sweep() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_secs(3600), 2, counting_loader(&probe, &[]));
    let pipe = mgr.get("llm:a.gguf", Some(0)).await.unwrap();
    drop(pipe);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        mgr.evict_idle(),
        1,
        "ttl_seconds = 0 evicts at the next sweep"
    );
    assert_eq!(probe.live.load(Ordering::Relaxed), 0);
}

/// `ttl_seconds = 0` cannot thrash an entry another touch shielded: the
/// effective window is `max(remaining, own)` (NEW-8 anti-thrash), so the
/// entry only use-then-unloads once the remaining budget is spent.
#[tokio::test]
async fn ttl_zero_cannot_shrink_a_live_window() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_millis(100), 2, counting_loader(&probe, &[]));
    let pipe = mgr.get("llm:a.gguf", None).await.unwrap(); // default 100ms budget
    drop(pipe);
    let pipe = mgr.get("llm:a.gguf", Some(0)).await.unwrap();
    drop(pipe);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        mgr.evict_idle(),
        0,
        "ttl 0 must not shorten the remaining window"
    );
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(mgr.evict_idle(), 1, "the window decays and the entry frees");
    assert_eq!(probe.live.load(Ordering::Relaxed), 0);
}

/// A `ttl_seconds` override raises the window past the configured default;
/// with `max(remaining, own)` semantics a later default touch neither
/// extends nor shortens it, and once the override budget decays the entry
/// evicts again. (Advice is whole seconds: `Some(1)` = a 1s window against
/// the 100ms configured default.)
#[tokio::test]
async fn ttl_override_extends_then_decays_back_to_default() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_millis(100), 2, counting_loader(&probe, &[]));
    let pipe = mgr.get("llm:a.gguf", Some(1)).await.unwrap();
    drop(pipe);
    std::thread::sleep(Duration::from_millis(300)); // well past the default TTL
    mgr.evict_idle();
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        1,
        "the request override shields the entry past the default TTL"
    );
    let pipe = mgr.get("llm:a.gguf", None).await.unwrap();
    drop(pipe);
    // ~700ms of override budget remains: a default-budget touch keeps it.
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(
        mgr.evict_idle(),
        0,
        "remaining override budget still shields after a default touch"
    );
    std::thread::sleep(Duration::from_millis(500)); // past the remaining budget
    assert_eq!(mgr.evict_idle(), 1, "the window decays back to eviction");
    assert_eq!(probe.live.load(Ordering::Relaxed), 0);
}

/// Over-cap `ttl_seconds` advice is capped by the manager's `max_ttl` even
/// if the route clamp were bypassed — the manager owns the clock (the
/// route-level clamp is covered by `ttl_above_cap_is_clamped_not_rejected`).
#[tokio::test]
async fn ttl_above_manager_cap_is_capped() {
    let probe = Arc::new(Probe::default());
    let mgr = manager_with(
        settings(
            Duration::from_secs(3600),
            2,
            Duration::from_millis(200),
            &[],
        ),
        counting_loader(&probe, &[]),
    );
    let pipe = mgr.get("llm:a.gguf", Some(u64::MAX)).await.unwrap();
    drop(pipe);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(
        mgr.evict_idle(),
        1,
        "window capped at max_ttl, not the requested u64::MAX seconds"
    );
    assert_eq!(probe.live.load(Ordering::Relaxed), 0);
}

/// Pinned slugs ignore per-request `ttl_seconds` entirely and never
/// auto-evict (TTL or LRU capacity trims) — pin outranks client advice.
#[tokio::test]
async fn pinned_slug_ignores_ttl_and_capacity_trims() {
    let probe = Arc::new(Probe::default());
    let mgr = manager_with(
        settings(
            Duration::from_millis(100),
            1,
            Duration::from_hours(24),
            &["llm:pin.gguf"],
        ),
        counting_loader(&probe, &[]),
    );
    let pin = mgr.get("llm:pin.gguf", Some(0)).await.unwrap(); // advice ignored
    drop(pin);
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(mgr.evict_idle(), 0, "pinned entry never TTL-evicts");
    assert_eq!(probe.live.load(Ordering::Relaxed), 1);
    // Capacity `max_loaded = 1` (keep = 0): the pinned resident survives the
    // pre-load trim while the incoming load swaps in.
    let other = mgr.get("llm:other.gguf", None).await.unwrap();
    drop(other);
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        2,
        "pinned resident survives the LRU trim"
    );
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(mgr.evict_idle(), 1, "only the non-pinned entry TTL-evicts");
    assert_eq!(probe.live.load(Ordering::Relaxed), 1);
}

/// Numeric tail of the first exposition line with the given prefix.
fn metric_value(body: &str, prefix: &str) -> u64 {
    body.lines()
        .find(|line| line.starts_with(prefix))
        .unwrap_or_else(|| panic!("missing series line: {prefix}"))
        .rsplit(' ')
        .next()
        .expect("series line has a value")
        .parse()
        .expect("series value parses as u64")
}

/// Swap events emit the kind-labeled metrics (kind `llm` — the manager's
/// only modality): load commit + duration on load, ttl unload on sweep,
/// eviction on the capacity trim.
#[tokio::test]
async fn swap_events_emit_kind_metrics() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_secs(3600), 1, counting_loader(&probe, &[]));
    let pipe = mgr.get("llm:a.gguf", None).await.unwrap();
    drop(pipe);
    let other = mgr.get("llm:b.gguf", Some(0)).await.unwrap(); // use-then-unload; LRU-evicts a
    drop(other);
    let body = metrics::exposition();
    assert!(
        metric_value(
            &body,
            "honkbox_model_loads_total{kind=\"llm\",reason=\"request\"}"
        ) >= 2,
        "both load commits counted"
    );
    assert!(
        metric_value(
            &body,
            "honkbox_model_load_duration_seconds_count{kind=\"llm\"}"
        ) >= 2,
        "both load durations observed"
    );
    assert!(
        metric_value(
            &body,
            "honkbox_model_unloads_total{kind=\"llm\",reason=\"evict\"}"
        ) >= 1,
        "the LRU swap-out counted as an evict unload"
    );
    assert!(
        metric_value(&body, "honkbox_swap_evictions_total{kind=\"llm\"}") >= 1,
        "the LRU swap-out counted on the evictions family"
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        mgr.evict_idle(),
        1,
        "b evicts by TTL for the unload counter"
    );
    let body = metrics::exposition();
    assert!(
        metric_value(
            &body,
            "honkbox_model_unloads_total{kind=\"llm\",reason=\"ttl\"}"
        ) >= 1,
        "the idle sweep counted as a ttl unload"
    );
}

/// Route-level contract: constructor failures 500 with the cache staying
/// consistent, plain unknown slugs still 400, and the warm entry keeps
/// serving. (Slug-kind refusal is the production loader's job — tested at
/// loader level below; the counting loader accepts every slug it is given.)
#[tokio::test]
async fn slug_errors_map_to_status_codes_and_cache_stays_consistent() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let app = swap_app(
        catalog_with(&[
            ("llm:ok.gguf", ModelKind::Llm),
            ("llm:bad.gguf", ModelKind::Llm),
        ]),
        counting_loader(&probe, &["llm:bad.gguf"]),
        Duration::from_secs(3600),
        2,
        Duration::from_hours(24),
    );
    let (status, _) = chat(app.clone(), "llm:ok.gguf", None).await;
    assert_eq!(status, StatusCode::OK, "warm the cache first");
    // Constructor failure: 500 naming the cause (the slug is a valid
    // catalog entry, so validate_model passes and the loader is reached).
    let (status, body) = chat(app.clone(), "llm:bad.gguf", None).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        body.contains("intentional load failure"),
        "load failure: {body}"
    );
    // Unknown slug: still a 400 (validate_model, unchanged).
    let (status, _) = chat(app.clone(), "nope", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // The cache is untouched by the failures: the warm entry answers
    // without a re-construction.
    let (status, _) = chat(app, "llm:ok.gguf", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(probe.loads.load(Ordering::Relaxed), 1);
}

/// The production loader refuses unknown slugs and non-LLM catalog entries
/// with `UnknownSlug` (mapped to 400 at the route) and wraps real backend
/// load failures as `Load` (mapped to 500). `LlamaCppLlm::load` checks the
/// path before initializing the backend, so the fake path errors cleanly.
#[test]
fn catalog_loader_rejects_unknown_and_non_llm_slugs() {
    let catalog = Arc::new(RwLock::new(Arc::new(catalog_with(&[
        ("llm:ok.gguf", ModelKind::Llm),
        ("stt:voice.bin", ModelKind::Stt),
    ]))));
    let loader = swap::catalog_llm_loader(Arc::clone(&catalog));
    let Err(missing) = loader("llm:not-in-catalog.gguf") else {
        panic!("unknown slug must not load");
    };
    assert!(
        matches!(missing, SwapError::UnknownSlug(_)),
        "unknown slug: {missing}"
    );
    let Err(wrong_kind) = loader("stt:voice.bin") else {
        panic!("non-LLM slug must not load");
    };
    assert!(
        matches!(wrong_kind, SwapError::UnknownSlug(_)),
        "non-LLM slug: {wrong_kind}"
    );
    assert!(
        wrong_kind.to_string().contains("not LLM weights"),
        "kind refusal names the modality: {wrong_kind}"
    );
    // A real LLM slug whose file cannot be opened is a load failure, not a
    // slug error (the catalog path here does not exist).
    let Err(load_fail) = loader("llm:ok.gguf") else {
        panic!("a nonexistent path must not load");
    };
    assert!(
        matches!(load_fail, SwapError::Load(_)),
        "bad path: {load_fail}"
    );
}

/// The seeded startup pipeline (main.rs pre-warms the configured slug) is
/// reused without re-construction and evicts like any other entry.
#[tokio::test]
async fn seeded_pipeline_is_reused_then_evicted() {
    let probe = Arc::new(Probe::default());
    let loader = counting_loader(&probe, &[]);
    let mgr = manager(Duration::from_millis(150), 1, Arc::clone(&loader));
    let boot = loader("llm:boot.gguf").unwrap();
    mgr.seed("llm:boot.gguf", boot);
    let pipe = mgr.get("llm:boot.gguf", None).await.unwrap();
    assert_eq!(
        probe.loads.load(Ordering::Relaxed),
        1,
        "seeded entry answers without re-construction"
    );
    drop(pipe);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(mgr.evict_idle(), 1, "seeded entry evicts like any other");
    assert_eq!(probe.live.load(Ordering::Relaxed), 0);
    let _reload = mgr.get("llm:boot.gguf", None).await.unwrap();
    assert_eq!(probe.loads.load(Ordering::Relaxed), 2, "reload on demand");
}

/// The reaper is plain tokio machinery: spawning returns a controllable
/// handle (the sweep never fires during the test — the period for a 1h TTL
/// clamps to 30s), and the observable eviction logic is `evict_idle`,
/// covered directly above.
#[tokio::test]
async fn reaper_task_spawns_and_aborts_cleanly() {
    let probe = Arc::new(Probe::default());
    let mgr = manager(Duration::from_secs(3600), 2, counting_loader(&probe, &[]));
    let handle = swap::spawn_reaper(Arc::clone(&mgr));
    handle.abort();
    let joined = handle.await;
    assert!(
        matches!(joined, Err(ref err) if err.is_cancelled()),
        "aborted reaper: {joined:?}"
    );
    let pipe = mgr.get("llm:a.gguf", None).await.unwrap();
    drop(pipe);
    assert_eq!(
        probe.live.load(Ordering::Relaxed),
        1,
        "manager unaffected by the aborted reaper"
    );
}

/// `/v1/chat/stream` resolves through the swap manager the same way.
#[tokio::test]
async fn chat_stream_resolves_through_swap_manager() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let app = swap_app(
        catalog_with(&[("llm:fake.gguf", ModelKind::Llm)]),
        counting_loader(&probe, &[]),
        Duration::from_secs(3600),
        1,
        Duration::from_hours(24),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/stream")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "llm:fake.gguf",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        body.contains("\"token\":\"hello \""),
        "streamed tokens: {body}"
    );
    assert!(body.contains("[DONE]"), "terminator present: {body}");
    assert_eq!(probe.loads.load(Ordering::Relaxed), 1);
}

/// Stub path unaffected: with no swap manager the static stub pipeline
/// answers and nothing else changes.
#[tokio::test]
async fn stub_path_unaffected_without_swap_manager() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let app = app(AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog_with(&[])))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: None,
        tts_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        swap: None,
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        downloads: std::sync::Arc::new(
            downloads::DownloadManager::new(None).expect("download manager"),
        ),
    });
    let (status, body) = chat(app, "", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"stub-llm\""), "stub reply: {body}");
}

/// Route-level clamp: `ttl_seconds` above `[swap] max_ttl_secs` is clamped
/// with a `warn!` and the request is served — never a 400.
#[tokio::test]
async fn ttl_above_cap_is_clamped_not_rejected() {
    let _slot_guard = SLOT_LOCK.lock().await;
    let probe = Arc::new(Probe::default());
    let app = swap_app(
        catalog_with(&[("llm:fake.gguf", ModelKind::Llm)]),
        counting_loader(&probe, &[]),
        Duration::from_secs(3600),
        2,
        Duration::from_secs(1),
    );
    let (status, body) = chat(app, "llm:fake.gguf", Some(999_999)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "over-cap ttl clamps, never rejects: {body}"
    );
    assert_eq!(probe.loads.load(Ordering::Relaxed), 1);
}
