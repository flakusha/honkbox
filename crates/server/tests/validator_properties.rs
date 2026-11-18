//! Contract-hardening property tests (proptest) for the server's request
//! validators and their HTTP contract, driven through the actual routes
//! (`tower::ServiceExt::oneshot`, stub backends) so the asserted behavior is
//! the real validator → status/code mapping, not a copy of it.
//!
//! Surfaces and their invariants:
//! - `POST /v1/embeddings`: JSON-shape / model-resolution / capability /
//!   input checks — every branch maps to its named rejection, valid input
//!   passes with order-preserving echoes, nothing panics;
//! - `POST /v1/images/generations`: size, `response_format`, recipe /
//!   sampler / scheduler / `cfg_scale` / `steps`, hires-fix, `output_format`
//!   and `output_compression` — out-of-range or unknown values are rejected
//!   BY NAME (never silently clamped), `n` is the documented deliberate
//!   clamp, and the output-size cap is enforced by the BACKEND, not the
//!   validator (an over-cap size is a 500 `generation_failed`, never a 400
//!   `invalid_size`);
//! - recipe precedence (explicit field > recipe default > server default)
//!   lands verbatim in the job-bundle sidecar;
//! - `POST /v1/chat/completions`: `validate_model` semantics, the
//!   `ttl_seconds` deliberate clamp (over-cap advice is clamped with a
//!   warn, NEVER a rejection), and the widened `ChatMessage.content`
//!   contract (string echo, unknown part type named in the 400, malformed
//!   variants mapped to named errors, never a panic);
//! - `POST /v1/downloads` repo validation: every implausible `org/name`
//!   shape is rejected with the named `BadRepo` message — the ACCEPT side
//!   cannot be asserted here because a plausible repo proceeds to a real
//!   `HuggingFace` metadata fetch (network), so only the reject side is
//!   property-tested; there is no `@rev` support (as built, `@` is outside
//!   the charset and correctly rejected).
//!
//! # Resource contract (parallel-safety)
//!
//! This is a MARKED-SERIAL integration suite: the routes under test all
//! reserve the process-wide residency slot (`ledger::global()`), which is
//! one shared GPU budget per process — the same shared-resource shape as
//! `tests/embeddings.rs`/`tests/completions.rs`. The one deliberate
//! cross-test coupling is `SLOT_LOCK` (a std mutex held in sync frames;
//! poison-healed so a failed test cannot wedge the others). Everything
//! else is per-test: fixtures are read-only `LazyLock` routers, proptest
//! cases are independent, downloads/recipe requests never mutate manager
//! state (rejects happen before insertion), and the recipe sidecar uses a
//! pid-unique tmp dir released by a drop guard (also on panic).
//! Every test passes alone and in any order; run `-j 1` for a
//! deterministic shakeout, parallel `-j` for speed (the lock keeps
//! routed cases mutually exclusive, other tests scale out).
//!
//! proptest macro note: `prop_assert_eq!` wraps its message through
//! `concat!`, which disables implicit format-args capture — assertions
//! there stay message-less (proptest prints the shrunk case anyway), while
//! `prop_assert!` keeps its contextual messages.

// No lib target on this binary crate: include the modules directly (the
// `tests/embeddings.rs` pattern).
#[allow(dead_code)]
#[path = "../src/admission.rs"]
mod admission;
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

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use proptest::prelude::*;
use routes::{AppConfig, app};
use serde_json::{Value, json};
use tower::ServiceExt as _;

/// Serializes the process-global GPU slot across this binary's tests
/// (parallel routed requests would otherwise race the one slot into
/// spurious 429s — same pattern as `completions.rs`/`embeddings.rs`).
/// The test GPU-slot lock: a plain `std` mutex acquired in the SYNC proptest
/// frame (never across an await) so the shared current-thread runtime never
/// blocks inside `block_on` on a tokio-internal wakeup — 11 concurrent
/// `block_on` callers parking on a tokio Mutex starve each other forever.
static SLOT_LOCK: LazyLock<std::sync::Mutex<()>> = LazyLock::new(|| std::sync::Mutex::new(()));

/// One current-thread runtime shared by the (sync) proptest case bodies.
static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
});

/// POST `body` to `uri`, return `(status, parsed JSON)` (null for non-JSON
/// bodies). Lock-free: callers hold `SLOT_LOCK` (sync bodies hold the std
/// guard across `block_on`; `#[tokio::test]` bodies do the same).
async fn post_json(router: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("oneshot");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .expect("body reads");
    let parsed = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, parsed)
}

/// Sync-side request helper for proptest bodies: acquires the GPU-slot std
/// guard, then drives the lock-free async POST on the shared current-thread
/// runtime. The guard lives in the SYNC frame (never across an await).
fn post_json_sync(router: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let _slot = SLOT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    RT.block_on(post_json(router, uri, body))
}

/// POST without the slot lock, returning the RAW body (downloads never
/// touch the GPU slot and their errors are plain text, not envelopes).
async fn post_text(router: &Router, uri: &str, body: &Value) -> (StatusCode, String) {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("oneshot");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .expect("body reads");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// `error.code` of an OpenAI-shaped error envelope (empty when the body is
/// not an envelope, e.g. an axum Json rejection's plain text).
fn error_code(payload: &Value) -> String {
    payload["error"]["code"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// `error.message` of an OpenAI-shaped error envelope.
fn error_message(payload: &Value) -> String {
    payload["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// `true` when `{s:?}` renders without escapes, so a
/// `message.contains(s)` assertion is exact (Debug escapes quotes,
/// backslashes and control chars).
fn debug_safe(s: &str) -> bool {
    s.chars().all(|c| !c.is_control() && c != '"' && c != '\\')
}

/// One fake catalog entry (kind decides the inferred defaults; the explicit
/// flags win, mirroring the REQ-2 override semantics).
fn entry(
    id: &str,
    slug: &str,
    kind: ModelKind,
    chat: bool,
    embeddings: bool,
    vision: bool,
) -> ModelCatalogEntry {
    let mut capabilities = ModelCapabilities::for_kind(kind);
    capabilities.chat = chat;
    capabilities.completion = chat;
    capabilities.embeddings = embeddings;
    capabilities.vision = vision;
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: slug.to_owned(),
        kind,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities,
        category: "root".to_owned(),
        path: format!("/models/{slug}"),
        bytes: 1,
        metadata: None,
    }
}

/// Config skeleton shared by all fixtures (stub pipelines everywhere, no
/// auth, no swap unless the caller installs one).
fn base_config(catalog: Vec<ModelCatalogEntry>, data_dir: Option<PathBuf>) -> AppConfig {
    AppConfig {
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(ModelCatalog {
            object: "list".to_owned(),
            data: catalog,
        }))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
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
        swap_max_ttl: Duration::from_secs(3600),
        data_dir,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
        auth: config::AuthConfig::default(),
    }
}

/// Embeddings fixture: an embeddings-capable model with an alias plus a
/// chat-only model (the alias, the raw slug and the incapable model cover
/// every resolution branch of `validate_embeddings_request`).
static EMBEDDINGS_APP: LazyLock<Router> = LazyLock::new(|| {
    app(base_config(
        vec![
            entry(
                "emb-alias",
                "emb-model",
                ModelKind::Embedding,
                false,
                true,
                false,
            ),
            entry(
                "chat-model",
                "chat-model",
                ModelKind::Llm,
                true,
                false,
                false,
            ),
        ],
        None,
    ))
});

fn embeddings_app() -> Router {
    EMBEDDINGS_APP.clone()
}

/// Completions fixture: a chat-capable model with an alias, served through
/// an ACTIVE swap manager with a stub loader — required because the
/// `ttl_seconds` clamp is only observable on the swap path.
static COMPLETIONS_APP: LazyLock<Router> = LazyLock::new(|| {
    let loader: swap::LlmLoader = Arc::new(|_slug| {
        Ok(Arc::new(honkbox_llm_worker::StubLlm::default())
            as Arc<dyn honkbox_llm_worker::LlmPipeline>)
    });
    let mut config = base_config(
        vec![entry(
            "chat-alias",
            "chat-model",
            ModelKind::Llm,
            true,
            false,
            false,
        )],
        None,
    );
    config.swap = Some(Arc::new(swap::SwapManager::new(
        swap::SwapSettings {
            llm_ttl: Duration::from_secs(600),
            llm_max_loaded: 2,
            max_ttl: Duration::from_secs(3600),
            pinned: HashSet::new(),
        },
        loader,
    )));
    app(config)
});

fn completions_app() -> Router {
    COMPLETIONS_APP.clone()
}

/// Images fixture: no catalog, stub SD pipeline (the validator properties
/// exercise the request surface; the stub renders deterministic pixels).
static IMAGES_APP: LazyLock<Router> = LazyLock::new(|| app(base_config(Vec::new(), None)));

fn images_app() -> Router {
    IMAGES_APP.clone()
}

/// Images fixture with a per-test data dir (only the sidecar-precedence
/// test needs one; the caller owns cleanup).
fn images_app_with_dir(data_dir: PathBuf) -> Router {
    app(base_config(Vec::new(), Some(data_dir)))
}

/// Downloads fixture (`models_root: None` — only the repo-validation
/// rejection path is asserted; a plausible repo would proceed to a real
/// `HuggingFace` metadata fetch, which tests must not perform).
static DOWNLOADS_APP: LazyLock<Router> = LazyLock::new(|| app(base_config(Vec::new(), None)));

fn downloads_app() -> Router {
    DOWNLOADS_APP.clone()
}

// ---------------------------------------------------------------------
// Reference models (independent re-derivations of the documented rules).
// ---------------------------------------------------------------------

/// `POST /v1/embeddings` reference decision: `(status, code)` — mirrors
/// `validate_embeddings_request`'s branch order (JSON shape → model
/// resolution → capability → inputs).
fn expected_embeddings_outcome(model: &Value, input: &Value) -> (u16, &'static str) {
    let Some(model_str) = model.as_str() else {
        return (400, "invalid_json_body");
    };
    // `EmbeddingsInput` is an untagged string | array-of-strings; anything
    // else fails deserialization → the `invalid_json_body` envelope.
    let inputs: Option<Vec<&str>> = match input {
        Value::String(s) => Some(vec![s]),
        Value::Array(items) if items.iter().all(Value::is_string) => Some(
            items
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>()
                .expect("all strings"),
        ),
        _ => None,
    };
    let Some(inputs) = inputs else {
        return (400, "invalid_json_body");
    };
    let trimmed = model_str.trim();
    let capable = trimmed == "emb-model" || trimmed == "emb-alias";
    if !(capable || trimmed == "chat-model") {
        return (404, "model_not_found");
    }
    if !capable {
        return (400, "not_embeddings_capable");
    }
    if inputs.is_empty() || inputs.iter().any(|s| s.is_empty()) {
        return (400, "input_required");
    }
    (200, "")
}

/// `parse_size` reference: `WxH` (either case separator), both parseable as
/// u32, both nonzero.
fn parse_size_ref(raw: &str) -> Option<(u32, u32)> {
    let (w, h) = raw.split_once(['x', 'X'])?;
    let w = w.parse::<u32>().ok()?;
    let h = h.parse::<u32>().ok()?;
    (w > 0 && h > 0).then_some((w, h))
}

/// `validate_hires_fix` reference decision for a parsed `hires_fix` object.
/// The f64→f32 narrowing mirrors the route's own `Option<f32>`
/// deserialization, so the truncation is part of the mirrored contract.
#[allow(clippy::cast_possible_truncation)]
fn hires_fix_ref(hires: &Value, base: (u32, u32)) -> Result<(), &'static str> {
    let Some(object) = hires.as_object() else {
        return Err("invalid_hires_fix_shape");
    };
    match object.get("scale") {
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(f64::NAN) as f32;
            if !((1.0..=4.0).contains(&v) && v.is_finite()) {
                return Err("invalid_hires_scale");
            }
        }
        Some(_) => return Err("invalid_hires_fix_shape"),
        None => {}
    }
    let target = match object.get("target") {
        Some(Value::String(raw)) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(parse_size_ref(trimmed).ok_or("invalid_hires_target")?)
            }
        }
        Some(_) => return Err("invalid_hires_fix_shape"),
        None => None,
    };
    match object.get("denoise") {
        Some(Value::Number(n)) => {
            let v = n.as_f64().unwrap_or(f64::NAN) as f32;
            if !(v.is_finite() && v > 0.0 && v <= 1.0) {
                return Err("invalid_hires_denoise");
            }
        }
        Some(_) => return Err("invalid_hires_fix_shape"),
        None => {}
    }
    // Defaults the validator applies for omitted fields (scale 2.0,
    // denoise 0.7; target overrides scale).
    let scale = object
        .get("scale")
        .map_or(2.0, |v| v.as_f64().unwrap_or(2.0) as f32);
    // The final output edge (target wins over scale) must stay under the
    // sd-core cap — the validator enforces it for the hires pass.
    // Scale-scaled edges: `max(1.0)` proves positivity, so the u32 casts
    // cannot lose sign.
    #[allow(clippy::cast_sign_loss)]
    let (tw, th) = target.unwrap_or((
        (f64::from(base.0) * f64::from(scale)).round().max(1.0) as u32,
        (f64::from(base.1) * f64::from(scale)).round().max(1.0) as u32,
    ));
    let edge = base.0.max(base.1).max(tw).max(th);
    if edge > honkbox_sd_core::MAX_EDGE {
        return Err("invalid_hires_target");
    }
    Ok(())
}

/// `POST /v1/images/generations` reference decision over a type-correct
/// body: `Ok(expected n)` or `Err(400 code)` — mirrors
/// `validate_images_request`'s field order (first failure wins). The
/// f64→f32 narrowing mirrors the route's `Option<f32>` deserialization.
#[allow(clippy::cast_possible_truncation)]
fn expected_images_outcome(body: &Value) -> Result<u32, &'static str> {
    const KNOWN_RECIPES: [&str; 7] = [
        "sd15",
        "sdxl",
        "sd35",
        "turbo",
        "lcm",
        "flux-dev",
        "flux-schnell",
    ];
    let size = body["size"].as_str().map_or(Ok((1024, 1024)), |raw| {
        parse_size_ref(raw).ok_or("invalid_size")
    })?;
    if let Some(format) = body["response_format"].as_str()
        && format.trim() != "b64_json"
    {
        return Err("unsupported_response_format");
    }
    let n = body["n"].as_u64().unwrap_or(1);
    let recipe = body["recipe"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if let Some(name) = recipe
        && !KNOWN_RECIPES.contains(&name)
    {
        return Err("unknown_recipe");
    }
    let samplers = honkbox_sd_core::SAMPLER_NAMES;
    if let Some(name) = body["sampler"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        && !samplers.contains(&name)
    {
        return Err("unknown_sampler");
    }
    let schedulers = honkbox_sd_core::SCHEDULER_NAMES;
    if let Some(name) = body["scheduler"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        && !schedulers.contains(&name)
    {
        return Err("unknown_scheduler");
    }
    if let Some(cfg) = body["cfg_scale"].as_f64() {
        let cfg = cfg as f32;
        if !(1.0..=30.0).contains(&cfg) || !cfg.is_finite() {
            return Err("invalid_cfg_scale");
        }
    }
    if let Some(steps) = body["steps"].as_u64()
        && !(1..=100).contains(&steps)
    {
        return Err("invalid_steps");
    }
    if body.get("hires_fix").is_some_and(Value::is_object) {
        hires_fix_ref(&body["hires_fix"], size)?;
    }
    let format = body["output_format"].as_str().unwrap_or("png").trim();
    if !["png", "jpeg", "webp"].contains(&format) {
        return Err("unsupported_output_format");
    }
    if let Some(compression) = body.get("output_compression")
        && !compression.is_null()
        && !matches!(compression.as_u64(), Some(q) if q <= 100)
    {
        return Err("invalid_output_compression");
    }
    Ok(u32::try_from(n).unwrap_or(u32::MAX).clamp(1, 4))
}

/// `DownloadManager::validate_repo` reference predicate: exactly one `/`,
/// both parts nonempty, ≤96 bytes, no leading dot, charset
/// `[A-Za-z0-9._-]`. Note: there is no `@rev` support — `@` is outside the
/// charset, so `org/name@rev` is (correctly, as built) a rejection.
fn plausible_repo(repo: &str) -> bool {
    let plausible = |part: &str| {
        !part.is_empty()
            && part.len() <= 96
            && !part.starts_with('.')
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    matches!(repo.split_once('/'), Some((owner, name)) if plausible(owner) && plausible(name))
}

/// A compact arbitrary `serde_json::Value` (`Value` itself is not
/// `Arbitrary`): scalars, strings, null, and small number arrays.
fn arbitrary_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        2 => any::<u64>().prop_map(|v| json!(v)),
        1 => any::<i64>().prop_map(|v| json!(v)),
        1 => (any::<f64>()).prop_map(|v| json!(v)),
        1 => any::<bool>().prop_map(|v| json!(v)),
        1 => (any::<String>()).prop_map(|v| json!(v)),
        1 => Just(Value::Null),
        1 => proptest::collection::vec(any::<u64>(), 0..3).prop_map(|v| json!(v)),
    ]
}

// ---------------------------------------------------------------------
// Embeddings validator properties.
// ---------------------------------------------------------------------

fn embeddings_body_strategy() -> impl Strategy<Value = serde_json::Value> {
    let model = prop_oneof![
        2 => Just(json!("emb-model")),
        2 => Just(json!("emb-alias")),
        2 => Just(json!(" emb-model ")),
        2 => Just(json!("chat-model")),
        2 => (any::<String>()).prop_map(|s| json!(s)),
        1 => Just(json!("")),
        1 => Just(json!(42u64)),
    ];
    let input = prop_oneof![
        3 => (any::<String>()).prop_map(|s| json!(s)),
        3 => (any::<String>(), any::<String>()).prop_map(|(a, b)| json!([a, b])),
        1 => Just(json!(["a", ""])),
        1 => Just(json!([])),
        1 => Just(json!(42u64)),
        1 => Just(json!(null)),
        1 => Just(json!({"deep": [1]})),
    ];
    (model, input)
        .prop_map(|(model, input)| json!({"model": model, "input": input}))
        // The 404 assertion quotes the model via `{model:?}`; keep the
        // generated strings Debug-unescaped so `contains` is exact.
        .prop_filter("debug-safe model", |body| {
            debug_safe(body["model"].as_str().unwrap_or_default())
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Every embeddings request lands on exactly the documented branch:
    /// shape → resolution → capability → inputs, with the named rejection
    /// code and (for unknown models) the offending model quoted.
    #[test]
    fn embeddings_validation_contract(body in embeddings_body_strategy()) {
        let router = embeddings_app();
        let (expected_status, expected_code) =
            expected_embeddings_outcome(&body["model"], &body["input"]);
        let (status, payload) = post_json_sync(&router, "/v1/embeddings", &body);
        prop_assert_eq!(status.as_u16(), expected_status);
        prop_assert_eq!(error_code(&payload), expected_code);
        if expected_code == "model_not_found" {
            let model = body["model"].as_str().unwrap_or_default().trim();
            prop_assert!(
                error_message(&payload).contains(&format!("{model:?}")),
                "the 404 must name the model: {}", error_message(&payload)
            );
        }
        if expected_status == 200 {
            let inputs = match &body["input"] {
                Value::String(_) => 1usize,
                Value::Array(items) => items.len(),
                _ => 0,
            };
            let data = payload["data"].as_array().expect("data array").clone();
            prop_assert_eq!(data.len(), inputs);
            for (index, item) in data.iter().enumerate() {
                prop_assert_eq!(&item["index"], &json!(index));
                prop_assert!(!item["embedding"].as_array().expect("vector").is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------
// Images validator properties.
// ---------------------------------------------------------------------

// One strategy per field; the tuple fold assembles the request body.
#[allow(clippy::too_many_lines)]
fn images_body_strategy() -> impl Strategy<Value = serde_json::Value> {
    // Sizes stay ≤ 256 px so accepted cases render cheaply against the
    // stub; the over-cap backend behavior is covered by a targeted test.
    let size = prop_oneof![
        3 => Just(Value::Null),
        2 => (1u32..=256u32, 1u32..=256u32).prop_map(|(w, h)| Value::String(format!("{w}x{h}"))),
        1 => Just(json!(" 128x128 ")),
        1 => Just(json!("0x64")),
        1 => Just(json!("64x0")),
        1 => Just(json!("-3x64")),
        2 => (any::<String>()).prop_map(|s| json!(s)),
    ];
    let n = prop_oneof![3 => Just(Value::Null), 3 => any::<u32>().prop_map(|v| json!(v))];
    let response_format = prop_oneof![
        3 => Just(Value::Null),
        2 => Just(json!("b64_json")),
        1 => Just(json!(" b64_json ")),
        1 => (any::<String>()).prop_map(|s| json!(s)),
    ];
    let output_format = prop_oneof![
        3 => Just(Value::Null),
        1 => Just(json!("png")),
        1 => Just(json!("jpeg")),
        1 => Just(json!("webp")),
        1 => Just(json!("jpg")),
        1 => Just(json!("")),
        1 => (any::<String>()).prop_map(|s| json!(s)),
    ];
    let compression = prop_oneof![
        3 => Just(Value::Null),
        2 => (0u64..=100u64).prop_map(|v| json!(v)),
        1 => (101u64..u64::MAX).prop_map(|v| json!(v)),
        1 => Just(json!(-1)),
        1 => Just(json!(3.5)),
        1 => Just(json!("90")),
        1 => Just(json!(true)),
    ];
    let recipe = prop_oneof![
        3 => Just(Value::Null),
        3 => proptest::sample::select(vec![
            "sd15", "sdxl", "sd35", "turbo", "lcm", "flux-dev", "flux-schnell",
        ])
        .prop_map(|s| Value::String(s.to_owned())),
        2 => Just(json!("")),
        2 => Just(json!(" sdxl ")),
        2 => (any::<String>()).prop_map(|s| json!(s)),
    ];
    let sampler = prop_oneof![
        3 => Just(Value::Null),
        2 => proptest::sample::select(
            honkbox_sd_core::SAMPLER_NAMES.to_vec(),
        )
        .prop_map(|s| Value::String(s.to_owned())),
        2 => (any::<String>()).prop_map(|s| json!(s)),
    ];
    let scheduler = prop_oneof![
        3 => Just(Value::Null),
        2 => proptest::sample::select(
            honkbox_sd_core::SCHEDULER_NAMES.to_vec(),
        )
        .prop_map(|s| Value::String(s.to_owned())),
        2 => (any::<String>()).prop_map(|s| json!(s)),
    ];
    let cfg_scale = prop_oneof![
        3 => Just(Value::Null),
        2 => (1.0f64..=30.0f64).prop_map(|v| json!(v)),
        1 => Just(json!(0.999)),
        1 => Just(json!(30.001)),
        1 => Just(json!(-5.0)),
        1 => (any::<f64>()).prop_map(|v| json!(v)),
    ];
    let steps = prop_oneof![
        3 => Just(Value::Null),
        2 => (1u64..=100u64).prop_map(|v| json!(v)),
        1 => Just(json!(0u64)),
        1 => (101u64..=u64::from(u32::MAX)).prop_map(|v| json!(v)),
    ];
    let hires = prop_oneof![
        6 => Just(Value::Null),
        2 => Just(json!({})),
        2 => (1.0f64..=4.0f64).prop_map(|v| json!({"scale": v})),
        1 => Just(json!({"scale": 4.5})),
        1 => Just(json!({"scale": 0.5})),
        2 => Just(json!({"target": "192x192"})),
        1 => Just(json!({"target": " 192x192 "})),
        1 => Just(json!({"target": "192"})),
        1 => Just(json!({"target": "0x192"})),
        1 => Just(json!({"target": "4096x4096"})),
        2 => (0.0f64..=1.0f64).prop_map(|v| json!({"denoise": v})),
        1 => Just(json!({"denoise": 0.0})),
        1 => Just(json!({"denoise": 1.5})),
        1 => Just(json!({"denoise": -0.5})),
        1 => Just(json!({"scale": 2.0, "denoise": 0.5})),
        1 => Just(json!({"scale": 3.0, "denoise": 2.0})),
        1 => Just(json!({"target": "4096x2", "scale": 1.0})),
    ];
    (
        size,
        n,
        response_format,
        output_format,
        compression,
        recipe,
        sampler,
        scheduler,
        cfg_scale,
        steps,
        hires,
    )
        .prop_map(
            |(
                size,
                n,
                response_format,
                output_format,
                compression,
                recipe,
                sampler,
                scheduler,
                cfg_scale,
                steps,
                hires,
            )| {
                let mut body = json!({"prompt": "property test"});
                for (key, value) in [
                    ("size", size),
                    ("n", n),
                    ("response_format", response_format),
                    ("output_format", output_format),
                    ("output_compression", compression),
                    ("recipe", recipe),
                    ("sampler", sampler),
                    ("scheduler", scheduler),
                    ("cfg_scale", cfg_scale),
                    ("steps", steps),
                    ("hires_fix", hires),
                ] {
                    if !value.is_null() {
                        body[key] = value;
                    }
                }
                body
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// The images contract: every out-of-range/unknown value is rejected by
    /// name with the exact documented code, every in-range value passes
    /// with `n` deliberately clamped to [1, 4] (never rejected), and
    /// accepted requests render exactly `n` images. Cases are reduced from
    /// 256: each accepted case renders real stub pixels + PNG encoding.
    #[test]
    fn images_validation_contract(body in images_body_strategy()) {
        let router = images_app();
        let outcome = expected_images_outcome(&body);
        let (status, payload) = post_json_sync(&router, "/v1/images/generations", &body);
        match outcome {
            Ok(expected_n) => {
                prop_assert_eq!(status.as_u16(), 200u16);
                let data = payload["data"].as_array().expect("data array").clone();
                prop_assert_eq!(data.len(), usize::try_from(expected_n).unwrap_or(0));
                for image in &data {
                    prop_assert!(!image["b64_json"].as_str().expect("b64").is_empty());
                }
            }
            Err(expected_code) => {
                prop_assert_eq!(status.as_u16(), 400u16);
                prop_assert_eq!(error_code(&payload), expected_code);
            }
        }
    }

    /// `response_format` accepts only `b64_json` (trimmed), naming any
    /// other value in the 400.
    #[test]
    fn images_response_format_is_b64_json_only(
        value in (any::<String>()).prop_filter("debug-safe", |value| debug_safe(value)),
    ) {
        let router = images_app();
        let body = json!({"prompt": "p", "response_format": value});
        let (status, payload) = post_json_sync(&router, "/v1/images/generations", &body);
        if value.trim() == "b64_json" {
            prop_assert_eq!(status.as_u16(), 200u16);
        } else {
            prop_assert_eq!(status.as_u16(), 400u16);
            prop_assert_eq!(error_code(&payload), "unsupported_response_format");
            // The route names the value with `{:?}` (Debug), which escapes
            // non-printable characters; accept raw or Debug-escaped form.
            let trimmed = value.trim();
            let message = error_message(&payload);
            prop_assert!(
                message.contains(trimmed) || message.contains(&format!("{trimmed:?}")),
                "must name the value: {message}"
            );
        }
    }

    /// `output_compression` accepts exactly an integer JSON number in
    /// [0, 100]; anything else (fractional, negative, over 100, string,
    /// bool, object) is a named 400, never a clamp. A JSON `null` counts
    /// as the field being absent (serde `Option<Value>`).
    #[test]
    fn images_output_compression_rejects_without_clamping(raw in arbitrary_value()) {
        let router = images_app();
        let body = json!({"prompt": "p", "output_compression": raw});
        let (status, payload) = post_json_sync(&router, "/v1/images/generations", &body);
        if raw.is_null() || matches!(raw.as_u64(), Some(q) if q <= 100) {
            prop_assert_eq!(status.as_u16(), 200u16);
        } else {
            prop_assert_eq!(status.as_u16(), 400u16);
            prop_assert_eq!(error_code(&payload), "invalid_output_compression");
            prop_assert!(error_message(&payload).contains("rejected, not clamped"));
        }
    }
}

// ---------------------------------------------------------------------
// Chat completions: validate_model, ttl clamp, content-parts contract.
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// `validate_model` on the completions surface: empty/whitespace → 400
    /// `model_required`, anything else (unknown names included) → 404
    /// `model_not_found` quoting the trimmed value.
    #[test]
    fn completions_model_contract(
        model in (any::<String>()).prop_filter(
            "known names and debug-safe strings only",
            |model| debug_safe(model.trim()) && !matches!(model.trim(), "chat-model" | "chat-alias"),
        ),
    ) {
        let router = completions_app();
        let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
        let (status, payload) = post_json_sync(&router, "/v1/chat/completions", &body);
        let trimmed = model.trim();
        if trimmed.is_empty() {
            prop_assert_eq!(status.as_u16(), 400u16);
            prop_assert_eq!(error_code(&payload), "model_required");
        } else {
            prop_assert_eq!(status.as_u16(), 404u16);
            prop_assert_eq!(error_code(&payload), "model_not_found");
            prop_assert!(
                error_message(&payload).contains(&format!("{trimmed:?}")),
                "404 must name the model: {}", error_message(&payload)
            );
        }
    }

    /// Known models (alias AND raw slug, including padded) resolve to 200
    /// with the stub reply — the positive half of `validate_model`.
    #[test]
    fn completions_known_model_names_resolve(
        model in proptest::sample::select(vec![
            "chat-model", "chat-alias", " chat-alias ", "chat-model ",
        ]),
    ) {
        let router = completions_app();
        let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
        let (status, payload) = post_json_sync(&router, "/v1/chat/completions", &body);
        prop_assert_eq!(status.as_u16(), 200u16);
        prop_assert_eq!(&payload["choices"][0]["message"]["role"], &json!("assistant"));
    }

    /// The `ttl_seconds` advice is the documented deliberate clamp: ANY
    /// value — including far over the cap and `u64::MAX` — is clamped with
    /// a warn and never rejected (200, identical to the under-cap case).
    #[test]
    fn ttl_seconds_is_clamped_never_rejected(ttl in any::<u64>()) {
        let router = completions_app();
        let body = json!({
            "model": "chat-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "ttl_seconds": ttl,
        });
        let (status, payload) = post_json_sync(&router, "/v1/chat/completions", &body);
        prop_assert_eq!(status.as_u16(), 200u16);
        prop_assert_eq!(&payload["choices"][0]["finish_reason"], &json!("stop"));
    }

    /// The string content form echoes byte-identically through the whole
    /// route stack: the stub reply is `stub reply to: {content}` — the
    /// assertion pins the EXACT flattened string for arbitrary content.
    #[test]
    fn content_string_echoes_byte_identically(
        content in (0u64..64).prop_map(|seed| {
            const ATOMS: [&str; 10] = [
                "", "a", "hi there", "\n", "\t", "é", "🎉", "語", "  spaced  ", "data:x",
            ];
            let mut state = seed ^ 0x6C6E;
            let mut text = String::new();
            for _ in 0..=(seed % 5) {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let idx = usize::try_from(state % u64::try_from(ATOMS.len()).unwrap_or(u64::MAX))
                    .unwrap_or(0);
                text.push_str(ATOMS[idx]);
            }
            text
        }),
    ) {
        let router = completions_app();
        let body = json!({
            "model": "chat-alias",
            "messages": [{"role": "user", "content": content}],
        });
        let (status, payload) = post_json_sync(&router, "/v1/chat/completions", &body);
        prop_assert_eq!(status.as_u16(), 200u16);
        let expected = json!(format!("stub reply to: {content}"));
        prop_assert_eq!(&payload["choices"][0]["message"]["content"], &expected);
    }

    /// The parts-array contract on the route: text parts flatten and pass
    /// the vision gate, image parts hit the explicit vision rejection (the
    /// fixture model is not vision-capable), and every malformed variant —
    /// unknown type, missing/wrong-typed fields, non-object parts, nested
    /// arrays — maps to a named 400 that quotes the offender, never a
    /// panic and never a silent accept.
    #[test]
    fn content_parts_contract(
        part in prop_oneof![
            4 => (any::<String>()).prop_map(|t| json!({"type": "text", "text": t})),
            2 => proptest::string::string_regex("[a-zA-Z0-9_ ./-]{0,24}")
                .expect("type-name regex")
                .prop_filter("known names", |t| !matches!(t.as_str(), "text" | "image_url"))
                .prop_map(|t| json!({"type": t, "text": "x"})),
            1 => Just(json!({"type": "text"})),
            1 => Just(json!({"type": "text", "text": 42})),
            1 => Just(json!({"type": "image_url", "image_url": {}})),
            1 => Just(json!({"type": "image_url", "image_url": {"url": 42}})),
            1 => Just(json!("bare string part")),
            1 => Just(json!(42)),
            1 => Just(json!([{"type": "text", "text": "nested"}])),
            1 => Just(json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}})),
        ],
    ) {
        let router = completions_app();
        let body = json!({
            "model": "chat-alias",
            "messages": [{"role": "user", "content": [part]}],
        });
        let (status, payload) = post_json_sync(&router, "/v1/chat/completions", &body);
        let message = error_message(&payload);
        match part["type"].as_str() {
            Some("text") => {
                if part["text"].is_string() {
                    prop_assert!(status.as_u16() == 200u16, "expected 200, got {} for {part}", status);
                } else {
                    prop_assert!(status.as_u16() == 400u16, "expected 400, got {} for {part}", status);
                    prop_assert!(
                        message.contains("text part requires a string `text` field"),
                        "{message}"
                    );
                }
            }
            Some("image_url") => {
                prop_assert!(status.as_u16() == 400u16, "expected 400, got {} for {part}", status);
                if part["image_url"]["url"].is_string() {
                    prop_assert_eq!(error_code(&payload), "vision_unsupported");
                } else {
                    prop_assert!(
                        message.contains("image_url part requires a string `image_url.url` field"),
                        "{message}"
                    );
                }
            }
            Some(other) => {
                prop_assert!(status.as_u16() == 400u16, "expected 400, got {} for {part}", status);
                prop_assert!(message.contains("unsupported content part type"), "{message}");
                // Named via `{:?}` in api-types; accept raw or Debug-escaped.
                prop_assert!(
                    message.contains(other) || message.contains(&format!("{other:?}")),
                    "must name the type: {message}"
                );
            }
            None if part.is_object() => {
                prop_assert!(status.as_u16() == 400u16, "expected 400, got {} for {part}", status);
                prop_assert!(
                    message.contains("content part requires a string `type` field"),
                    "{message}"
                );
            }
            None => {
                // Non-object parts and nested arrays.
                prop_assert!(status.as_u16() == 400u16, "expected 400, got {} for {part}", status);
                prop_assert!(
                    message.contains("content part must be an object with a `type` field"),
                    "{message}"
                );
            }
        }
    }

    /// Scalar (non string/null/array) content is a named 400, never a
    /// panic and never a silent flatten.
    #[test]
    fn content_scalar_is_a_named_400(content in prop_oneof![
        1 => Just(json!(42)),
        1 => Just(json!(true)),
        1 => Just(json!({"type": "text", "text": "x"})),
        1 => Just(json!(1.5)),
    ]) {
        let router = completions_app();
        let body = json!({
            "model": "chat-alias",
            "messages": [{"role": "user", "content": content}],
        });
        let (status, payload) = post_json_sync(&router, "/v1/chat/completions", &body);
        prop_assert_eq!(status.as_u16(), 400u16);
        prop_assert!(
            error_message(&payload)
                .contains("content must be a string, null, or an array of content parts"),
            "{}", error_message(&payload)
        );
    }
}

// ---------------------------------------------------------------------
// Downloads repo validation (reject-side property; the accept side
// performs a real HF metadata fetch and must not run in tests).
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Every implausible `org/name` repo string is rejected with the named
    /// `BadRepo` message quoting the repo — charset, length, dot-leading,
    /// slash-count and `@rev` shapes included. Plausible repos are skipped
    /// (they would trigger the real HF metadata fetch).
    #[test]
    fn downloads_repo_rejects_every_implausible_shape(
        repo in (any::<String>()).prop_filter(
            "plausible repos hit the network; unsafe chars escape in Debug",
            |repo| !plausible_repo(repo) && debug_safe(repo),
        ),
    ) {
        let router = downloads_app();
        let body = json!({"kind": "llm", "repo": repo, "file": "m.gguf"});
        let (status, raw) = RT.block_on(post_text(&router, "/v1/downloads", &body));
        prop_assert_eq!(status.as_u16(), 400u16);
        prop_assert!(raw.contains("bad HuggingFace repo id"), "{raw}");
        prop_assert!(raw.contains(&format!("{repo:?}")), "must quote the repo: {raw}");
    }
}

// ---------------------------------------------------------------------
// Targeted boundary tests (exhaustive over small finite sets).
// ---------------------------------------------------------------------

/// Charset + boundary sweep for repo validation: every printable ASCII at
/// three positions of both parts, plus the exact length boundary (96
/// plausible — skipped, network — 97 rejected by name), the documented
/// `@rev` shapes (NOT supported: `@` is outside the charset, so they are
/// named rejections), and the structural rejects (missing/extra slashes,
/// dot-leading components).
#[tokio::test]
async fn downloads_repo_boundary_sweep() {
    let router = downloads_app();
    let mut rejected = 0;
    for byte in 0x20u8..=0x7E {
        let c = char::from(byte);
        for (owner, name) in [
            (format!("aa{c}bb"), "ok-name".to_owned()),
            ("ok-owner".to_owned(), format!("nn{c}mm")),
            (format!("{c}aabb"), "ok-name".to_owned()),
        ] {
            let repo = format!("{owner}/{name}");
            let body = json!({"kind": "llm", "repo": repo, "file": "m.gguf"});
            let (status, raw) = post_text(&router, "/v1/downloads", &body).await;
            if plausible_repo(&repo) {
                // Accept side would fetch HF metadata; skip the assertion.
                continue;
            }
            assert_eq!(status, StatusCode::BAD_REQUEST, "repo {repo:?}");
            assert!(raw.contains("bad HuggingFace repo id"), "{raw}");
            rejected += 1;
        }
    }
    assert!(rejected > 0, "the sweep must exercise the rejection path");

    for repo in ["org/name@rev", "org@rev/name", "org/name@"] {
        let body = json!({"kind": "llm", "repo": repo, "file": "m.gguf"});
        let (status, raw) = post_text(&router, "/v1/downloads", &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "repo {repo:?}");
        assert!(raw.contains("bad HuggingFace repo id"), "{raw}");
    }

    for len in [97usize, 200] {
        let repo = format!("{}/{}", "a".repeat(len), "name");
        let body = json!({"kind": "llm", "repo": repo, "file": "m.gguf"});
        let (status, raw) = post_text(&router, "/v1/downloads", &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "len {len}");
        assert!(raw.contains("bad HuggingFace repo id"), "{raw}");
    }
    for repo in [
        "",
        "/",
        "owner/",
        "/name",
        "a/b/c",
        ".hidden/name",
        "org/.name",
        "org/..",
    ] {
        let body = json!({"kind": "llm", "repo": repo, "file": "m.gguf"});
        let (status, raw) = post_text(&router, "/v1/downloads", &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "repo {repo:?}");
        assert!(raw.contains("bad HuggingFace repo id"), "{raw}");
    }
}

/// The output-size cap is enforced by the BACKEND, not the validator: a
/// size over `MAX_EDGE` passes validation (never `invalid_size`) and is
/// rejected at generation time as `generation_failed` — while the exact
/// boundary `MAX_EDGE` edge still renders.
#[tokio::test]
// Deliberate: the std guard serializes this test's requests against the
// proptest threads; the single-threaded test runtime never re-enters it.
#[allow(clippy::await_holding_lock)]
async fn images_base_size_cap_is_backend_enforced() {
    let router = images_app();
    let slot = SLOT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (status, payload) = post_json(
        &router,
        "/v1/images/generations",
        &json!({"prompt": "p", "size": "4096x64"}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(error_code(&payload), "generation_failed");
    let (status, _) = post_json(
        &router,
        "/v1/images/generations",
        &json!({"prompt": "p", "size": "2048x2"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    drop(slot);
}

/// Recipe precedence (explicit field > recipe default > server default)
/// lands verbatim in the job-bundle sidecar `request.json`.
#[tokio::test]
// Deliberate: same std-guard serialization as above.
#[allow(clippy::await_holding_lock)]
async fn recipe_precedence_lands_in_sidecar() {
    let dir =
        std::env::temp_dir().join(format!("honkbox-contract-hardening-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let _guard = TempDirGuard(dir.clone());
    let slot = SLOT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // One fresh app dir per request: `latest_bundle_request` then sees
    // exactly one bundle per phase — no newest-by-mtime tie to race on
    // (back-to-back writes under `-j 1` can share an mtime tick).
    let phase_app = |n: usize| {
        let phase = dir.join(format!("phase-{n}"));
        std::fs::create_dir_all(&phase).ok();
        (images_app_with_dir(phase.clone()), phase)
    };

    // Explicit fields win over the sdxl recipe defaults (dpm++2m/karras,
    // cfg 7.0, 25 steps): euler sampler explicit, karras from the recipe,
    // cfg 5.0 and 42 steps explicit.
    let (router, phase) = phase_app(1);
    let (status, _) = post_json(
        &router,
        "/v1/images/generations",
        &json!({
            "prompt": "p",
            "size": "64x64",
            "recipe": "sdxl",
            "sampler": "euler",
            "cfg_scale": 5.0,
            "steps": 42,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let request = latest_bundle_request(&phase).expect("bundle written");
    assert_eq!(request["sampler"], json!("euler"), "explicit override wins");
    assert_eq!(
        request["scheduler"],
        json!("karras"),
        "recipe default fills the rest"
    );
    assert_eq!(request["cfg_scale"], json!(5.0));
    assert_eq!(request["steps"], json!(42));
    assert_eq!(request["recipe"], json!("sdxl"));

    // Recipe alone: all four bundle fields come from the recipe table.
    let (router, phase) = phase_app(2);
    let (status, _) = post_json(
        &router,
        "/v1/images/generations",
        &json!({"prompt": "p", "size": "64x64", "recipe": "sdxl"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let request = latest_bundle_request(&phase).expect("bundle written");
    assert_eq!(request["sampler"], json!("dpm++2m"));
    assert_eq!(request["scheduler"], json!("karras"));
    assert_eq!(request["cfg_scale"], json!(7.0));
    assert_eq!(request["steps"], json!(25));

    // Bare request: the server defaults (the pre-recipe fixed recipe).
    let (router, phase) = phase_app(3);
    let (status, _) = post_json(&router, "/v1/images/generations", &json!({"prompt": "p"})).await;
    assert_eq!(status, StatusCode::OK);
    let request = latest_bundle_request(&phase).expect("bundle written");
    assert_eq!(request["sampler"], json!("euler"));
    // The discrete schedule keeps its historical sidecar recording name
    // (`normal`); sd.cpp wire spellings apply only to explicit requests.
    assert_eq!(request["scheduler"], json!("normal"));
    assert_eq!(request["cfg_scale"], json!(7.0));
    assert_eq!(request["steps"], json!(20));
    assert_eq!(request["recipe"], json!(null));

    drop(slot);
    // `_guard` removes the tmp dir here (and on any earlier panic).
}

/// Drop guard for the recipe test's pid-unique tmp dir: removes the tree
/// on scope exit — including panic unwinds — so a failed test cannot leak
/// fixtures that poison later runs.
struct TempDirGuard(std::path::PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The `request.json` sidecar of the most recently written job bundle.
fn latest_bundle_request(data_dir: &Path) -> Option<Value> {
    let jobs = data_dir.join("jobs");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&jobs).ok()? {
        let entry = entry.ok()?;
        let modified = entry.metadata().ok()?.modified().ok()?;
        if newest.as_ref().is_none_or(|(t, _)| modified > *t) {
            newest = Some((modified, entry.path()));
        }
    }
    let path = newest?.1.join("request.json");
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}
