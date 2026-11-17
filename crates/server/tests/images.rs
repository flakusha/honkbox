//! `POST /v1/images/generations` (REQ-10): `OpenAI` image shape over the
//! startup-configured SD pipeline (stub in tests). Covers the binding
//! checklist: base64 decodes to a PNG whose dimensions match `size`, `n`
//! defaults to 1 / honors 2 / clamps above 4, 400 bad `size` and
//! `response_format`, 404 unknown-model envelope, and the per-generation
//! job bundle landing under `<data_dir>/jobs/`.

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
#[path = "../src/downloads.rs"]
mod downloads;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;
#[allow(dead_code)]
#[path = "../src/jobs_registry.rs"]
mod jobs_registry;
#[allow(dead_code)]
#[path = "../src/metrics.rs"]
mod metrics;
#[path = "../src/routes.rs"]
mod routes;
#[allow(dead_code)]
#[path = "../src/swap.rs"]
mod swap;

use std::sync::{Arc, LazyLock, RwLock};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use base64::Engine as _;
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's tests.
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One SD catalog entry (registry-scan shape: id == slug here).
fn sd_entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: id.to_owned(),
        kind: ModelKind::Sd,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["image".to_owned()],
        capabilities: ModelCapabilities::for_kind(ModelKind::Sd),
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

fn catalog_with(entries: &[ModelCatalogEntry]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries.to_vec(),
    }
}

fn app_with(catalog: ModelCatalog, data_dir: Option<std::path::PathBuf>) -> Router {
    app(AppConfig {
        auth: AuthConfig::default(),
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
        sd_model: None,
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// `POST /v1/images/generations`; returns (status, parsed JSON body).
async fn images(app: Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let parsed = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("response not JSON: {err}: {text}"));
    (status, parsed)
}

/// Decode one `b64_json` entry and assert the PNG magic + IHDR dimensions.
fn assert_png(entry: &Value, width: u32, height: u32) {
    let b64 = entry["b64_json"].as_str().expect("b64_json string");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("valid base64");
    assert_eq!(
        &bytes[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
    );
    // IHDR: length(4) + "IHDR" + big-endian width + big-endian height.
    assert_eq!(&bytes[12..16], b"IHDR");
    assert_eq!(u32::from_be_bytes(bytes[16..20].try_into().unwrap()), width);
    assert_eq!(
        u32::from_be_bytes(bytes[20..24].try_into().unwrap()),
        height
    );
}

/// Single image: base64 decodes to a PNG matching the requested size;
/// `created` is a plausible Unix timestamp.
#[tokio::test]
async fn generation_returns_png_matching_size() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(
            catalog_with(&[sd_entry("comfy:checkpoints/fake.safetensors")]),
            None,
        ),
        json!({
            "model": "comfy:checkpoints/fake.safetensors",
            "prompt": "a red panda wearing a hat",
            "size": "256x128"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    let created = body["created"].as_u64().expect("created");
    assert!(created > 1_500_000_000, "unix seconds, got {created}");
    assert_png(&body["data"][0], 256, 128);
}

/// `n` defaults to 1 without an explicit field.
#[tokio::test]
async fn n_defaults_to_one() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
}

/// `n: 2` returns two images with distinct per-image seeds.
#[tokio::test]
async fn n_two_returns_two_distinct_images() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi", "n": 2, "size": "64x64"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_png(&data[0], 64, 64);
    assert_png(&data[1], 64, 64);
    assert_ne!(data[0]["b64_json"], data[1]["b64_json"], "distinct seeds");
}

/// `n` above the cap of 4 is clamped (never rejected): 99 → 4 images.
#[tokio::test]
async fn n_above_cap_clamps_to_four() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi", "n": 99, "size": "16x16"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 4);
}

/// Bad `size` strings are a 400 `OpenAI` envelope.
#[tokio::test]
async fn bad_size_is_400_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    for size in ["abc", "0x0", "1024", "1024x0"] {
        let (status, body) = images(app.clone(), json!({"prompt": "hi", "size": size})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "size {size}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    }
}

/// A `response_format` other than `b64_json` is an explicit 400, never a
/// silent substitution.
#[tokio::test]
async fn unsupported_response_format_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi", "response_format": "url"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["code"], "unsupported_response_format",
        "{body}"
    );
}

/// Unknown model ids — and known non-SD ids — are a 404 `OpenAI` envelope
/// (`model_not_found`), the same contract as chat completions.
#[tokio::test]
async fn unknown_model_is_404_envelope() {
    let _slot = SLOT_LOCK.lock().await;
    let mut catalog = catalog_with(&[sd_entry("comfy:f.safetensors")]);
    catalog.data.push(ModelCatalogEntry {
        kind: ModelKind::Llm,
        capabilities: ModelCapabilities::for_kind(ModelKind::Llm),
        modalities: vec!["text".to_owned()],
        ..sd_entry("llm:fake.gguf")
    });
    let app = app_with(catalog, None);
    for model in ["nope", "llm:fake.gguf"] {
        let (status, body) = images(app.clone(), json!({"prompt": "hi", "model": model})).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "model {model}: {body}");
        assert_eq!(body["error"]["code"], "model_not_found", "{body}");
    }
}

/// Each generation lands a job bundle under `<data_dir>/jobs/` (CLI GC +
/// recordkeeping cover HTTP generations too).
#[tokio::test]
async fn job_bundle_written_per_generation() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-img-test-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let (status, body) = images(
        app_with(
            catalog_with(&[sd_entry("comfy:f.safetensors")]),
            Some(data_dir.clone()),
        ),
        json!({"prompt": "bundle me", "size": "32x32", "n": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dirs: Vec<_> = jobs.filter_map(Result::ok).collect();
    assert_eq!(dirs.len(), 2, "one bundle per image");
    for dir in dirs {
        let png = std::fs::read(dir.path().join("final_00.png")).expect("bundle png");
        assert_eq!(&png[..4], &[0x89, b'P', b'N', b'G']);
        let request: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("request.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(request["prompt"], "bundle me");
    }
    std::fs::remove_dir_all(&data_dir).ok();
}

/// An unknown sampler name is an explicit 400 naming it (with the accepted
/// list), never a silent substitution of the default recipe.
#[tokio::test]
async fn unknown_sampler_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi", "sampler": "snooze"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unknown_sampler", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("snooze"), "{message}");
    assert!(message.contains("euler"), "lists accepted: {message}");
}

/// An unknown scheduler name is an explicit 400 naming it.
#[tokio::test]
async fn unknown_scheduler_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi", "scheduler": "sundial"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unknown_scheduler", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("sundial"), "{message}");
    assert!(message.contains("karras"), "lists accepted: {message}");
}

/// `cfg_scale` outside `[1.0, 30.0]` is rejected with a 400 (never
/// silently clamped — the opposite policy of `n`, documented on the DTO).
#[tokio::test]
async fn cfg_scale_out_of_range_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    for value in [0.9, 30.5] {
        let (status, body) = images(app.clone(), json!({"prompt": "hi", "cfg_scale": value})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "cfg {value}: {body}");
        assert_eq!(body["error"]["code"], "invalid_cfg_scale", "{body}");
    }
    // The accepted bounds themselves pass.
    for value in [1.0, 30.0] {
        let (status, _) = images(
            app.clone(),
            json!({"prompt": "hi", "cfg_scale": value, "size": "16x16"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "cfg {value}");
    }
}

/// Omitting every recipe field is byte-identical to passing the documented
/// defaults explicitly (euler + discrete + CFG 7.0 + seed base 42).
#[tokio::test]
async fn defaults_match_explicit_recipe() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    let (s1, b1) = images(app.clone(), json!({"prompt": "hi", "size": "16x16"})).await;
    let (s2, b2) = images(
        app,
        json!({
            "prompt": "hi",
            "size": "16x16",
            "sampler": "euler",
            "scheduler": "discrete",
            "cfg_scale": 7.0,
            "seed": 42
        }),
    )
    .await;
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
    assert_eq!(b1["data"][0]["b64_json"], b2["data"][0]["b64_json"]);
}

/// Same explicit seed = byte-identical stub output; a different seed
/// differs (seed determinism through the new passthrough field).
#[tokio::test]
async fn seed_determinism() {
    let _slot = SLOT_LOCK.lock().await;
    let app = app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None);
    let body_for = |app: Router, seed: Value| async move {
        images(app, json!({"prompt": "hi", "size": "16x16", "seed": seed})).await
    };
    let (_, a) = body_for(app.clone(), json!(7)).await;
    let (_, b) = body_for(app.clone(), json!(7)).await;
    let (_, c) = body_for(app, json!(8)).await;
    assert_eq!(a["data"][0]["b64_json"], b["data"][0]["b64_json"]);
    assert_ne!(a["data"][0]["b64_json"], c["data"][0]["b64_json"]);
}

/// Recipe overrides land in the job bundle sidecar: the actual recipe used
/// is recorded (sampler, scheduler, CFG, negative prompt, explicit seed).
#[tokio::test]
async fn bundle_records_requested_recipe() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-img-recipe-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let (status, _) = images(
        app_with(
            catalog_with(&[sd_entry("comfy:f.safetensors")]),
            Some(data_dir.clone()),
        ),
        json!({
            "prompt": "recipe me",
            "size": "16x16",
            "sampler": "dpm++2m",
            "scheduler": "karras",
            "cfg_scale": 12.0,
            "negative_prompt": "blurry",
            "seed": 9
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let mut jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    let dir = jobs.find_map(Result::ok).expect("one bundle").path();
    let side: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("final_00.json")).unwrap()).unwrap();
    assert_eq!(side["sampler"], "dpm++2m");
    assert_eq!(side["scheduler"], "karras");
    assert_eq!(side["cfg_scale"], 12.0);
    assert_eq!(side["negative_prompt"], "blurry");
    assert_eq!(side["seed"], 9);
    std::fs::remove_dir_all(&data_dir).ok();
}

/// One `LoRA` catalog entry (registry-scan shape; `path` mirrors the
/// absolute weight-file path a real scan produces).
fn lora_entry(id: &str, path: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        kind: ModelKind::Lora,
        capabilities: ModelCapabilities::for_kind(ModelKind::Lora),
        modalities: vec!["image".to_owned()],
        category: "loras".to_owned(),
        path: path.to_owned(),
        ..sd_entry(id)
    }
}

/// An unknown `LoRA` slug — or a known non-`LoRA` slug — is a 400 naming the
/// offender (resolution is alias-or-slug over `ModelKind::Lora` only).
#[tokio::test]
async fn unknown_lora_slug_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let mut catalog = catalog_with(&[sd_entry("comfy:f.safetensors")]);
    catalog.data.push(lora_entry(
        "comfy:loras/detail.safetensors",
        "/models/loras/detail.safetensors",
    ));
    let app = app_with(catalog, None);
    for slug in ["comfy:loras/missing.safetensors", "comfy:f.safetensors"] {
        let (status, body) = images(
            app.clone(),
            json!({"prompt": "hi", "loras": [{"slug": slug}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "slug {slug}: {body}");
        assert_eq!(body["error"]["code"], "unknown_lora", "slug {slug}: {body}");
    }
}

/// More than two adapters per request is a documented 400 (`too_many_loras`).
#[tokio::test]
async fn too_many_loras_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry("comfy:loras/a.safetensors", "/models/loras/a.safetensors"),
        lora_entry("comfy:loras/b.safetensors", "/models/loras/b.safetensors"),
    ]);
    let (status, body) = images(
        app_with(catalog, None),
        json!({"prompt": "hi", "loras": [
            {"slug": "comfy:loras/a.safetensors"},
            {"slug": "comfy:loras/b.safetensors"},
            {"slug": "comfy:loras/c.safetensors"}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "too_many_loras", "{body}");
}

/// A `LoRA` strength outside `[-2.0, 2.0]` (or non-finite) is a 400, never a
/// silent clamp — the same reject-not-clamp policy as `cfg_scale`.
#[tokio::test]
async fn lora_strength_out_of_range_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry("comfy:loras/a.safetensors", "/models/loras/a.safetensors"),
    ]);
    let app = app_with(catalog, None);
    for strength in [5.0_f64, -3.0] {
        let (status, body) = images(
            app.clone(),
            json!({"prompt": "hi", "loras": [
                {"slug": "comfy:loras/a.safetensors", "strength": strength}
            ]}),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "strength {strength}: {body}"
        );
        assert_eq!(body["error"]["code"], "invalid_lora_strength", "{body}");
    }
}

/// Adapters spanning more than one directory are a 400: the vendor applies
/// a single `LoRA` dir per model config, so honoring both is impossible.
#[tokio::test]
async fn mixed_lora_dirs_is_400() {
    let _slot = SLOT_LOCK.lock().await;
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry(
            "comfy:loras/a.safetensors",
            "/roots/one/loras/a.safetensors",
        ),
        lora_entry(
            "comfy:loras/b.safetensors",
            "/roots/two/loras/b.safetensors",
        ),
    ]);
    let (status, body) = images(
        app_with(catalog, None),
        json!({"prompt": "hi", "loras": [
            {"slug": "comfy:loras/a.safetensors"},
            {"slug": "comfy:loras/b.safetensors"}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "mixed_lora_dirs", "{body}");
}

/// A valid `LoRA` request renders (stub backend) with the license warning
/// recorded (no curated manifest row ⇒ all-rights-reserved note), and the
/// bundle sidecar records the resolved slug + strength.
#[tokio::test]
async fn loras_warn_and_record() {
    let _slot = SLOT_LOCK.lock().await;
    let data_dir = std::env::temp_dir().join(format!("honkbox-img-lora-{}", ulid::Ulid::new()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let catalog = catalog_with(&[
        sd_entry("comfy:f.safetensors"),
        lora_entry(
            "comfy:loras/detail.safetensors",
            "/models/loras/detail.safetensors",
        ),
    ]);
    let (status, body) = images(
        app_with(catalog, Some(data_dir.clone())),
        json!({"prompt": "hi", "loras": [
            {"slug": "comfy:loras/detail.safetensors", "strength": 0.8}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let warnings = body["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 1, "{body}");
    assert!(
        warnings[0]
            .as_str()
            .is_some_and(|w| w.contains("no license recorded")
                && w.contains("comfy:loras/detail.safetensors")),
        "{body}"
    );
    let jobs = std::fs::read_dir(data_dir.join("jobs")).expect("jobs dir");
    for dir in jobs.filter_map(Result::ok) {
        let request: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("request.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            request["loras"][0]["slug"],
            "comfy:loras/detail.safetensors"
        );
        // f32 strengths widen to f64 in the sidecar JSON: compare with a
        // tolerance, not equality.
        assert!((request["loras"][0]["strength"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    }
    std::fs::remove_dir_all(&data_dir).ok();
}

/// Without `loras` the `warnings` key is omitted entirely (skip-if-none),
/// keeping the response byte-compatible with pre-`LoRA` consumers.
#[tokio::test]
async fn warnings_absent_without_loras() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, body) = images(
        app_with(catalog_with(&[sd_entry("comfy:f.safetensors")]), None),
        json!({"prompt": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("warnings").is_none(), "{body}");
}
