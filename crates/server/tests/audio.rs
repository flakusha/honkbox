//! `/v1/audio/transcriptions` + `/v1/audio/speech` (REQ-9): the `OpenAI`
//! wire shapes over the existing real STT/TTS workers. Covers the
//! acceptance checklist: multipart transcription matching the native
//! `/v1/transcribe` transcript for the same audio, `verbose_json` with the
//! worker's timestamped segments, `audio/wav` speech bytes matching the
//! native `/v1/synthesize` output for the same input, the octet-stream
//! RIFF sniff (the `openai` SDK's file content type), pcm-s16le +
//! `sample_rate`, and the explicit error envelopes. The native routes are
//! exercised only for parity — they are unchanged.

// No lib target on this binary crate, so include the modules directly.
// This binary exercises the audio surfaces of `routes.rs`; every included
// module carries `allow(dead_code)` for the unused rest.
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
use honkbox_api_types::{ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind};
use honkbox_audio_worker::{StubStt, StubTts, encode_wav};
use honkbox_llm_worker::StubLlm;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use routes::{AppConfig, AuthConfig, app};

/// Serializes the process-global GPU slot across this binary's tests:
/// parallel requests could otherwise race the one slot into a spurious
/// 429 (same pattern as `completions.rs`/`embeddings.rs`).
static SLOT_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

/// One fake STT/TTS catalog entry (the audio routes validate against the
/// catalog the same way the native routes do).
fn entry(id: &str) -> ModelCatalogEntry {
    ModelCatalogEntry {
        id: id.to_owned(),
        slug: id.to_owned(),
        kind: ModelKind::Llm,
        owned_by: "local".to_owned(),
        context_length: None,
        max_output: None,
        thinking: false,
        tool_calling: false,
        modalities: vec!["text".to_owned()],
        capabilities: ModelCapabilities::for_kind(ModelKind::Llm),
        category: "root".to_owned(),
        path: format!("/models/{id}"),
        bytes: 1,
        metadata: None,
    }
}

fn catalog_with(ids: &[&str]) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: ids.iter().map(|id| entry(id)).collect(),
    }
}

fn test_app() -> Router {
    app(AppConfig {
        auth: AuthConfig::default(),
        dev: false,
        web_dist_dir: "web-dist".into(),
        catalog: Arc::new(RwLock::new(Arc::new(catalog_with(&[
            "llm:stt-fake.gguf",
            "llm:tts-fake.gguf",
        ])))),
        roots: honkbox_registry::ModelRoots {
            llm_dirs: Vec::new(),
            comfyui_dirs: Vec::new(),
            stt_dirs: Vec::new(),
            tts_dirs: Vec::new(),
        },
        llm_model: None,
        stt_model: Some("llm:stt-fake.gguf".to_owned()),
        tts_model: Some("llm:tts-fake.gguf".to_owned()),
        llm: Arc::new(StubLlm::default()),
        stt: Arc::new(StubStt),
        tts: Arc::new(StubTts),
        sd: Arc::new(honkbox_sd_core::StubPipeline),
        sd_model: None,
        swap: None,
        embed: Arc::new(swap::stub_embed_cache()),
        swap_max_ttl: std::time::Duration::from_hours(24),
        data_dir: None,
        vram_budget: None,
        model_entries: Vec::new(),
        downloads: Arc::new(downloads::DownloadManager::new(None).expect("download manager")),
    })
}

/// One multipart form part.
struct Part {
    name: &'static str,
    filename: Option<&'static str>,
    content_type: Option<&'static str>,
    data: Vec<u8>,
}

fn part(name: &'static str, text: &str) -> Part {
    Part {
        name,
        filename: None,
        content_type: None,
        data: text.as_bytes().to_vec(),
    }
}

/// Encode parts into a multipart/form-data body (boundary included in the
/// returned content-type value).
fn multipart_body(parts: &[Part]) -> (String, Vec<u8>) {
    let boundary = "honkbox-audio-test";
    let mut body = Vec::new();
    for one in parts {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\ncontent-disposition: form-data; name=\"{}\"",
                one.name
            )
            .as_bytes(),
        );
        if let Some(filename) = one.filename {
            body.extend_from_slice(format!("; filename=\"{filename}\"\r\n").as_bytes());
        } else {
            body.extend_from_slice(b"\r\n");
        }
        if let Some(content_type) = one.content_type {
            body.extend_from_slice(format!("content-type: {content_type}\r\n").as_bytes());
        }
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(&one.data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// 100 ms of 16 kHz s16 sine, as a wav container (valid RIFF payload for
/// the STT workers).
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn test_wav() -> Vec<u8> {
    let rate = 16_000.0_f32;
    let pcm: Vec<i16> = (0..1_600)
        .map(|i| {
            let t = i as f32 / rate;
            (0.5 * (2.0 * std::f32::consts::PI * t * 440.0).sin() * f32::from(i16::MAX)) as i16
        })
        .collect();
    encode_wav(16_000, &pcm)
}

/// Fire a request at the app; returns (status, content-type, body bytes).
async fn send(app: Router, request: Request<Body>) -> (StatusCode, String, Vec<u8>) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, content_type, bytes.to_vec())
}

async fn post_json(app: Router, uri: &str, body: Value) -> (StatusCode, String, Vec<u8>) {
    send(
        app,
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

/// TASK-speech-input-cap: `input` over the 4096-character cap is a named
/// 400 (`input_too_long`) — rejected, never truncated. The cap is counted
/// in CHARACTERS: a multibyte string of 4097 characters is over the cap
/// even though it is far smaller than any byte-based guess.
#[tokio::test]
async fn speech_input_over_cap_is_rejected_not_clamped() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, _, bytes) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"input": "a".repeat(4097)}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert_eq!(parsed["error"]["code"], "input_too_long", "{parsed}");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("4097") && m.contains("cap is 4096")),
        "message names actual and cap: {parsed}"
    );
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("not truncated")),
        "reject-not-clamp is explicit: {parsed}"
    );

    // Multibyte counting: 4097 CJK characters (12 KB of UTF-8) is over the
    // cap — the limit is not a byte limit.
    let (status, _, _) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"input": "字".repeat(4097)}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// TASK-speech-input-cap: at-cap (4096 characters) requests are unchanged —
/// the stub TTS synthesizes them normally, and whitespace-only handling is
/// untouched (the pre-existing `input is required` 400).
#[tokio::test]
async fn speech_input_at_cap_still_synthesizes() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, content_type, bytes) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"input": "h".repeat(4096)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "audio/wav");
    assert!(bytes.starts_with(b"RIFF"), "wav container intact");
}

/// REQ-9 acceptance: the multipart transcription matches the native
/// `/v1/transcribe` transcript for the same wav bytes.
#[tokio::test]
async fn transcriptions_multipart_matches_native() {
    let _slot = SLOT_LOCK.lock().await;
    let wav = test_wav();
    // Multipart, model named.
    let (content_type, body) = multipart_body(&[
        part("model", "llm:stt-fake.gguf"),
        Part {
            name: "file",
            filename: Some("clip.wav"),
            content_type: Some("audio/wav"),
            data: wav.clone(),
        },
    ]);
    let (status, response_type, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert!(
        response_type.starts_with("application/json"),
        "{response_type}"
    );
    let parsed: Value = serde_json::from_slice(&bytes).expect("JSON");
    assert_eq!(parsed["text"], "stub transcription", "{parsed}");
    assert!(parsed.get("segments").is_none(), "plain shape: {parsed}");

    // Native route, same bytes (plus its required model query), same
    // transcript.
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/transcribe?model=llm%3Astt-fake.gguf")
            .header("content-type", "audio/wav")
            .body(Body::from(wav))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let native: Value = serde_json::from_slice(&bytes).expect("JSON");
    assert_eq!(native["text"], parsed["text"], "{native}");
}

/// REQ-9 acceptance: `response_format=verbose_json` adds the worker's
/// timestamped segments (honkbox ms-integer segment shape).
#[tokio::test]
async fn transcriptions_verbose_json_carries_segments() {
    let _slot = SLOT_LOCK.lock().await;
    let (content_type, body) = multipart_body(&[
        part("response_format", "verbose_json"),
        Part {
            name: "file",
            filename: Some("clip.wav"),
            content_type: Some("audio/wav"),
            data: test_wav(),
        },
    ]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let parsed: Value = serde_json::from_slice(&bytes).expect("JSON");
    assert_eq!(parsed["text"], "stub transcription");
    let segments = parsed["segments"].as_array().expect("segments");
    assert_eq!(segments.len(), 1, "{parsed}");
    assert_eq!(segments[0]["start_ms"], 0);
    assert_eq!(segments[0]["end_ms"], 100, "1600 samples @16kHz");
    assert_eq!(segments[0]["text"], "stub transcription");
}

/// The `openai` SDK sends file parts as `application/octet-stream`: the
/// RIFF sniff accepts a real wav container under that content type.
#[tokio::test]
async fn transcriptions_octet_stream_sniffs_riff() {
    let _slot = SLOT_LOCK.lock().await;
    let (content_type, body) = multipart_body(&[Part {
        name: "file",
        filename: Some("clip.bin"),
        content_type: Some("application/octet-stream"),
        data: test_wav(),
    }]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let parsed: Value = serde_json::from_slice(&bytes).expect("JSON");
    assert_eq!(parsed["text"], "stub transcription");
}

/// pcm-s16le uploads ride the native contract: `sample_rate` from the
/// form field, or the query param carried over.
#[tokio::test]
async fn transcriptions_pcm_s16le_with_sample_rate() {
    let _slot = SLOT_LOCK.lock().await;
    let pcm: Vec<u8> = [0_i16, 100, -100, 500]
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    // Form field.
    let (content_type, body) = multipart_body(&[
        part("sample_rate", "16000"),
        Part {
            name: "file",
            filename: None,
            content_type: Some("audio/pcm-s16le"),
            data: pcm.clone(),
        },
    ]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    // Query param carried over.
    let (content_type, body) = multipart_body(&[Part {
        name: "file",
        filename: None,
        content_type: Some("audio/pcm-s16le"),
        data: pcm,
    }]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions?sample_rate=8000")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
}

/// Explicit rejections: missing `file`, unknown `model`, unsupported
/// `response_format`, non-audio payload — each an `OpenAI` envelope.
#[tokio::test]
async fn transcriptions_rejections_are_envelopes() {
    let _slot = SLOT_LOCK.lock().await;
    // Missing file.
    let (content_type, body) = multipart_body(&[part("model", "llm:stt-fake.gguf")]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(parsed["error"]["message"].is_string(), "{parsed}");

    // Unknown model.
    let (content_type, body) = multipart_body(&[
        part("model", "llm:nope.gguf"),
        Part {
            name: "file",
            filename: None,
            content_type: Some("audio/wav"),
            data: test_wav(),
        },
    ]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unknown model")),
        "{parsed}"
    );

    // Unsupported response_format (`srt`/`vtt` are supported now; an
    // unknown format stays an explicit 400 envelope).
    let (content_type, body) = multipart_body(&[
        part("response_format", "html"),
        Part {
            name: "file",
            filename: None,
            content_type: Some("audio/wav"),
            data: test_wav(),
        },
    ]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unsupported response_format")),
        "{parsed}"
    );

    // Non-audio payload under octet-stream.
    let (content_type, body) = multipart_body(&[Part {
        name: "file",
        filename: None,
        content_type: Some("application/octet-stream"),
        data: b"definitely not audio".to_vec(),
    }]);
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(parsed["error"]["message"].is_string(), "{parsed}");
}

/// The native `/v1/synthesize` twin enforces the SAME boundary as
/// `/v1/audio/speech` (TASK-speech-input-cap): same worker, same cap,
/// reject-not-clamp — a route without the check would be the bypass.
#[tokio::test]
async fn synthesize_input_over_cap_is_rejected() {
    let app = test_app();
    let over_cap = "a".repeat(4097);
    let (status, _, body) = post_json(
        app.clone(),
        "/v1/synthesize",
        json!({"model": "llm:tts-fake.gguf", "text": over_cap, "voice": "af_heart"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = String::from_utf8(body).expect("utf8 error body");
    assert!(body.contains("cap is 4096"), "{body}");
    assert!(body.contains("rejected, not truncated"), "{body}");
    // Blank text is rejected on the twin too (speech parity). Full valid
    // shape so deserialization succeeds and the ROUTE boundary runs.
    let (status, _, body) = post_json(
        app,
        "/v1/synthesize",
        json!({"model": "llm:tts-fake.gguf", "text": "   ", "voice": "af_heart"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body = String::from_utf8(body).expect("utf8 error body");
    assert!(body.contains("text is required"), "{body}");
}

/// REQ-9 acceptance: `/v1/audio/speech` returns `audio/wav` bytes that
/// match the native `/v1/synthesize` output for the same input.
#[tokio::test]
async fn speech_wav_matches_native_synthesize() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, content_type, bytes) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({
            "model": "llm:tts-fake.gguf",
            "input": "hello honkbox",
            "voice": "af_heart"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "audio/wav", "{content_type}");
    assert!(bytes.starts_with(b"RIFF"), "not a wav container");
    // Native route, same text + voice → byte-identical output.
    let (native_status, native_type, native_bytes) = post_json(
        test_app(),
        "/v1/synthesize",
        json!({
            "model": "llm:tts-fake.gguf",
            "text": "hello honkbox",
            "voice": "af_heart"
        }),
    )
    .await;
    assert_eq!(native_status, StatusCode::OK);
    assert_eq!(native_type, "audio/wav");
    assert_eq!(bytes, native_bytes, "speech bytes must match synthesize");
    // Server-default model (no `model` field) rides the configured default.
    let (status, content_type, bytes) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"input": "default model"}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(content_type, "audio/wav");
    assert!(bytes.starts_with(b"RIFF"));
}

/// Speech rejections: unknown model, empty input, non-wav `response_format`
/// — each an `OpenAI` envelope.
#[tokio::test]
async fn speech_rejections_are_envelopes() {
    let _slot = SLOT_LOCK.lock().await;
    // Unknown model.
    let (status, _, bytes) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"model": "llm:nope.gguf", "input": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unknown model")),
        "{parsed}"
    );

    // Empty input.
    let (status, _, bytes) =
        post_json(test_app(), "/v1/audio/speech", json!({"input": "   "})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("input is required")),
        "{parsed}"
    );

    // Unsupported response_format.
    let (status, _, bytes) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"input": "hi", "response_format": "mp3"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).expect("envelope");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("unsupported response_format")),
        "{parsed}"
    );
}

/// Speech `response_format: "pcm"`: raw little-endian s16 mono samples —
/// the exact bytes the wav path wraps, minus the 44-byte RIFF header (`OpenAI`
/// pcm semantics). The stub output is deterministic, so both requests here
/// synthesize identical samples.
#[tokio::test]
async fn speech_pcm_is_raw_s16le() {
    let _slot = SLOT_LOCK.lock().await;
    let (status, ct, pcm) = post_json(
        test_app(),
        "/v1/audio/speech",
        json!({"input": "hi", "response_format": "pcm"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ct, "audio/pcm-s16le");
    assert!(pcm.len() % 2 == 0, "s16le payload must be even");

    let (status, ct, wav) = post_json(test_app(), "/v1/audio/speech", json!({"input": "hi"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ct, "audio/wav");
    assert!(wav.starts_with(b"RIFF"), "wav path keeps the container");
    assert_eq!(pcm.len() + 44, wav.len(), "44-byte header difference");
    assert_eq!(&wav[44..], &pcm[..], "same samples, container omitted");
}

/// `GET /v1/audio/voices`: the statically documented supported Kokoro set,
/// served as an OpenAI-style list without any model loaded.
#[tokio::test]
async fn voices_lists_supported_kokoro_set() {
    let (status, ct, bytes) = send(
        test_app(),
        Request::builder()
            .method("GET")
            .uri("/v1/audio/voices")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.starts_with("application/json"), "{ct}");
    let parsed: Value = serde_json::from_slice(&bytes).expect("list envelope");
    assert_eq!(parsed["object"], "list");
    let data = parsed["data"].as_array().expect("data array");
    assert!(!data.is_empty(), "supported set is non-empty");
    assert!(data.iter().all(|v| v["object"] == "voice"), "{parsed}");
    let heart = data
        .iter()
        .find(|v| v["id"] == "af_heart")
        .expect("af_heart in the supported set");
    assert_eq!(heart["language"], "en-US");
    assert_eq!(heart["gender"], "female");
}

/// H3 (ReviewedAudio): the served `OpenAPI` document must carry the REQ-9
/// audio routes (translations included), the voices route, and the model
/// deletion route — clients codegen from this document, so unlisted
/// endpoints are undiscoverable.
#[tokio::test]
async fn openapi_documents_audio_and_delete_routes() {
    let (status, _, bytes) = send(
        test_app(),
        Request::builder()
            .method("GET")
            .uri("/v1/openapi.json")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let doc: Value = serde_json::from_slice(&bytes).expect("openapi json");
    let paths = doc["paths"].as_object().expect("paths");
    for path in [
        "/v1/audio/voices",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/jobs",
        "/v1/jobs/{id}/cancel",
        "/v1/models/{slug}",
    ] {
        assert!(paths.contains_key(path), "missing {path} in {paths:?}");
    }
    assert!(
        paths["/v1/models/{slug}"]
            .as_object()
            .is_some_and(|ops| ops.contains_key("delete")),
        "DELETE /v1/models/{{slug}} not documented"
    );
}

/// SRT/VTT render deterministically from the same worker segments the
/// `verbose_json` path serves: `HH:MM:SS,mmm` (`SubRip`) vs `HH:MM:SS.mmm`
/// (`WebVTT`, with the `WEBVTT` header). The test wav is 100 ms of 16 kHz,
/// so the stub's single segment spans 0→100 ms.
#[tokio::test]
async fn transcriptions_srt_and_vtt_render_from_segments() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app();
    let wav = test_wav();
    let cases = [
        (
            "srt",
            "application/x-subrip",
            "1\n00:00:00,000 --> 00:00:00,100\nstub transcription\n\n".as_bytes(),
        ),
        (
            "vtt",
            "text/vtt; charset=utf-8",
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:00.100\nstub transcription\n\n".as_bytes(),
        ),
    ];
    for (format, want_type, want_body) in cases {
        let (content_type, body) = multipart_body(&[
            part("model", "llm:stt-fake.gguf"),
            Part {
                name: "file",
                filename: Some("a.wav"),
                content_type: Some("audio/wav"),
                data: wav.clone(),
            },
            part("response_format", format),
        ]);
        let (status, got_type, bytes) = send(
            app.clone(),
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header("content-type", content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "({format})");
        assert_eq!(got_type, want_type, "({format})");
        assert_eq!(bytes, want_body, "({format})");
    }

    // An unsupported format stays an explicit OpenAI 400 envelope.
    let (content_type, body) = multipart_body(&[
        part("model", "llm:stt-fake.gguf"),
        Part {
            name: "file",
            filename: Some("a.wav"),
            content_type: Some("audio/wav"),
            data: wav.clone(),
        },
        part("response_format", "html"),
    ]);
    let (status, _, bytes) = send(
        app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("srt")),
        "{parsed}"
    );
}

/// `POST /v1/audio/translations`: same decode + STT worker, `translate`
/// wired on; the plain shape mirrors `/v1/audio/transcriptions` and the
/// verbose shape carries the timestamped segments.
#[tokio::test]
async fn translations_return_english_json_and_verbose() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app();
    let wav = test_wav();

    // Plain json (default format).
    let (content_type, body) = multipart_body(&[
        part("model", "llm:stt-fake.gguf"),
        Part {
            name: "file",
            filename: Some("a.wav"),
            content_type: Some("audio/wav"),
            data: wav.clone(),
        },
    ]);
    let (status, got_type, bytes) = send(
        app.clone(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/translations")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(got_type, "application/json");
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["text"], "stub transcription");

    // verbose_json carries the segments (same shape as transcriptions).
    let (content_type, body) = multipart_body(&[
        part("model", "llm:stt-fake.gguf"),
        Part {
            name: "file",
            filename: Some("a.wav"),
            content_type: Some("audio/wav"),
            data: wav,
        },
        part("response_format", "verbose_json"),
    ]);
    let (status, _, bytes) = send(
        app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/translations")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["text"], "stub transcription");
    let segments = parsed["segments"].as_array().expect("segments");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0]["start_ms"], 0);
    assert_eq!(segments[0]["end_ms"], 100);
}

/// Translations rejections: missing file, the transcriptions-only
/// `language` field, and an unsupported `response_format` are explicit
/// `OpenAI` 400 envelopes.
#[tokio::test]
async fn translations_rejections_are_envelopes() {
    let _slot = SLOT_LOCK.lock().await;
    let app = test_app();

    // Missing file.
    let (content_type, body) = multipart_body(&[part("model", "llm:stt-fake.gguf")]);
    let (status, _, bytes) = send(
        app.clone(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/translations")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("file is required")),
        "{parsed}"
    );

    // `language` is not a translations field (output is always English).
    let (content_type, body) = multipart_body(&[
        part("model", "llm:stt-fake.gguf"),
        Part {
            name: "file",
            filename: Some("a.wav"),
            content_type: Some("audio/wav"),
            data: test_wav(),
        },
        part("language", "de"),
    ]);
    let (status, _, bytes) = send(
        app.clone(),
        Request::builder()
            .method("POST")
            .uri("/v1/audio/translations")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("always English")),
        "{parsed}"
    );

    // Unsupported response_format.
    let (content_type, body) = multipart_body(&[
        part("model", "llm:stt-fake.gguf"),
        Part {
            name: "file",
            filename: Some("a.wav"),
            content_type: Some("audio/wav"),
            data: test_wav(),
        },
        part("response_format", "tsv"),
    ]);
    let (status, _, bytes) = send(
        app,
        Request::builder()
            .method("POST")
            .uri("/v1/audio/translations")
            .header("content-type", content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("response_format")),
        "{parsed}"
    );
}
