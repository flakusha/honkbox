//! HTTP surface: `/health`, `/api/version`, `/v1/openapi.json`, UI fallback.
//!
//! The `OpenAPI` document is served as plain JSON via `Json` — no
//! `utoipa-axum` router glue.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Query, State},
    http::{HeaderMap, StatusCode, Uri, header},
    response::{
        Html, IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use rust_embed::Embed;
use sd_rip_api_types::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, Health, SynthesizeRequest, Transcript,
    TranscriptSegment, Version,
};
use sd_rip_audio_worker::{
    SttPipeline, SynthesizeRequest as WorkerSynthRequest, TranscribeRequest as WorkerTranscribeRequest,
    Transcript as WorkerTranscript, TtsPipeline, decode_wav, encode_wav,
};
use sd_rip_llm_worker::{
    ChatMessage as WorkerChatMessage, ChatRequest as WorkerChatRequest, LlmPipeline, Role,
};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use utoipa::OpenApi;

/// Compile-time UI bundle (release always embeds; dev serves from disk instead).
#[derive(Embed)]
#[folder = "../../web-dist/"]
struct EmbeddedUi;

/// `OpenAPI` document for the whole HTTP surface.
#[derive(OpenApi)]
#[openapi(
    paths(
        health,
        get_version,
        openapi_json,
        post_chat,
        post_chat_stream,
        post_transcribe,
        post_synthesize
    ),
    components(schemas(
        Health,
        Version,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        ChatRole,
        Transcript,
        TranscriptSegment,
        SynthesizeRequest
    ))
)]
pub struct ApiDoc;

/// Runtime switch for the UI fallback plus the shared inference pipelines.
///
/// Pipelines are `Clone`-safe (`Arc`): stub unit structs cost nothing, file
/// backends load once at startup and are shared across requests.
#[derive(Clone)]
pub struct AppConfig {
    /// Serve `web_dist_dir` from disk instead of the embedded bundle.
    pub dev: bool,
    /// Directory served in dev mode (Vite output).
    pub web_dist_dir: PathBuf,
    /// Shared chat pipeline (`stub-llm` or `llama-cpp-*`).
    pub llm: Arc<dyn LlmPipeline>,
    /// Shared transcription pipeline (`stub-stt` or `whisper-*`).
    pub stt: Arc<dyn SttPipeline>,
    /// Shared synthesis pipeline (`stub-tts` or `kokoro-*`).
    pub tts: Arc<dyn TtsPipeline>,
}

/// Build the full router.
pub fn app(config: AppConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/version", get(get_version))
        .route("/v1/openapi.json", get(openapi_json))
        .route("/v1/chat", post(post_chat))
        .route("/v1/chat/stream", post(post_chat_stream))
        .route("/v1/transcribe", post(post_transcribe))
        .route("/v1/synthesize", post(post_synthesize))
        .fallback(fallback)
        .with_state(config)
}

/// Liveness probe.
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Server is alive", body = Health)))]
async fn health() -> Json<Health> {
    Json(Health {
        status: "ok".to_owned(),
    })
}

/// Binary name + version.
#[utoipa::path(get, path = "/api/version", responses((status = 200, description = "Binary name and version", body = Version)))]
async fn get_version() -> Json<Version> {
    Json(Version {
        name: "sd-rip".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

/// `OpenAPI` 3.1 document as JSON.
#[utoipa::path(get, path = "/v1/openapi.json", responses((status = 200, description = "OpenAPI document")))]
async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

/// Embedded `web-dist/` asset when present, dev-disk file in `--dev`,
/// otherwise the static fallback page.
async fn fallback(State(config): State<AppConfig>, uri: Uri) -> Response {
    let raw = uri.path().trim_start_matches('/');
    let file = if raw.is_empty() { "index.html" } else { raw };
    if config.dev {
        if let Some(response) = serve_from_disk(&config.web_dist_dir, file).await {
            return response;
        }
    } else if let Some(asset) = EmbeddedUi::get(file) {
        return asset_response(file, asset.data.into_owned());
    }
    Html(include_str!("../fallback.html")).into_response()
}

/// Serve one file from the dev web-dist directory; `None` = miss or escape attempt.
async fn serve_from_disk(dir: &Path, file: &str) -> Option<Response> {
    if file.contains("..") {
        return None;
    }
    let bytes = tokio::fs::read(dir.join(file)).await.ok()?;
    Some(asset_response(file, bytes))
}

/// Raw bytes with a content type guessed from the extension (no extra deps).
fn asset_response(file: &str, bytes: Vec<u8>) -> Response {
    let mime = match file.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    };
    ([(header::CONTENT_TYPE, mime)], bytes).into_response()
}

/// `POST /v1/transcribe` query: model slug + language for the worker request;
/// `sample_rate` is required only for raw `audio/pcm-s16le` bodies.
#[derive(Debug, Deserialize)]
struct TranscribeQuery {
    model: Option<String>,
    language: Option<String>,
    sample_rate: Option<u32>,
}

/// `api-types` message → worker message (only the role enum differs).
fn worker_message(message: &ChatMessage) -> WorkerChatMessage {
    let role = match message.role {
        ChatRole::User => Role::User,
        ChatRole::Assistant => Role::Assistant,
        ChatRole::System => Role::System,
    };
    WorkerChatMessage {
        role,
        content: message.content.clone(),
    }
}

/// `POST /v1/chat`: sync worker call inside `spawn_blocking`; the residency
/// lease is held across the call. Only the server edge is async.
#[utoipa::path(
    post,
    path = "/v1/chat",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "Chat completion", body = ChatResponse),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Inference failed")
    )
)]
async fn post_chat(
    State(config): State<AppConfig>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let worker_req = WorkerChatRequest {
        model: req.model,
        messages: req.messages.iter().map(worker_message).collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };
    let pipe = config.llm.clone();
    let reply = tokio::task::spawn_blocking(move || match sd_rip_residency::global().try_reserve()
    {
        Ok(lease) => {
            let _lease = lease;
            pipe.complete(&worker_req).map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("chat failed: {err}"),
                )
            })
        }
        Err(err) => Err((StatusCode::TOO_MANY_REQUESTS, err.to_string())),
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("chat task failed: {err}"),
        )
    })??;
    Ok(Json(ChatResponse {
        backend_id: reply.backend_id,
        text: reply.text,
        tokens: reply.tokens,
        millis: reply.millis,
    }))
}

/// `POST /v1/chat/stream`: same blocking call, tokens forwarded as SSE
/// (`data: {"token":"..."}` + `data: [DONE]`) over an unbounded channel so
/// the sink never blocks the worker thread.
#[utoipa::path(
    post,
    path = "/v1/chat/stream",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "SSE token stream"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Inference failed")
    )
)]
async fn post_chat_stream(
    State(config): State<AppConfig>,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<UnboundedReceiverStream<Result<Event, Infallible>>>, (StatusCode, String)> {
    let worker_req = WorkerChatRequest {
        model: req.model,
        messages: req.messages.iter().map(worker_message).collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let pipe = config.llm.clone();
    // The lease lives inside the blocking task so a client disconnect cannot
    // free the slot mid-generation; the handler returns the SSE body without
    // awaiting the worker, so tokens stream as they are produced.
    tokio::task::spawn_blocking(move || match sd_rip_residency::global().try_reserve() {
        Ok(lease) => {
            let _lease = lease;
            let result = pipe.complete_stream(&worker_req, &mut |token: &str| {
                let data = serde_json::json!({ "token": token }).to_string();
                let _ = tx.send(Ok(Event::default().data(data)));
            });
            if let Err(err) = result {
                let _ = tx.send(Ok(
                    Event::default().event("error").data(format!("chat failed: {err}")),
                ));
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        }
        Err(err) => {
            let _ = tx.send(Ok(Event::default().event("error").data(err.to_string())));
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
        }
    });
    Ok(Sse::new(UnboundedReceiverStream::new(rx)))
}

/// `POST /v1/transcribe`: content-type dispatch (`audio/wav` → `decode_wav`,
/// `audio/pcm-s16le` → raw LE bytes + required `?sample_rate=`), then the
/// sync worker call inside `spawn_blocking`.
#[utoipa::path(
    post,
    path = "/v1/transcribe",
    responses(
        (status = 200, description = "Transcription", body = Transcript),
        (status = 400, description = "Bad audio or missing sample_rate"),
        (status = 415, description = "Use audio/wav or audio/pcm-s16le"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Transcription failed")
    )
)]
async fn post_transcribe(
    State(config): State<AppConfig>,
    Query(query): Query<TranscribeQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Transcript>, (StatusCode, String)> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let (sample_rate, pcm): (u32, Vec<i16>) = match content_type.as_str() {
        "audio/wav" | "audio/wave" | "audio/x-wav" => decode_wav(&body)
            .map_err(|err| (StatusCode::BAD_REQUEST, format!("invalid wav: {err}")))?,
        "audio/pcm-s16le" => {
            let sample_rate = query.sample_rate.ok_or((
                StatusCode::BAD_REQUEST,
                "audio/pcm-s16le requires ?sample_rate=".to_owned(),
            ))?;
            if body.len() % 2 != 0 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "odd audio/pcm-s16le length".to_owned(),
                ));
            }
            let pcm = body
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            (sample_rate, pcm)
        }
        other => {
            return Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!("unsupported content-type {other:?}; use audio/wav or audio/pcm-s16le"),
            ));
        }
    };
    let worker_req = WorkerTranscribeRequest {
        model: query.model.unwrap_or_default(),
        language: query.language,
    };
    let pipe = config.stt.clone();
    let out: WorkerTranscript =
        tokio::task::spawn_blocking(move || match sd_rip_residency::global().try_reserve() {
            Ok(lease) => {
                let _lease = lease;
                pipe.transcribe(&pcm, sample_rate, &worker_req).map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("transcribe failed: {err}"),
                    )
                })
            }
            Err(err) => Err((StatusCode::TOO_MANY_REQUESTS, err.to_string())),
        })
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("transcribe task failed: {err}"),
            )
        })??;
    Ok(Json(Transcript {
        backend_id: out.backend_id,
        text: out.text,
        segments: out
            .segments
            .into_iter()
            .map(|segment| TranscriptSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: segment.text,
            })
            .collect(),
        millis: out.millis,
    }))
}

/// `POST /v1/synthesize`: JSON in, WAV bytes out (`audio/wav`).
#[utoipa::path(
    post,
    path = "/v1/synthesize",
    request_body = SynthesizeRequest,
    responses(
        (status = 200, description = "WAV audio"),
        (status = 429, description = "GPU slot busy"),
        (status = 500, description = "Synthesis failed")
    )
)]
async fn post_synthesize(
    State(config): State<AppConfig>,
    Json(req): Json<SynthesizeRequest>,
) -> Result<Response, (StatusCode, String)> {
    let worker_req = WorkerSynthRequest {
        model: req.model,
        text: req.text,
        voice: req.voice,
        sample_rate: req.sample_rate,
    };
    let pipe = config.tts.clone();
    let audio = tokio::task::spawn_blocking(move || match sd_rip_residency::global().try_reserve()
    {
        Ok(lease) => {
            let _lease = lease;
            pipe.synthesize(&worker_req).map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("synthesize failed: {err}"),
                )
            })
        }
        Err(err) => Err((StatusCode::TOO_MANY_REQUESTS, err.to_string())),
    })
    .await
    .map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("synthesize task failed: {err}"),
        )
    })??;
    let wav = encode_wav(audio.sample_rate, &audio.pcm16);
    Ok(([(header::CONTENT_TYPE, "audio/wav")], wav).into_response())
}
