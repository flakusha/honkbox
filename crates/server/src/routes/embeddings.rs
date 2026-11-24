//! Embeddings routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handler and its helpers
//! live here.

use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    EmbeddingsData, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage, ModelCatalogEntry,
    OpenAiError,
};

use super::chat_prep::{admit_then_register, with_job_id};
use super::openai_constraints::openai_error_response;
use super::{AppConfig, resolve_model_entry};

/// Small error payload from [`validate_embeddings_request`]; the route
/// turns it into the `OpenAI` error envelope.
struct EmbeddingsRejection {
    status: StatusCode,
    message: String,
    code: &'static str,
}

impl EmbeddingsRejection {
    fn response(self) -> Response {
        openai_error_response(self.status, self.message, Some(self.code))
    }
}

/// Parse and validate an embeddings request: JSON shape, known model,
/// embeddings capability, and non-empty inputs.
fn validate_embeddings_request(
    config: &AppConfig,
    body: &Bytes,
) -> Result<(ModelCatalogEntry, Vec<String>), EmbeddingsRejection> {
    let req: EmbeddingsRequest =
        serde_json::from_slice(body).map_err(|err| EmbeddingsRejection {
            status: StatusCode::BAD_REQUEST,
            message: format!("invalid JSON body: {err}"),
            code: "invalid_json_body",
        })?;
    let model = req.model.trim();
    let catalog = config.catalog_snapshot();
    let entry = resolve_model_entry(model, &catalog).ok_or_else(|| EmbeddingsRejection {
        status: StatusCode::NOT_FOUND,
        message: format!(
            "unknown model {model:?}: not an alias or registry slug (see GET /v1/models)"
        ),
        code: "model_not_found",
    })?;
    // Capability gate (REQ-4 acceptance): only models the operator declared
    // embeddings-capable load at all — a chat GGUF handed to the embedding
    // path would produce garbage vectors, so it is refused by name.
    if !entry.capabilities.embeddings {
        return Err(EmbeddingsRejection {
            status: StatusCode::BAD_REQUEST,
            message: format!(
                "model {:?} is not embeddings-capable: declare capabilities.embeddings = true for it ([[models.entry]]) and use an embedding GGUF",
                entry.id
            ),
            code: "not_embeddings_capable",
        });
    }
    let inputs = req.input.into_vec();
    if inputs.is_empty() || inputs.iter().any(String::is_empty) {
        return Err(EmbeddingsRejection {
            status: StatusCode::BAD_REQUEST,
            message: "input is required: at least one non-empty string".to_owned(),
            code: "input_required",
        });
    }
    Ok((entry.clone(), inputs))
}

/// `POST /v1/embeddings` (REQ-4): OpenAI-shaped embeddings over the same
/// residency model as chat — one GPU slot, `spawn_blocking` inference.
/// Resolution: alias/slug → 404 envelope when unknown; a model without
/// `capabilities.embeddings` is refused with an explicit 400 (never
/// loaded, never garbage tokens). The pipeline is cached per slug
/// (`EmbedCache`: load-once, idle-TTL reuse, capacity 1) outside the
/// slot; the embed call itself runs inside the lease.
///
/// Errors: 400 bad body / not-embeddings-capable, 404 unknown model, 429
/// busy slot (+`Retry-After` via the shared middleware), 500 embed failure.
#[utoipa::path(
    post,
    path = "/v1/embeddings",
    request_body = EmbeddingsRequest,
    responses(
        (status = 200, description = "Embeddings", body = EmbeddingsResponse),
        (status = 400, description = "Bad request or not an embeddings-capable model", body = OpenAiError),
        (status = 404, description = "Unknown model", body = OpenAiError),
        (status = 429, description = "GPU slot busy", body = OpenAiError),
        (status = 500, description = "Embedding failed", body = OpenAiError)
    )
)]
pub(crate) async fn post_embeddings(State(config): State<AppConfig>, body: Bytes) -> Response {
    let (entry, inputs) = match validate_embeddings_request(&config, &body) {
        Ok(pair) => pair,
        Err(rejection) => return rejection.response(),
    };
    let served_id = entry.id.clone();
    // Pipeline resolution (cache hit = cheap; miss = blocking GGUF load)
    // runs on the blocking pool, outside the GPU slot — same order as the
    // chat routes (load first, then reserve).
    let cache = Arc::clone(&config.embed);
    let load_entry = entry.clone();
    let pipe = match tokio::task::spawn_blocking(move || cache.get(&load_entry)).await {
        Ok(Ok(pipe)) => pipe,
        Ok(Err(message)) => {
            return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
        }
        Err(err) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embed task failed: {err}"),
                None,
            );
        }
    };
    // NEW-3: the slot comes through the admission queue first (REQ-6
    // Option A); only an admitted request gets a registry-visible job
    // (never interruptible: one blocking FFI call). A 429 therefore has no
    // lifecycle events and no X-Job-Id, keeping the shared envelope.
    let (lease, job) = match admit_then_register(
        crate::jobs_registry::global(),
        crate::admission::acquire(honkbox_residency::ResourceKind::Embed).await,
        crate::jobs_registry::JobKind::Embeddings,
        Some(entry.slug.clone()),
        false,
    ) {
        Ok(pair) => pair,
        Err((status, message)) => {
            return openai_error_response(status, message, Some("gpu_slot_busy"));
        }
    };
    let reply = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        pipe.embed(&inputs)
            .map(|embeddings| (embeddings, pipe.prompt_tokens(&inputs)))
            .map_err(|err| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embed failed: {err}"),
                )
            })
    })
    .await
    .map_err(|err| format!("embed task failed: {err}"));
    let (embeddings, prompt_tokens) = match reply {
        Ok(Ok(pair)) => pair,
        Ok(Err((StatusCode::TOO_MANY_REQUESTS, message))) => {
            return openai_error_response(
                StatusCode::TOO_MANY_REQUESTS,
                message,
                Some("gpu_slot_busy"),
            );
        }
        Ok(Err((_, message))) | Err(message) => {
            return openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, message, None);
        }
    };
    with_job_id(
        Json(EmbeddingsResponse {
            object: "list".to_owned(),
            data: embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| EmbeddingsData {
                    object: "embedding".to_owned(),
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    embedding,
                })
                .collect(),
            model: served_id,
            usage: EmbeddingsUsage {
                prompt_tokens,
                total_tokens: u64::from(prompt_tokens),
            },
        })
        .into_response(),
        job.id(),
    )
}

/// REQ-4 reverse gate: an embeddings-capable model never runs through the
/// chat pipeline (garbage tokens dressed up as a completion). `Some`
/// carries the explicit 400 response.
pub(super) fn embeddings_model_gate(entry: &ModelCatalogEntry, model: &str) -> Option<Response> {
    entry.capabilities.embeddings.then(|| {
        openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "model {model:?} is an embeddings model (capabilities.embeddings = true) and cannot serve chat completions"
            ),
            Some("embeddings_model_for_chat"),
        )
    })
}
