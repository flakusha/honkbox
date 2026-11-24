//! Salvo endpoints for the job/SSE/chat/multipart/streamed-body route batch.
//!
//! Every wrapper calls the existing axum-style handler via `salvo_support`;
//! SSE and streamed response bodies are forwarded frame-by-frame with
//! `Response::stream`, never buffered through `IntoResponse`.

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Multipart, Request as AxumRequest, State};
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse as _, Response as AxumResponse};
use honkbox_api_types::ChatRequest;
use salvo::http::StatusCode as SalvoStatusCode;
use salvo::prelude::{Depot, Request, Response, Router};

use crate::routes::AUDIO_BODY_LIMIT_BYTES;
use crate::routes::audio_transcriptions::AudioSampleRateQuery;
use crate::salvo_support::{config, path, query, respond};

/// Copy the axum status/headers, then forward the body as a live stream.
/// Dropping the salvo stream drops the axum body, which keeps the existing
/// SSE disconnect/cancel guards and file-blob lifetimes intact.
fn stream_response(res: &mut Response, response: AxumResponse) {
    let status = response.status();
    let headers = response.headers().clone();
    res.status_code =
        Some(SalvoStatusCode::from_u16(status.as_u16()).expect("axum status maps to salvo"));
    res.headers.clear();
    for (name, value) in headers {
        if let Some(name) = name {
            res.headers.append(name, value);
        }
    }
    res.stream(response.into_body().into_data_stream());
}

/// Salvo payloads default to a 64 KiB cap; the axum routes used lifted
/// body limits, so read the whole body under an explicit unlimited cap.
async fn raw_bytes(req: &mut Request) -> Result<Bytes, String> {
    Ok(req
        .payload_with_max_size(usize::MAX)
        .await
        .map_err(|err| format!("unreadable request body: {err}"))?
        .clone())
}

async fn json_body<T: serde::de::DeserializeOwned>(req: &mut Request) -> Result<T, String> {
    req.parse_json_with_max_size(usize::MAX)
        .await
        .map_err(|err| err.to_string())
}

/// `axum::Multipart` is not constructible from parts, but its `FromRequest`
/// impl is: feed it a synthetic axum request carrying salvo's headers and
/// payload, with the same body limit the axum router applied. All existing
/// field parsing, validation and error envelopes stay in one place.
async fn axum_multipart(req: &mut Request, limit: usize) -> Result<Multipart, String> {
    let bytes = raw_bytes(req).await?;
    let headers = req.headers().clone();
    let mut axum_req = AxumRequest::builder()
        .method(Method::POST)
        .uri("/")
        .body(Body::from(bytes))
        .expect("a static POST request is valid");
    *axum_req.headers_mut() = headers;
    DefaultBodyLimit::max(limit).apply(&mut axum_req);
    <Multipart as axum::extract::FromRequest<()>>::from_request(axum_req, &())
        .await
        .map_err(|err| format!("invalid multipart body: {err}"))
}

async fn bad_request(res: &mut Response, message: String) {
    respond(res, (StatusCode::BAD_REQUEST, message)).await;
}

#[salvo::handler]
async fn jobs(_req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    respond(res, crate::routes::jobs::get_jobs().await).await;
}

#[salvo::handler]
async fn job_cancel(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    respond(
        res,
        crate::routes::jobs::post_job_cancel(axum::extract::Path(id)).await,
    )
    .await;
}

#[salvo::handler]
async fn job_delete(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    respond(
        res,
        crate::routes::jobs::delete_job(axum::extract::Path(id)).await,
    )
    .await;
}

#[salvo::handler]
async fn events(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    // T4 cursor: forward `?since=` / `Last-Event-ID` so the salvo front keeps
    // the reconnect-replay contract of the axum handler (unparseable cursors
    // resume live-only, same as axum).
    let params = query(req).unwrap_or_default();
    let headers = req.headers().clone();
    stream_response(
        res,
        crate::routes::jobs::get_events(axum::extract::Query(params), headers)
            .await
            .into_response(),
    );
}

#[salvo::handler]
async fn chat(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let request = match json_body::<ChatRequest>(req).await {
        Ok(request) => request,
        Err(message) => return bad_request(res, message).await,
    };
    respond(
        res,
        crate::routes::chat::post_chat(State(config(depot)), axum::Json(request)).await,
    )
    .await;
}

#[salvo::handler]
async fn chat_stream(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let request = match json_body::<ChatRequest>(req).await {
        Ok(request) => request,
        Err(message) => return bad_request(res, message).await,
    };
    match crate::routes::chat::post_chat_stream(State(config(depot)), axum::Json(request)).await {
        Ok(response) => stream_response(res, response),
        Err(err) => respond(res, err).await,
    }
}

#[salvo::handler]
async fn chat_completions(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let bytes = match raw_bytes(req).await {
        Ok(bytes) => bytes,
        Err(message) => return bad_request(res, message).await,
    };
    stream_response(
        res,
        crate::routes::openai_chat::post_chat_completions(State(config(depot)), bytes).await,
    );
}

#[salvo::handler]
async fn audio_transcriptions(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let query = match query::<AudioSampleRateQuery>(req) {
        Ok(query) => query,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    let multipart = match axum_multipart(req, AUDIO_BODY_LIMIT_BYTES).await {
        Ok(multipart) => multipart,
        Err(message) => return bad_request(res, message).await,
    };
    stream_response(
        res,
        crate::routes::audio_transcriptions::post_audio_transcriptions(
            State(config(depot)),
            axum::extract::Query(query),
            multipart,
        )
        .await,
    );
}

#[salvo::handler]
async fn audio_translations(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let query = match query::<AudioSampleRateQuery>(req) {
        Ok(query) => query,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    let multipart = match axum_multipart(req, AUDIO_BODY_LIMIT_BYTES).await {
        Ok(multipart) => multipart,
        Err(message) => return bad_request(res, message).await,
    };
    stream_response(
        res,
        crate::routes::audio_transcriptions::post_audio_translations(
            State(config(depot)),
            axum::extract::Query(query),
            multipart,
        )
        .await,
    );
}

#[salvo::handler]
async fn images_edits(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let headers = req.headers().clone();
    let multipart = match axum_multipart(req, AUDIO_BODY_LIMIT_BYTES).await {
        Ok(multipart) => multipart,
        Err(message) => return bad_request(res, message).await,
    };
    respond(
        res,
        crate::routes::images_edit::post_images_edits(State(config(depot)), headers, multipart)
            .await,
    )
    .await;
}

#[salvo::handler]
async fn job_bundle(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let id = match path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    let signed = match query::<crate::routes::signed_urls::SignedQuery>(req) {
        Ok(signed) => signed,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    stream_response(
        res,
        crate::routes::job_content::get_job_bundle(
            State(config(depot)),
            axum::extract::Path(id),
            axum::extract::Query(signed),
        )
        .await,
    );
}

#[salvo::handler]
async fn job_content(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let id = match path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    let name = match path::<String>(req, "name") {
        Ok(name) => name,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    let signed = match query::<crate::routes::signed_urls::SignedQuery>(req) {
        Ok(signed) => signed,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    stream_response(
        res,
        crate::routes::job_content::get_job_content(
            State(config(depot)),
            axum::extract::Path((id, name)),
            axum::extract::Query(signed),
        )
        .await,
    );
}

#[salvo::handler]
async fn artifact_post(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let bytes = match raw_bytes(req).await {
        Ok(bytes) => bytes,
        Err(message) => return bad_request(res, message).await,
    };
    let headers = req.headers().clone();
    let mut axum_req = AxumRequest::builder()
        .method(Method::POST)
        .uri("/")
        .body(Body::from(bytes))
        .expect("a static POST request is valid");
    *axum_req.headers_mut() = headers;
    respond(
        res,
        crate::artifacts::post_artifact(State(config(depot)), axum_req).await,
    )
    .await;
}

#[salvo::handler]
async fn artifact_bytes(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let id = match path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => return bad_request(res, err.to_string()).await,
    };
    let headers = req.headers().clone();
    stream_response(
        res,
        crate::artifacts::get_artifact_bytes(
            State(config(depot)),
            Method::GET,
            headers,
            axum::extract::Path(id),
        )
        .await,
    );
}

/// All paths use salvo's native router with the same `{param}` templates
/// as axum; no axum router or middleware is involved.
pub fn router() -> Router {
    Router::new()
        .push(Router::with_path("v1/jobs").get(jobs))
        .push(Router::with_path("v1/jobs/{id}/cancel").post(job_cancel))
        .push(Router::with_path("v1/jobs/{id}").delete(job_delete))
        .push(Router::with_path("v1/events").get(events))
        .push(Router::with_path("v1/chat").post(chat))
        .push(Router::with_path("v1/chat/stream").post(chat_stream))
        .push(Router::with_path("v1/chat/completions").post(chat_completions))
        .push(Router::with_path("v1/audio/transcriptions").post(audio_transcriptions))
        .push(Router::with_path("v1/audio/translations").post(audio_translations))
        .push(Router::with_path("v1/images/edits").post(images_edits))
        .push(Router::with_path("v1/jobs/{id}/bundle.zip").get(job_bundle))
        .push(Router::with_path("v1/jobs/{id}/content/{name}").get(job_content))
        .push(Router::with_path("v1/artifacts").post(artifact_post))
        .push(Router::with_path("v1/artifacts/{id}/bytes").get(artifact_bytes))
}
