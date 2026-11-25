//! Salvo endpoints for the text/image generation family. Handler logic stays
//! in the axum-style modules; this module only adapts request parts and
//! renders the existing `IntoResponse` results byte-for-byte.

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Query as AxumQuery, State};
use honkbox_api_types::{
    BlenderJobRequest, ImageOpRequest, ImagesGenerationRequest, RecognizeRequest, SynthesizeRequest,
};
use salvo::prelude::*;
use serde::de::DeserializeOwned;

use crate::routes::chat_prep::TranscribeQuery;
use crate::routes::{
    audio_upload, blender, embeddings, image_ops_route, images_generate, recognition, rerank,
    shader_validate, texture_postprocess,
};
use crate::salvo_support::{config, json, query, respond};

/// Route table for this batch.
pub fn router() -> Router {
    Router::new()
        .push(Router::with_path("/v1/transcribe").post(post_transcribe))
        .push(Router::with_path("/v1/synthesize").post(post_synthesize))
        .push(Router::with_path("/v1/rerank").post(post_rerank))
        .push(Router::with_path("/v1/rerank/").post(post_rerank_slash))
        .push(Router::with_path("/v1/embeddings").post(post_embeddings))
        .push(Router::with_path("/v1/image-ops").post(post_image_ops))
        .push(Router::with_path("/v1/images/generations").post(post_images_generations))
        .push(Router::with_path("/v1/jobs/blender").post(post_blender_job))
        .push(Router::with_path("/v1/texture-postprocess").post(post_texture_postprocess))
        .push(Router::with_path("/v1/shader-validate").post(post_shader_validate))
        .push(Router::with_path("/v1/recognize").post(post_recognize))
}

/// Raw body with axum's lifted ceiling; salvo's 64 KiB default would truncate.
async fn raw_body(req: &mut Request) -> Result<Bytes, StatusError> {
    let payload = req
        .payload_with_max_size(usize::MAX)
        .await
        .map_err(|error| StatusError::bad_request().detail(error.to_string()))?;
    Ok(Bytes::from(payload.to_vec()))
}

/// Parse a JSON body once and hand the handler the `Ok` arm it expects.
async fn json_ok<T: DeserializeOwned>(req: &mut Request) -> Result<Json<T>, StatusError> {
    json::<T>(req).await.map(Json)
}

/// Named 400 for failed salvo-side extraction, using axum's response shape.
async fn bad_request(res: &mut Response, error: StatusError) {
    let detail = error.detail.unwrap_or_default();
    respond(res, (axum::http::StatusCode::BAD_REQUEST, detail)).await;
}

#[salvo::handler]
async fn post_transcribe(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let query = match query::<TranscribeQuery>(req) {
        Ok(query) => AxumQuery(query),
        Err(error) => return bad_request(res, error).await,
    };
    let body = match raw_body(req).await {
        Ok(body) => body,
        Err(error) => return bad_request(res, error).await,
    };
    let result =
        audio_upload::post_transcribe(State(config(depot)), query, req.headers().clone(), body)
            .await;
    respond(res, result).await;
}

#[salvo::handler]
async fn post_synthesize(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<SynthesizeRequest>(req).await {
        Ok(body) => body,
        Err(error) => return bad_request(res, error).await,
    };
    let result = audio_upload::post_synthesize(State(config(depot)), body).await;
    respond(res, result).await;
}

#[salvo::handler]
async fn post_rerank(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let body = match raw_body(req).await {
        Ok(body) => body,
        Err(error) => return bad_request(res, error).await,
    };
    respond(res, rerank::post_rerank(body).await).await;
}

#[salvo::handler]
async fn post_rerank_slash(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let body = match raw_body(req).await {
        Ok(body) => body,
        Err(error) => return bad_request(res, error).await,
    };
    respond(res, rerank::post_rerank_slash(body).await).await;
}

#[salvo::handler]
async fn post_embeddings(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match raw_body(req).await {
        Ok(body) => body,
        Err(error) => return bad_request(res, error).await,
    };
    respond(
        res,
        embeddings::post_embeddings(State(config(depot)), body).await,
    )
    .await;
}

#[salvo::handler]
async fn post_image_ops(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<ImageOpRequest>(req).await {
        Ok(body) => Ok(body),
        Err(error) => return bad_request(res, error).await,
    };
    respond(
        res,
        image_ops_route::post_image_ops(State(config(depot)), req.headers().clone(), body).await,
    )
    .await;
}

#[salvo::handler]
async fn post_images_generations(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<ImagesGenerationRequest>(req).await {
        Ok(body) => body,
        Err(error) => return bad_request(res, error).await,
    };
    let result =
        images_generate::post_images_generations(State(config(depot)), req.headers().clone(), body)
            .await;
    respond(res, result).await;
}

#[salvo::handler]
async fn post_blender_job(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<BlenderJobRequest>(req).await {
        Ok(body) => Ok(body),
        Err(error) => return bad_request(res, error).await,
    };
    respond(
        res,
        blender::post_blender_job(State(config(depot)), req.headers().clone(), body).await,
    )
    .await;
}

#[salvo::handler]
async fn post_texture_postprocess(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<texture_postprocess::TexturePostprocessRequest>(req).await {
        Ok(body) => Ok(body),
        Err(error) => return bad_request(res, error).await,
    };
    respond(
        res,
        texture_postprocess::post_texture_postprocess(State(config(depot)), body).await,
    )
    .await;
}

#[salvo::handler]
async fn post_shader_validate(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<shader_validate::ShaderValidateRequest>(req).await {
        Ok(body) => Ok(body),
        Err(error) => return bad_request(res, error).await,
    };
    respond(res, shader_validate::post_shader_validate(body).await).await;
}

#[salvo::handler]
async fn post_recognize(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match json_ok::<RecognizeRequest>(req).await {
        Ok(body) => Ok(body),
        Err(error) => return bad_request(res, error).await,
    };
    respond(
        res,
        recognition::post_recognize(State(config(depot)), body).await,
    )
    .await;
}
