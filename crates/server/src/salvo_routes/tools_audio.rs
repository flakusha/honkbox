//! Salvo endpoints for external tools, Fizgig jobs, asset packs, audio and OCR.
//!
//! Wrappers render the existing axum handlers' `IntoResponse` byte-for-byte
//! through `salvo_support::respond`; no handler logic is duplicated here.

use axum::{
    Json,
    extract::{Path, Query, State},
};
use honkbox_api_types::{FizgigTrainRequest, OcrIngestRequest, SpeechRequest};
use salvo::prelude::*;

use crate::catalog::asset_packs::FetchRequest;
use crate::routes::{asset_packs, audio_speech, audio_voices, external_tools, fizgig, ocr_ingest};

/// Route table for this batch. Static paths precede `/{id}` so a POST to
/// `/v1/external-tools/reload` never falls through to the dynamic GET route.
pub fn router() -> Router {
    Router::new()
        .push(Router::with_path("/v1/external-tools").get(get_external_tools))
        .push(Router::with_path("/v1/external-tools/reload").post(post_external_tools_reload))
        .push(Router::with_path("/v1/external-tools/{id}").get(get_external_tool))
        .push(Router::with_path("/v1/fizgig/train").post(post_fizgig_train))
        .push(Router::with_path("/v1/fizgig/jobs/{id}").get(get_fizgig_job))
        .push(Router::with_path("/v1/fizgig/jobs/{id}/cancel").post(cancel_fizgig_job))
        .push(Router::with_path("/v1/asset-packs/fetch").post(post_asset_pack_fetch))
        .push(Router::with_path("/v1/audio/speech").post(post_audio_speech))
        .push(Router::with_path("/v1/audio/voices").get(get_audio_voices))
        .push(Router::with_path("/v1/ocr/ingest").post(post_ocr_ingest))
}

#[salvo::handler]
async fn get_external_tools(_req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let result = external_tools::get_external_tools().await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_external_tools_reload(_req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let result = external_tools::post_external_tools_reload().await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn get_external_tool(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let query = match crate::salvo_support::query::<external_tools::ExternalToolQuery>(req) {
        Ok(query) => query,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let result = external_tools::get_external_tool(Path(id), Query(query)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_fizgig_train(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match crate::salvo_support::json::<FizgigTrainRequest>(req).await {
        Ok(body) => body,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let config = crate::salvo_support::config(depot);
    let result = fizgig::post_fizgig_train(State(config), Ok(Json(body))).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn get_fizgig_job(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let result = fizgig::get_fizgig_job(Path(id)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn cancel_fizgig_job(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let result = fizgig::cancel_fizgig_job(Path(id)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_asset_pack_fetch(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match crate::salvo_support::json::<FetchRequest>(req).await {
        Ok(body) => body,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let config = crate::salvo_support::config(depot);
    let result = asset_packs::post_asset_pack_fetch(State(config), Json(body)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_audio_speech(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match crate::salvo_support::json::<SpeechRequest>(req).await {
        Ok(body) => body,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let config = crate::salvo_support::config(depot);
    let result = audio_speech::post_audio_speech(State(config), Json(body)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn get_audio_voices(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let config = crate::salvo_support::config(depot);
    let result = audio_voices::get_audio_voices(State(config)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_ocr_ingest(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let body = match crate::salvo_support::json::<OcrIngestRequest>(req).await {
        Ok(body) => body,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let config = crate::salvo_support::config(depot);
    let result = ocr_ingest::post_ocr_ingest(State(config), Ok(Json(body))).await;
    crate::salvo_support::respond(res, result).await;
}
