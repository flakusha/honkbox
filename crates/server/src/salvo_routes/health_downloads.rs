//! Salvo endpoints for health, version, `OpenAPI`, metrics, downloads, queue
//! and content URLs during the mixed axum → salvo cutover. Each wrapper
//! calls the existing axum handler unchanged and renders its response through
//! the `salvo_support` bridge.

use axum::Json;
use axum::extract::{Path, State};
use honkbox_api_types::DownloadRequest;
use salvo::prelude::{Depot, Request, Response, Router, StatusError};

use crate::routes::content_urls::ContentUrlRequest;
use crate::salvo_support;

/// GET /health: content negotiation stays in the axum handler.
#[salvo::handler]
async fn health(req: &mut Request, res: &mut Response) {
    let result = crate::routes::health::health(req.headers().clone()).await;
    salvo_support::respond(res, result).await;
}

/// GET /v1/openapi.json: same schema document as the axum route.
#[salvo::handler]
async fn openapi_json(res: &mut Response) {
    let result = crate::routes::health::openapi_json().await;
    salvo_support::respond(res, result).await;
}

/// GET /metrics: Prometheus exposition from the shared metric sources.
#[salvo::handler]
async fn metrics(res: &mut Response) {
    let result = crate::routes::health::metrics_endpoint().await;
    salvo_support::respond(res, result).await;
}

/// POST /v1/downloads.
#[salvo::handler]
async fn post_download(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let config = salvo_support::config(depot);
    let request = salvo_support::json::<DownloadRequest>(req).await?;
    let result = crate::routes::downloads::post_download(State(config), Json(request)).await;
    salvo_support::respond(res, result).await;
    Ok(())
}

/// GET /v1/downloads.
#[salvo::handler]
async fn list_downloads(depot: &mut Depot, res: &mut Response) {
    let config = salvo_support::config(depot);
    let result = crate::routes::downloads::list_downloads(State(config)).await;
    salvo_support::respond(res, result).await;
}

/// GET /v1/downloads/{id}.
#[salvo::handler]
async fn get_download(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let id = salvo_support::path::<String>(req, "id")?;
    let config = salvo_support::config(depot);
    let result = crate::routes::downloads::get_download(State(config), Path(id)).await;
    salvo_support::respond(res, result).await;
    Ok(())
}

/// DELETE /v1/downloads/{id}.
#[salvo::handler]
async fn delete_download(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let id = salvo_support::path::<String>(req, "id")?;
    let config = salvo_support::config(depot);
    let result = crate::routes::downloads::delete_download(State(config), Path(id)).await;
    salvo_support::respond(res, result).await;
    Ok(())
}

/// GET /v1/status/warnings: same bounded document as the axum route.
#[salvo::handler]
async fn status_warnings(res: &mut Response) {
    let result = axum::Json(crate::routes::status::warnings_document());
    salvo_support::respond(res, result).await;
}

/// GET /v1/queue.
#[salvo::handler]
async fn get_queue(res: &mut Response) {
    let result = crate::routes::queue::get_queue().await;
    salvo_support::respond(res, result).await;
}

/// POST /v1/content-urls.
#[salvo::handler]
async fn post_content_urls(req: &mut Request, res: &mut Response) -> Result<(), StatusError> {
    let request = salvo_support::json::<ContentUrlRequest>(req).await?;
    let result = crate::routes::content_urls::post_content_urls(Json(request)).await;
    salvo_support::respond(res, result).await;
    Ok(())
}

/// All routes owned by this batch.
pub fn router() -> Router {
    Router::new()
        .push(Router::with_path("health").get(health))
        .push(Router::with_path("api/version").get(crate::salvo_version::get_version))
        .push(Router::with_path("v1/openapi.json").get(openapi_json))
        .push(Router::with_path("metrics").get(metrics))
        .push(
            Router::with_path("v1/downloads")
                .get(list_downloads)
                .post(post_download),
        )
        .push(
            Router::with_path("v1/downloads/{id}")
                .get(get_download)
                .delete(delete_download),
        )
        .push(Router::with_path("v1/queue").get(get_queue))
        .push(Router::with_path("v1/status/warnings").get(status_warnings))
        .push(Router::with_path("v1/content-urls").post(post_content_urls))
}
