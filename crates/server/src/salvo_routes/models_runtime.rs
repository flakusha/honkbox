//! Salvo endpoints for the model catalog, residency controls and mesh reroute.
//!
//! Thin wrappers only: the axum handlers keep every behavior, including the
//! response body and headers, which `salvo_support::respond` renders byte for
//! byte. The mesh state is built once with the router and injected on that
//! route so its cooldowns persist across requests.

use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::response::IntoResponse;
use honkbox_api_types::OpenAiError;
use salvo::affix_state;
use salvo::prelude::*;

use crate::salvo_support::{config, path, respond};

/// GET /v1/models
#[salvo::handler]
async fn get_models_endpoint(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let result = crate::routes::models::get_models(State(config(depot))).await;
    respond(res, result).await;
}

/// GET /v1/catalog
#[salvo::handler]
async fn get_catalog_endpoint(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let result = crate::routes::models::get_catalog(State(config(depot))).await;
    respond(res, result).await;
}

/// POST /v1/models/refresh
#[salvo::handler]
async fn refresh_models_endpoint(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let result = crate::routes::models::refresh_models(State(config(depot))).await;
    respond(res, result).await;
}

/// POST /v1/models/{slug}/unload
#[salvo::handler]
async fn model_unload_endpoint(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let model = match path::<String>(req, "slug") {
        Ok(model) => model,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let result =
        crate::routes::models::post_model_unload(State(config(depot)), AxumPath(model)).await;
    respond(res, result).await;
}

/// DELETE /v1/models/{slug}
#[salvo::handler]
async fn delete_model_endpoint(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let model = match path::<String>(req, "slug") {
        Ok(model) => model,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let result = crate::routes::models::delete_model(State(config(depot)), AxumPath(model)).await;
    respond(res, result).await;
}

/// GET /v1/memory
#[salvo::handler]
async fn get_memory_endpoint(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let result = crate::routes::runtime::get_memory(State(config(depot))).await;
    respond(res, result).await;
}

/// GET /running
#[salvo::handler]
async fn get_running_endpoint(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let result = crate::routes::runtime::get_running(State(config(depot))).await;
    respond(res, result).await;
}

/// POST /api/models/unload
#[salvo::handler]
async fn unload_all_endpoint(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let result = crate::routes::runtime::post_unload_all(State(config(depot))).await;
    respond(res, result).await;
}

/// POST /api/models/unload/{model}
#[salvo::handler]
async fn unload_one_endpoint(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let model = match path::<String>(req, "model") {
        Ok(model) => model,
        Err(err) => {
            res.render(err);
            return;
        }
    };
    let result =
        crate::routes::runtime::post_unload_one(State(config(depot)), AxumPath(model)).await;
    respond(res, result).await;
}

/// POST /v1/mesh/reroute: rebuild the axum request the existing handler
/// expects (method, URI, headers and bounded body bytes) and render its
/// streamed response through the shared bridge.
#[salvo::handler]
#[allow(clippy::single_match_else)] // named mesh error path is intentionally verbose
async fn mesh_reroute_endpoint(req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let state = depot
        .get_typed::<Arc<crate::routes::mesh::RerouteState>>()
        .expect("RerouteState must be affixed to the mesh route")
        .clone();
    // Mirror the mesh handler's own body ceiling instead of salvo's 64KB
    // default; a read failure becomes the same named error the handler uses.
    let body = match req.payload_with_max_size(64 * 1024 * 1024).await {
        Ok(payload) => payload.clone(),
        Err(_) => {
            let err = crate::config::mesh::MeshError::InvalidBody;
            crate::config::mesh::record_mesh_metric(crate::config::mesh::MeshMetricKind::Rejected);
            let status = err.response_status();
            let code = err.code().to_owned();
            let result = (
                status,
                axum::Json(OpenAiError::new(status.as_u16(), err.to_string(), Some(code))),
            )
                .into_response();
            respond(res, result).await;
            return;
        }
    };
    let mut axum_request = axum::extract::Request::builder()
        .method(req.method().clone())
        .uri(req.uri().clone())
        .version(req.version())
        .body(axum::body::Body::from(body))
        .expect("salvo request parts map to axum");
    *axum_request.headers_mut() = req.headers().clone();

    let result = crate::routes::mesh::post_mesh_reroute(State(state), axum_request).await;
    respond(res, result).await;
}

/// Route table; salvo 1.2.1 keeps axum `{param}` paths verbatim.
pub fn router() -> Router {
    let mesh_state = Arc::new(
        crate::routes::mesh::RerouteState::new(crate::routes::mesh::runtime_mesh_config())
            .expect("mesh proxy client builds"),
    );

    Router::new()
        .push(Router::with_path("/v1/models").get(get_models_endpoint))
        .push(Router::with_path("/v1/catalog").get(get_catalog_endpoint))
        .push(Router::with_path("/v1/models/refresh").post(refresh_models_endpoint))
        .push(Router::with_path("/v1/models/{slug}/unload").post(model_unload_endpoint))
        .push(Router::with_path("/v1/models/{slug}").delete(delete_model_endpoint))
        .push(Router::with_path("/v1/memory").get(get_memory_endpoint))
        .push(Router::with_path("/running").get(get_running_endpoint))
        .push(Router::with_path("/api/models/unload").post(unload_all_endpoint))
        .push(Router::with_path("/api/models/unload/{model}").post(unload_one_endpoint))
        .push(
            Router::with_path(crate::routes::mesh::MESH_REROUTE_PATH)
                .hoop(affix_state::inject(mesh_state))
                .post(mesh_reroute_endpoint),
        )
}