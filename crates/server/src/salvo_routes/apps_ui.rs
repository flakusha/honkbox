//! Salvo wrappers for the managed apps / `ComfyUI` read surface, the UI
//! fallback and `/v1/capabilities`. Behaviour stays in the axum handlers;
//! this module adapts request parts and renders through the shared bridge.

use axum::extract::{Path as AxumPath, Query as AxumQuery, State};
use salvo::prelude::*;

use crate::routes::{apps, capabilities};

/// Routes owned by this batch; axum and salvo use the same `{id}` syntax.
pub fn router() -> Router {
    Router::new()
        .push(Router::with_path("/v1/apps").get(get_apps))
        .push(Router::with_path("/v1/apps/{id}").get(get_app))
        .push(Router::with_path("/v1/apps/{id}/start").post(post_app_start))
        .push(Router::with_path("/v1/apps/{id}/stop").post(post_app_stop))
        .push(Router::with_path("/comfyui/start").post(post_comfyui_start))
        .push(Router::with_path("/comfyui/status").get(get_comfyui_status))
        .push(Router::with_path("/comfyui/queue").get(get_comfyui_queue))
        .push(Router::with_path("/comfyui/history/{id}").get(get_comfyui_history))
        .push(Router::with_path("/comfyui/view").get(get_comfyui_view))
        .push(Router::with_path("/v1/capabilities").get(get_capabilities))
}

/// Static/UI fallback. Kept outside the API hoops, like the axum route-layer exemption.
pub fn fallback() -> impl Handler + Clone + 'static {
    #[salvo::handler]
    async fn ui_fallback(req: &mut Request, depot: &mut Depot, res: &mut Response) {
        // axum 0.8 exports this URI as `axum::http::Uri` (formerly `extract::Uri`).
        let uri: axum::http::Uri = req.uri().clone();
        let result = crate::routes::fallback(State(crate::salvo_support::config(depot)), uri).await;
        crate::salvo_support::respond(res, result).await;
    }

    ui_fallback.arc()
}

#[salvo::handler]
async fn get_apps(_req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    crate::salvo_support::respond(res, apps::get_apps().await).await;
}

#[salvo::handler]
async fn get_app(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(error) => return res.render(error),
    };
    crate::salvo_support::respond(res, apps::get_app(AxumPath(id)).await).await;
}

#[salvo::handler]
async fn post_app_start(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(error) => return res.render(error),
    };
    let result = apps::post_app_start(req.headers().clone(), AxumPath(id)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_app_stop(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(error) => return res.render(error),
    };
    let result = apps::post_app_stop(req.headers().clone(), AxumPath(id)).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn post_comfyui_start(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let result = apps::post_comfyui_start(req.headers().clone()).await;
    crate::salvo_support::respond(res, result).await;
}

#[salvo::handler]
async fn get_comfyui_status(_req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    crate::salvo_support::respond(res, apps::get_comfyui_status().await).await;
}

#[salvo::handler]
async fn get_comfyui_queue(_req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    crate::salvo_support::respond(res, apps::get_comfyui_queue().await).await;
}

#[salvo::handler]
async fn get_comfyui_history(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let id = match crate::salvo_support::path::<String>(req, "id") {
        Ok(id) => id,
        Err(error) => return res.render(error),
    };
    crate::salvo_support::respond(res, apps::get_comfyui_history(AxumPath(id)).await).await;
}

#[salvo::handler]
async fn get_comfyui_view(req: &mut Request, _depot: &mut Depot, res: &mut Response) {
    let query = match crate::salvo_support::query::<apps::ViewQuery>(req) {
        Ok(query) => query,
        Err(error) => return res.render(error),
    };
    crate::salvo_support::respond(res, apps::get_comfyui_view(AxumQuery(query)).await).await;
}

#[salvo::handler]
async fn get_capabilities(_req: &mut Request, depot: &mut Depot, res: &mut Response) {
    let config = crate::salvo_support::config(depot);
    crate::salvo_support::respond(res, capabilities::get_capabilities(State(config)).await).await;
}
