//! The salvo router under construction. The API branch carries all hoops
//! (metrics → compression → auth) and the affixed config; the UI fallback
//! stays outside them, exactly like axum's `route_layer` exemption.

use salvo::affix_state;
use salvo::prelude::*;
use std::sync::Arc;

use crate::routes::AppConfig;
use crate::{salvo_middleware, salvo_routes};

/// Build the full salvo router with the given shared config.
pub fn router(config: &Arc<AppConfig>) -> Router {
    let mut api = Router::new()
        .push(salvo_routes::apps_ui::router())
        .push(salvo_routes::generation::router())
        .push(salvo_routes::health_downloads::router())
        .push(salvo_routes::models_runtime::router())
        .push(salvo_routes::specials::router())
        .push(salvo_routes::tools_audio::router());
    for hoop in salvo_middleware::api_hoops(config.clone()) {
        api = api.hoop(hoop);
    }

    // Affix at the root: the catch-all UI branch is a sibling of `api` and
    // its handlers need the same config from the Depot.
    Router::new()
        .hoop(affix_state::inject((*config).clone()))
        .push(api)
        .push(Router::with_filter_fn(|_, _| true).goal(salvo_routes::apps_ui::fallback()))
}
