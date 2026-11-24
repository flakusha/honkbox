//! Salvo endpoints for the mixed cutover. One module per disjoint route
//! batch; each exposes `router()` with its own path filters. `AppConfig` is
//! affixed once by `crate::salvo_app`, so handlers read it from the `Depot`.

pub(crate) mod apps_ui;
pub(crate) mod generation;
pub(crate) mod health_downloads;
pub(crate) mod models_runtime;
pub(crate) mod specials;
pub(crate) mod tools_audio;
