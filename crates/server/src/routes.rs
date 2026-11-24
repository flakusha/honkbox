//! HTTP surface: `/health`, `/api/version`, `/v1/models`(+`/refresh`),
//! `/metrics`, `/v1/openapi.json`, chat/transcribe/synthesize, image
//! generations (REQ-10, incl. the `ControlNet` conditioned render),
//! `/v1/jobs` (+`DELETE /v1/jobs/{id}`, `POST …/cancel`) and the
//! `/v1/events` SSE lifecycle feed, UI fallback.
//!
//! The `OpenAPI` document is served as plain JSON via `Json` — no
//! `utoipa-axum` router glue.

#[allow(unused_imports)] // handler imports vary across #[path]-included test binaries
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, Request, State},
    http::{Extensions, HeaderMap, StatusCode, Version as HttpVersion, header},
    middleware::{self as axum_middleware, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{delete, get, post},
};

/// Re-exported auth settings (REQ-7): `app` consumers (integration tests,
/// `main`) build `AppConfig.auth` from these.
#[allow(unused_imports)] // which name is used varies per #[path]-included test binary
pub use crate::config::{AuthConfig, AuthMode};
use tower_http::compression::predicate::SizeAbove;
use tower_http::compression::{CompressionLayer, DefaultPredicate, Predicate as _};

#[path = "routes/external_tools.rs"]
mod external_tools;

#[path = "routes/apps.rs"]
mod apps;

#[allow(unused_imports)] // handler names are used by .route() and the OpenApi derive
use self::apps::{
    __path_get_app, __path_get_apps, __path_get_comfyui_history, __path_get_comfyui_queue,
    __path_get_comfyui_status, __path_get_comfyui_view, __path_post_app_start,
    __path_post_app_stop, get_app, get_apps, get_comfyui_history, get_comfyui_queue,
    get_comfyui_status, get_comfyui_view, post_app_start, post_app_stop,
};

#[allow(unused_imports)] // main.rs constructs the manager for install_manager
pub(crate) use self::apps::Manager as ManagedAppManager;
#[allow(unused_imports)] // main.rs installs the managed app manager at startup
pub(crate) use self::apps::install_manager;
#[cfg(test)]
#[allow(dead_code, unused_imports)]
pub(crate) use self::apps::{AppState, set_manager_for_tests};
#[allow(unused_imports)] // main.rs is the only caller; #[path] test harnesses omit it
pub(crate) use self::external_tools::init_external_tools;
#[cfg(test)] // the fn itself is test-gated; the plain bin build must not see this
#[allow(unused_imports)] // the capabilities contract test is the only caller
pub(crate) use self::external_tools::set_registry_for_tests;
#[allow(unused_imports)] // referenced by the OpenApi derive's generated types
use self::external_tools::{
    __path_get_external_tool, __path_get_external_tools, __path_post_external_tools_reload,
    get_external_tool, get_external_tools, post_external_tools_reload,
};

#[path = "routes/health.rs"]
mod health;

#[path = "routes/capabilities.rs"]
mod capabilities;

#[path = "routes/models.rs"]
mod models;

#[path = "routes/runtime.rs"]
mod runtime;

#[allow(unused_imports)]
use self::capabilities::{__path_get_capabilities, get_capabilities};
#[allow(unused_imports)] // used by capability contract tests via routes::
pub use self::capabilities::{ADMITTING_ROUTES, admitting_route};
#[allow(unused_imports)]
// handler names are used by .route() and the OpenApi derive's generated types
use self::health::{
    __path_get_version, __path_health, __path_metrics_endpoint, __path_openapi_json, get_version,
    health, metrics_endpoint, openapi_json,
};
#[allow(unused_imports)]
use self::models::{
    __path_delete_model, __path_get_catalog, __path_get_models, __path_post_model_unload,
    __path_refresh_models, delete_model, get_catalog, get_models, post_model_unload,
    refresh_models,
};
#[allow(unused_imports)]
use self::runtime::{
    __path_get_memory, __path_get_running, __path_post_unload_all, __path_post_unload_one,
    get_memory, get_running, post_unload_all, post_unload_one,
};

#[path = "routes/catalog.rs"]
mod catalog;

#[path = "routes/bindings.rs"]
mod bindings;

#[path = "routes/downloads.rs"]
mod downloads;

#[path = "routes/asset_packs.rs"]
mod asset_packs;

#[path = "routes/ui.rs"]
mod ui;

#[path = "routes/blender.rs"]
mod blender;

#[path = "routes/blender_stages.rs"]
mod blender_stages;

#[path = "routes/blender_exec.rs"]
mod blender_exec;

#[path = "routes/job_content.rs"]
pub(super) mod job_content;

#[path = "routes/job_records.rs"]
mod job_records;

#[allow(unused_imports)] // used by .route() and the OpenApi derive's generated types
use self::job_records::{
    __path_get_job_record, __path_get_job_records, get_job_record, get_job_records,
};

#[path = "routes/texture_postprocess.rs"]
mod texture_postprocess;

#[path = "routes/mesh.rs"]
mod mesh;

#[allow(unused_imports)] // main.rs installs the operator config at startup
pub(crate) use self::mesh::install_mesh_config;
#[allow(unused_imports)] // the OpenApi derive references the generated path symbol
use self::mesh::{__path_post_mesh_reroute, post_mesh_reroute};

#[path = "routes/shader_validate.rs"]
mod shader_validate;

#[allow(unused_imports)]
// handler names are used by .route() and the OpenApi derive's generated types
use self::blender::{__path_post_blender_job, post_blender_job};
#[allow(unused_imports)]
// handler names are used by .route() and the OpenApi derive's generated types
use self::job_content::{
    __path_get_job_bundle, __path_get_job_content, get_job_bundle, get_job_content,
};
#[allow(unused_imports)]
use self::shader_validate::{__path_post_shader_validate, post_shader_validate};
#[allow(unused_imports)]
use self::texture_postprocess::{__path_post_texture_postprocess, post_texture_postprocess};

// Supervised external processes (TASK-process-supervisor): the process
// group teardown is a unix mechanism, so the module is unix-only. It is
// declared HERE (not at the crate root) because the `#[path]`-included
// test harnesses pull in `routes.rs`, and the Blender job route is the
// supervisor's first caller.
#[cfg(unix)]
#[path = "supervisor.rs"]
mod supervisor;

#[allow(unused_imports)]
use self::asset_packs::{__path_post_asset_pack_fetch, post_asset_pack_fetch};
#[allow(unused_imports)] // main.rs and in-file tests use these via routes::
pub use self::bindings::{ModelBindings, bind_entries, build_catalog};
#[allow(unused_imports)] // main.rs and in-file tests use these via routes::
pub use self::catalog::{refresh_catalog, rescan_catalog, rescan_on_downloads};
#[allow(unused_imports)]
use self::downloads::{
    __path_delete_download, __path_get_download, __path_list_downloads, __path_post_download,
    delete_download, get_download, list_downloads, post_download,
};
#[allow(unused_imports)] // shared by later route slices and the UI fallback registration
pub(super) use self::ui::{fallback, resolve_model_slug, validate_model};

#[path = "routes/chat_prep.rs"]
mod chat_prep;

#[path = "routes/chat.rs"]
mod chat;

#[path = "routes/openai_constraints.rs"]
mod openai_constraints;

#[path = "routes/embeddings.rs"]
mod embeddings;

#[path = "routes/images_recipe.rs"]
mod images_recipe;

#[path = "routes/images_inputs.rs"]
mod images_inputs;

#[path = "routes/images_controlnet.rs"]
mod images_controlnet;

#[path = "routes/images_generate.rs"]
mod images_generate;

#[path = "routes/images_edit.rs"]
mod images_edit;

#[path = "routes/openai_chat.rs"]
mod openai_chat;

#[path = "routes/openai_stream.rs"]
mod openai_stream;

#[path = "routes/audio_upload.rs"]
mod audio_upload;

#[path = "routes/audio_multipart.rs"]
mod audio_multipart;

#[path = "routes/audio_transcriptions.rs"]
mod audio_transcriptions;

#[path = "routes/audio_subtitles.rs"]
mod audio_subtitles;

#[path = "routes/jobs.rs"]
mod jobs;

#[path = "routes/audio_voices.rs"]
mod audio_voices;

#[path = "routes/image_ops_limits.rs"]
mod image_ops_limits;

#[path = "routes/image_ops_dedup.rs"]
mod image_ops_dedup;

#[path = "routes/image_ops_params.rs"]
mod image_ops_params;

#[path = "routes/image_ops_exec.rs"]
mod image_ops_exec;

#[path = "routes/image_ops_settle.rs"]
mod image_ops_settle;

#[path = "routes/image_ops_route.rs"]
mod image_ops_route;

#[path = "routes/recognition.rs"]
mod recognition;

#[path = "routes/ocr_ingest.rs"]
pub(crate) mod ocr_ingest;

#[path = "routes/audio_speech.rs"]
mod audio_speech;

#[path = "routes/openapi.rs"]
mod openapi;

#[path = "routes/state.rs"]
mod state;

#[path = "routes/middleware.rs"]
mod middleware;

#[path = "runtime_config.rs"]
pub(crate) mod runtime_config;

#[path = "nsfw.rs"]
pub(crate) mod nsfw;

#[path = "mesh_discovery.rs"]
pub(crate) mod mesh_discovery;

#[path = "request_cache.rs"]
pub(crate) mod request_cache;

#[path = "signed_urls.rs"]
pub(crate) mod signed_urls;

#[path = "routes/queue.rs"]
mod queue;

#[path = "routes/rerank.rs"]
mod rerank;

#[path = "routes/python_sidecar.rs"]
mod python_sidecar;

#[path = "routes/fizgig.rs"]
mod fizgig;

#[cfg(unix)]
#[path = "routes/fizgig_run.rs"]
mod fizgig_run;

#[path = "routes/content_urls.rs"]
mod content_urls;

#[allow(unused_imports)]
use self::content_urls::{__path_post_content_urls, post_content_urls};
#[allow(unused_imports)]
use self::fizgig::{
    __path_cancel_fizgig_job, __path_get_fizgig_job, __path_post_fizgig_train, cancel_fizgig_job,
    get_fizgig_job, post_fizgig_train,
};
#[allow(unused_imports)]
use self::queue::{__path_get_queue, get_queue};
#[allow(unused_imports)]
use self::rerank::{__path_post_rerank, __path_post_rerank_slash, post_rerank, post_rerank_slash};

#[allow(unused_imports)] // re-exported for integration tests
#[cfg(test)]
pub use self::middleware::set_env_var;
pub use self::openapi::{ApiDoc, EmbeddedUi};
pub use self::state::AppConfig;
pub(crate) use self::state::{
    AUDIO_BODY_LIMIT_BYTES, COMPRESSION_MIN_BYTES, SPEECH_INPUT_MAX_CHARS,
    partial_content_stays_plain,
};

#[allow(unused_imports)]
use self::audio_speech::{__path_post_audio_speech, post_audio_speech};
#[allow(unused_imports)] // subtitle helpers are pinned by in-file tests
use self::audio_subtitles::{render_srt, render_vtt, srt_timestamp, vtt_timestamp};
#[allow(unused_imports)]
use self::audio_transcriptions::{
    __path_post_audio_transcriptions, __path_post_audio_translations, post_audio_transcriptions,
    post_audio_translations,
};
#[allow(unused_imports)]
use self::audio_upload::{
    __path_post_synthesize, __path_post_transcribe, post_synthesize, post_transcribe, run_synthesis,
};
#[allow(unused_imports)]
use self::audio_voices::{__path_get_audio_voices, get_audio_voices};
#[allow(unused_imports)]
use self::chat::{
    __path_post_chat, __path_post_chat_stream, CancelOnDrop, free_gpu_lease, post_chat,
    post_chat_stream,
};
#[allow(unused_imports)] // helpers are consumed by remaining route families and in-file tests
use self::chat_prep::{
    JOB_ID_HEADER, TranscribeQuery, admit_then_register, clamp_ttl_seconds,
    completions_request_gates, with_job_id, worker_messages,
};
#[allow(unused_imports)]
use self::embeddings::{__path_post_embeddings, embeddings_model_gate, post_embeddings};
#[allow(unused_imports)]
use self::image_ops_route::{__path_post_image_ops, post_image_ops};
#[allow(unused_imports)]
use self::images_controlnet::resolve_controlnet;
#[allow(unused_imports)]
use self::images_edit::{__path_post_images_edits, post_images_edits};
#[allow(unused_imports)] // shared with sibling route modules and in-file tests
use self::images_generate::{
    __path_post_images_generations, GenerateFailure, IMAGE_SEED_BASE, IMAGE_STEPS,
    generate_image_bytes, image_cancel_before_start, image_cancel_mid_generation,
    image_scheduler_name, new_image_cancel_flag, post_images_generations, write_generation_bundle,
};
#[allow(unused_imports)]
use self::images_inputs::{
    parse_output_compression, parse_output_format, resolve_loras, validate_images_request,
};
#[allow(unused_imports)]
use self::images_recipe::{resolve_sd_model, validate_recipe};
#[allow(unused_imports)]
use self::jobs::{
    __path_delete_job, __path_get_events, __path_get_jobs, __path_post_job_cancel, delete_job,
    get_events, get_jobs, post_job_cancel,
};
#[allow(unused_imports)] // used by .route() and the OpenApi derive's generated types
use self::ocr_ingest::{__path_post_ocr_ingest, post_ocr_ingest};
#[allow(unused_imports)] // shared with sibling route modules
use self::openai_chat::{
    __path_post_chat_completions, completions_json, epoch_secs, finish_reason,
    post_chat_completions, resolve_model_entry, usage_of,
};
#[allow(unused_imports)] // crate::artifacts calls this through routes::
pub use self::openai_constraints::openai_error_response;
#[allow(unused_imports)]
use self::openai_constraints::{ChatConstraints, resolve_constraints};
#[allow(unused_imports)] // shared with sibling route modules
use self::openai_stream::{
    argument_frames, chunk_frame, completions_stream, emit_content_stream, emit_tool_stream,
    id_frame, send_delta, send_token_delta, tool_delta_frames,
};
#[allow(unused_imports)] // pinned by in-file tests
use self::recognition::{__path_post_recognize, post_recognize, recognition_501_message};

/// Build the full router.
// The route list is the map of the HTTP surface; splitting it for line count
// hides which handler serves which path from the same view.
#[allow(clippy::too_many_lines)]
pub fn app(config: AppConfig) -> Router {
    let mesh_router = Router::new()
        .route(mesh::MESH_REROUTE_PATH, post(post_mesh_reroute))
        .with_state(std::sync::Arc::new(
            mesh::RerouteState::new(mesh::runtime_mesh_config()).expect("mesh proxy client builds"),
        ));
    Router::new()
        .route("/health", get(health))
        .route("/api/version", get(get_version))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/v1/models", get(get_models))
        .route("/v1/models/refresh", post(refresh_models))
        .route("/v1/models/{slug}/unload", post(post_model_unload))
        .route("/v1/models/{slug}", delete(delete_model))
        .route("/v1/catalog", get(get_catalog))
        .route("/v1/memory", get(get_memory))
        .route("/v1/openapi.json", get(openapi_json))
        .route(
            "/v1/external-tools",
            get(external_tools::get_external_tools),
        )
        .route(
            "/v1/external-tools/reload",
            post(external_tools::post_external_tools_reload),
        )
        .route(
            "/v1/external-tools/{id}",
            get(external_tools::get_external_tool),
        )
        .route("/v1/apps", get(get_apps))
        .route("/v1/apps/{id}", get(get_app))
        .route("/v1/apps/{id}/start", post(post_app_start))
        .route("/v1/apps/{id}/stop", post(post_app_stop))
        .route("/comfyui/status", get(get_comfyui_status))
        .route("/comfyui/queue", get(get_comfyui_queue))
        .route("/comfyui/history/{id}", get(get_comfyui_history))
        .route("/comfyui/view", get(get_comfyui_view))
        .route("/v1/chat", post(post_chat))
        .route("/v1/chat/stream", post(post_chat_stream))
        .route("/v1/chat/completions", post(post_chat_completions))
        .route("/v1/embeddings", post(post_embeddings))
        .route("/v1/images/generations", post(post_images_generations))
        .route("/v1/images/edits", post(post_images_edits))
        .route("/v1/image-ops", post(post_image_ops))
        .route("/v1/recognize", post(post_recognize))
        .route("/v1/ocr/ingest", post(post_ocr_ingest))
        .route("/v1/transcribe", post(post_transcribe))
        .route("/v1/synthesize", post(post_synthesize))
        .route("/v1/audio/voices", get(get_audio_voices))
        .route("/v1/audio/transcriptions", post(post_audio_transcriptions))
        .route("/v1/audio/translations", post(post_audio_translations))
        .route("/v1/audio/speech", post(post_audio_speech))
        .route("/v1/jobs", get(get_jobs))
        .route("/v1/jobs/{id}/cancel", post(post_job_cancel))
        .route("/v1/jobs/{id}", delete(delete_job))
        .route("/v1/jobs/blender", post(post_blender_job))
        .route("/v1/texture-postprocess", post(post_texture_postprocess))
        .route("/v1/shader-validate", post(post_shader_validate))
        .route("/v1/jobs/{id}/bundle.zip", get(get_job_bundle))
        .route("/v1/jobs/{id}/content/{name}", get(get_job_content))
        .route("/v1/jobs/records", get(get_job_records))
        .route("/v1/jobs/records/{id}", get(get_job_record))
        .route("/v1/events", get(get_events))
        .route("/running", get(get_running))
        .route("/api/models/unload", post(post_unload_all))
        .route("/api/models/unload/{model}", post(post_unload_one))
        .route("/v1/downloads", post(post_download).get(list_downloads))
        .route("/v1/asset-packs/fetch", post(post_asset_pack_fetch))
        .route(
            "/v1/downloads/{id}",
            get(get_download).delete(delete_download),
        )
        .route(
            "/v1/artifacts",
            // TASK-binary-artifact-api: bodies stream straight to disk under
            // the chunk/artifact caps, so the POST route raises the router's
            // 64 MiB audio ceiling to the configured total cap (plus per-
            // chunk framing headroom) instead of rejecting large uploads
            // before the streaming caps can name their 413s.
            post(crate::artifacts::post_artifact).layer(DefaultBodyLimit::max(
                usize::try_from(
                    config
                        .artifacts
                        .max_artifact_bytes()
                        .saturating_add(crate::artifacts::CHUNK_MAX_BYTES),
                )
                .unwrap_or(usize::MAX),
            )),
        )
        .route(
            "/v1/artifacts/{id}/bytes",
            get(crate::artifacts::get_artifact_bytes),
        )
        .route("/v1/queue", get(get_queue))
        .route("/v1/rerank", post(post_rerank))
        .route("/v1/rerank/", post(post_rerank_slash))
        .route("/v1/fizgig/train", post(post_fizgig_train))
        .route("/v1/fizgig/jobs/{id}", get(get_fizgig_job))
        .route("/v1/fizgig/jobs/{id}/cancel", post(cancel_fizgig_job))
        .route("/v1/content-urls", post(post_content_urls))
        .route("/metrics", get(metrics_endpoint))
        .merge(mesh_router)
        .route_layer(axum_middleware::from_fn_with_state(
            config.clone(),
            middleware::require_bearer,
        ))
        .fallback(fallback)
        // H1 (ReviewedAudio): lift the silent 2 MiB default; see
        // [`AUDIO_BODY_LIMIT_BYTES`] for the byte math.
        .layer(DefaultBodyLimit::max(AUDIO_BODY_LIMIT_BYTES))
        // FEAT-response-compression: gzip/brotli on ordinary JSON/binary
        // routes; the predicate is DefaultPredicate's exclusion set (gRPC,
        // `image/*`, `text/event-stream` — BOTH SSE routes stay unbuffered)
        // with a 1 KiB size floor. Mounted INSIDE `track_metrics` so
        // status/`Retry-After` bookkeeping still sees the final response.
        .layer(
            CompressionLayer::new().compress_when(
                DefaultPredicate::new()
                    .and(SizeAbove::new(COMPRESSION_MIN_BYTES))
                    .and(partial_content_stays_plain),
            ),
        )
        .layer(axum_middleware::from_fn(middleware::track_metrics))
        .with_state(config)
}

#[cfg(test)]
#[path = "routes/tests_formats.rs"]
mod tests_formats;

#[cfg(test)]
#[path = "routes/tests_bindings.rs"]
mod tests_bindings;

#[cfg(test)]
#[path = "routes/tests_capabilities.rs"]
mod tests_capabilities;

#[cfg(test)]
#[path = "routes/tests_behavior.rs"]
mod tests_behavior;
