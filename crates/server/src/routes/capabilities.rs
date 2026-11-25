//! Capability-contract route, extracted from `routes.rs`.
//!
//! The router registration stays in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handler and the capability
//! document builder live here.

use axum::{Json, extract::State};
use honkbox_api_types::{
    AudioCppCapabilities, AuthCapabilities, Capabilities, CompressionRange, ControlNetFeature,
    FeatureFlags, FormatLimits, Formats, ImageOpKind, ImageOpsCapability, LicenseCapability,
    LicenseSummaryRow, ModelKind, OutputFormatCapability, QueueCapabilities, QueueKindCapacity,
    RecognitionFeature, ServerInfo, StorageCapability, ToolCallingCapability, VisionFeature,
};
use honkbox_audio_worker::{
    AUDIOCPP_COMPILED, AUDIOCPP_STT_FAMILIES, AUDIOCPP_TTS_FAMILIES, VAD_COMPILED,
};
use honkbox_recognition_worker::RecKind as WorkerRecKind;

/// Whether the Ash Stable Diffusion backend is compiled in.
///
/// Mirrors the compile-reachable `backends::make_sd` Ash arm: the
/// `ash-sd` feature is the single switch that enables that arm, so
/// this flag can never disagree with the backend the server can
/// actually build. Declared locally (instead of reading
/// `crate::backends`) so the route module also compiles in the
/// standalone route-module test crates, which do not declare a
/// `backends` module.
const ASH_SD_COMPILED: bool = cfg!(feature = "ash-sd");

/// `GET /v1/capabilities`: the machine-readable server contract (FEAT-
/// capabilities-api). Everything is DERIVED at request time — compiled
/// features, runtime config, the live catalog snapshot, and the resource
/// ledger's actual admission knobs — never a prose hard-code, so the
/// document is always true of the process answering and `/v1/models/refresh`
/// or a download commit are reflected with no cache to invalidate.
///
/// Includes `tools`, the chat tool-choice fidelity note (REQ-3): which
/// `tool_choice` values `/v1/chat/completions` honors verbatim vs clamps
/// to `auto`, and that `parallel_tool_calls` is accepted but ignored —
/// disclosed here instead of silently downgraded at request time.
#[utoipa::path(
    get,
    path = "/v1/capabilities",
    responses((status = 200, description = "Server capability contract (feature flags, formats, queue knobs, and the chat tool-choice fidelity note)", body = Capabilities))
)]
pub(crate) async fn get_capabilities(State(config): State<super::AppConfig>) -> Json<Capabilities> {
    Json(capabilities_document(&config))
}

/// The route that admits a job of each admission kind. A kind WITHOUT an
/// entry has no admitting route: the ledger still carries its queue
/// capacity, so `/v1/capabilities` advertises it as a RESERVED row
/// (`reserved: true`) — capacity for a job class that does not exist yet.
/// The contract test cross-checks every entry against the generated
/// `OpenAPI` document, so a row can never claim a route the server does
/// not serve.
pub const ADMITTING_ROUTES: &[(&str, &str)] = &[
    ("sd", "/v1/images/generations"),
    ("llm", "/v1/chat/completions"),
    ("embed", "/v1/embeddings"),
    // The CPU-only bucket: image ops and the audio routes take the `rag`
    // bucket (0 GPU slots) — the fixed 9-kind vocabulary has no CPU kind.
    ("rag", "/v1/image-ops"),
    // Blender asset jobs (`POST /v1/jobs/blender`): the route admits a job
    // of the `blender` kind, so the reserved row becomes callable.
    ("blender", "/v1/jobs/blender"),
    // OCR ingest (`POST /v1/ocr/ingest`) is the first real consumer of the
    // `ocr` admission bucket; the route keys its 501 on the same `serves()`
    // predicate this document reports as `recognition.ocr`.
    ("ocr", "/v1/ocr/ingest"),
    // Long Fizgig training runs take the dedicated exclusive lease; this
    // route is their only admission point.
    ("training", "/v1/fizgig/train"),
];

/// The route that admits a job of `kind`, if any (see [`ADMITTING_ROUTES`]).
pub fn admitting_route(kind: honkbox_residency::ResourceKind) -> Option<&'static str> {
    let label = kind.label();
    ADMITTING_ROUTES
        .iter()
        .find_map(|(candidate, path)| (*candidate == label).then_some(*path))
}

/// Build the format half of the capability document from the registry so a
/// missing encoder never appears as writable.
fn formats_capability(config: &super::AppConfig) -> Formats {
    // The router-wide audio ceiling and the artifact upload route's raised
    // ceiling are both real; advertise the larger one instead of a number
    // that understates what this binary accepts.
    let audio_limit = u64::try_from(super::AUDIO_BODY_LIMIT_BYTES).unwrap_or(u64::MAX);
    let artifact_limit = config
        .artifacts
        .max_artifact_bytes()
        .saturating_add(crate::artifacts::CHUNK_MAX_BYTES);
    Formats {
        // The registry (`crate::formats`) is the single source of truth:
        // `read` = rows the SD ingest validators accept (edits `image`/
        // `mask`, ControlNet `control_image` — magic-sniffed through
        // `identify_png`); vision chat bytes pass through to the
        // projector's own loader and are deliberately not listed.
        read: crate::formats::sd_ingest_names()
            .map(str::to_owned)
            .collect(),
        // `write` = the registry's compiled-encoder rows (locked to
        // sd-core `OutputFormat::ALL` by a routes-level unit test) — a
        // missing encoder never appears as writable.
        write: crate::formats::output_rows()
            .map(|row| OutputFormatCapability {
                format: row.name.to_owned(),
                compression: row.output_compression,
            })
            .collect(),
        limits: FormatLimits {
            output_compression: CompressionRange {
                min: 0,
                max: 100,
                applies_to: "jpeg".to_owned(),
            },
            // Largest upload any route accepts (audio ceiling or artifact
            // route ceiling, whichever is higher).
            body_bytes: audio_limit.max(artifact_limit),
        },
    }
}

/// Env-key arm liveness, read at request time like the middleware does: an
/// unset/empty key env var means only the JWT arm can admit (fail-closed).
fn env_key_usable(config: &super::AppConfig) -> bool {
    std::env::var(&config.auth.api_key_env).is_ok_and(|key| !key.is_empty())
}

/// Build the capability document (see [`get_capabilities`]). `pub(crate)`
/// so the contract tests can assert on the exact structure the route
/// serves.
// Flat, exhaustive field mapping: splitting it adds indirection without
// making the capability contract any clearer.
#[allow(clippy::too_many_lines)]
pub(crate) fn capabilities_document(config: &super::AppConfig) -> Capabilities {
    // The LIVE catalog snapshot: vision/controlnet model counts refresh
    // with `/v1/models/refresh` and download commits for free.
    let catalog = config.catalog_snapshot();
    let vision_models = u64::try_from(
        catalog
            .data
            .iter()
            .filter(|entry| entry.capabilities.vision)
            .count(),
    )
    .unwrap_or(u64::MAX);
    let controlnet_models = u64::try_from(
        catalog
            .data
            .iter()
            .filter(|entry| entry.kind == ModelKind::ControlNet)
            .count(),
    )
    .unwrap_or(u64::MAX);
    // ControlNet is honest about the 501: the flag is `true` only when the
    // configured SD pipeline actually executes control requests (the sd-cpp
    // backend loads the named weights, applies the conditioning type, and
    // unloads after; `resolve_controlnet` answers `controlnet_unsupported`
    // for everything else) AND conditioning targets exist in the catalog.
    let controlnet_available =
        controlnet_models > 0 && config.sd.backend_id() == honkbox_sd_core::SD_CPP_BACKEND_ID;
    let ledger = honkbox_residency::global().admission_info();
    // Derived at request time: the JWT arm counts when `[auth] jwt_*`
    // configures material, the env-key arm counts while its env var holds a
    // value — so rotation/unset flips `both` to `jwt` with no restart.
    let auth_mode = match config.auth.mode {
        super::AuthMode::None => "none",
        super::AuthMode::Bearer => match (env_key_usable(config), config.auth.jwt.enabled()) {
            (true, true) => "both",
            (false, true) => "jwt",
            _ => "bearer",
        },
    };
    Capabilities {
        api_version: honkbox_api_types::API_VERSION.to_owned(),
        server: ServerInfo {
            name: "honkbox".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        },
        features: FeatureFlags {
            // Same compile probe the catalog's `capabilities.vision` uses
            // (routes' vision gate) and the named `vision_unsupported` 400
            // backs — a text-only build reports `false` here.
            vision: VisionFeature {
                built: honkbox_llm_worker::VISION_COMPILED,
                models: vision_models,
            },
            controlnet: ControlNetFeature {
                available: controlnet_available,
                models: controlnet_models,
            },
            // Per-request LoRA stacks: accepted and validated on BOTH image
            // routes (`resolve_loras`); the contract test cross-checks the
            // flag against the generated OpenAPI paths.
            lora: true,
            // No video job class exists; the FEAT ticket pins this `false`.
            video: false,
            // No system-one decision endpoint exists yet; the flag stays
            // false until one lands (FEAT-model-modality-advertisement).
            decisions: false,
            // Compile-time audio.cpp surface from the worker's OWN features,
            // never a prose copy: the family lists are the same constants
            // the named load error cites, so no family here is unloadable.
            audio_cpp: AudioCppCapabilities {
                built: AUDIOCPP_COMPILED,
                stt_families: AUDIOCPP_STT_FAMILIES
                    .iter()
                    .map(|family| (*family).to_owned())
                    .collect(),
                tts_families: AUDIOCPP_TTS_FAMILIES
                    .iter()
                    .map(|family| (*family).to_owned())
                    .collect(),
            },
            // Unconditional routes (`GET /v1/jobs`, `GET /v1/events`); the
            // contract test cross-checks both against the OpenAPI paths.
            jobs: true,
            events: true,
            vad: VAD_COMPILED,
            experimental_ash_sd: ASH_SD_COMPILED,
            // Blender engine jobs (`POST /v1/jobs/blender`): `true` only
            // when a Blender engine is installed and probed — the route
            // answers a named 501 otherwise, so the flag never promises
            // an engine the server cannot drive.
            blender: super::external_tools::blender_engine().is_some(),
            // Artifact delivery stores generated bytes in the binary
            // artifact registry, which needs a resolved data dir.
            artifact_delivery: config.data_dir.is_some(),
            // Same gate the rerank route uses: the Python switch alone
            // would promise a route that answers a named 501.
            rerank: super::runtime_config::rerank_supported(),
            // The sidecar flag keeps its Python-switch semantics.
            python_sidecar: super::runtime_config::python_enabled(),
            fizgig: super::fizgig::available(),
            texture_postprocess: crate::texture_postprocess::capability().available,
            // The existing unload/running routes only mean something with a
            // swap manager installed (the same fact `runtime.rs` gates on).
            residency_control: config.swap.is_some(),
        },
        formats: formats_capability(config),
        auth: AuthCapabilities {
            mode: auth_mode.to_owned(),
        },
        queue: QueueCapabilities {
            gpu_slots: ledger.gpu_slots,
            // The LIVE admission bound — 0 = queueing disabled.
            queue_wait_secs: crate::admission::queue_wait_secs(),
            retry_after_secs: crate::admission::RETRY_AFTER_HINT_SECS,
            kinds: ledger
                .queue_caps
                .into_iter()
                .map(|(kind, cap)| QueueKindCapacity {
                    kind: kind.label().to_owned(),
                    queue_cap: u64::try_from(cap).unwrap_or(u64::MAX),
                    ttl_secs: ledger.queue_ttl.as_secs(),
                    // A kind with no admitting route is RESERVED capacity:
                    // the ledger carries the queue, but no route accepts a
                    // job of that kind (the closed job registry has none).
                    reserved: admitting_route(kind).is_none(),
                })
                .collect(),
        },
        openapi: "/v1/openapi.json".to_owned(),
        models: "/v1/models".to_owned(),
        image_ops: Some(ImageOpsCapability {
            // The closed vocabulary, derived from the enum itself (never a
            // prose copy — add an op to `ImageOpKind` and this follows).
            ops: ImageOpKind::ALL
                .iter()
                .map(|op| op.label().to_owned())
                .collect(),
            // Sidecar liveness, read at request time like the other
            // availability flags: loop-lore `resolveMattingProvider`
            // reads this as its first choice (local providers are the
            // fallback when `false`).
            matting_available: super::image_ops_exec::matting_sidecar_url().is_some(),
        }),
        // Tool-choice fidelity (TASK-llm-tool-choice-contract): derived
        // from the `ToolChoiceSpec` vocabulary consts — the same type the
        // `resolve_constraints` clamp is defined against — so the document
        // cannot drift from what `/v1/chat/completions` actually does.
        tools: Some(ToolCallingCapability::chat_routes()),
        // Recognition honesty: the SAME `serves` predicate the route uses
        // for its named 501 drives both flags. A stub fallback therefore
        // advertises nothing, and `ocr` stays false until a real backend is
        // compiled and selected.
        recognition: Some(RecognitionFeature {
            barcode: config.recognition.serves(WorkerRecKind::Barcode),
            ocr: config.recognition.serves(WorkerRecKind::Ocr),
            // Live only when every face gate passes (compiled + weights +
            // opt-in + data dir) — the same predicate the route 403s on.
            face: super::recognition::face_serving(config),
        }),
        external_tools: Some(super::external_tools::external_tools_capability()),
        // Additive probe so clients can check storage before using artifact
        // delivery; derived from the same config fact as the feature flag.
        storage: Some(StorageCapability {
            data_dir_present: config.data_dir.is_some(),
        }),
        nsfw: {
            let (enabled, policy, model_version, threshold, _) =
                super::runtime_config::nsfw_facts();
            Some(honkbox_api_types::NsfwCapability {
                available: enabled,
                policy,
                model_version,
                threshold,
            })
        },
        mesh: Some(super::mesh_discovery::capability()),
        signed_urls: {
            let (enabled, ttl_secs, _) = super::runtime_config::signed_url_facts();
            Some(honkbox_api_types::SignedUrlCapability { enabled, ttl_secs })
        },
        scopes: {
            let (enabled, scopes) = super::runtime_config::scope_facts();
            Some(honkbox_api_types::ScopeCapability {
                enabled,
                scopes: scopes.into_iter().map(|(name, _)| name).collect(),
            })
        },
        texture_postprocess: Some(crate::texture_postprocess::capability()),
        // Curated license records are the authoritative, review-controlled
        // catalog: the same rows the model-license gate and bundle sidecars
        // read. Additive so older clients ignore the block entirely.
        licenses: Some(LicenseCapability {
            rows: crate::catalog::licenses::records()
                .map(|record| LicenseSummaryRow {
                    license: record.license.clone(),
                    license_url: record.license_url.clone(),
                    territory: record.territory.clone(),
                    commercial_terms: record.commercial_terms.clone(),
                    gated: record.gated,
                    source_url: record.source_url.clone(),
                    sha256: record.sha256.clone(),
                    attribution_text: record.attribution_text.clone(),
                })
                .collect(),
        }),
    }
}
