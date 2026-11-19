//! Capability-contract route, extracted from `routes.rs`.
//!
//! The router registration stays in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the handler and the capability
//! document builder live here.

use axum::{Json, extract::State};
use honkbox_api_types::{
    AuthCapabilities, Capabilities, CompressionRange, ControlNetFeature, FeatureFlags,
    FormatLimits, Formats, ImageOpKind, ImageOpsCapability, ModelKind, OutputFormatCapability,
    QueueCapabilities, QueueKindCapacity, RecognitionFeature, ServerInfo, StorageCapability,
    ToolCallingCapability, VisionFeature,
};
use honkbox_audio_worker::VAD_COMPILED;
use honkbox_recognition_worker::{BARCODE_COMPILED, RecKind as WorkerRecKind};

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
pub(super) async fn get_capabilities(State(config): State<super::AppConfig>) -> Json<Capabilities> {
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
    // bucket (0 GPU slots) — the fixed 8-kind vocabulary has no CPU kind.
    ("rag", "/v1/image-ops"),
    // Blender asset jobs (`POST /v1/jobs/blender`): the route admits a job
    // of the `blender` kind, so the reserved row becomes callable.
    ("blender", "/v1/jobs/blender"),
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
fn formats_capability() -> Formats {
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
            // The router-wide body ceiling (`AUDIO_BODY_LIMIT_BYTES`,
            // H1) — the largest upload any route accepts.
            body_bytes: u64::try_from(super::AUDIO_BODY_LIMIT_BYTES).unwrap_or(u64::MAX),
        },
    }
}

/// Build the capability document (see [`get_capabilities`]). `pub(crate)`
/// so the contract tests can assert on the exact structure the route
/// serves.
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
    let auth_mode = match config.auth.mode {
        super::AuthMode::None => "none",
        super::AuthMode::Bearer => "bearer",
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
            // Unconditional routes (`GET /v1/jobs`, `GET /v1/events`); the
            // contract test cross-checks both against the OpenAPI paths.
            jobs: true,
            events: true,
            vad: VAD_COMPILED,
            experimental_ash_sd: cfg!(feature = "experimental-ash-sd"),
            // Blender engine jobs (`POST /v1/jobs/blender`): `true` only
            // when a Blender engine is installed and probed — the route
            // answers a named 501 otherwise, so the flag never promises
            // an engine the server cannot drive.
            blender: super::external_tools::blender_engine().is_some(),
            // Artifact delivery stores generated bytes in the binary
            // artifact registry, which needs a resolved data dir.
            artifact_delivery: config.data_dir.is_some(),
        },
        formats: formats_capability(),
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
        }),
        // Tool-choice fidelity (TASK-llm-tool-choice-contract): derived
        // from the `ToolChoiceSpec` vocabulary consts — the same type the
        // `resolve_constraints` clamp is defined against — so the document
        // cannot drift from what `/v1/chat/completions` actually does.
        tools: Some(ToolCallingCapability::chat_routes()),
        // Recognition honesty (recognition tickets): each flag is the same
        // gate the `/v1/recognize` route's named 501 keys on — the pipeline's
        // served kinds AND the compile probe. `ocr` stays `false` until an
        // OCR backend lands (the stub/barcode backend answers
        // `recognition_kind_unsupported` for it), so a client can trust the
        // absence.
        recognition: Some(RecognitionFeature {
            barcode: BARCODE_COMPILED
                && config.recognition.kinds().contains(&WorkerRecKind::Barcode),
            ocr: config.recognition.kinds().contains(&WorkerRecKind::Ocr),
        }),
        external_tools: Some(super::external_tools::external_tools_capability()),
        // Additive probe so clients can check storage before using artifact
        // delivery; derived from the same config fact as the feature flag.
        storage: Some(StorageCapability {
            data_dir_present: config.data_dir.is_some(),
        }),
    }
}
