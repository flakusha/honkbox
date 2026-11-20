//! Catalog refresh routes, extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`: the `OpenAPI` coverage gate
//! scans that file for `.route()` calls. Only the rescan handlers and the
//! registry→catalog mapping live here.

use std::path::Path;
use std::sync::Arc;

use honkbox_api_types::{
    GgufMetadata, ModelCapabilities, ModelCatalog, ModelCatalogEntry, ModelKind, ModelMetadata,
    ModelModalities, SafetensorsMetadata,
};
use tokio::sync::Notify;

use super::AppConfig;

/// Registry re-scan + shared catalog snapshot swap: the body of
/// `POST /v1/models/refresh`, also driven by the download manager's
/// completion signal (`rescan_on_downloads`). An operator-binding conflict
/// in the fresh scan leaves the previous snapshot in place and fails the
/// request — never a silently wrong alias.
pub async fn rescan_catalog(config: &AppConfig) -> Result<ModelCatalog, String> {
    let roots = config.roots.clone();
    let found = tokio::task::spawn_blocking(move || honkbox_registry::scan_roots(&roots))
        .await
        .map_err(|err| format!("refresh scan failed: {err}"))?;
    let catalog = refresh_catalog(&found, config.data_dir.as_deref(), &config.model_entries)?;
    *config
        .catalog
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(catalog.clone());
    Ok(catalog)
}

/// Rescan the registry whenever the download manager commits a file, so a
/// completed download appears in `GET /v1/models` under its normal slug
/// without a manual refresh. Detached loop; failures log and keep waiting.
/// (`allow(dead_code)` only silences the route-test binaries that include
/// `routes.rs` without driving this loop.)
#[cfg_attr(test, allow(dead_code))]
pub async fn rescan_on_downloads(config: AppConfig, signal: Arc<Notify>) {
    loop {
        signal.notified().await;
        if let Err(err) = rescan_catalog(&config).await {
            tracing::warn!(%err, "post-download rescan failed");
        }
    }
}

/// Registry scan hits → `/v1/models` body, with SQLite cache sync.
///
/// The file walk stays the **source of truth**; `HONKBOX_REGISTRY_DB` (off
/// by default via `"off"`/empty, otherwise a cache path) only adds an
/// incremental cache: a successful non-empty scan upserts its entries plus
/// the operator bindings and prunes rows whose files vanished; when a scan
/// comes back **empty** (roots unavailable) the last-known cached catalog
/// is served as an offline fallback — with the cached aliases, so a restart
/// against empty roots still resolves operator aliases. Every cache failure
/// is logged and non-fatal — the response is always the plain scan result
/// when caching is disabled or the cache is unusable. Operator-binding
/// conflicts (duplicate alias across files, `match` hitting several files)
/// are `Err` — a hard failure at startup, a 500 on refresh.
pub fn refresh_catalog(
    found: &[honkbox_registry::FoundModel],
    data_dir: Option<&Path>,
    model_entries: &[crate::config::ModelEntryConfig],
) -> Result<ModelCatalog, String> {
    let db_default = data_dir.map(|base| base.join("registry.sqlite3"));
    if found.is_empty() {
        // Offline fallback: roots unavailable → serve the cached catalog.
        match honkbox_registry::Db::open_from_env(db_default.as_deref()) {
            Ok(Some(db)) => match db.load_entries() {
                Ok(cached) if !cached.is_empty() => {
                    let bindings = match db.load_bindings() {
                        Ok((aliases, overrides)) => {
                            super::bindings::bindings_from_cache(&aliases, &overrides)
                        }
                        Err(err) => {
                            tracing::warn!(%err, "registry cache bindings read failed");
                            super::bindings::ModelBindings::default()
                        }
                    };
                    tracing::warn!(
                        count = cached.len(),
                        "scan found no models; serving cached registry catalog"
                    );
                    return Ok(catalog_from_entries(&cached, &bindings));
                }
                Ok(_) => {}
                Err(err) => tracing::warn!(%err, "registry cache read failed"),
            },
            Ok(None) => {}
            Err(err) => tracing::warn!(%err, "registry cache open failed"),
        }
        return Ok(super::bindings::build_catalog(
            found,
            &super::bindings::ModelBindings::default(),
        ));
    }
    // Bind the operator entries against the fresh hits; a conflict fails
    // loudly (startup: exit; refresh: 500) instead of binding a wrong file.
    let bindings = super::bindings::bind_entries(model_entries, found)?;
    // Cache sync: upsert the fresh scan + bindings, prune entries whose
    // files vanished. Best-effort; a slow/broken cache never fails the
    // request. Entries are built once and reused for the response (header
    // probes are file IO).
    let entries: Vec<honkbox_registry::CatalogEntry> = found
        .iter()
        .map(honkbox_registry::CatalogEntry::from)
        .collect();
    match honkbox_registry::Db::open_from_env(db_default.as_deref()) {
        Ok(Some(db)) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            if let Err(err) = db.upsert_entries(&entries, now) {
                tracing::warn!(%err, "registry cache upsert failed");
            }
            if let Err(err) = db.sync_bindings(&bindings.alias_rows(), &bindings.override_rows()) {
                tracing::warn!(%err, "registry cache binding sync failed");
            }
            if let Err(err) = db.prune_missing_roots() {
                tracing::warn!(%err, "registry cache prune failed");
            }
        }
        Ok(None) => {}
        Err(err) => tracing::warn!(%err, "registry cache open failed"),
    }
    Ok(catalog_from_entries(&entries, &bindings))
}

/// Registry catalog entries (e.g. the SQLite cache) → `/v1/models` body.
fn catalog_from_entries(
    entries: &[honkbox_registry::CatalogEntry],
    bindings: &super::bindings::ModelBindings,
) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: entries
            .iter()
            .map(|e| catalog_entry(e.clone(), bindings))
            .collect(),
    }
}

/// Registry catalog entry → response DTO (the only registry→wire mapping).
/// The operator alias (when bound) becomes `id`; the slug always rides its
/// own field. Capability defaults derive from the (possibly reclassified)
/// kind, then on-disk vision reality (a sibling mmproj projector; a
/// projector file itself is not chat weights), then operator declarations
/// win; loop-lore's flat fields mirror the nested `capabilities` object.
pub(super) fn catalog_entry(
    entry: honkbox_registry::CatalogEntry,
    bindings: &super::bindings::ModelBindings,
) -> ModelCatalogEntry {
    let overrides = bindings.capabilities.get(&entry.slug).copied();
    let kind = if overrides
        .is_some_and(crate::config::ModelCapabilitiesOverride::reclassifies_embedding)
    {
        ModelKind::Embedding
    } else {
        match entry.kind {
            honkbox_registry::ModelKind::Llm => ModelKind::Llm,
            honkbox_registry::ModelKind::Sd => ModelKind::Sd,
            honkbox_registry::ModelKind::ControlNet => ModelKind::ControlNet,
            honkbox_registry::ModelKind::Lora => ModelKind::Lora,
            honkbox_registry::ModelKind::Embedding => ModelKind::Embedding,
            honkbox_registry::ModelKind::Vae => ModelKind::Vae,
            honkbox_registry::ModelKind::Stt => ModelKind::Stt,
            honkbox_registry::ModelKind::Tts => ModelKind::Tts,
            honkbox_registry::ModelKind::Ocr => ModelKind::Ocr,
        }
    };
    let mut capabilities = ModelCapabilities::for_kind(kind);
    // Vision chat (REQ-8): a projector file named `mmproj*.gguf` is not
    // chat weights (loading one as a chat model would produce garbage), so
    // its chat default flips off; a text model with a sibling mmproj
    // projector is vision-capable — but only when this build compiles the
    // mtmd path (`llm-vision` feature), so `/v1/models` never advertises
    // vision the pipeline cannot serve. Operator overrides below still win.
    if honkbox_registry::is_mmproj_file(&entry.path) {
        capabilities.chat = false;
    } else if kind == ModelKind::Llm
        && honkbox_llm_worker::VISION_COMPILED
        && honkbox_registry::sibling_mmproj(&entry.path).is_some()
    {
        capabilities.vision = true;
    }
    capabilities.completion = capabilities.chat;
    if let Some(overrides) = overrides {
        if let Some(chat) = overrides.chat {
            capabilities.chat = chat;
        }
        if let Some(embeddings) = overrides.embeddings {
            capabilities.embeddings = embeddings;
        }
        if let Some(tool_calling) = overrides.tool_calling {
            capabilities.tool_calling = tool_calling;
        }
        if let Some(vision) = overrides.vision {
            capabilities.vision = vision;
        }
        capabilities.completion = capabilities.chat;
    }
    let context_length = match entry.metadata.as_ref() {
        Some(honkbox_registry::ModelFileMetadata::Gguf(meta)) => meta.context_length,
        _ => None,
    };
    ModelCatalogEntry {
        id: bindings
            .display_alias(&entry.slug)
            .map_or_else(|| entry.slug.clone(), str::to_owned),
        slug: entry.slug,
        kind,
        owned_by: "local".to_owned(),
        context_length,
        // honkbox has no per-model output ceiling; the field exists for the
        // loop-lore flat contract and stays `null`.
        max_output: None,
        // No honkbox model declares reasoning/thinking support today.
        thinking: false,
        tool_calling: capabilities.tool_calling,
        modalities: if capabilities.vision {
            vec!["text".to_owned(), "vision".to_owned()]
        } else {
            vec!["text".to_owned()]
        },
        // Additive nested modality from the (possibly reclassified) kind and
        // the resolved capabilities; the flat `modalities` above stays as-is
        // for loop-lore (its `vision` is a different vocabulary).
        modality: modality_for(kind, capabilities),
        capabilities,
        category: entry.category,
        path: entry.path.display().to_string(),
        bytes: entry.bytes,
        metadata: entry.metadata.map(catalog_metadata),
    }
}

/// Input/output modalities from the (possibly reclassified) on-disk kind and
/// resolved capabilities. Unknown kinds degrade to empty lists, never a
/// made-up modality.
fn modality_for(kind: ModelKind, capabilities: ModelCapabilities) -> ModelModalities {
    match kind {
        ModelKind::Llm => {
            let mut input = vec!["text".to_owned()];
            if capabilities.vision {
                input.push("image".to_owned());
            }
            ModelModalities {
                input,
                output: vec!["text".to_owned()],
            }
        }
        ModelKind::Sd => ModelModalities {
            input: vec!["text".to_owned()],
            output: vec!["image".to_owned()],
        },
        ModelKind::Stt => ModelModalities {
            input: vec!["audio".to_owned()],
            output: vec!["text".to_owned()],
        },
        ModelKind::Tts => ModelModalities {
            input: vec!["text".to_owned()],
            output: vec!["audio".to_owned()],
        },
        ModelKind::Embedding => ModelModalities {
            input: vec!["text".to_owned()],
            output: Vec::new(),
        },
        ModelKind::Ocr => ModelModalities {
            input: vec!["image".to_owned()],
            output: vec!["text".to_owned()],
        },
        // ControlNet/LoRA/VAE have no I/O content contract today.
        ModelKind::ControlNet | ModelKind::Lora | ModelKind::Vae => ModelModalities {
            input: Vec::new(),
            output: Vec::new(),
        },
    }
}

/// Registry header metadata → response DTO (part of the registry→wire mapping).
fn catalog_metadata(metadata: honkbox_registry::ModelFileMetadata) -> ModelMetadata {
    match metadata {
        honkbox_registry::ModelFileMetadata::Gguf(meta) => ModelMetadata::Gguf(GgufMetadata {
            version: meta.version,
            tensor_count: meta.tensor_count,
            name: meta.name,
            architecture: meta.architecture,
        }),
        honkbox_registry::ModelFileMetadata::Safetensors(meta) => {
            ModelMetadata::Safetensors(SafetensorsMetadata {
                tensor_count: meta.tensor_count,
                dtypes: meta.dtypes,
            })
        }
    }
}
