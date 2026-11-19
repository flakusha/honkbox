//! Operator bindings and the registry→catalog mapping, extracted from
//! `routes.rs`.
//!
//! The router registrations stay in `routes.rs`; this module owns the
//! binding conflict rules, the `ModelBindings` state they produce, and the
//! `build_catalog` projection the sibling `catalog` module reuses.

use std::collections::HashMap;
use std::path::Path;

use honkbox_api_types::ModelCatalog;

/// Operator bindings bound to scan hits: per-slug declared aliases (in
/// declaration order; the first is the served `id`) and capability
/// overrides. Built by [`bind_entries`]; cached rows rebuild it for the
/// offline fallback.
#[derive(Debug, Default)]
pub struct ModelBindings {
    /// slug → declared aliases (declaration order preserved).
    aliases: HashMap<String, Vec<String>>,
    /// slug → operator capability overrides.
    pub(super) capabilities: HashMap<String, crate::config::ModelCapabilitiesOverride>,
}

impl ModelBindings {
    /// The alias served as `id` for `slug` (first declared), if any.
    pub(super) fn display_alias(&self, slug: &str) -> Option<&str> {
        self.aliases.get(slug)?.first().map(String::as_str)
    }

    /// Cache rows: alias→slug pairs, grouped per slug in declaration order
    /// so the loaded catalog's `id` (first alias) is stable across restarts.
    pub(super) fn alias_rows(&self) -> Vec<honkbox_registry::AliasRow> {
        let mut slugs: Vec<&String> = self.aliases.keys().collect();
        slugs.sort_unstable();
        slugs
            .into_iter()
            .flat_map(|slug| {
                self.aliases[slug]
                    .iter()
                    .map(|alias| honkbox_registry::AliasRow {
                        alias: alias.clone(),
                        slug: slug.clone(),
                    })
            })
            .collect()
    }

    /// Cache rows: per-slug capability overrides (slug-sorted; the JSON
    /// column carries the declaration verbatim).
    pub(super) fn override_rows(&self) -> Vec<honkbox_registry::EntryOverride> {
        let mut rows: Vec<(String, String)> = self
            .capabilities
            .iter()
            .map(|(slug, caps)| (slug.clone(), caps.cache_json()))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        rows.into_iter()
            .map(|(slug, json)| honkbox_registry::EntryOverride {
                slug,
                capabilities_json: json,
            })
            .collect()
    }
}

/// Rebuild [`ModelBindings`] from the registry cache (offline fallback).
/// Rows arrive slug-grouped in declaration order (`sync_bindings`), so the
/// first alias per slug is again the served `id`. Orphan rows were dropped
/// by `load_bindings`.
pub(super) fn bindings_from_cache(
    aliases: &[honkbox_registry::AliasRow],
    overrides: &[honkbox_registry::EntryOverride],
) -> ModelBindings {
    let mut bindings = ModelBindings::default();
    for row in aliases {
        bindings
            .aliases
            .entry(row.slug.clone())
            .or_default()
            .push(row.alias.clone());
    }
    for row in overrides {
        bindings.capabilities.insert(
            row.slug.clone(),
            crate::config::ModelCapabilitiesOverride::from_cache_json(&row.capabilities_json),
        );
    }
    bindings
}

/// Bind operator `[[models.entry]]` declarations to scan hits.
///
/// Targeting: `slug` binds exactly that registry slug; `match` binds every
/// hit whose weight file **name** (final path component) equals the value.
/// Rules, all **hard errors** naming the offending files:
///
/// - two entries binding the same file (a config-level duplicate, or a
///   `match` that catches another entry's `slug` target);
/// - an alias-bearing `match` hitting several files (the alias would be
///   ambiguous — capabilities-only entries apply to every hit instead);
/// - the same alias declared for two different files, or an alias colliding
///   with a *different* model's registry slug.
///
/// Declarations whose target is absent from this scan bind nothing (the
/// file may appear later); they resurface on the next refresh.
///
/// # Errors
/// Any of the conflict rules above; the message names both paths.
pub fn bind_entries(
    entries: &[crate::config::ModelEntryConfig],
    found: &[honkbox_registry::FoundModel],
) -> Result<ModelBindings, String> {
    let mut bindings = ModelBindings::default();
    // slug → descriptor of the entry that bound it (for duplicate reports).
    let mut bound_by: HashMap<&str, String> = HashMap::new();
    for entry in entries {
        let descriptor = entry_descriptor(entry);
        let targets = targets_of(entry, found);
        if !entry.aliases.is_empty() && targets.len() > 1 {
            return Err(format!(
                "[[models.entry]] {descriptor}: match binds {} files; aliases must name one: {}",
                targets.len(),
                paths_of(&targets)
            ));
        }
        for target in targets {
            if let Some(previous) = bound_by.get(target.slug.as_str()) {
                return Err(format!(
                    "two [[models.entry]] entries bind {}: {} and {descriptor}",
                    target.path.display(),
                    previous
                ));
            }
            bound_by.insert(target.slug.as_str(), descriptor.clone());
            bind_target(&mut bindings, entry, target, found)?;
        }
    }
    Ok(bindings)
}

/// Scan hits one entry targets: the exact `slug`, plus every hit whose
/// weight file name equals `match`.
fn targets_of<'a>(
    entry: &crate::config::ModelEntryConfig,
    found: &'a [honkbox_registry::FoundModel],
) -> Vec<&'a honkbox_registry::FoundModel> {
    found
        .iter()
        .filter(|m| {
            entry.slug.as_deref().is_some_and(|slug| slug == m.slug)
                || entry
                    .file_match
                    .as_deref()
                    .is_some_and(|file_match| file_name(&m.path) == Some(file_match))
        })
        .collect()
}

/// Bind one scan hit: capability overrides (unless nothing is declared) and
/// the entry's aliases, each checked against aliases of other slugs and
/// against other models' registry slugs.
fn bind_target(
    bindings: &mut ModelBindings,
    entry: &crate::config::ModelEntryConfig,
    target: &honkbox_registry::FoundModel,
    found: &[honkbox_registry::FoundModel],
) -> Result<(), String> {
    if entry.capabilities != crate::config::ModelCapabilitiesOverride::default() {
        bindings
            .capabilities
            .insert(target.slug.clone(), entry.capabilities);
    }
    for alias in &entry.aliases {
        let conflicting = bindings
            .aliases
            .iter()
            .find(|(slug, list)| slug.as_str() != target.slug && list.contains(alias))
            .map(|(slug, _)| slug.clone());
        if let Some(other_slug) = conflicting {
            return Err(format!(
                "alias {alias:?} is declared twice: for {other_slug:?} and {} ({})",
                target.slug,
                target.path.display()
            ));
        }
        if let Some(other) = found
            .iter()
            .find(|m| m.slug == *alias && m.slug != target.slug)
        {
            return Err(format!(
                "alias {alias:?} collides with the registry slug of another model ({})",
                other.path.display()
            ));
        }
        bindings
            .aliases
            .entry(target.slug.clone())
            .or_default()
            .push(alias.clone());
    }
    Ok(())
}

/// Stable descriptor for error messages: what the entry targets.
fn entry_descriptor(entry: &crate::config::ModelEntryConfig) -> String {
    if let Some(slug) = &entry.slug {
        format!("slug = {slug:?}")
    } else {
        format!("match = {:?}", entry.file_match.as_deref().unwrap_or(""))
    }
}

/// Final path component (the weight file name `match` compares).
fn file_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(|name| name.to_str())
}

/// Comma-joined display paths for conflict messages.
fn paths_of(targets: &[&honkbox_registry::FoundModel]) -> String {
    targets
        .iter()
        .map(|m| m.path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Registry scan hits → `/v1/models` body. Also feeds `models show --json`
/// so the CLI output and the endpoint agree (P0 exit of the parity plan).
#[must_use]
pub fn build_catalog(
    found: &[honkbox_registry::FoundModel],
    bindings: &ModelBindings,
) -> ModelCatalog {
    ModelCatalog {
        object: "list".to_owned(),
        data: found
            .iter()
            .map(|m| super::catalog::catalog_entry(m.into(), bindings))
            .collect(),
    }
}
