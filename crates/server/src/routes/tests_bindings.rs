//! Tests for model entry binding: slug/file-match resolution and conflict
//! detection.

use super::*;
use crate::config::{ModelCapabilitiesOverride, ModelEntryConfig};
use honkbox_registry::RootKind;
use std::path::PathBuf;

/// One scan hit for a synthetic file path.
pub(super) fn scan_hit(slug: &str, path: &str) -> honkbox_registry::FoundModel {
    honkbox_registry::FoundModel {
        root: RootKind::Llm,
        category: "root".to_owned(),
        path: PathBuf::from(path),
        slug: slug.to_owned(),
        bytes: 1,
    }
}

/// One `[[models.entry]]` declaration.
pub(super) fn model_entry(
    slug: Option<&str>,
    file_match: Option<&str>,
    aliases: &[&str],
    capabilities: ModelCapabilitiesOverride,
) -> ModelEntryConfig {
    ModelEntryConfig {
        slug: slug.map(str::to_owned),
        file_match: file_match.map(str::to_owned),
        aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
        capabilities,
    }
}

#[test]
fn bind_binds_slug_and_match_targets() {
    let found = vec![
        scan_hit("llm:a/snap/sha1/m.gguf", "/r/a/snap/sha1/m.gguf"),
        scan_hit("llm:b/w.bin", "/r/b/w.bin"),
    ];
    let entries = vec![
        model_entry(
            Some("llm:a/snap/sha1/m.gguf"),
            None,
            &["alpha"],
            ModelCapabilitiesOverride::default(),
        ),
        model_entry(
            None,
            Some("w.bin"),
            &["whisper"],
            ModelCapabilitiesOverride::default(),
        ),
    ];
    let bindings = bind_entries(&entries, &found).expect("bind");
    let catalog = build_catalog(&found, &bindings);
    assert_eq!(catalog.data.len(), 2);
    assert_eq!(catalog.data[0].id, "alpha");
    assert_eq!(catalog.data[0].slug, "llm:a/snap/sha1/m.gguf");
    assert_eq!(catalog.data[1].id, "whisper");
    // The flat loop-lore fields mirror the capabilities object.
    assert_eq!(catalog.data[0].modalities, vec!["text"]);
    assert!(catalog.data[0].capabilities.chat);
    assert!(!catalog.data[0].tool_calling);
    assert_eq!(catalog.data[0].owned_by, "local");
    assert_eq!(catalog.data[0].context_length, None);
    assert_eq!(catalog.data[0].max_output, None);
    assert!(!catalog.data[0].thinking);
}

/// Alias conflicts are hard errors naming both paths (binding owner
/// decision: never a silent first-wins).
#[test]
fn bind_conflicts_are_errors() {
    let found = vec![
        scan_hit("llm:a/m1.gguf", "/r/a/m1.gguf"),
        scan_hit("llm:b/m2.gguf", "/r/b/m2.gguf"),
    ];

    // The same alias for two files.
    let entries = vec![
        model_entry(
            Some("llm:a/m1.gguf"),
            None,
            &["same"],
            ModelCapabilitiesOverride::default(),
        ),
        model_entry(
            Some("llm:b/m2.gguf"),
            None,
            &["same"],
            ModelCapabilitiesOverride::default(),
        ),
    ];
    let err = bind_entries(&entries, &found).expect_err("duplicate alias");
    assert!(err.contains("m1.gguf") && err.contains("m2.gguf"), "{err}");

    // An alias that collides with another model's registry slug.
    let entries = vec![model_entry(
        Some("llm:a/m1.gguf"),
        None,
        &["llm:b/m2.gguf"],
        ModelCapabilitiesOverride::default(),
    )];
    let err = bind_entries(&entries, &found).expect_err("slug collision");
    assert!(err.contains("collides"), "{err}");

    // Two entries binding the same file.
    let entries = vec![
        model_entry(
            Some("llm:a/m1.gguf"),
            None,
            &[],
            ModelCapabilitiesOverride::default(),
        ),
        model_entry(
            None,
            Some("m1.gguf"),
            &[],
            ModelCapabilitiesOverride::default(),
        ),
    ];
    let err = bind_entries(&entries, &found).expect_err("double bind");
    assert!(err.contains("two [[models.entry]] entries bind"), "{err}");

    // An alias-bearing match hitting several files.
    let multi = vec![
        scan_hit("llm:a/m1.gguf", "/r/a/m1.gguf"),
        scan_hit("llm:b/m1.gguf", "/r/b/m1.gguf"),
    ];
    let entries = vec![model_entry(
        None,
        Some("m1.gguf"),
        &["x"],
        ModelCapabilitiesOverride::default(),
    )];
    let err = bind_entries(&entries, &multi).expect_err("ambiguous match");
    assert!(err.contains("binds 2 files"), "{err}");
}
