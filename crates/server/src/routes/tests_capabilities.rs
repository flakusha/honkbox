//! Capability and catalog-derivation tests split out of `routes::tests`.

use super::tests_bindings::{model_entry, scan_hit};
use super::*;
use crate::config::ModelCapabilitiesOverride;
use honkbox_api_types::ModelKind;

/// Operator capability declarations override the kind-inferred
/// defaults; `chat = false` + `embeddings = true` reclassifies the
/// emitted `kind` (embedding GGUF discoverability).
#[test]
fn capabilities_override_kind_defaults() {
    let found = vec![scan_hit("llm:e/embed.gguf", "/r/e/embed.gguf")];
    let entries = vec![model_entry(
        Some("llm:e/embed.gguf"),
        None,
        &["embed-0.6b"],
        ModelCapabilitiesOverride {
            chat: Some(false),
            embeddings: Some(true),
            ..Default::default()
        },
    )];
    let bindings = bind_entries(&entries, &found).expect("bind");
    let catalog = build_catalog(&found, &bindings);
    let entry = &catalog.data[0];
    assert_eq!(entry.id, "embed-0.6b");
    assert_eq!(entry.kind, ModelKind::Embedding);
    assert!(!entry.capabilities.chat);
    assert!(!entry.capabilities.completion);
    assert!(entry.capabilities.embeddings);
    assert!(!entry.tool_calling);

    // A partial declaration keeps the inferred defaults for the rest.
    let entries = vec![model_entry(
        Some("llm:e/embed.gguf"),
        None,
        &[],
        ModelCapabilitiesOverride {
            tool_calling: Some(true),
            vision: Some(true),
            ..Default::default()
        },
    )];
    let bindings = bind_entries(&entries, &found).expect("bind");
    let catalog = build_catalog(&found, &bindings);
    let entry = &catalog.data[0];
    assert_eq!(entry.kind, ModelKind::Llm);
    assert!(entry.capabilities.chat);
    assert!(entry.capabilities.tool_calling);
    assert!(entry.capabilities.vision);
    assert_eq!(entry.modalities, vec!["text", "vision"]);
    assert!(entry.tool_calling);
}

/// Default-on vision chat: a chat model with a sibling `mmproj*.gguf`
/// projector advertises `capabilities.vision` with no operator
/// declaration, because the default feature set compiles the mtmd
/// path. `/v1/models` must never advertise vision the build cannot
/// serve, so a text-only build (no `llm-vision`) keeps the same
/// layout text-only — the assertion tracks `VISION_COMPILED` for
/// both configs.
#[test]
fn sibling_mmproj_derives_vision_by_default() {
    let base = std::env::temp_dir().join(format!("honkbox-vision-catalog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("test dir");
    let model_path = base.join("m1.gguf");
    std::fs::write(&model_path, b"m").expect("write model file");
    let mmproj_path = base.join("mmproj-m1-f16.gguf");
    std::fs::write(&mmproj_path, b"p").expect("write mmproj file");

    let found = vec![scan_hit("llm:v/m1.gguf", &model_path.display().to_string())];
    let bindings = bind_entries(&[], &found).expect("bind");
    let catalog = build_catalog(&found, &bindings);
    let entry = &catalog.data[0];
    std::fs::remove_dir_all(&base).ok();
    assert_eq!(
        entry.capabilities.vision,
        honkbox_llm_worker::VISION_COMPILED
    );
    let expected = if entry.capabilities.vision {
        vec!["text".to_owned(), "vision".to_owned()]
    } else {
        vec!["text".to_owned()]
    };
    assert_eq!(entry.modalities, expected);
}
