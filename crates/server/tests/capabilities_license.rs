//! Product-license consistency gate (TASK-model-license-registry).
//!
//! Every curated `catalog.json` license id must be documented in the NOTICE
//! template (and the generated NOTICE), and the additive capabilities block
//! must publish the full summary-row shape. No network, no shared state: the
//! inputs are compile-time `include_str!` snapshots.

// No lib target: include the modules the bundle writer and catalog's
// license_gate need. catalog.rs declares `licenses` itself.
#[allow(dead_code)]
#[path = "../src/catalog.rs"]
mod catalog;
#[allow(dead_code)]
#[path = "../src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../src/jobs.rs"]
mod jobs;

use std::collections::BTreeSet;

use honkbox_api_types::{LicenseCapability, LicenseSummaryRow};

/// Compile-time assertion that the additive block is part of the `Capabilities`
/// wire DTO with the documented type. Never called: merely type-checking this
/// accessor fails the build if the field is removed or retyped.
#[allow(dead_code)]
fn capabilities_expose_licenses(
    capabilities: &honkbox_api_types::Capabilities,
) -> Option<&LicenseCapability> {
    capabilities.licenses.as_ref()
}

/// The shipped curated catalog's license ids are all documented in about.hbs
/// and NOTICE, and the additive DTO block carries every summary field with
/// honest absence for unknown source facts.
#[test]
fn shipped_license_ids_are_documented_and_capabilities_expose_the_block() {
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("../src/catalog.json")).expect("shipped catalog parses");
    let ids: BTreeSet<&str> = catalog
        .as_array()
        .expect("catalog is a JSON list")
        .iter()
        .map(|row| {
            row["license"]
                .as_str()
                .expect("every curated row carries a license id")
        })
        .collect();
    assert!(!ids.is_empty(), "shipped catalog must carry license ids");

    let template = include_str!("../../../about.hbs");
    let notice = include_str!("../../../NOTICE");
    for id in &ids {
        assert!(
            template.contains(id),
            "about.hbs is missing curated license id {id}"
        );
        assert!(
            notice.contains(id),
            "NOTICE is missing curated license id {id}"
        );
    }

    let id = (*ids.iter().next().expect("non-empty")).to_owned();
    let full = LicenseCapability {
        rows: vec![LicenseSummaryRow {
            license: id.clone(),
            license_url: Some("https://example.test/license".to_owned()),
            territory: Some("excludes-eu-uk-kr".to_owned()),
            commercial_terms: "non-commercial".to_owned(),
            gated: true,
            source_url: Some("https://example.test/model".to_owned()),
            sha256: Some("ab".repeat(32)),
            attribution_text: Some("Attribution text".to_owned()),
        }],
    };
    let wire = serde_json::to_value(&full).expect("license block serializes");
    let row = &wire["rows"][0];
    for field in [
        "license",
        "license_url",
        "territory",
        "commercial_terms",
        "gated",
        "source_url",
        "sha256",
        "attribution_text",
    ] {
        assert!(row.get(field).is_some(), "summary row is missing {field}");
    }
    assert_eq!(row["license"], id);
    assert!(row["gated"].as_bool().expect("gated is a boolean"));

    // Unknown source facts are omitted, never turned into empty strings that
    // could be mistaken for curated values.
    let sparse = LicenseCapability {
        rows: vec![LicenseSummaryRow {
            license: id,
            license_url: None,
            territory: None,
            commercial_terms: "commercial".to_owned(),
            gated: false,
            source_url: None,
            sha256: None,
            attribution_text: None,
        }],
    };
    let wire = serde_json::to_value(&sparse).expect("sparse license block serializes");
    let row = &wire["rows"][0];
    assert_eq!(row.as_object().expect("row is an object").len(), 3);
    for field in [
        "license_url",
        "territory",
        "source_url",
        "sha256",
        "attribution_text",
    ] {
        assert!(row.get(field).is_none(), "unknown {field} must stay absent");
    }
}

/// Bundle sidecars stamp the curated record beside the existing provenance
/// keys, and NC weights carry the flag. Each run owns a fresh temp dir.
#[test]
fn bundle_sidecar_stamps_curated_license_and_nc_flag() {
    let dir = std::env::temp_dir().join(format!(
        "honkbox-license-bundle-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("per-test temp dir");

    let bundle_for = |slug: &str| jobs::BundleInputs {
        model_slug: slug.to_owned(),
        prompt: "a red fox".to_owned(),
        seed: 42,
        steps: 20,
        width: 16,
        height: 16,
        format: honkbox_sd_core::OutputFormat::Png,
        backend_id: "stub",
        model_sha256: None,
        engine_version: None,
        sampler: "euler".to_owned(),
        scheduler: "normal".to_owned(),
        cfg_scale: 7.0,
        negative_prompt: None,
        loras: Vec::new(),
        strength: None,
        recipe: None,
        hires: None,
        annotations: Vec::new(),
    };

    jobs::write_bundle_into(&dir, &bundle_for("sdxl-base-fp16"), &[0; 8], 7)
        .expect("commercial bundle");
    let sidecar: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("final_00.json")).expect("final_00.json"))
            .expect("sidecar parses");
    assert_eq!(sidecar["model_license"]["license"], "openrail++");
    assert!(!sidecar["model_license"]["gated"].as_bool().unwrap());
    assert!(
        !sidecar["model_license"]["non_commercial"]
            .as_bool()
            .unwrap()
    );

    jobs::write_bundle_into(&dir, &bundle_for("flux2-dev-vae"), &[0; 8], 7).expect("NC bundle");
    let sidecar: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("final_00.json")).expect("final_00.json"))
            .expect("sidecar parses");
    assert_eq!(
        sidecar["model_license"]["license"],
        "flux-non-commercial-license"
    );
    assert!(
        sidecar["model_license"]["non_commercial"]
            .as_bool()
            .unwrap()
    );
    assert!(sidecar["model_license"]["gated"].as_bool().unwrap());

    std::fs::remove_dir_all(&dir).expect("cleanup per-test temp dir");
}
