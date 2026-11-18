//! TASK-determinism-contract tests.
//!
//! The harness includes ONLY `src/jobs.rs` (which registers
//! `src/determinism.rs`), so this suite compiles and runs while the
//! binary-formats wave's in-flight files (routes/formats/svg/ora) are red
//! — the same `#[path]` include pattern `tests/jobs.rs` uses. Scope is
//! L0/L1 on the stub backend: byte-identical bundle *inputs* for the same
//! request, insertion-order-stable canonical hashing, named PRNG stage
//! streams, stable fusion ordering. By contract there is NO
//! cross-backend or cross-hardware byte-equality assertion here (L3 is
//! unsupported — see `src/determinism.rs` module docs).

#[allow(dead_code)] // the whole module tree, not just the items used below
#[path = "../src/jobs.rs"]
mod jobs;

use jobs::{BundleInputs, determinism, write_bundle_into};

fn fixture() -> BundleInputs {
    BundleInputs {
        model_slug: "comfy:checkpoints/x.safetensors".to_string(),
        prompt: "a red fox".to_string(),
        seed: 42,
        steps: 20,
        width: 16,
        height: 16,
        format: honkbox_sd_core::OutputFormat::Png,
        backend_id: "stub",
        sampler: "euler".to_string(),
        scheduler: "normal".to_string(),
        cfg_scale: 7.0,
        negative_prompt: None,
        loras: Vec::new(),
        strength: None,
        recipe: None,
        hires: None,
    }
}

/// Empty-input SHA-256, pinned so the digest form can never drift.
#[test]
fn sha256_hex_matches_reference() {
    assert_eq!(
        determinism::sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

/// Acceptance: the same request written twice on the stub backend lands
/// byte-identical bundle INPUT files — `request.json` and the sidecar.
/// `metrics.jsonl` is telemetry (carries `elapsed_ms`) and is deliberately
/// not compared; the ULID dir names differ and are excluded by design.
#[test]
fn same_request_twice_writes_byte_identical_inputs() {
    let a = std::env::temp_dir().join(format!("honkbox-det-a-{}", std::process::id()));
    let b = std::env::temp_dir().join(format!("honkbox-det-b-{}", std::process::id()));
    for dir in [&a, &b] {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).expect("test dir");
    }
    write_bundle_into(&a, &fixture(), &[0u8; 8], 7).expect("bundle a");
    write_bundle_into(&b, &fixture(), &[0u8; 8], 999).expect("bundle b");

    let req_a = std::fs::read(a.join("request.json")).expect("request a");
    let req_b = std::fs::read(b.join("request.json")).expect("request b");
    assert_eq!(
        req_a, req_b,
        "request.json byte-identical for the same inputs"
    );
    let side_a = std::fs::read(a.join("final_00.json")).expect("sidecar a");
    let side_b = std::fs::read(b.join("final_00.json")).expect("sidecar b");
    assert_eq!(side_a, side_b, "sidecar byte-identical for the same inputs");

    // The recorded key is exactly the canonical hash of the same inputs.
    let req: serde_json::Value = serde_json::from_slice(&req_a).expect("request json");
    assert_eq!(
        req["inputs_sha256"],
        determinism::bundle_record_hash(&fixture())
    );

    for dir in [&a, &b] {
        std::fs::remove_dir_all(dir).ok();
    }
}

/// Hashing is stable across map insertion order: the same data textualized
/// in two key orders (and built key-by-key in two orders, which bites on
/// `preserve_order` `serde_json` builds) canonicalizes and hashes identically.
#[test]
fn canonical_hash_stable_across_insertion_order() {
    let late_first: serde_json::Value =
        serde_json::from_str(r#"{"b":2,"a":1,"c":{"y":2,"x":1},"d":[3,1,2]}"#)
            .expect("late-first json");
    let early_first: serde_json::Value =
        serde_json::from_str(r#"{"d":[3,1,2],"c":{"x":1,"y":2},"a":1,"b":2}"#)
            .expect("early-first json");
    assert_eq!(
        determinism::canonical_json(&late_first),
        determinism::canonical_json(&early_first)
    );
    assert_eq!(
        determinism::canonical_hash(&late_first),
        determinism::canonical_hash(&early_first)
    );

    let mut z_first = serde_json::Map::new();
    z_first.insert("z".to_string(), serde_json::json!(1));
    z_first.insert("a".to_string(), serde_json::json!(2));
    let mut a_first = serde_json::Map::new();
    a_first.insert("a".to_string(), serde_json::json!(2));
    a_first.insert("z".to_string(), serde_json::json!(1));
    assert_eq!(
        determinism::canonical_hash(&serde_json::Value::Object(z_first)),
        determinism::canonical_hash(&serde_json::Value::Object(a_first))
    );
}

/// The canonical form is compact, byte-order-sorted, and strips volatile
/// keys at every object level.
#[test]
fn canonical_form_is_sorted_compact_and_volatile_free() {
    let v = serde_json::json!({
        "bb": 1,
        "a": [1, 2],
        "c": "x\"y",
        "ulid": "01XYZ",
        "elapsed_ms": 12,
        "nested": { "timestamp": 5, "keep": true }
    });
    assert_eq!(
        determinism::canonical_json(&v),
        r#"{"a":[1,2],"bb":1,"c":"x\"y","nested":{"keep":true}}"#
    );
    let clean = serde_json::json!({ "keep": 1 });
    let dirty = serde_json::json!({ "keep": 1, "queue_position": 3, "job_id": "j" });
    assert_eq!(
        determinism::canonical_hash(&clean),
        determinism::canonical_hash(&dirty),
        "volatile keys never change the hash"
    );
}

/// The record pins versions, keeps the reserved provenance keys (wiring
/// deferred to the bundle construction sites), and reacts to input changes.
#[test]
fn bundle_record_pins_versions_and_notices_input_changes() {
    let rec = determinism::bundle_record(&fixture());
    assert_eq!(rec["record_version"], determinism::BUNDLE_RECORD_VERSION);
    assert_eq!(rec["code_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(rec["backend_id"], "stub");
    assert_eq!(rec["seed"], 42);
    assert_eq!(rec["cfg_scale"], 7.0);
    assert_eq!(rec["loras"], serde_json::json!([]));
    assert!(
        rec["model_sha256"].is_null(),
        "model digest wiring deferred"
    );
    assert!(
        rec["engine_version"].is_null(),
        "engine version wiring deferred"
    );

    let mut other = fixture();
    other.seed = 43;
    assert_ne!(
        determinism::bundle_record_hash(&fixture()),
        determinism::bundle_record_hash(&other),
        "a changed input must change the key"
    );
}

/// Two independently built records of the same full request (negative
/// prompt, strength, recipe, `LoRA` entries, hires pass) share the key.
#[test]
fn equal_full_requests_share_the_key() {
    let build = || BundleInputs {
        negative_prompt: Some("blurry".to_string()),
        loras: vec![jobs::LoraRecord {
            slug: "comfy:loras/detail.safetensors".to_string(),
            strength: 0.8,
        }],
        strength: Some(0.6),
        recipe: Some("sdxl".to_string()),
        hires: Some(jobs::HiresRecord {
            upscaler: "latent",
            target: "32x16".to_string(),
            scale: 2.0,
            denoising_strength: 0.55,
        }),
        ..fixture()
    };
    assert_eq!(
        determinism::bundle_record_hash(&build()),
        determinism::bundle_record_hash(&build())
    );
}

/// Named stage streams: same (seed, stage) replays identically; stage and
/// seed changes give independent streams; children are stable, distinct
/// from the parent, and per-label. Float draws stay in [0, 1).
#[test]
fn stage_streams_are_deterministic_and_independent() {
    let mut s = determinism::stage_stream(42, "denoise");
    let mut twin = determinism::stage_stream(42, "denoise");
    assert_eq!(s.next_u64(), twin.next_u64(), "first draws replay");
    assert_eq!(s.next_u64(), twin.next_u64(), "subsequent draws replay");

    let first = |seed: u64, stage: &str| determinism::stage_stream(seed, stage).next_u64();
    assert_ne!(first(42, "denoise"), first(42, "tile"), "stages differ");
    assert_ne!(first(42, "denoise"), first(43, "denoise"), "seeds differ");

    let child = |label: &str| {
        determinism::stage_stream(42, "denoise")
            .child(label)
            .next_u64()
    };
    assert_eq!(child("tile-0"), child("tile-0"), "children replay");
    assert_ne!(child("tile-0"), child("tile-1"), "children differ");
    assert_ne!(child("tile-0"), first(42, "denoise"), "child != parent");

    let mut probe = determinism::stage_stream(1, "probe");
    for _ in 0..64 {
        let u = probe.next_f32();
        assert!((0.0..1.0).contains(&u), "draw out of [0,1): {u}");
    }
}

/// RRF-style fusion order: score descending, chunk id ascending on ties —
/// so input list order (volatile) cannot flip the output order.
#[test]
fn rrf_tie_breaks_are_score_desc_then_chunk_id_asc() {
    let mut items: Vec<(f64, u64)> = vec![(3.0, 2), (1.0, 9), (3.0, 1), (1.0, 4), (2.5, 7)];
    determinism::sort_score_desc_id_asc(&mut items, |i| i.0, |i| i.1);
    let ids: Vec<u64> = items.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids, vec![1, 2, 7, 4, 9], "ties by id asc, then score desc");
    for pair in items.windows(2) {
        let (score_a, id_a) = (pair[0].0, pair[0].1);
        let (score_b, id_b) = (pair[1].0, pair[1].1);
        let order = score_a.total_cmp(&score_b);
        assert!(
            order.is_gt() || (order.is_eq() && id_a < id_b),
            "adjacent pair not score-desc/id-asc: {pair:?}"
        );
    }
}
