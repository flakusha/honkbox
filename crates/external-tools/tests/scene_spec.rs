//! Integration coverage for the public scene-spec surface of external-tools.

use honkbox_external_tools::{SCHEMA_ID, canonical_json, stage_seed, validate};
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// One unique temp directory per test; removed on every exit path.
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "honkbox-scene-spec-{}-{}-{label}",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for TempDir {
    type Target = PathBuf;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn digest(n: u64) -> String {
    format!("{n:064x}")
}

fn base_value() -> serde_json::Value {
    let mesh = digest(1);
    let albedo = digest(2);
    let normal = digest(3);
    json!({
        "schema_version": SCHEMA_ID,
        "seed": 42,
        "engine_target": "blender_cycles_cpu",
        "assets": [
            { "asset_id": mesh, "sha256": mesh, "role": "mesh" },
            { "asset_id": albedo, "sha256": albedo, "role": "albedo" },
            { "asset_id": normal, "sha256": normal, "role": "normal" }
        ],
        "objects": [{
            "id": "cube",
            "asset_id": mesh,
            "parent_id": null,
            "material_id": "cube_mat",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0]
        }],
        "materials": [{
            "id": "cube_mat",
            "base_color": [0.8, 0.2, 0.1],
            "metallic": 0.0,
            "roughness": 0.5,
            "texture_slots": [
                { "role": "albedo", "asset_id": albedo },
                { "role": "normal", "asset_id": normal }
            ]
        }],
        "cameras": [{
            "id": "hero",
            "position": [0.0, -5.0, 1.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "projection": "perspective",
            "lens": 50.0,
            "clip_near": 0.1,
            "clip_far": 100.0
        }],
        "views": [
            { "id": "front", "camera_id": "hero", "targets": ["cube"], "width": 512, "height": 512 },
            { "id": "side", "camera_id": "hero", "targets": ["cube"], "width": 512, "height": 512 },
            { "id": "top", "camera_id": "hero", "targets": ["cube"], "width": 512, "height": 512 }
        ],
        "lights": [{
            "id": "key",
            "light_type": "point",
            "position": [2.0, -2.0, 4.0],
            "direction": [0.0, 0.0, -1.0],
            "color": [1.0, 1.0, 1.0],
            "energy": 1000.0,
            "attenuation": 2.0
        }]
    })
}

#[test]
fn cube_scene_validates_to_stable_canonical_bytes_and_digest() {
    assert_eq!(SCHEMA_ID, "scene-spec/v1");
    let first = validate(&base_value()).expect("cube scene validates");
    let second = validate(&base_value()).expect("repeat validates");

    assert_eq!(first.spec, second.spec);
    assert_eq!(first.canonical, second.canonical);
    assert_eq!(first.digest, second.digest);
    assert_eq!(first.view_plan, second.view_plan);
    assert_eq!(first.digest.len(), 64);
    assert!(
        first
            .digest
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
    );

    let spec: honkbox_external_tools::SceneSpec =
        serde_json::from_value(base_value()).expect("SceneSpec is publicly deserializable");
    assert_eq!(spec, first.spec);
    assert_eq!(first.canonical, canonical_json(&first.spec));
    assert_eq!(first.spec.schema_version, SCHEMA_ID);
    assert_eq!(first.spec.objects.len(), 1);
    assert_eq!(first.spec.materials.len(), 1);
    assert_eq!(first.spec.cameras.len(), 1);
    assert_eq!(first.spec.lights.len(), 1);
    assert_eq!(first.spec.views.len(), 3);

    assert!(first.canonical.starts_with("{\"assets\":"));
    assert!(!first.canonical.contains(char::is_whitespace));
}

#[test]
fn view_plan_keeps_view_order_and_stage_seeds() {
    let validated = validate(&base_value()).expect("valid scene");
    let plans: &[honkbox_external_tools::ViewPlan] = &validated.view_plan;
    let view_ids = ["front", "side", "top"];
    let stages = ["setup", "render", "post"];
    assert_eq!(plans.len(), view_ids.len());

    for (plan, view_id) in plans.iter().zip(view_ids) {
        assert_eq!(plan.view_id, view_id);
        assert_eq!(plan.seeds.len(), stages.len());
        for ((stage, seed), name) in plan.seeds.iter().zip(stages) {
            assert_eq!(stage.as_str(), name);
            let expected = format!("{SCHEMA_ID}/{name}/{view_id}");
            assert_eq!(*seed, stage_seed(42, &expected));
        }
    }

    let root_a = stage_seed(42, "scene-spec/v1/setup/front");
    let root_b = stage_seed(43, "scene-spec/v1/setup/front");
    let other_stage = stage_seed(42, "scene-spec/v1/render/front");
    assert_ne!(root_a, root_b);
    assert_ne!(root_a, other_stage);
}

#[test]
fn canonical_bytes_round_trip_in_unique_temp_dir() {
    let dir = TempDir::new("canonical");
    let path = dir.join("scene.canonical.json");
    let validated = validate(&base_value()).expect("valid scene");
    std::fs::write(&path, validated.canonical.as_bytes()).expect("write canonical");
    let bytes = std::fs::read(&path).expect("read canonical");
    assert_eq!(bytes, validated.canonical.as_bytes());
    assert_eq!(bytes.len(), validated.canonical.len());
    assert_eq!(
        std::str::from_utf8(&bytes).expect("utf8"),
        canonical_json(&validated.spec)
    );
}

#[test]
fn public_enums_use_snake_case_vocabulary() {
    use honkbox_external_tools::{AssetRole, EngineTarget, LightType, Projection};
    assert_eq!(
        serde_json::to_value(EngineTarget::BlenderCyclesCpu).expect("serialize"),
        json!("blender_cycles_cpu")
    );
    for (role, name) in [
        (AssetRole::Mesh, "mesh"),
        (AssetRole::Albedo, "albedo"),
        (AssetRole::Normal, "normal"),
        (AssetRole::Roughness, "roughness"),
        (AssetRole::Metalness, "metalness"),
        (AssetRole::Ao, "ao"),
        (AssetRole::Height, "height"),
    ] {
        assert_eq!(serde_json::to_value(role).expect("serialize"), json!(name));
    }
    assert_eq!(
        serde_json::to_value(Projection::Perspective).expect("serialize"),
        json!("perspective")
    );
    assert_eq!(
        serde_json::to_value(Projection::Orthographic).expect("serialize"),
        json!("orthographic")
    );
    for (kind, name) in [
        (LightType::Point, "point"),
        (LightType::Sun, "sun"),
        (LightType::Area, "area"),
    ] {
        assert_eq!(serde_json::to_value(kind).expect("serialize"), json!(name));
    }
}

struct Case {
    name: String,
    value: serde_json::Value,
    code: String,
}

fn case(name: &str, mutate: impl FnOnce(&mut serde_json::Value), code: &str) -> Case {
    let mut value = base_value();
    mutate(&mut value);
    Case {
        name: name.to_owned(),
        value,
        code: code.to_owned(),
    }
}

fn shape_cases() -> Vec<Case> {
    vec![
        case(
            "unknown_field",
            |v| {
                v["bogus"] = json!(1);
            },
            "unknown_field",
        ),
        case(
            "parse_error",
            |v| {
                v["seed"] = json!("not-a-number");
            },
            "parse_error",
        ),
        case(
            "schema_version_unsupported",
            |v| {
                v["schema_version"] = json!("scene-spec/v2");
            },
            "schema_version_unsupported",
        ),
        case(
            "engine_target_unsupported",
            |v| {
                v["engine_target"] = json!("cycles_cpu");
            },
            "engine_target_unsupported",
        ),
        case(
            "invalid_sha256",
            |v| {
                v["assets"][0]["asset_id"] = json!("not-a-digest");
                v["assets"][0]["sha256"] = json!("not-a-digest");
                v["objects"][0]["asset_id"] = json!("not-a-digest");
            },
            "invalid_sha256",
        ),
        case(
            "asset_id_digest_mismatch",
            |v| {
                v["assets"][0]["sha256"] = json!(digest(99));
            },
            "asset_id_digest_mismatch",
        ),
        case(
            "duplicate_id",
            |v| {
                let first = v["objects"][0].clone();
                v["objects"]
                    .as_array_mut()
                    .expect("objects array")
                    .push(first);
            },
            "duplicate_id",
        ),
    ]
}

fn reference_cases() -> Vec<Case> {
    vec![
        case(
            "invalid_role",
            |v| {
                v["materials"][0]["texture_slots"][0]["role"] = json!("mesh");
            },
            "invalid_role",
        ),
        case(
            "unknown_asset",
            |v| {
                v["objects"][0]["asset_id"] = json!(digest(999));
            },
            "unknown_asset",
        ),
        case(
            "unknown_material",
            |v| {
                v["objects"][0]["material_id"] = json!("missing-material");
            },
            "unknown_material",
        ),
        case(
            "unknown_camera",
            |v| {
                v["views"][0]["camera_id"] = json!("missing-camera");
            },
            "unknown_camera",
        ),
        case(
            "unknown_parent",
            |v| {
                v["objects"][0]["parent_id"] = json!("missing-parent");
            },
            "unknown_parent",
        ),
        case(
            "parent_cycle",
            |v| {
                let mut child = v["objects"][0].clone();
                child["id"] = json!("cube-child");
                child["parent_id"] = json!("cube");
                v["objects"][0]["parent_id"] = json!("cube-child");
                v["objects"]
                    .as_array_mut()
                    .expect("objects array")
                    .push(child);
            },
            "parent_cycle",
        ),
    ]
}

fn numeric_cases() -> Vec<Case> {
    vec![
        case(
            "non_positive_scale",
            |v| {
                v["objects"][0]["scale"] = json!([0.0, 1.0, 1.0]);
            },
            "non_positive_scale",
        ),
        case(
            "invalid_camera",
            |v| {
                v["cameras"][0]["clip_near"] = json!(0.0);
            },
            "invalid_camera",
        ),
        case(
            "invalid_light",
            |v| {
                v["lights"][0]["energy"] = json!(-1.0);
            },
            "invalid_light",
        ),
        case(
            "invalid_dimensions",
            |v| {
                v["views"][0]["height"] = json!(0u64);
            },
            "invalid_dimensions",
        ),
    ]
}

fn path_and_limit_cases() -> Vec<Case> {
    vec![
        case(
            "unsafe_path",
            |v| {
                v["views"][0]["targets"][0] = json!("../outside");
            },
            "unsafe_path",
        ),
        case(
            "limit_exceeded",
            |v| {
                let target = v["views"][0]["targets"][0].clone();
                let targets = v["views"][0]["targets"]
                    .as_array_mut()
                    .expect("targets array");
                for _ in 0..32 {
                    targets.push(target.clone());
                }
            },
            "limit_exceeded",
        ),
    ]
}

#[test]
fn validation_rejects_bad_specs_with_exact_error_codes() {
    let cases = shape_cases()
        .into_iter()
        .chain(reference_cases())
        .chain(numeric_cases())
        .chain(path_and_limit_cases())
        .collect::<Vec<_>>();
    assert!(cases.len() >= 10);
    for item in cases {
        let err = validate(&item.value).expect_err(item.name.as_str());
        assert_eq!(err.code, item.code.as_str(), "case: {}", item.name);
    }
}
