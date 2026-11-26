//! Versioned, deterministic scene specification (`scene-spec/v1`).
//!
//! A [`SceneSpec`] is the whole input to one render job: hash-addressed
//! assets, ordered objects and transforms, materials, cameras, lights and a
//! fixed set of views. Nothing here resolves an asset, spawns Blender, reads
//! a clock or touches the filesystem. [`validate`] turns a caller-supplied
//! [`serde_json::Value`] into a [`ValidatedScene`] carrying a canonical byte
//! string, its digest and a per-view seed plan, or one named
//! [`SceneSpecError`].
//!
//! Canonical bytes are compact JSON with object keys sorted by UTF-8 byte
//! order and arrays kept in declaration order. Because the spec holds no
//! job ids, timestamps or other volatile fields, the same input always
//! produces the same bytes, digest and seeds. Seeds reuse the server's
//! `stage_stream` KDF so this crate and the job layer derive identical
//! values without depending on each other.

#![allow(clippy::module_name_repetitions)] // the public names are fixed by the schema contract

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Schema identifier carried by every v1 scene spec.
pub const SCHEMA_ID: &str = "scene-spec/v1";

/// The only engine target v1 accepts.
const ENGINE_TARGET_V1: &str = "blender_cycles_cpu";

/// Hard ceiling on canonical bytes (1 MiB).
const MAX_CANONICAL_BYTES: usize = 1024 * 1024;

/// Seed stages, in the order every [`ViewPlan`] records them.
const STAGES: [&str; 3] = ["setup", "render", "post"];

/// Rendering backend a scene targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineTarget {
    /// Headless Blender Cycles on CPU.
    BlenderCyclesCpu,
}

/// Semantic role of a hash-addressed asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetRole {
    /// Geometry to place in the scene.
    Mesh,
    /// Base colour texture.
    Albedo,
    /// Tangent-space normal texture.
    Normal,
    /// Roughness texture.
    Roughness,
    /// Metalness texture.
    Metalness,
    /// Ambient-occlusion texture.
    Ao,
    /// Height or displacement texture.
    Height,
}

/// Camera projection model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    /// Perspective projection, scaled by the camera lens.
    Perspective,
    /// Orthographic projection.
    Orthographic,
}

/// Kind of light in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightType {
    /// Omnidirectional point light.
    Point,
    /// Directional sun light.
    Sun,
    /// Area light with a facing direction.
    Area,
}

/// One hash-addressed input asset.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    /// Stable id, equal to the lowercase hex `SHA-256` of the stored bytes.
    pub asset_id: String,
    /// Lowercase hex `SHA-256` of the stored bytes.
    pub sha256: String,
    /// What the asset is used for.
    pub role: AssetRole,
}

/// A texture binding on a material.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureSlot {
    /// Semantic slot; `Mesh` is not a texture role.
    pub role: AssetRole,
    /// Id of an asset declared in [`SceneSpec::assets`].
    pub asset_id: String,
}

/// One placed object in the scene.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectSpec {
    /// Stable id unique across every collection.
    pub id: String,
    /// Id of a mesh asset.
    pub asset_id: String,
    /// Optional id of a parent object; parenting must be acyclic.
    pub parent_id: Option<String>,
    /// Optional id of a material.
    pub material_id: Option<String>,
    /// Translation in declared scene units.
    pub translation: [f64; 3],
    /// Euler rotation in radians, XYZ order.
    pub rotation: [f64; 3],
    /// Per-axis scale; every component must be positive.
    pub scale: [f64; 3],
}

/// One physically based material.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    /// Stable id unique across every collection.
    pub id: String,
    /// Linear base colour, three finite components.
    pub base_color: [f64; 3],
    /// Metallic scalar.
    pub metallic: f64,
    /// Roughness scalar.
    pub roughness: f64,
    /// Ordered texture bindings, at most 16.
    pub texture_slots: Vec<TextureSlot>,
}

/// One camera definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    /// Stable id unique across every collection.
    pub id: String,
    /// Camera origin.
    pub position: [f64; 3],
    /// Point the camera looks at.
    pub target: [f64; 3],
    /// Camera up vector.
    pub up: [f64; 3],
    /// Projection model.
    pub projection: Projection,
    /// Focal length in millimetres; must be positive.
    pub lens: f64,
    /// Near clip plane; must be positive.
    pub clip_near: f64,
    /// Far clip plane; must exceed the near plane.
    pub clip_far: f64,
}

/// One fixed output view.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    /// Stable id unique across every collection; used in seed stage names.
    pub id: String,
    /// Id of a declared camera.
    pub camera_id: String,
    /// Ordered, safe scene-relative output names.
    pub targets: Vec<String>,
    /// Output width in pixels, 1 through 8192.
    pub width: u32,
    /// Output height in pixels, 1 through 8192.
    pub height: u32,
}

/// One light definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Light {
    /// Stable id unique across every collection.
    pub id: String,
    /// Light kind.
    pub light_type: LightType,
    /// Light origin.
    pub position: [f64; 3],
    /// Facing direction.
    pub direction: [f64; 3],
    /// Linear light colour.
    pub color: [f64; 3],
    /// Radiant power; must be non-negative.
    pub energy: f64,
    /// Distance attenuation; must be non-negative.
    pub attenuation: f64,
}

/// The complete input to one render job.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneSpec {
    /// Must equal [`SCHEMA_ID`].
    pub schema_version: String,
    /// Root seed for every derived stage seed.
    pub seed: u64,
    /// Closed engine target.
    pub engine_target: EngineTarget,
    /// Hash-addressed inputs, at most 256.
    pub assets: Vec<Asset>,
    /// Placed objects, at most 4096.
    pub objects: Vec<ObjectSpec>,
    /// Materials, at most 256.
    pub materials: Vec<Material>,
    /// Cameras, at most 64.
    pub cameras: Vec<Camera>,
    /// Fixed views, at most 64.
    pub views: Vec<View>,
    /// Lights, at most 256.
    pub lights: Vec<Light>,
}

/// A named validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneSpecError {
    /// Stable machine-readable code, for example `unknown_asset`.
    pub code: &'static str,
    /// Field, id or path that failed validation.
    pub field: String,
}

impl fmt::Display for SceneSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.field)
    }
}

impl std::error::Error for SceneSpecError {}

/// Deterministic seed plan for one view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewPlan {
    /// Id of the view this plan belongs to.
    pub view_id: String,
    /// `(stage, seed)` pairs in `setup`, `render`, `post` order.
    pub seeds: Vec<(String, u64)>,
}

/// A [`SceneSpec`] that passed every check.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedScene {
    /// The parsed spec, verbatim.
    pub spec: SceneSpec,
    /// Compact canonical JSON with keys sorted by UTF-8 byte order.
    pub canonical: String,
    /// Lowercase hex `SHA-256` of the canonical bytes.
    pub digest: String,
    /// One seed plan per view, in view order.
    pub view_plan: Vec<ViewPlan>,
}

/// Validate a caller-supplied value and derive its canonical form and seeds.
///
/// The function is pure: no network, no IO, no clock reads. Every rejection
/// carries a stable [`SceneSpecError::code`] so callers can map it to a
/// specific HTTP status without string matching.
pub fn validate(value: &Value) -> Result<ValidatedScene, SceneSpecError> {
    validate_spec(parse_spec(value)?)
}

/// Canonical bytes for a spec: compact JSON, object keys in UTF-8 byte
/// order, arrays in input order.
#[must_use]
pub fn canonical_json(spec: &SceneSpec) -> String {
    let value = serde_json::to_value(spec).expect("scene spec serialization is infallible");
    let mut out = String::new();
    write_canonical(&mut out, &value);
    out
}

/// First `u64` draw of the named server PRNG stage stream.
///
/// Expands `SHA-256(b"honkbox/prng/v1" || seed.to_le_bytes() || stage)` and
/// then `SHA-256(state || 0_u64.to_le_bytes())`, returning the first eight
/// bytes as a little-endian `u64`. This is exactly `stage_stream(seed,
/// stage).next_u64()` on the server.
#[must_use]
pub fn stage_seed(root_seed: u64, stage: &str) -> u64 {
    let mut expansion = Sha256::new();
    expansion.update(b"honkbox/prng/v1");
    expansion.update(root_seed.to_le_bytes());
    expansion.update(stage.as_bytes());
    let seed_state: [u8; 32] = expansion.finalize().into();

    let mut draw = Sha256::new();
    draw.update(seed_state);
    draw.update(0_u64.to_le_bytes());
    let block: [u8; 32] = draw.finalize().into();
    u64::from_le_bytes(block[..8].try_into().expect("8 of 32 bytes"))
}

/// Parse the value, mapping shape failures onto named codes.
fn parse_spec(value: &Value) -> Result<SceneSpec, SceneSpecError> {
    match serde_json::from_value::<SceneSpec>(value.clone()) {
        Ok(spec) => Ok(spec),
        Err(source) => {
            if value
                .get("engine_target")
                .and_then(Value::as_str)
                .is_some_and(|target| target != ENGINE_TARGET_V1)
            {
                return Err(failure("engine_target_unsupported", "engine_target"));
            }
            let message = source.to_string();
            if message.contains("unknown field") {
                return Err(failure("unknown_field", unknown_field_name(&message)));
            }
            Err(failure("parse_error", "root"))
        }
    }
}

/// Run every semantic check on an already parsed spec.
fn validate_spec(spec: SceneSpec) -> Result<ValidatedScene, SceneSpecError> {
    check_header(&spec)?;
    check_limits(&spec)?;
    check_ids(&spec)?;
    check_assets(&spec)?;
    check_objects(&spec)?;
    check_materials(&spec)?;
    check_cameras(&spec)?;
    check_views(&spec)?;
    check_lights(&spec)?;
    check_parent_cycles(&spec)?;

    let canonical = canonical_json(&spec);
    if canonical.len() > MAX_CANONICAL_BYTES {
        return Err(failure("limit_exceeded", "canonical"));
    }
    Ok(ValidatedScene {
        digest: sha256_hex(canonical.as_bytes()),
        view_plan: build_view_plan(&spec),
        spec,
        canonical,
    })
}

/// Build one named error.
fn failure(code: &'static str, field: impl Into<String>) -> SceneSpecError {
    SceneSpecError {
        code,
        field: field.into(),
    }
}

/// Pull the rejected field name out of a serde `unknown field` message.
fn unknown_field_name(message: &str) -> String {
    message
        .split_once('`')
        .and_then(|(_, rest)| rest.split_once('`'))
        .map_or("root", |(name, _)| name)
        .to_owned()
}

/// Check the closed version and engine target.
fn check_header(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    if spec.schema_version != SCHEMA_ID {
        return Err(failure("schema_version_unsupported", "schema_version"));
    }
    if spec.engine_target != EngineTarget::BlenderCyclesCpu {
        return Err(failure("engine_target_unsupported", "engine_target"));
    }
    Ok(())
}

/// Check every collection-size limit.
fn check_limits(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    for (field, len, max) in [
        ("assets", spec.assets.len(), 256),
        ("objects", spec.objects.len(), 4096),
        ("materials", spec.materials.len(), 256),
        ("cameras", spec.cameras.len(), 64),
        ("views", spec.views.len(), 64),
        ("lights", spec.lights.len(), 256),
    ] {
        if len > max {
            return Err(failure("limit_exceeded", field));
        }
    }
    Ok(())
}

/// Require ids to be unique within and across all collections.
fn check_ids(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    let ids = spec
        .assets
        .iter()
        .map(|asset| asset.asset_id.as_str())
        .chain(spec.objects.iter().map(|object| object.id.as_str()))
        .chain(spec.materials.iter().map(|material| material.id.as_str()))
        .chain(spec.cameras.iter().map(|camera| camera.id.as_str()))
        .chain(spec.views.iter().map(|view| view.id.as_str()))
        .chain(spec.lights.iter().map(|light| light.id.as_str()));
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(failure("duplicate_id", id));
        }
    }
    Ok(())
}

/// Check asset digests.
fn check_assets(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    for asset in &spec.assets {
        if !is_hex64(&asset.asset_id) || !is_hex64(&asset.sha256) {
            return Err(failure("invalid_sha256", &asset.asset_id));
        }
        if asset.asset_id != asset.sha256 {
            return Err(failure("asset_id_digest_mismatch", &asset.asset_id));
        }
    }
    Ok(())
}

/// Whether a string is exactly 64 lowercase hex digits.
fn is_hex64(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Check object references and transforms.
fn check_objects(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    let assets: HashSet<&str> = spec
        .assets
        .iter()
        .map(|asset| asset.asset_id.as_str())
        .collect();
    let materials: HashSet<&str> = spec
        .materials
        .iter()
        .map(|material| material.id.as_str())
        .collect();
    let objects: HashSet<&str> = spec
        .objects
        .iter()
        .map(|object| object.id.as_str())
        .collect();
    for (index, object) in spec.objects.iter().enumerate() {
        let at = |part: &str| format!("objects[{index}].{part}");
        if !assets.contains(object.asset_id.as_str()) {
            return Err(failure("unknown_asset", at("asset_id")));
        }
        if let Some(id) = object.material_id.as_deref()
            && !materials.contains(id)
        {
            return Err(failure("unknown_material", at("material_id")));
        }
        if let Some(id) = object.parent_id.as_deref()
            && !objects.contains(id)
        {
            return Err(failure("unknown_parent", at("parent_id")));
        }
        check_finite(object.translation, at("translation"))?;
        check_finite(object.rotation, at("rotation"))?;
        check_finite(object.scale, at("scale"))?;
        if !object.scale.iter().all(|value| *value > 0.0) {
            return Err(failure("non_positive_scale", at("scale")));
        }
    }
    Ok(())
}

/// Check material scalars, slot count and texture references.
fn check_materials(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    let assets: HashSet<&str> = spec
        .assets
        .iter()
        .map(|asset| asset.asset_id.as_str())
        .collect();
    for (index, material) in spec.materials.iter().enumerate() {
        let at = |part: &str| format!("materials[{index}].{part}");
        if material.texture_slots.len() > 16 {
            return Err(failure("limit_exceeded", at("texture_slots")));
        }
        check_finite(material.base_color, at("base_color"))?;
        check_finite([material.metallic], at("metallic"))?;
        check_finite([material.roughness], at("roughness"))?;
        for (slot_index, slot) in material.texture_slots.iter().enumerate() {
            let slot_at = |part: &str| at(&format!("texture_slots[{slot_index}].{part}"));
            if slot.role == AssetRole::Mesh {
                return Err(failure("invalid_role", slot_at("role")));
            }
            if !assets.contains(slot.asset_id.as_str()) {
                return Err(failure("unknown_asset", slot_at("asset_id")));
            }
        }
    }
    Ok(())
}

/// Check camera vectors and clip/lens invariants.
fn check_cameras(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    for (index, camera) in spec.cameras.iter().enumerate() {
        let at = |part: &str| format!("cameras[{index}].{part}");
        check_finite(camera.position, at("position"))?;
        check_finite(camera.target, at("target"))?;
        check_finite(camera.up, at("up"))?;
        check_finite([camera.lens], at("lens"))?;
        check_finite([camera.clip_near], at("clip_near"))?;
        check_finite([camera.clip_far], at("clip_far"))?;
        if !(camera.lens > 0.0 && camera.clip_near > 0.0 && camera.clip_far > camera.clip_near) {
            return Err(failure("invalid_camera", at("clip_far")));
        }
    }
    Ok(())
}

/// Check view camera references, dimensions and output names.
fn check_views(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    let cameras: HashSet<&str> = spec
        .cameras
        .iter()
        .map(|camera| camera.id.as_str())
        .collect();
    for (index, view) in spec.views.iter().enumerate() {
        let at = |part: &str| format!("views[{index}].{part}");
        if !cameras.contains(view.camera_id.as_str()) {
            return Err(failure("unknown_camera", at("camera_id")));
        }
        if !(1..=8192).contains(&view.width) || !(1..=8192).contains(&view.height) {
            return Err(failure("invalid_dimensions", at("dimensions")));
        }
        if view.targets.is_empty() {
            return Err(failure("unsafe_path", at("targets")));
        }
        if view.targets.len() > 32 {
            return Err(failure("limit_exceeded", at("targets")));
        }
        if view.targets.iter().any(|target| !is_safe_target(target)) {
            return Err(failure("unsafe_path", at("targets")));
        }
    }
    Ok(())
}

/// Whether a target is one safe scene-relative path component.
fn is_safe_target(target: &str) -> bool {
    target != "."
        && target != ".."
        && !target.is_empty()
        && !target
            .chars()
            .any(|ch| matches!(ch, '/' | '\\' | '%') || ch.is_control())
}

/// Check light vectors and energy bounds.
fn check_lights(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    for (index, light) in spec.lights.iter().enumerate() {
        let at = |part: &str| format!("lights[{index}].{part}");
        check_finite(light.position, at("position"))?;
        check_finite(light.direction, at("direction"))?;
        check_finite(light.color, at("color"))?;
        check_finite([light.energy], at("energy"))?;
        check_finite([light.attenuation], at("attenuation"))?;
        if light.energy < 0.0 || light.attenuation < 0.0 {
            return Err(failure("invalid_light", at("energy")));
        }
    }
    Ok(())
}

/// Reject a cycle in the single-parent object graph.
fn check_parent_cycles(spec: &SceneSpec) -> Result<(), SceneSpecError> {
    let parents: HashMap<&str, Option<&str>> = spec
        .objects
        .iter()
        .map(|object| (object.id.as_str(), object.parent_id.as_deref()))
        .collect();
    let mut state: HashMap<&str, u8> = HashMap::new();
    for object in &spec.objects {
        if state.get(object.id.as_str()) == Some(&2) {
            continue;
        }
        let mut chain = Vec::new();
        let mut current = object.id.as_str();
        loop {
            match state.get(current).copied() {
                Some(2) => break,
                Some(1) => return Err(failure("parent_cycle", current)),
                _ => {
                    state.insert(current, 1);
                    chain.push(current);
                    match parents.get(current) {
                        Some(Some(parent_id)) => current = *parent_id,
                        _ => break,
                    }
                }
            }
        }
        for id in chain {
            state.insert(id, 2);
        }
    }
    Ok(())
}

/// Require every supplied number to be finite.
fn check_finite(
    values: impl IntoIterator<Item = f64>,
    field: impl Into<String>,
) -> Result<(), SceneSpecError> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(failure("non_finite_number", field))
    }
}

/// Derive the ordered seed plan of every view.
fn build_view_plan(spec: &SceneSpec) -> Vec<ViewPlan> {
    spec.views
        .iter()
        .map(|view| {
            let seeds = STAGES
                .iter()
                .map(|stage| {
                    let name = format!("{SCHEMA_ID}/{stage}/{}", view.id);
                    (stage.to_string(), stage_seed(spec.seed, &name))
                })
                .collect();
            ViewPlan {
                view_id: view.id.clone(),
                seeds,
            }
        })
        .collect()
}

/// Recursive canonical writer; objects are sorted, arrays are not.
fn write_canonical(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => {
            out.push_str(&serde_json::to_string(text).expect("string serialization is infallible"));
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_unstable_by_key(|(left, _)| *left);
            out.push('{');
            for (index, (key, child)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).expect("key serialization is infallible"));
                out.push(':');
                write_canonical(out, child);
            }
            out.push('}');
        }
    }
}

/// Lowercase hex `SHA-256` of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        hex.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    hex
}

/// In-memory tests: every case owns only local `Value`/`SceneSpec` data.
/// No paths, ports, globals, or ordering dependence.
#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_unrelated)] // each case starts from a fresh fixture

    use super::*;
    use serde_json::json;

    /// A lowercase hex digest of `fill` repeated 64 times.
    fn digest(fill: char) -> String {
        std::iter::repeat_n(fill, 64).collect()
    }

    /// One cube, one material, one camera, one light and three views.
    fn valid_spec_value() -> Value {
        let mesh = digest('a');
        let albedo = digest('b');
        json!({
            "schema_version": SCHEMA_ID,
            "seed": 42_u64,
            "engine_target": ENGINE_TARGET_V1,
            "assets": [
                { "asset_id": mesh.clone(), "sha256": mesh.clone(), "role": "mesh" },
                { "asset_id": albedo.clone(), "sha256": albedo.clone(), "role": "albedo" },
            ],
            "objects": [{
                "id": "cube",
                "asset_id": mesh,
                "parent_id": Value::Null,
                "material_id": "mat",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }],
            "materials": [{
                "id": "mat",
                "base_color": [0.8, 0.8, 0.8],
                "metallic": 0.0,
                "roughness": 0.5,
                "texture_slots": [{ "role": "albedo", "asset_id": albedo }],
            }],
            "cameras": [{
                "id": "cam",
                "position": [0.0, 0.0, 5.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "projection": "perspective",
                "lens": 50.0,
                "clip_near": 0.1,
                "clip_far": 100.0,
            }],
            "views": [
                { "id": "front", "camera_id": "cam", "targets": ["front_00.png"], "width": 512, "height": 512 },
                { "id": "side", "camera_id": "cam", "targets": ["side_00.png"], "width": 512, "height": 512 },
                { "id": "top", "camera_id": "cam", "targets": ["top_00.png"], "width": 512, "height": 512 },
            ],
            "lights": [{
                "id": "key",
                "light_type": "area",
                "position": [2.0, 3.0, 4.0],
                "direction": [0.0, 0.0, -1.0],
                "color": [1.0, 1.0, 1.0],
                "energy": 100.0,
                "attenuation": 0.0,
            }],
        })
    }

    /// Assert that validation fails with one named code.
    fn assert_code(value: &Value, expected: &'static str) {
        let error = validate(value).expect_err("expected a validation failure");
        assert_eq!(error.code, expected, "field was {}", error.field);
    }

    #[test]
    fn positive_fixture_validates() {
        let validated = validate(&valid_spec_value()).expect("positive fixture must validate");
        assert_eq!(validated.spec.schema_version, SCHEMA_ID);
        assert_eq!(validated.spec.seed, 42);
        assert_eq!(validated.spec.views.len(), 3);
        assert_eq!(validated.digest.len(), 64);
        assert_eq!(validated.canonical, canonical_json(&validated.spec));
        assert_eq!(validated.view_plan.len(), 3);
        assert_eq!(validated.view_plan[0].view_id, "front");
        assert_eq!(validated.view_plan[0].seeds.len(), 3);
    }

    #[test]
    fn canonical_and_digest_are_stable_across_calls() {
        let value = valid_spec_value();
        let first = validate(&value).expect("first validation");
        let second = validate(&value).expect("second validation");
        assert_eq!(first.canonical, second.canonical);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.view_plan, second.view_plan);
        assert_eq!(first.digest, sha256_hex(first.canonical.as_bytes()));
        assert_eq!(canonical_json(&first.spec), first.canonical);
    }

    #[test]
    fn view_seed_plan_matches_independent_constants() {
        // Values pinned from an independent SHA-256 computation, not from
        // stage_seed itself: rows are views (front, side, top), columns are
        // stages (setup, render, post), root seed 42.
        let expected = [
            [
                7_366_411_686_369_306_787,
                3_085_301_732_685_394_913,
                16_844_505_829_992_170_166,
            ],
            [
                7_958_119_777_050_355_179,
                9_994_731_610_443_526_223,
                12_813_418_754_949_853_527,
            ],
            [
                3_025_430_085_169_004_684,
                4_790_290_300_609_109_471,
                3_403_621_640_268_555_316,
            ],
        ];
        let validated = validate(&valid_spec_value()).expect("positive fixture");
        for (view_index, plan) in validated.view_plan.iter().enumerate() {
            assert_eq!(plan.seeds.len(), STAGES.len());
            for (stage_index, (stage, seed)) in plan.seeds.iter().enumerate() {
                assert_eq!(stage, STAGES[stage_index]);
                assert_eq!(
                    *seed, expected[view_index][stage_index],
                    "view {view_index} stage {stage_index}"
                );
            }
        }
    }

    #[test]
    fn malformed_values_are_parse_errors() {
        assert_code(&json!("not an object"), "parse_error");
        let mut value = valid_spec_value();
        value["seed"] = json!("forty-two");
        assert_code(&value, "parse_error");
    }

    #[test]
    fn unsupported_schema_and_target_are_rejected() {
        let mut value = valid_spec_value();
        value["schema_version"] = json!("scene-spec/v2");
        assert_code(&value, "schema_version_unsupported");

        let mut value = valid_spec_value();
        value["engine_target"] = json!("cycles_gpu");
        assert_code(&value, "engine_target_unsupported");
    }

    #[test]
    fn unknown_fields_are_rejected_at_every_level() {
        let mut value = valid_spec_value();
        value["unexpected"] = json!(1);
        assert_code(&value, "unknown_field");

        let mut value = valid_spec_value();
        value["objects"][0]["unexpected"] = json!(1);
        assert_code(&value, "unknown_field");

        let mut value = valid_spec_value();
        value["materials"][0]["texture_slots"][0]["unexpected"] = json!(1);
        assert_code(&value, "unknown_field");
    }

    #[test]
    fn duplicate_ids_are_rejected_within_and_across_collections() {
        let mut value = valid_spec_value();
        let asset = value["assets"][0].clone();
        value["assets"]
            .as_array_mut()
            .expect("asset array")
            .push(asset);
        assert_code(&value, "duplicate_id");

        let mut value = valid_spec_value();
        value["objects"][0]["id"] = json!(digest('a'));
        assert_code(&value, "duplicate_id");
    }

    #[test]
    fn bad_sha_and_digest_mismatch_are_rejected() {
        let mut value = valid_spec_value();
        value["assets"][0]["sha256"] = json!(digest('c'));
        assert_code(&value, "asset_id_digest_mismatch");

        let mut value = valid_spec_value();
        value["assets"][0]["sha256"] = json!("not-hex");
        assert_code(&value, "invalid_sha256");

        let mut value = valid_spec_value();
        value["assets"][0]["asset_id"] = json!(digest('A'));
        assert_code(&value, "invalid_sha256");
    }

    #[test]
    fn unknown_references_are_rejected() {
        let mut value = valid_spec_value();
        value["objects"][0]["asset_id"] = json!(digest('f'));
        assert_code(&value, "unknown_asset");

        let mut value = valid_spec_value();
        value["objects"][0]["material_id"] = json!("missing");
        assert_code(&value, "unknown_material");

        let mut value = valid_spec_value();
        value["objects"][0]["parent_id"] = json!("missing");
        assert_code(&value, "unknown_parent");

        let mut value = valid_spec_value();
        value["views"][0]["camera_id"] = json!("missing");
        assert_code(&value, "unknown_camera");

        let mut value = valid_spec_value();
        value["materials"][0]["texture_slots"][0]["asset_id"] = json!(digest('f'));
        assert_code(&value, "unknown_asset");
    }

    #[test]
    fn a_mesh_texture_role_is_rejected() {
        let mut value = valid_spec_value();
        value["materials"][0]["texture_slots"][0]["role"] = json!("mesh");
        assert_code(&value, "invalid_role");
    }

    #[test]
    fn parenting_cycles_are_rejected() {
        let mut value = valid_spec_value();
        let object = value["objects"][0].clone();
        let mut other = object.clone();
        other["id"] = json!("cube-two");
        value["objects"]
            .as_array_mut()
            .expect("object array")
            .push(other);
        value["objects"][0]["parent_id"] = json!("cube-two");
        value["objects"][1]["parent_id"] = json!("cube");
        assert_code(&value, "parent_cycle");
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        let mut spec: SceneSpec =
            serde_json::from_value(valid_spec_value()).expect("parse fixture");
        spec.objects[0].translation[0] = f64::INFINITY;
        assert_eq!(
            validate_spec(spec).expect_err("infinite transform").code,
            "non_finite_number"
        );

        let mut spec: SceneSpec =
            serde_json::from_value(valid_spec_value()).expect("parse fixture");
        spec.lights[0].energy = f64::NAN;
        assert_eq!(
            validate_spec(spec).expect_err("NaN light energy").code,
            "non_finite_number"
        );
    }

    #[test]
    fn non_positive_scale_is_rejected() {
        for scale in [[0.0, 1.0, 1.0], [-1.0, 1.0, 1.0]] {
            let mut value = valid_spec_value();
            value["objects"][0]["scale"] = json!(scale);
            assert_code(&value, "non_positive_scale");
        }
    }

    #[test]
    fn invalid_camera_and_light_values_are_rejected() {
        let mut value = valid_spec_value();
        value["cameras"][0]["clip_far"] = json!(0.05);
        assert_code(&value, "invalid_camera");

        let mut value = valid_spec_value();
        value["lights"][0]["energy"] = json!(-1.0);
        assert_code(&value, "invalid_light");
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        for (width, height) in [(0_u64, 512_u64), (512, 8193)] {
            let mut value = valid_spec_value();
            value["views"][0]["width"] = json!(width);
            value["views"][0]["height"] = json!(height);
            assert_code(&value, "invalid_dimensions");
        }
    }

    #[test]
    fn unsafe_targets_are_rejected() {
        for target in [
            "",
            ".",
            "..",
            "/abs.png",
            "a/b.png",
            "a\\b.png",
            "%2f.png",
            "a\u{0001}b",
        ] {
            let mut value = valid_spec_value();
            value["views"][0]["targets"] = json!([target]);
            assert_code(&value, "unsafe_path");
        }
        let mut value = valid_spec_value();
        value["views"][0]["targets"] = json!([]);
        assert_code(&value, "unsafe_path");
    }

    #[test]
    fn collection_limits_are_enforced() {
        let mut value = valid_spec_value();
        let asset = value["assets"][0].clone();
        for _ in 0..256 {
            value["assets"]
                .as_array_mut()
                .expect("asset array")
                .push(asset.clone());
        }
        assert_code(&value, "limit_exceeded");

        let mut value = valid_spec_value();
        let slot = value["materials"][0]["texture_slots"][0].clone();
        for _ in 0..16 {
            value["materials"][0]["texture_slots"]
                .as_array_mut()
                .expect("slot array")
                .push(slot.clone());
        }
        assert_code(&value, "limit_exceeded");

        let targets: Vec<String> = (0..33).map(|index| format!("target-{index}.png")).collect();
        let mut value = valid_spec_value();
        value["views"][0]["targets"] = json!(targets);
        assert_code(&value, "limit_exceeded");

        let mut value = valid_spec_value();
        let view = value["views"][0].clone();
        for index in 0..64 {
            let mut extra = view.clone();
            extra["id"] = json!(format!("view-{index}"));
            value["views"]
                .as_array_mut()
                .expect("view array")
                .push(extra);
        }
        assert_code(&value, "limit_exceeded");
    }

    #[test]
    fn canonical_byte_limit_is_enforced() {
        let mut value = valid_spec_value();
        let object = value["objects"][0].clone();
        let objects: Vec<Value> = (0..300)
            .map(|index| {
                let mut item = object.clone();
                item["id"] = json!("x".repeat(3_900) + &index.to_string());
                item
            })
            .collect();
        value["objects"] = json!(objects);
        assert_code(&value, "limit_exceeded");
    }
}
