//! Reproducibility contract (TASK-determinism-contract): the definitions,
//! canonical hashing, and seeded-stream primitives every generation stage
//! builds on. Pure functions only — no I/O, no clocks, no global state.
//!
//! # Reproducibility levels
//!
//! Recorded artifacts (job bundles) publish *which* level their backend
//! honors. Consumers must never assume a higher level than the bundle
//! records (`backend_id` + this module's provenance keys).
//!
//! - **L0 — stable identifiers.** The same logical request always produces
//!   the same canonical input record, `inputs_sha256` key, and command
//!   hash. Always true: this module is the implementation (pure functions
//!   over the recorded fields).
//! - **L1 — same backend, same machine, bit-identical re-run.** The stub
//!   backend guarantees L1 today. GPU diffusion backends target L1 for a
//!   fixed seed/shape but are not certified: kernel autotuning, atomic
//!   reduction order, and driver updates can flip low bits between runs.
//! - **L2 — same engine + code versions, stable across processes and
//!   restarts on the same hardware class.** Backends reach L2 only when
//!   the bundle's engine/code provenance keys pin the exact versions that
//!   produced a run (wiring deferred — see [`bundle_record`]).
//! - **L3 — cross-hardware / cross-backend byte equality: explicitly
//!   UNSUPPORTED.** Never assert byte equality across backends, GPUs, or
//!   CPU instruction sets; cross-backend parity tests are structural
//!   (shapes, dtypes, metadata), never bytes. There is no L3 assertion in
//!   this repository and none may be added.
//!
//! Deterministic kernels (fixed reduction order, no atomics in
//! accumulations) can be measurably slower than the fastest
//! non-deterministic equivalent; honkbox does not opt into them today,
//! which is exactly why L2/L3 are not claimed for GPU backends.
//!
//! # Canonical hashing
//!
//! [`canonical_json`] is the canonical form: compact, object keys sorted
//! by UTF-8 byte order, arrays kept in order (array order is data), and
//! [`VOLATILE_KEYS`] dropped (ULIDs, wall-clock timestamps, queue
//! positions — values that differ between two runs of the same request).
//! [`canonical_hash`] is SHA-256 over that form, so hashes are stable
//! across map insertion order and serialization quirks. Hash dep: `sha2`
//! (MIT/Apache-2.0), already a workspace dependency — no new hash family.
//!
//! The bundle record ([`bundle_record`]) is version-scoped on purpose: it
//! embeds `record_version` and the server's code version, and its
//! `model_sha256`/`engine_version` keys carry provenance when known (the
//! verified digest sidecar of a checksum-verified download; the vendored
//! engine identity) and stay null otherwise — honest absence, never a
//! fabricated hash. A hash never means "this pixel output equals that
//! one" — it means "this recorded input set equals that one" (L0/L1
//! evidence, never L3).

// The stage-stream and fusion-order primitives land ahead of their
// consumers on purpose: the contract reserves them for future stages
// (per-stage noise, RRF fusion) without retrofitting today's pipelines.
// Dead code is tolerated at module scope until a stage adopts them; drop
// this allow once the last item gains a caller.
#![allow(dead_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::BundleInputs;

/// Schema version of [`bundle_record`]'s payload. Bump on any change to
/// the key set so consumers can tell historical records apart.
pub const BUNDLE_RECORD_VERSION: u32 = 1;

/// Object keys excluded from the canonical form because they vary between
/// two runs of the same logical request.
///
/// - `ulid`, `job_id`, `request_id`: per-run identifiers (the bundle dir
///   name is a fresh ULID every write; `request_id` is the idempotency
///   key of the *call*, not of the *inputs*).
/// - `timestamp`, `created_at`, `started_at`, `finished_at`,
///   `elapsed_ms`: wall-clock values.
/// - `queue_position`, `queue_number`: scheduler bookkeeping.
pub const VOLATILE_KEYS: &[&str] = &[
    "ulid",
    "job_id",
    "request_id",
    "queue_position",
    "queue_number",
    "timestamp",
    "created_at",
    "started_at",
    "finished_at",
    "elapsed_ms",
];

/// Canonical JSON: compact, object keys in UTF-8 byte order, arrays in
/// order, [`VOLATILE_KEYS`] dropped from objects (recursively).
///
/// Stable across map insertion order by construction — keys are sorted
/// here, never trusted from the container. Numbers use `serde_json`'s
/// shortest round-trip form (deterministic for a given `Value`); `f32`
/// fields widen to `f64` exactly, as the sidecars already do.
#[must_use]
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

/// Recursive canonical writer behind [`canonical_json`]. Objects: keys
/// sorted by UTF-8 byte order, [`VOLATILE_KEYS`] dropped (the relative
/// order of the remaining keys is preserved). Arrays: element order kept
/// — array order is data, not container state. Strings: standard JSON
/// escaping via `serde_json`; numbers: `serde_json`'s shortest round-trip
/// form.
///
/// # Panics
///
/// Never: serializing a string key or value that is already a Rust
/// `String`/`&str` cannot fail.
fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push_str(&serde_json::to_string(s).expect("str serialization is infallible"));
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            let mut first = true;
            for (key, val) in entries {
                if VOLATILE_KEYS.contains(&key.as_str()) {
                    continue;
                }
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&serde_json::to_string(key).expect("key serialization is infallible"));
                out.push(':');
                write_value(out, val);
            }
            out.push('}');
        }
    }
}

/// SHA-256 of [`canonical_json`], lowercase hex — the canonical input
/// key. Equal hashes mean equal canonical inputs (L0); this is never a
/// cross-hardware output-equality claim.
#[must_use]
pub fn canonical_hash(value: &Value) -> String {
    sha256_hex(canonical_json(value).as_bytes())
}

/// Lowercase hex SHA-256 of `bytes` (the same digest form the sidecars
/// and the image-ops bundle already record).
///
/// # Panics
///
/// Never: each byte's nibbles are 0-15, always valid radix-16 digits, so
/// `char::from_digit` cannot return `None` here.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        hex.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    hex
}

/// The canonical input record of a generation bundle: every recorded
/// input (model, seed, sampler/scheduler, steps, cfg, dimensions,
/// negative prompt, strength, recipe, hires pass, `LoRA` entries, annotations) plus the
/// version/provenance context (`record_version`, `code_version`,
/// `backend_id`).
///
/// `model_sha256` carries the verified digest sidecar of a
/// checksum-verified download and `engine_version` the vendored engine
/// identity (`sd-core::engine_version_for`); both are null when unknown —
/// locally-placed or unverified weights, and the stub backend, which has
/// no separate engine (its identity is `code_version`). Null is honest
/// absence; a hash is never fabricated.
///
/// Deliberately absent: ULID, wall-clock values, queue position,
/// `elapsed_ms` — the record covers *inputs*, not telemetry.
#[must_use]
pub fn bundle_record(inputs: &BundleInputs) -> Value {
    json!({
        "record_version": BUNDLE_RECORD_VERSION,
        "code_version": env!("CARGO_PKG_VERSION"),
        "backend_id": inputs.backend_id,
        "model": inputs.model_slug,
        "model_sha256": inputs.model_sha256,
        "engine_version": inputs.engine_version,
        "prompt": inputs.prompt,
        "negative_prompt": inputs.negative_prompt,
        "seed": inputs.seed,
        "steps": inputs.steps,
        "width": inputs.width,
        "height": inputs.height,
        "format": inputs.format.name(),
        "sampler": inputs.sampler,
        "scheduler": inputs.scheduler,
        "cfg_scale": inputs.cfg_scale,
        "loras": inputs.loras,
        "strength": inputs.strength,
        "mask_mode": inputs.mask_mode,
        "mask_sha256": inputs.mask_sha256,
        "recipe": inputs.recipe,
        "hires": inputs.hires,
        "annotations": inputs.annotations,
    })
}

/// SHA-256 hex over [`bundle_record`] — the `inputs_sha256` the bundle
/// sidecars carry. Two runs with equal inputs and equal versions produce
/// equal keys regardless of map insertion order (L0).
#[must_use]
pub fn bundle_record_hash(inputs: &BundleInputs) -> String {
    canonical_hash(&bundle_record(inputs))
}

/// A named PRNG stream derived from a job's root seed (L0/L1 plumbing
/// for future stages: per-stage noise, tile offsets, crop choices —
/// anything a stage must reproduce exactly for a given seed).
///
/// Not cryptographically secure output — *seeded, deterministic*
/// expansion. Construction (the KDF, part of the contract):
/// `state = SHA-256("honkbox/prng/v1" || root_seed.to_le_bytes() ||
/// stage)` — the fixed-length seed makes the stage name unambiguous
/// without a separator. Different stages therefore get independent
/// streams, and one stage's draws never depend on another stage's call
/// order.
///
/// Consumption rule for stages: draw a request-shape-fixed number of
/// values per run (never a count that depends on wall-clock, queue
/// state, or other volatile input), so a re-run replays identically.
#[must_use]
pub fn stage_stream(root_seed: u64, stage: &str) -> StageStream {
    let mut h = Sha256::new();
    h.update(b"honkbox/prng/v1");
    h.update(root_seed.to_le_bytes());
    h.update(stage.as_bytes());
    StageStream {
        state: h.finalize().into(),
        counter: 0,
    }
}

/// A deterministic byte stream (see [`stage_stream`]).
#[derive(Clone, Debug)]
pub struct StageStream {
    state: [u8; 32],
    counter: u64,
}

impl StageStream {
    /// Next 64 bits, little-endian, from
    /// `SHA-256(state || counter.to_le_bytes())`; the counter advances so
    /// the stream never repeats a block within `u64` draws.
    ///
    /// # Panics
    ///
    /// Never: the digest is 32 bytes and the extraction takes its first
    /// 8, so `try_into` cannot fail.
    pub fn next_u64(&mut self) -> u64 {
        let mut h = Sha256::new();
        h.update(self.state);
        h.update(self.counter.to_le_bytes());
        let block: [u8; 32] = h.finalize().into();
        self.counter = self.counter.wrapping_add(1);
        u64::from_le_bytes(block[..8].try_into().expect("8 of 32 bytes"))
    }

    /// Uniform in `[0, 1)`: 24 significant bits per draw, so every
    /// returned value is exactly representable and no backend-dependent
    /// rounding is involved.
    #[allow(clippy::cast_precision_loss)] // a 24-bit integer draw is exact in f32
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Derived child stream for a sub-stage (e.g. one stream per tile
    /// inside a stage), so adding a sub-stage never shifts the parent's
    /// draw sequence.
    #[must_use]
    pub fn child(&self, label: &str) -> StageStream {
        let mut h = Sha256::new();
        h.update(self.state);
        h.update(b"\x00child\x00");
        h.update(label.as_bytes());
        StageStream {
            state: h.finalize().into(),
            counter: 0,
        }
    }
}

/// Stable ordering for rank-fusion inputs (RRF and friends): score
/// **descending**, then chunk id **ascending** on ties.
///
/// Ranked-list positions are volatile (they depend on which lists ran
/// and in what order); chunk ids are stable identifiers. Ties broken by
/// id make fused output order independent of input list order — the
/// same contract the canonical hash gives inputs.
pub fn sort_score_desc_id_asc<T>(
    items: &mut [T],
    score: impl Fn(&T) -> f64,
    id: impl Fn(&T) -> u64,
) {
    items.sort_unstable_by(|a, b| {
        // `total_cmp` gives a total order even with NaN scores, so the
        // sort key is fully deterministic.
        score(b)
            .total_cmp(&score(a))
            .then_with(|| id(a).cmp(&id(b)))
    });
}
