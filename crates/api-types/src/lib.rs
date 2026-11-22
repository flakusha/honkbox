//! Shared DTOs: the only types that cross the Rust/TS boundary.
//!
//! No routes, DB, tokio, or backend logic — serde + `ts-rs` + `utoipa`
//! derives only, so codegen never pulls in the server dependency tree.

use serde::{Deserialize, Serialize};
use ts_rs::TS;
use utoipa::ToSchema;

/// Liveness probe response.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct Health {
    /// Always `"ok"` while the server is alive.
    pub status: String,
}

/// Binary name + version response.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct Version {
    /// Binary name, always `"honkbox"`.
    pub name: String,
    /// Crate version, from `CARGO_PKG_VERSION`.
    pub version: String,
}

/// Wire-contract version of the HTTP API (surfaced as `api_version` on
/// `GET /v1/capabilities`). This is the CONTRACT version, deliberately
/// independent of the crate version: crate `version` tracks releases, not
/// the wire shape.
///
/// API version policy (also embedded in the [`Capabilities`] schema
/// description, so clients discover it from the generated OpenAPI):
/// bump `api_version` when the wire contract changes incompatibly — a DTO
/// field removal or type change, an error-code change, or a semantic change
/// to an advertised capability flag. Purely ADDITIVE optional fields may
/// land under the existing `api_version` (the `codegen:check` idempotence
/// gate keeps the generated schema honest).
///
/// Deliberately NO `min_client_api_version`: the server serves exactly one
/// contract and additive optional fields are feature-detected by presence
/// (`skip_serializing_if`), so a min field would just duplicate
/// `api_version` until the first breaking bump. Introduce it only if a
/// bump ships with a real compatibility window (see `docs/README.md`,
/// "API version policy").
pub const API_VERSION: &str = "v1";

/// Machine-readable server contract (`GET /v1/capabilities`). Everything is
/// DERIVED at request time from compiled features, runtime config, and the
/// live catalog snapshot — never a prose hard-code — so the document is
/// always true of the process answering.
///
/// API version policy: `api_version` is the wire-contract version,
/// independent of the crate version. Bump it when the contract changes
/// incompatibly (DTO field removal/type change, error-code change, or an
/// advertised-flag semantic change); purely additive optional fields may
/// land under the existing version. Crate `version` tracks releases, not
/// the wire contract.
// No `Eq`: `nsfw` carries an `f64` threshold, and `Eq` is not derivable
// for a struct containing one.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct Capabilities {
    /// Wire-contract version (`"v1"`); see the version policy above.
    pub api_version: String,
    /// Server identity (name + crate version).
    pub server: ServerInfo,
    /// Feature flags derived from compile features, runtime config, and the
    /// live catalog.
    pub features: FeatureFlags,
    /// Image containers accepted on ingest and producible on output, with
    /// the size/quality limits the server actually enforces.
    pub formats: Formats,
    /// Auth mode in effect (`"none"` or `"bearer"`); the key VALUE is never
    /// exposed.
    pub auth: AuthCapabilities,
    /// Resource-ledger admission knobs a client needs to plan retries:
    /// per-kind queue caps, the queue wait bound, and the `Retry-After`
    /// hint the server's 429s carry.
    pub queue: QueueCapabilities,
    /// CPU image-ops surface (`POST /v1/image-ops`): the supported op
    /// vocabulary. Present whenever the ops route is compiled (always
    /// today) — additive optional field per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_ops: Option<ImageOpsCapability>,
    /// Chat tool-calling fidelity of the `OpenAI` chat surface (REQ-3):
    /// which `tool_choice` values are honored vs clamped, and whether
    /// `parallel_tool_calls` is honored. Present whenever the chat routes
    /// are mounted (always today) — additive optional field per the
    /// version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCallingCapability>,
    /// Recognition surface (`POST /v1/recognize`): probe-driven flags per
    /// kind. Present whenever the recognize route is mounted (always
    /// today) — additive optional field per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recognition: Option<RecognitionFeature>,
    /// Externally discovered engines (`GET /v1/external-tools`): versions,
    /// probed capabilities and honest availability reasons. Additive optional
    /// field per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_tools: Option<ExternalToolsCapability>,
    /// Local storage surface: whether a data dir is configured, so a client
    /// can probe artifact delivery before using it. Additive optional field
    /// per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageCapability>,
    /// Texture post-processing surface (`oiiotool`/`maketx`): whether the
    /// server can run texture conversion passes, and which tool versions
    /// are present. Additive optional field per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub texture_postprocess: Option<TexturePostprocessCapability>,
    /// NSFW classification surface: the policy in effect, the model
    /// version, and the score threshold. Additive optional field per the
    /// version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nsfw: Option<NsfwCapability>,
    /// External mesh surface: what is enabled, which upstreams and models
    /// are known, and how fresh the observation is. Additive optional field
    /// per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh: Option<MeshExternalCapability>,
    /// Signed-URL delivery surface: whether signed URLs are enabled and
    /// their time-to-live. Additive optional field per the version policy
    /// above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signed_urls: Option<SignedUrlCapability>,
    /// Scope surface: whether scoped credentials are enabled and which
    /// scopes exist. Additive optional field per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes: Option<ScopeCapability>,
    /// Machine-readable model/asset license records from the live curated
    /// catalog. Additive optional field per the version policy above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub licenses: Option<LicenseCapability>,
    /// Where the generated `OpenAPI` document lives.
    pub openapi: String,
    /// Where the model catalog lives.
    pub models: String,
}

/// Machine-readable model/asset license surface (`Capabilities::licenses`):
/// one summary row per curated license record. Additive optional field per
/// the API version policy above.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct LicenseCapability {
    /// One summary row per curated license record.
    pub rows: Vec<LicenseSummaryRow>,
}

/// One curated license record published on `GET /v1/capabilities`. Nullable
/// source facts stay absent on the wire, never fabricated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct LicenseSummaryRow {
    /// Curated license id (SPDX-style where one exists).
    pub license: String,
    /// License terms URL, when curated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license_url: Option<String>,
    /// Territory restriction tag, when curated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub territory: Option<String>,
    /// `commercial` / `non-commercial` / `all-rights-reserved`.
    pub commercial_terms: String,
    /// Whether the upstream weights are gated behind accepted terms/token.
    pub gated: bool,
    /// Upstream source URL, when curated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// Published weight digest, when known (never fabricated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Required attribution text, for CC-BY-family sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribution_text: Option<String>,
}

/// Local storage surface (`Capabilities::storage`): whether the server has
/// a data dir, i.e. can persist artifacts for `X-Honkbox-Delivery: artifact`.
/// A `false` here means artifact mode is unavailable, not merely degraded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct StorageCapability {
    /// A data dir is configured and writable.
    pub data_dir_present: bool,
}

/// Compile-time audio.cpp surface (`Capabilities::features.audio_cpp`).
/// Family lists come from the audio-worker crate's own features, so the
/// document cannot advertise a family this binary cannot load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct AudioCppCapabilities {
    /// The audio.cpp engine is compiled into this binary.
    pub built: bool,
    /// Speech-to-text families compiled in (empty when `built` is false).
    pub stt_families: Vec<String>,
    /// Text-to-speech families compiled in (empty: no audio.cpp TTS exists).
    pub tts_families: Vec<String>,
}

/// Server identity (`Capabilities::server`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ServerInfo {
    /// Binary name, always `"honkbox"` (mirrors `GET /api/version`).
    pub name: String,
    /// Crate version, from `CARGO_PKG_VERSION` (release version, NOT the
    /// wire-contract version).
    pub version: String,
}

/// Feature flags (`Capabilities::features`). Every `true` here is backed by
/// a working route (contract-tested); `video` stays `false` until a real
/// video job class lands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
#[allow(clippy::struct_excessive_bools)] // the wire contract IS a flag map (llama-swap shape)
pub struct FeatureFlags {
    /// Vision chat (mtmd/mmproj image parts).
    pub vision: VisionFeature,
    /// ControlNet-conditioned renders on `POST /v1/images/generations`.
    pub controlnet: ControlNetFeature,
    /// Per-request `LoRA` stacks on image generations AND edits.
    pub lora: bool,
    /// Video generation. Always `false` today — there is no video job
    /// class; a client must not probe for video routes.
    pub video: bool,
    /// System-one decision endpoint. `false` until that endpoint lands.
    pub decisions: bool,
    /// Compile-time audio.cpp surface. Default builds report `built: false`
    /// and empty family lists.
    pub audio_cpp: AudioCppCapabilities,
    /// Job registry (`GET /v1/jobs`, cancel/delete routes).
    pub jobs: bool,
    /// Server-sent event stream (`GET /v1/events`).
    pub events: bool,
    /// Silero VAD in the audio worker (compile feature `vad`).
    pub vad: bool,
    /// Experimental ash SD backend (compile feature
    /// `experimental-ash-sd`); empty until that backend's first code lands.
    pub experimental_ash_sd: bool,
    /// Typed Blender asset jobs (`POST /v1/jobs/blender`). `true` only when
    /// a Blender engine is installed and probed: the route answers a named
    /// 501 otherwise.
    pub blender: bool,
    /// Artifact delivery (artifact store routes). `true` exactly when the
    /// artifact store (data dir) is present; without a resolved data dir
    /// there is nowhere to store or serve generated bytes, so the flag never
    /// promises delivery the server cannot perform.
    pub artifact_delivery: bool,
    /// Rerank surface (`POST /v1/rerank`): whether a rerank route is
    /// mounted and backed by a model.
    pub rerank: bool,
    /// Python sidecar surface: whether a Python worker sidecar is
    /// available to this build.
    pub python_sidecar: bool,
    /// Fizgig `LoRA` training (`POST /v1/fizgig/train`). `true` only when the
    /// backend is enabled AND its pinned CLI passed the startup probe;
    /// the route answers a named 501 otherwise.
    pub fizgig: bool,
    /// Texture post-processing: whether `oiiotool`/`maketx` passes can
    /// run in this build.
    pub texture_postprocess: bool,
    /// Residency control (`GET /running`, `POST /api/models/unload`): `true`
    /// exactly when a swap manager exists, so the warm-set readout and
    /// explicit unload have something to act on. The control routes never
    /// take a GPU slot themselves.
    pub residency_control: bool,
}

/// Vision chat capability (`FeatureFlags::vision`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct VisionFeature {
    /// `true` when THIS build compiles the mtmd path (`llm-vision`). A
    /// text-only build reports `false`; scan-derived entries then never
    /// advertise vision, and image parts on any entry whose
    /// `capabilities.vision` is `false` get the named `vision_unsupported`
    /// 400 — the gate and `/v1/models` key on the same catalog flag, so
    /// `models` counts exactly the entries the routes accept as
    /// vision-capable. Operator-declared vision overrides still win over
    /// the build probe (a real pipeline fails loudly at model load).
    pub built: bool,
    /// Catalog entries currently advertising `capabilities.vision` (what
    /// `/v1/models` serves and the web UI's model picker consumes).
    pub models: u64,
}

/// `ControlNet` capability (`FeatureFlags::controlnet`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ControlNetFeature {
    /// `true` only when the configured SD pipeline applies control images
    /// (the sd-cpp backend) AND the catalog holds at least one
    /// `controlnet`-kind entry; otherwise valid pairs get the named 501
    /// (`controlnet_unsupported`) and this flag is `false`.
    pub available: bool,
    /// `controlnet`-kind entries in the live catalog (conditioning inputs
    /// resolve by slug/alias against exactly these).
    pub models: u64,
}

/// Image container support (`Capabilities::formats`), derived from the
/// compiled encoders/validators — a missing decoder never appears.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct Formats {
    /// Containers the server validates on ingest (image edits' `image`/
    /// `mask` and `ControlNet` `control_image` are PNG magic-sniffed).
    pub read: Vec<String>,
    /// Containers producible on output (`output_format` on both image
    /// routes); one entry per compiled encoder.
    pub write: Vec<OutputFormatCapability>,
    /// Size/quality limits the server enforces.
    pub limits: FormatLimits,
}

/// One writable container (`Formats::write`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct OutputFormatCapability {
    /// `output_format` wire value (`png` | `jpeg` | `webp`).
    pub format: String,
    /// `true` when `output_compression` affects this container (JPEG
    /// quality only; PNG is deterministic and WebP encodes lossless-only).
    pub compression: bool,
}

/// Size/quality limits (`Formats::limits`), read from the validation code
/// that enforces them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct FormatLimits {
    /// `output_compression` accepted range (JPEG quality, rejected not
    /// clamped outside it).
    pub output_compression: CompressionRange,
    /// Request-body ceiling applied by the router (largest upload the
    /// server accepts, multipart framing included).
    pub body_bytes: u64,
}

/// Inclusive integer range (`FormatLimits::output_compression`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct CompressionRange {
    /// Smallest accepted value (inclusive).
    pub min: u32,
    /// Largest accepted value (inclusive).
    pub max: u32,
    /// The container the knob applies to (`"jpeg"`).
    pub applies_to: String,
}

/// Auth mode (`Capabilities::auth`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct AuthCapabilities {
    /// `"none"` (loopback-only operation) or `"bearer"` (key required on
    /// every route except the probes). The key VALUE is never in this
    /// document.
    pub mode: String,
}

/// Resource-ledger admission knobs (`Capabilities::queue`), read live from
/// the ledger + admission config — the same values the 429 contract uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct QueueCapabilities {
    /// Shared GPU slots the ledger admits against.
    pub gpu_slots: u32,
    /// Admission-queue wait bound in seconds (`[server] queue_wait_secs`);
    /// `0` = queueing disabled (busy requests 429 immediately). Doubles as
    /// the queued-waiter TTL.
    pub queue_wait_secs: u64,
    /// The `Retry-After` hint (seconds) the server's 429 responses carry.
    pub retry_after_secs: u64,
    /// One row per admission kind (fixed vocabulary), with its bounded
    /// queue length and TTL — a client can size its own retry cadence.
    pub kinds: Vec<QueueKindCapacity>,
}

/// Per-kind admission queue row (`QueueCapabilities::kinds`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct QueueKindCapacity {
    /// Fixed kind label (`sd`, `llm`, `ocr`, `embed`, `rag`, `comfy`,
    /// `blender`, `3d`, `training`) — never user data.
    pub kind: String,
    /// Bounded queue length for the kind.
    pub queue_cap: u64,
    /// Queue TTL in seconds (a waiter un-admitted after this long is
    /// evicted); `0` = never expires.
    pub ttl_secs: u64,
    /// The kind is RESERVED: the ledger carries capacity for it, but no
    /// route admits a job of that kind — the row is capacity only, never a
    /// callable job kind. Additive field: `api_version` is unchanged (a
    /// reserved row hides a job class, it does not change one).
    pub reserved: bool,
}

/// Chat role.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// Human / caller turn.
    User,
    /// Model turn.
    Assistant,
    /// System instructions.
    System,
    /// Tool result turn (`REQ-3`); pairs with `tool_call_id`.
    Tool,
}

/// One tool call as it appears on an assistant message (`REQ-3`).
/// `arguments` is a JSON-encoded **string**, exactly as `OpenAI` clients
/// exchange it.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ToolCall {
    /// Call id (unique per response on replies; echoed by `role: "tool"`
    /// messages as `tool_call_id`).
    #[serde(default)]
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub call_type: ToolCallType,
    /// The invoked function.
    pub function: FunctionCall,
}

/// Flavor of a tool call. Only `OpenAI` function calls are supported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallType {
    /// A named function call.
    #[default]
    Function,
}

/// Function invocation payload of a [`ToolCall`].
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct FunctionCall {
    /// Tool name.
    pub name: String,
    /// The arguments, as a JSON-encoded string ("{\"city\":\"Reykjavik\"}").
    #[serde(default)]
    pub arguments: String,
}

/// One `OpenAI` content part (REQ-8). Parsed from the widened
/// `ChatMessage.content` array form; text parts flatten into the message
/// text, `image_url` parts are flagged for the explicit vision refusal.
/// Requests naming any other `type` are a deserialization 400 naming the
/// offending type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text.
    Text {
        /// The text itself.
        text: String,
    },
    /// An image reference (data URI or http(s) URL). Vision-capable models
    /// (a sibling `mmproj*.gguf` projector or an operator
    /// `capabilities.vision` override) accept the part; models without a
    /// vision backend reject the request with an explicit 400
    /// (`vision not supported by backend`), never a silent drop.
    ImageUrl {
        /// The image reference.
        image_url: ImageUrl,
    },
}

/// `image_url` part payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct ImageUrl {
    /// Data URI or http(s) URL.
    pub url: String,
}

/// Widened `ChatMessage.content` (REQ-8). Deserializes from a plain
/// string, `null`, a missing field, or the `OpenAI` content-parts array:
/// `text` parts concatenate into `text` (order preserved, no separator —
/// the `OpenAI` flattening), `image_url` parts set `has_image` and are
/// retained (with their text offset) for the vision path, any other part
/// type is a deserialization error naming the type. Serializes **always**
/// as the plain string or `null` — the flattened text — so the response
/// wire shape (`string | null`), the TS bindings and every existing
/// consumer stay byte-for-byte identical, and image payloads never echo
/// back. A vision-capable model (an mmproj projector discovered next to
/// the weights, or an operator override) accepts the images; without one
/// the chat routes reject the request explicitly (400), never a silent
/// drop.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatContent {
    /// Flattened text of the message (`None` = absent/`null`).
    pub text: Option<String>,
    /// At least one `image_url` part was present in the request.
    pub has_image: bool,
    /// `image_url` parts in request order, with their flattened-text
    /// offsets; empty when the message carried none. Request-side only —
    /// never serialized (the wire form stays the flattened text, so image
    /// payloads never echo back).
    pub images: Vec<ChatImageRef>,
}

/// One `image_url` part as received (vision chat): the raw `url` string
/// plus the byte offset it occupied in the flattened message text — the
/// splice point for the media marker the worker tokenizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatImageRef {
    /// `image_url.url` verbatim (a `data:` URI, or whatever else the
    /// request carried).
    pub url: String,
    /// Byte offset into the flattened `text` where this part sat.
    pub text_offset: usize,
}

impl ChatContent {
    /// Plain-text content (reply shape; no image flag).
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            has_image: false,
            images: Vec::new(),
        }
    }

    /// Flattened text with `marker` spliced in at each `image_url` part's
    /// offset (vision chat): the string the worker tokenizes so media
    /// embeddings land where the client put the parts. A `None` text with
    /// images renders as one marker per image (image-only message).
    #[must_use]
    pub fn text_with_media_markers(&self, marker: &str) -> String {
        let mut out = self.text.clone().unwrap_or_default();
        // Offsets index the pre-splice text; inserting from the end keeps
        // every not-yet-inserted offset valid.
        for image in self.images.iter().rev() {
            let at = image.text_offset.min(out.len());
            out.insert_str(at, marker);
        }
        out
    }

    /// Parse the wire form: string | null | array of parts.
    fn parse(value: &serde_json::Value) -> Result<Self, String> {
        match value {
            serde_json::Value::Null => Ok(Self::default()),
            serde_json::Value::String(text) => Ok(Self::text(text.clone())),
            serde_json::Value::Array(parts) => {
                let mut flat = String::new();
                let mut images = Vec::new();
                for part in parts {
                    let Some(object) = part.as_object() else {
                        return Err(format!(
                            "content part must be an object with a `type` field, got {part}"
                        ));
                    };
                    match object.get("type").and_then(serde_json::Value::as_str) {
                        Some("text") => {
                            let text = object
                                .get("text")
                                .and_then(serde_json::Value::as_str)
                                .ok_or("text part requires a string `text` field")?;
                            flat.push_str(text);
                        }
                        Some("image_url") => {
                            let url = object
                                .get("image_url")
                                .and_then(|image| image.get("url"))
                                .and_then(serde_json::Value::as_str)
                                .ok_or("image_url part requires a string `image_url.url` field")?;
                            images.push(ChatImageRef {
                                url: url.to_owned(),
                                text_offset: flat.len(),
                            });
                        }
                        Some(other) => {
                            return Err(format!(
                                "unsupported content part type {other:?}: supported types are `text` and `image_url`"
                            ));
                        }
                        None => {
                            return Err("content part requires a string `type` field".to_owned());
                        }
                    }
                }
                let text = (!flat.is_empty()).then_some(flat);
                let has_image = !images.is_empty();
                Ok(Self {
                    text,
                    has_image,
                    images,
                })
            }
            other => Err(format!(
                "content must be a string, null, or an array of content parts, got {other}"
            )),
        }
    }
}

impl Serialize for ChatContent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.text {
            Some(text) => serializer.serialize_str(text),
            None => serializer.serialize_none(),
        }
    }
}

impl<'de> Deserialize<'de> for ChatContent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

/// One chat message.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatMessage {
    /// Who said it.
    pub role: ChatRole,
    /// Message content (REQ-8): accepts the plain string, `null` (the
    /// assistant-with-`tool_calls` shape), or the `OpenAI` content-parts
    /// array; always serializes as `string | null`. Carrying an
    /// `image_url` part is an explicit 400 upstream, never a silent drop.
    #[serde(default)]
    #[ts(type = "string | null")]
    #[schema(value_type = String, nullable)]
    pub content: ChatContent,
    /// Assistant turn: tool calls the model already made (`REQ-3`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// `role: "tool"` turn: id of the assistant tool call this message
    /// answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[cfg(test)]
mod chat_content_tests {
    use super::*;

    #[test]
    fn string_null_and_missing_round_trip_as_plain_text() {
        let message: ChatMessage =
            serde_json::from_str(r#"{"role":"user","content":"hi"}"#).expect("string form");
        assert_eq!(message.content.text.as_deref(), Some("hi"));
        assert!(!message.content.has_image);
        // Serialized form is the plain string again.
        assert_eq!(serde_json::to_string(&message.content).unwrap(), r#""hi""#);

        let message: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":null,"tool_calls":[]}"#)
                .expect("null form");
        assert_eq!(message.content.text, None);
        assert_eq!(serde_json::to_string(&message.content).unwrap(), "null");

        // Missing content deserializes as null (assistant tool-call shape).
        let message: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant"}"#).expect("missing content");
        assert_eq!(message.content.text, None);
    }

    #[test]
    fn parts_array_flattens_text_and_flags_images() {
        let message: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[
                {"type":"text","text":"What is in "},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,x"}},
                {"type":"text","text":"this image?"}
            ]}"#,
        )
        .expect("parts form");
        assert_eq!(
            message.content.text.as_deref(),
            Some("What is in this image?")
        );
        assert!(message.content.has_image);
        // Serialization stays the flattened string: the response wire shape
        // is unchanged, and the image payloads never echo back.
        assert_eq!(
            serde_json::to_string(&message.content).unwrap(),
            "\"What is in this image?\""
        );
        // The image ref records the offset its part occupied in the
        // flattened text (vision chat splice point).
        assert_eq!(message.content.images.len(), 1);
        assert_eq!(message.content.images[0].url, "data:image/png;base64,x");
        assert_eq!(message.content.images[0].text_offset, "What is in ".len());
    }

    #[test]
    fn media_markers_splice_at_part_offsets() {
        let content = ChatContent {
            text: Some("What is in this image?".to_owned()),
            has_image: true,
            images: vec![ChatImageRef {
                url: "data:image/png;base64,x".to_owned(),
                text_offset: "What is in ".len(),
            }],
        };
        assert_eq!(
            content.text_with_media_markers("<__media__>"),
            "What is in <__media__>this image?"
        );
        // Image-only message (no text): one marker per image.
        let images_only = ChatContent {
            text: None,
            has_image: true,
            images: vec![
                ChatImageRef {
                    url: "a".to_owned(),
                    text_offset: 0,
                },
                ChatImageRef {
                    url: "b".to_owned(),
                    text_offset: 0,
                },
            ],
        };
        assert_eq!(images_only.text_with_media_markers("M"), "MM");
        // Multiple images keep each marker at its own part's offset
        // (parts ["A", img, " ", img, "B"] → offsets 1 and 2).
        let two = ChatContent {
            text: Some("A B".to_owned()),
            has_image: true,
            images: vec![
                ChatImageRef {
                    url: "a".to_owned(),
                    text_offset: 1,
                },
                ChatImageRef {
                    url: "b".to_owned(),
                    text_offset: 2,
                },
            ],
        };
        assert_eq!(two.text_with_media_markers("-"), "A- -B");
        // Text shape without images is untouched.
        assert_eq!(
            ChatContent::text("plain").text_with_media_markers("M"),
            "plain"
        );
    }

    #[test]
    fn missing_image_url_field_is_a_named_error() {
        let error = serde_json::from_str::<ChatMessage>(
            r#"{"role":"user","content":[{"type":"image_url"}]}"#,
        )
        .expect_err("image_url without url");
        assert!(
            error
                .to_string()
                .contains("image_url part requires a string `image_url.url` field"),
            "{error}"
        );
    }

    #[test]
    fn unknown_part_type_names_the_type() {
        let error = serde_json::from_str::<ChatMessage>(
            r#"{"role":"user","content":[{"type":"input_audio"}]}"#,
        )
        .expect_err("unknown type");
        assert!(error.to_string().contains("input_audio"), "{error}");
        assert!(error.to_string().contains("supported types"), "{error}");
    }

    #[test]
    fn malformed_parts_are_rejected_with_named_causes() {
        // Non-object part.
        let error = serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":["plain"]}"#)
            .expect_err("non-object part");
        assert!(error.to_string().contains("`type`"), "{error}");
        // Missing type.
        let error =
            serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":[{"text":"x"}]}"#)
                .expect_err("missing type");
        assert!(error.to_string().contains("string `type` field"), "{error}");
        // Text part without text.
        let error =
            serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":[{"type":"text"}]}"#)
                .expect_err("text part without text");
        assert!(error.to_string().contains("string `text` field"), "{error}");
        // Scalar content.
        let error = serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":42}"#)
            .expect_err("scalar content");
        assert!(
            error.to_string().contains("string, null, or an array"),
            "{error}"
        );
    }
}

/// One `OpenAI` tool definition (`REQ-3`).
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatTool {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    /// The callable function.
    pub function: FunctionDefinition,
}

/// Flavor of a tool definition. Only `OpenAI` function tools are supported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    /// A named function tool.
    #[default]
    Function,
}

/// Callable function description (`tools[].function`).
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct FunctionDefinition {
    /// Tool name.
    pub name: String,
    /// What the function does (model-facing hint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema of the arguments object. Arbitrary JSON: it is forwarded
    /// to the schema→grammar converter verbatim.
    #[serde(default)]
    #[schema(value_type = Object)]
    pub parameters: serde_json::Value,
}

/// `OpenAI` `tool_choice` (`REQ-3`): a mode string or a named function.
/// Everything except `"none"` is clamped to `auto` behavior (tools are
/// offered; the model decides) — no rejection. The clamp is disclosed, not
/// silent: [`Capabilities::tools`] publishes the exact honored/clamped
/// vocabulary machine-readably before the client sends.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
#[serde(untagged)]
pub enum ToolChoiceSpec {
    /// `"auto"`, `"none"`, `"required"`, or `"any"`.
    Mode(String),
    /// `{"type":"function","function":{"name":…}}`.
    Named {
        /// The named function.
        function: NamedFunction,
    },
}

/// Named function payload of [`ToolChoiceSpec::Named`].
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct NamedFunction {
    /// Tool name to steer the model towards.
    pub name: String,
}

impl ToolChoiceSpec {
    /// `tool_choice` string modes the chat routes honor verbatim: `"auto"`
    /// (tools offered, the model decides — the default behavior) and
    /// `"none"` (tools rendered for context but never called). Vocabulary
    /// note: the vendored llama.cpp binding (`llama-cpp-4` 0.11.0,
    /// `ToolChoice`) additionally exposes a grammar-enforced `required`;
    /// honkbox keeps `"required"` clamped to `"auto"` pending an exact-
    /// choice consumer (TASK-llm-tool-choice-contract optional follow-up).
    pub const HONORED_MODES: [&str; 2] = ["auto", "none"];
    /// `tool_choice` string modes accepted but clamped to `"auto"`
    /// behavior — never a 400; disclosed in [`Capabilities::tools`].
    pub const CLAMPED_MODES: [&str; 2] = ["required", "any"];

    /// True when tool calls must not be offered (`"none"`). Any other
    /// value — `auto`, `required`, `any`, named — samples with tools
    /// available (clamped to `auto`).
    #[must_use]
    pub fn suppresses_tools(&self) -> bool {
        matches!(self, Self::Mode(mode) if mode == "none")
    }
}

/// Chat tool-calling fidelity (`Capabilities::tools`; REQ-3): the exact
/// `tool_choice` / `parallel_tool_calls` contract of the `OpenAI` chat
/// routes, published machine-readably so no request is silently downgraded
/// — a client can read the clamp and the ignored flag before it sends.
/// `tool_choice_honored` is exhaustive (`"none"` is the only mode that
/// changes generation); `tool_choice_clamped` lists the standard `OpenAI`
/// modes and is NOT exhaustive: this is an untagged string field, so any
/// other unrecognized value is accepted and behaves as `"auto"` too.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ToolCallingCapability {
    /// `tool_choice` string modes honored verbatim
    /// ([`ToolChoiceSpec::HONORED_MODES`]).
    pub tool_choice_honored: Vec<String>,
    /// `tool_choice` string modes accepted but clamped to `"auto"`
    /// behavior ([`ToolChoiceSpec::CLAMPED_MODES`]); never a 400. Not
    /// exhaustive: any unrecognized value is accepted and behaves as
    /// `"auto"` too. The named-function form clamps the same way.
    pub tool_choice_clamped: Vec<String>,
    /// The named-function form (`{"type":"function","function":{"name":…}}`)
    /// is accepted too — and likewise clamped to `"auto"` behavior.
    pub tool_choice_named_function_clamped: bool,
    /// `true` when `parallel_tool_calls` is honored. `false` today: at most
    /// one tool call is produced per turn; the field is still accepted so
    /// parallel-`true` clients are not rejected.
    pub parallel_tool_calls: bool,
}

impl ToolCallingCapability {
    /// The fidelity contract of the current chat routes. Built from the
    /// [`ToolChoiceSpec`] vocabulary so the capability document cannot
    /// drift from the wire behavior it describes.
    #[must_use]
    pub fn chat_routes() -> Self {
        Self {
            tool_choice_honored: ToolChoiceSpec::HONORED_MODES
                .iter()
                .map(ToString::to_string)
                .collect(),
            tool_choice_clamped: ToolChoiceSpec::CLAMPED_MODES
                .iter()
                .map(ToString::to_string)
                .collect(),
            tool_choice_named_function_clamped: true,
            parallel_tool_calls: false,
        }
    }
}

/// `OpenAI` `response_format` (`REQ-3`). `json_object` and `json_schema` are
/// **enforced**: the schema (or a generic any-JSON grammar for
/// `json_object`) is converted to GBNF and wired into the sampler chain, so
/// the model cannot emit a non-conforming string.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ResponseFormat {
    /// Format discriminator.
    #[serde(rename = "type")]
    pub format_type: ResponseFormatType,
    /// Required when `format_type` is `json_schema`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaSpec>,
}

/// Supported `response_format` types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    /// Plain text (unconstrained).
    Text,
    /// Any valid JSON object, enforced by grammar.
    JsonObject,
    /// One specific JSON schema, enforced by grammar.
    JsonSchema,
}

/// Schema payload of `response_format: {"type":"json_schema"}`.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct JsonSchemaSpec {
    /// Schema name (`OpenAI` requires one; only echoed, never enforced).
    #[serde(default)]
    pub name: String,
    /// Optional schema description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema itself; converted to a GBNF grammar on the llama.cpp
    /// path.
    #[serde(default)]
    #[schema(value_type = Object)]
    pub schema: serde_json::Value,
    /// `OpenAI` strict-mode flag; accepted (grammar enforcement is always
    /// strict here).
    #[serde(default)]
    pub strict: bool,
}

/// Chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatRequest {
    /// Model id (informational; the backend is chosen by the server).
    pub model: String,
    /// Conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Per-request residency budget for this model's swap entry, in seconds
    /// (NEW-8; the Ollama `keep_alive` / LM Studio `ttl` analog). The entry's
    /// effective window is `max(remaining, this)`, capped at
    /// `[swap] max_ttl_secs`; `0` = use-then-unload, still shielded by the
    /// remaining window so it cannot thrash. Values over the cap are clamped
    /// at the route (`warn!`, never a 400); `[swap] pin` entries ignore it
    /// entirely. Ignored when the swap manager is off (stub backends,
    /// `[swap] enabled = false`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ChatResponse {
    /// Which backend produced this.
    pub backend_id: String,
    /// Generated text.
    pub text: String,
    /// Number of generated tokens.
    pub tokens: u32,
    /// Wall-clock time in milliseconds.
    pub millis: u64,
}

/// OpenAI-compatible chat completion request (`POST /v1/chat/completions`;
/// REQ-1). A translation surface over honkbox's native chat pipeline — the
/// request is served by the same pipeline, GPU lease, and swap decision as
/// `POST /v1/chat`.
///
/// Undeclared keys (`top_p`, `stop`, `reasoning_budget`, `cache_prompt`, …)
/// are accepted and silently ignored (serde's default unknown-key behavior);
/// only the fields below have honkbox-side meaning.
///
/// REQ-3 additions and their fidelity:
/// - **Enforced** (never silently ignored): `tools` + `tool_calls`,
///   `response_format` (`json_object`/`json_schema` → GBNF grammar), raw
///   `grammar` (llama.cpp GBNF, honored verbatim), `tool_choice: "none"`.
///   `tools` combined with `grammar`/`response_format` is a 400 (mutually
///   exclusive constraint), matching llama.cpp's own hard error.
/// - **Clamped, disclosed**: `tool_choice` `"required"`/`"any"`/named
///   function all behave as `"auto"` (tools offered, model decides) —
///   accepted, never a 400; the clamp is published machine-readably on
///   `GET /v1/capabilities` (`tools`), so the contract is visible before
///   the client sends.
/// - **Accepted and ignored, disclosed**: `parallel_tool_calls` — one tool
///   call is produced per turn regardless; the field exists so clients
///   that always send it are not rejected. `capabilities.tools.
///   parallel_tool_calls = false` says so up front.
/// - **Extended sampler params, wired into the sampler chain**: `top_k`,
///   `min_p`, `typical_p`, `repeat_penalty`, `xtc_probability`,
///   `dry_multiplier`/`dry_base`/`dry_allowed_length`,
///   `dynatemp_range`/`dynatemp_exponent`. All are supported by the
///   llama-cpp-4 binding; each is applied only when present.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct OpenAiChatRequest {
    /// Model id: an operator alias or registry slug (`GET /v1/models`).
    /// Required on this route — an omitted/empty model is a 400 naming the
    /// requirement (the native route's empty-means-server-default fallback
    /// stays native-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Conversation turns (string content; roles user/assistant/system).
    pub messages: Vec<ChatMessage>,
    /// Stream `OpenAI` SSE chunks (`chat.completion.chunk` frames + a final
    /// `data: [DONE]` line) instead of one JSON body. Default false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Max tokens to generate; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Deterministic sampling seed, honored end-to-end (sampler chain): the
    /// same prompt + seed streams the same tokens the non-streaming path
    /// returns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Per-request residency advice; identical contract to
    /// [`ChatRequest::ttl_seconds`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    /// `OpenAI` tool definitions (`REQ-3`). Non-empty arrays switch the
    /// worker onto the tool-capable chat template: the model's tool-call
    /// grammar constrains generation and tool calls are parsed from the
    /// output (`finish_reason: "tool_calls"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    /// `"auto"` / `"none"` / `"required"` / named. Only `"none"` changes
    /// behavior; everything else is clamped to `auto` (accepted, never a
    /// 400 — see `capabilities.tools` on `GET /v1/capabilities` for the
    /// machine-readable honored/clamped vocabulary).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceSpec>,
    /// Accepted and ignored (disclosed: `capabilities.tools.
    /// parallel_tool_calls` is `false`): honkbox produces at most one tool
    /// call per turn. Present so parallel-`true` clients are not rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Structured output (`REQ-3`): `json_object` → any-JSON grammar,
    /// `json_schema` → schema converted to GBNF. Both enforced on the
    /// llama.cpp path, never a silently unconstrained string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Raw GBNF grammar (llama.cpp `grammar` field), honored verbatim as
    /// the sampler chain's head. An unparseable grammar is a 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// Distinct tokens considered per step (llama.cpp `top_k`; `<= 0`
    /// disables). Wired when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Minimum probability for a token to be kept (llama.cpp `min_p`).
    /// Wired when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Locally-typical sampling mass (llama.cpp `typical_p`). Wired when
    /// present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    /// Repetition penalty (llama.cpp `repeat_penalty`; 1.0 = off). Wired
    /// when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// Probability of excising the top token when a close runner-up exists
    /// (llama.cpp `xtc_probability`; 0 = off). Wired when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xtc_probability: Option<f32>,
    /// DRY repetition penalty strength (0 = off). Wired when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f32>,
    /// DRY penalty exponent base (llama.cpp default 1.75). Wired when
    /// `dry_multiplier` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f32>,
    /// DRY sequence length below which no penalty applies (llama.cpp
    /// default 2). Wired when `dry_multiplier` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<u32>,
    /// Dynamic temperature range (0 = off). Wired when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynatemp_range: Option<f32>,
    /// Dynamic temperature exponent (llama.cpp default 1.0). Wired when
    /// `dynatemp_range` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynatemp_exponent: Option<f32>,
}

/// Why generation stopped (`choices[].finish_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of generation.
    Stop,
    /// `max_tokens` reached (heuristic: reported tokens >= requested cap —
    /// the worker does not surface a native stop cause).
    Length,
    /// The reply carries `message.tool_calls` (`REQ-3`).
    ToolCalls,
}

/// Token accounting (`usage`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiUsage {
    /// Prompt tokens after tokenization (real count on the llama.cpp path;
    /// the stub reports 0).
    pub prompt_tokens: u32,
    /// Generated tokens.
    pub completion_tokens: u32,
    /// `prompt_tokens + completion_tokens`.
    pub total_tokens: u64,
}

/// One non-streaming choice (`choices[]`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiChoice {
    /// Always 0 (single choice).
    pub index: u8,
    /// Assistant turn.
    pub message: ChatMessage,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// Non-streaming response (`object: "chat.completion"`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiChatCompletion {
    /// `chatcmpl-<ulid>`.
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: String,
    /// Unix seconds at completion.
    pub created: u64,
    /// The resolved model's served id (alias when bound, else the slug);
    /// the server-default path echoes the request value.
    pub model: String,
    /// Exactly one choice.
    pub choices: Vec<OpenAiChoice>,
    /// Token accounting.
    pub usage: OpenAiUsage,
}

/// One streamed delta (`choices[].delta` of a chunk). All fields optional:
/// the first frame carries `role`, token frames carry `content`, tool-call
/// frames carry `tool_calls` (`REQ-3`), the final frame carries none.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiDelta {
    /// Assistant role marker, first frame only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    /// Next text piece, token frames only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool-call fragments (`REQ-3`): the first frame of a call carries
    /// `id`/`type`/`name` + empty `arguments`, later frames append to
    /// `arguments` at the same `index`. Concatenating the `arguments`
    /// fragments per index yields the complete JSON string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// One streamed tool-call fragment (`delta.tool_calls[]`). Mirrors the
/// `OpenAI` fragment shape: the first fragment of a call carries
/// `id`/`type` and a `function` object with `name` + empty `arguments`;
/// continuation fragments carry only `index` and `function.arguments`.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct DeltaToolCall {
    /// Fragment position within the turn's tool calls.
    pub index: u32,
    /// Call id, first fragment only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Always `"function"`, first fragment only.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub call_type: Option<ToolCallType>,
    /// The invoked function: `name` (+ empty `arguments`) on the first
    /// fragment, next `arguments` piece on every continuation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<DeltaFunction>,
}

/// Function payload of [`DeltaToolCall`] (streamed incrementally).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct DeltaFunction {
    /// Function name, first fragment only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Next piece of the JSON-encoded arguments string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// One streamed chunk choice (`choices[]` of a chunk).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiChunkChoice {
    /// Always 0 (single choice).
    pub index: u8,
    /// Piece of the turn.
    pub delta: OpenAiDelta,
    /// `null` on every frame except the last.
    pub finish_reason: Option<FinishReason>,
}

/// Streaming chunk (`object: "chat.completion.chunk"`). `usage` rides only
/// on the final frame.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiChunk {
    /// `chatcmpl-<ulid>`, shared by every frame of one completion.
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    /// Unix seconds at stream start.
    pub created: u64,
    /// The resolved model's served id.
    pub model: String,
    /// Exactly one choice per frame.
    pub choices: Vec<OpenAiChunkChoice>,
    /// Token accounting; present only on the final frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiUsage>,
}

/// `OpenAI` error envelope detail (`error`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiErrorDetail {
    /// Human-readable cause.
    pub message: String,
    /// OpenAI-style class: `invalid_request_error` (400/404),
    /// `rate_limit_error` (429), `server_error` (5xx).
    #[serde(rename = "type")]
    pub error_type: String,
    /// Offending request field, when known.
    pub param: Option<String>,
    /// Machine-readable code (`model_not_found`, `gpu_slot_busy`, …).
    pub code: Option<String>,
}

/// OpenAI-shaped error body (`{"error":{…}}`) for
/// `/v1/chat/completions`.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct OpenAiError {
    /// The error detail.
    pub error: OpenAiErrorDetail,
}

impl OpenAiError {
    /// Build an envelope with the `OpenAI` status-class type string.
    #[must_use]
    pub fn new(status: u16, message: String, code: Option<String>) -> Self {
        let error_type = match status {
            400 | 404 => "invalid_request_error",
            429 => "rate_limit_error",
            _ => "server_error",
        };
        Self {
            error: OpenAiErrorDetail {
                message,
                error_type: error_type.to_owned(),
                param: None,
                code,
            },
        }
    }
}

/// `POST /v1/embeddings` request (REQ-4). `model` is required (`OpenAI`
/// contract): an omitted model is a deserialization 400.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct EmbeddingsRequest {
    /// Model id: an operator alias or registry slug whose
    /// `capabilities.embeddings` is true (`REQ-2` override honored).
    pub model: String,
    /// One input or a batch of inputs; batch order maps onto
    /// `data[].index` one-to-one.
    pub input: EmbeddingsInput,
}

/// The `input` field: a bare string or an array of strings.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    /// A single input (one `data` entry, `index` 0).
    Text(String),
    /// A batch (one `data` entry per input, in order).
    Batch(Vec<String>),
}

impl EmbeddingsInput {
    /// The inputs as an ordered vector (batch order = index order).
    #[must_use]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Text(text) => vec![text],
            Self::Batch(batch) => batch,
        }
    }
}

/// One embedding vector (`data[]`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct EmbeddingsData {
    /// Always `"embedding"`.
    pub object: String,
    /// Position of the source input in the request batch.
    pub index: u32,
    /// The embedding vector: the model's dimension, L2-normalized.
    pub embedding: Vec<f32>,
}

/// Token accounting for an embeddings call (no completion side).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct EmbeddingsUsage {
    /// Input token count (real tokenizer on the llama.cpp path;
    /// deterministic word count on the stub).
    pub prompt_tokens: u32,
    /// `prompt_tokens` (embeddings generate nothing).
    pub total_tokens: u64,
}

/// Embeddings response (`object: "list"`), one `data` entry per input.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct EmbeddingsResponse {
    /// Always `"list"`.
    pub object: String,
    /// One entry per input, in request order.
    pub data: Vec<EmbeddingsData>,
    /// The resolved model's served id (alias when bound, else the slug).
    pub model: String,
    /// Token accounting.
    pub usage: EmbeddingsUsage,
}

/// `POST /v1/images/generations` request (REQ-10). `OpenAI` shape: `prompt`
/// required; `model` optional (server default SD pipeline); `size` is
/// `"WxH"` pixels (default `1024x1024`, longest edge capped by sd-core);
/// `n` defaults to 1 and is clamped (never rejected) to 4;
/// `response_format` currently only accepts `"b64_json"`.
///
/// Recipe overrides (all optional; omitting every one is byte-identical to
/// the pre-passthrough fixed recipe: euler sampler, `normal`/discrete
/// scheduler, CFG 7.0, server-chosen seed base):
/// - `sampler`: one of the sd.cpp sampler names; an unknown name is a 400
///   naming it (accepted list comes from sd-core `SAMPLER_NAMES`).
/// - `scheduler`: one of the sd.cpp scheduler names; unknown name = 400.
/// - `cfg_scale`: classifier-free guidance scale in `[1.0, 30.0]`; outside
///   the range is a 400 (rejected, never silently clamped).
/// - `negative_prompt`: text steered away from in CFG models.
/// - `seed`: RNG base; per-image seed = `seed + i`. Omitted = the
///   server-chosen fixed base (deterministic, like today).
/// - `loras`: per-request `LoRA` adapters (max 2; each `slug` resolves
///   against catalog entries of kind `lora` by slug or alias — unknown
///   names are a 400 naming the offender). `strength` is the adapter
///   multiplier, default 1.0, accepted range `[-2.0, 2.0]` (out of range
///   is a 400, never clamped). Weight licensing is the operator's
///   responsibility: honkbox never bundles `LoRA` weights, and the curated
///   catalog `license` is surfaced as a response warning when known — a
///   missing license means all-rights-reserved, i.e. local-only use
///   (ReviewedLicense-AGPL §2b/§2c). The high-noise adapter split is not
///   exposed; requests always apply adapters as standard `LoRA`s. Base-model
///   mismatch cannot be detected today (no base tags are recorded on scan
///   entries or `LoRA` files), so it is neither warned about nor rejected.
///
/// **Named recipes and precedence**: `recipe` selects a curated bundle
/// (sampler/scheduler/cfg/steps for a checkpoint family; names come from
/// sd-core's supported enums and mirror the vendored stable-diffusion.cpp
/// presets — e.g. `sd15`, `sdxl`, `sd35`, `turbo`, `lcm`, `flux-dev`,
/// `flux-schnell`; an unknown name is a 400 listing every accepted value).
/// Explicit per-field overrides ALWAYS win over the recipe defaults, which
/// in turn win over the server defaults: request field > recipe > server
/// default. Omitting `recipe` and every override field reproduces the
/// pre-recipe fixed recipe exactly (euler, discrete, CFG 7.0, 20 steps).
/// The applied recipe name — and which fields a recipe actually supplied —
/// land verbatim in the job bundle sidecar.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ImagesGenerationRequest {
    /// Model id (alias or registry slug) from the SD catalog; omitted →
    /// the startup-configured SD pipeline (`--sd-model`, stub default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Text prompt (required).
    pub prompt: String,
    /// Image size as `"WxH"` pixels (e.g. `"1024x1024"`); default
    /// `1024x1024`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Number of images (1-4; above 4 is clamped down with a warn, never a
    /// 400 — the same clamp-not-reject policy as `ttl_seconds`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Output encoding; only `"b64_json"` is supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Output container for the returned image bytes: `"png"` (default),
    /// `"jpeg"`, or `"webp"` (lossless). Unknown names are rejected with a
    /// 400 listing the accepted values. The `response_format` transport
    /// encoding stays `"b64_json"` either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// JPEG quality in `[0, 100]` (the `OpenAI` `output_compression`
    /// precedent). Only meaningful with `output_format: "jpeg"` — PNG is
    /// lossless and WebP is encoded lossless-only today, so the value is
    /// documented-and-ignored there. Non-integer or out-of-range values are
    /// rejected with a 400 (`invalid_output_compression`), never clamped.
    /// Omitted = JPEG quality 90.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<serde_json::Value>,
    /// Sampler override (sd.cpp name, e.g. `"euler"`, `"dpm++2m"`);
    /// unknown names are rejected with a 400 listing the accepted values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    /// Scheduler override (sd.cpp name, e.g. `"discrete"`, `"karras"`);
    /// unknown names are rejected with a 400 listing the accepted values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<String>,
    /// Classifier-free guidance scale; accepted range `[1.0, 30.0]`, values
    /// outside it are rejected with a 400 (never silently clamped).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    /// Negative prompt (empty/omitted = none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// RNG seed base; per-image seed = `seed + i`. Omitted = the fixed
    /// server-chosen base (deterministic across identical requests).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Per-request `LoRA` adapters (see the struct-level docs for the
    /// licensing and validation rules); omit = plain checkpoint render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loras: Option<Vec<ImagesLora>>,
    /// Named generation recipe (curated sampler/scheduler/cfg/steps bundle
    /// per checkpoint family); unknown names are rejected with a 400 listing
    /// every accepted value. Explicit per-field overrides win over recipe
    /// defaults (see the struct-level precedence docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// Diffusion steps override; accepted range `[1, 100]`, outside it is a
    /// 400 (rejected, never clamped). Omitted = the recipe default, else the
    /// server default of 20.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Optional hires-fix second pass: generate at `size`, interpolate the
    /// latent up to the target, re-denoise (see [`ImagesHiresFix`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hires_fix: Option<ImagesHiresFix>,
    /// `ControlNet` slug or alias of a `controlnet`-kind catalog entry
    /// (honkbox extension, not part of the `OpenAI` image shape). Accepted
    /// and validated (unknown slug = 400) together with
    /// [`ImagesGenerationRequest::control_image`]; a fully valid pair is
    /// applied by the sd-cpp pipeline (weights load with the model
    /// context, the conditioning image steers the render). A pipeline that
    /// cannot apply control inputs (the stub backend) is answered with an
    /// explicit 501 (`controlnet_unsupported`), never silently ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controlnet: Option<String>,
    /// Conditioning image for [`ImagesGenerationRequest::controlnet`]:
    /// base64-encoded PNG. Validated (bad base64 / not a PNG = 400) and
    /// applied as-is by the sd-cpp pipeline — pass a preprocessed
    /// conditioning map (depth, edges, pose); the engine resizes it to the
    /// generation size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_image: Option<String>,
    /// `ControlNet` guidance strength in `(0, 1]` (how hard the conditioning
    /// image steers the render); values outside the range are rejected with
    /// a 400 (`invalid_controlnet_strength`), never clamped. Omitted = the
    /// vendor default 0.9 (stable-diffusion.cpp `--control-strength`).
    /// Only meaningful with `controlnet` + `control_image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controlnet_strength: Option<f32>,
    /// `ControlNet` conditioning type (honkbox extension): what the
    /// conditioning image depicts — one of the engine's names or aliases
    /// (`none`, `openpose`, `depth`, `scribble`, `canny`, `normal`,
    /// `segment`, `tile`, `repaint`, plus engine aliases like `lineart`,
    /// `hed`, `mlsd`). Validated against the engine's own name table
    /// (unknown names are a 400 `unknown_control_type`); a union
    /// `ControlNet` requires a named type while a classic one accepts the
    /// default `none` (omitted = `none`, the preprocessed-map behavior).
    /// Only meaningful with `controlnet` + `control_image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_type: Option<String>,
}

/// One per-request `LoRA` adapter on [`ImagesGenerationRequest::loras`].
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ImagesLora {
    /// Catalog slug or alias of a `lora`-kind entry (e.g.
    /// `comfy:loras/detail.safetensors`); unknown names are a 400.
    pub slug: String,
    /// Adapter multiplier; default 1.0, accepted range `[-2.0, 2.0]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
}

/// Optional hires-fix second pass on [`ImagesGenerationRequest::hires_fix`].
/// Vendor-native two-pass upscale (stable-diffusion.cpp `sd_hires_params_t`,
/// latent upscaler): the image is generated at the requested `size`, the
/// latent is interpolated up to the target, and a second pass re-denoises at
/// `denoise`. `target` wins over `scale`. The final edge stays under the
/// same 2048 px cap as plain generation; violations are a 400.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct ImagesHiresFix {
    /// Upscale factor over the requested `size`; accepted range
    /// `[1.0, 4.0]`, default 2.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    /// Explicit target size `"WxH"` (e.g. `"2048x1152"`); overrides
    /// `scale` when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Second-pass denoising strength `(0, 1]`; default 0.7.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denoise: Option<f32>,
}

/// One generated image in the `POST /v1/images/generations` response.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ImageB64 {
    /// The generated image bytes (in the requested `output_format`),
    /// base64-encoded (`response_format: "b64_json"`).
    pub b64_json: String,
}

/// One generated image delivered as a content-addressed artifact instead
/// of inline base64 (`X-Honkbox-Delivery: artifact`). Additive DTO: inline
/// responses never carry this field.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ImageArtifact {
    /// Artifact id (content sha256 hex), the `{id}` of
    /// `GET /v1/artifacts/{id}/bytes`.
    pub artifact_id: String,
    /// MIME type of the stored bytes (`image/png`, `image/jpeg`, ...).
    pub content_type: String,
    /// Length of the stored bytes.
    pub size_bytes: u64,
    /// Registry id (`X-Job-Id`) of the job that produced the image.
    pub job_id: String,
}

/// `POST /v1/images/generations` response: Unix `created` seconds plus one
/// `data` entry per requested image (`n`). In artifact delivery mode
/// (`X-Honkbox-Delivery: artifact`) `data` stays empty and `artifacts`
/// carries the images instead.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ImagesGenerationResponse {
    /// Creation time, Unix seconds.
    pub created: u64,
    /// One entry per requested image, in order (empty in artifact mode).
    pub data: Vec<ImageB64>,
    /// Advisory notes (`LoRA` license surfacing); omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
    /// Artifact-mode image descriptors; omitted for inline delivery.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Vec<ImageArtifact>>,
    /// Job bundle id written alongside the images, when one was produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    /// Registry id (`X-Job-Id`) of the generating job, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

/// The bounded `POST /v1/image-ops` op vocabulary (epic-editor-integration).
/// Wire names are the serde names; [`ImageOpKind::ALL`] is the authoritative
/// order for the `GET /v1/capabilities` advertisement. Anything outside this
/// set is a 400 naming the accepted values — the vocabulary is closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageOpKind {
    /// Resize to an explicit `WxH` (Lanczos3, see [`ImageOpResize`]).
    Resize,
    /// Pixel-exact crop rect (see [`ImageOpCrop`]).
    Crop,
    /// Lossless 90/180/270 rotation (see [`ImageOpRotate`]).
    Rotate,
    /// Convert every pixel to grayscale (no params).
    Grayscale,
    /// Container conversion only — pixels untouched, `output_format`
    /// decides the encoding (no params).
    Format,
}

impl ImageOpKind {
    /// Every supported op, in advertisement order (`GET /v1/capabilities`
    /// `image_ops.ops` lists exactly these).
    pub const ALL: [ImageOpKind; 5] = [
        ImageOpKind::Resize,
        ImageOpKind::Crop,
        ImageOpKind::Rotate,
        ImageOpKind::Grayscale,
        ImageOpKind::Format,
    ];

    /// Wire label (the serde name; also the bundle sidecar `op` value).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ImageOpKind::Resize => "resize",
            ImageOpKind::Crop => "crop",
            ImageOpKind::Rotate => "rotate",
            ImageOpKind::Grayscale => "grayscale",
            ImageOpKind::Format => "format",
        }
    }
}

/// `resize` params. Both edges are required (no implicit aspect preservation
/// guess); accepted range `[1, 8192]` per edge with `width * height <=`
/// `16_777_216` px — outside it is a named 400, never a clamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageOpResize {
    /// Target width in pixels, `[1, 8192]`.
    pub width: u32,
    /// Target height in pixels, `[1, 8192]`.
    pub height: u32,
}

/// `crop` params. All four values are required; the rect must satisfy
/// `x + width <= input_width` and `y + height <= input_height` (out-of-bounds
/// crops are a named 400, never clamped to the image edge).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageOpCrop {
    /// Left edge of the crop rect, px from x=0.
    pub x: u32,
    /// Top edge of the crop rect, px from y=0.
    pub y: u32,
    /// Crop width in px (at least 1).
    pub width: u32,
    /// Crop height in px (at least 1).
    pub height: u32,
}

/// `rotate` params. Only lossless multiples of a quarter turn are accepted
/// (`90`, `180`, `270`); arbitrary angles would need resampling — rejected
/// with a 400 naming the accepted values, never approximated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageOpRotate {
    /// Rotation in degrees; accepted values are `90`, `180`, `270`
    /// (clockwise).
    pub degrees: u16,
}

/// One image-op command: the op tag plus exactly that op's params.
/// Param ops are externally tagged single-key objects —
/// `{"resize": {"width": 512, "height": 512}}` — while the no-param ops
/// (`grayscale`, `format`) are bare strings (`"grayscale"`), their one and
/// only wire form. The params structs carry `deny_unknown_fields`, so
/// unknown keys inside a params object are a 400; an unknown op tag or the
/// wrong shape (e.g. `{"format": {}}`) is a 400 from parsing itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageOpCommand {
    /// Resize to `width`x`height` (Lanczos3 resample — the single fixed
    /// filter keeps bytes deterministic for identical inputs).
    Resize(ImageOpResize),
    /// Crop the rect `x, y, width, height`.
    Crop(ImageOpCrop),
    /// Lossless quarter-turn rotation.
    Rotate(ImageOpRotate),
    /// Grayscale conversion.
    Grayscale,
    /// Re-encode only (container conversion via `output_format`).
    Format,
}

/// One input image for `POST /v1/image-ops`: a data URI or a prior job
/// bundle's artifact — never a filesystem path, `file://` URL, or remote
/// URL (there is no fetch, so no SSRF surface by construction).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageOpInput {
    /// `data:image/png;base64,…` — only `image/png`, `image/jpeg`, and
    /// `image/webp` payloads are accepted (magic-sniffed after decode).
    DataUri(String),
    /// A prior bundle's output artifact (see [`ImageOpArtifact`]).
    Artifact(ImageOpArtifact),
}

/// An artifact input: one output file of a finished job bundle written by
/// this server (under `<data_dir>/jobs/{bundle}/`). The bundle id is the
/// `ImageOpResponse.bundle` ULID; the index is the zero-based position in
/// that response's `outputs`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageOpArtifact {
    /// Bundle id (ULID string, exactly as a prior response reported it).
    pub bundle: String,
    /// Zero-based artifact index within that bundle.
    pub index: u32,
}

/// `POST /v1/image-ops` request (epic-editor-integration). Bounds are
/// reject-not-clamp everywhere: unknown fields (`deny_unknown_fields`),
/// out-of-range params, oversized batches, and malformed inputs are named
/// 400s — defaults never silently repair an explicit request.
///
/// `request_id` deduplicates retries: the same `request_id` with the same
/// canonical command replays the first job's response (same job id, same
/// bundle, no double execution); the same `request_id` with a DIFFERENT
/// command is a named 400 conflict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageOpRequest {
    /// The op plus its params (see [`ImageOpCommand`]).
    pub command: ImageOpCommand,
    /// Input images; `1..=8` (the image-ops batch cap). Each is a data URI
    /// or an artifact reference (see [`ImageOpInput`]); one output is
    /// produced per input, in order.
    pub inputs: Vec<ImageOpInput>,
    /// Output container: `"png"` (default), `"jpeg"`, or `"webp"` — the
    /// same vocabulary as the generation routes; unknown names are a 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// JPEG quality in `[0, 100]` (`output_compression` precedent);
    /// non-integer or out-of-range values are a 400, never clamped. Only
    /// applied to `output_format: "jpeg"` (PNG/WebP are lossless here).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<serde_json::Value>,
    /// Free-form caller identity recorded in the bundle sidecar (plugin
    /// name/version, e.g. `"krita-honkbox/1.0"`); `1..=128` chars when
    /// present. Never interpreted, never a credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    /// Caller-chosen idempotency key, `1..=128` chars. Required so every
    /// submission is retry-safe by construction.
    pub request_id: String,
}

/// One produced image in the `POST /v1/image-ops` response.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct ImageOpOutput {
    /// The produced image bytes (in the requested `output_format`),
    /// base64-encoded.
    pub b64_json: String,
    /// SHA-256 hex digest of the produced bytes (determinism checks).
    pub sha256: String,
    /// Container actually produced (`png` | `jpeg` | `webp`).
    pub format: String,
}

/// `POST /v1/image-ops` response: one output per input, in order, plus the
/// job/bundle ids a client needs to poll, cancel, or reference artifacts in
/// follow-up ops.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct ImageOpResponse {
    /// Completion time, Unix seconds.
    pub created: u64,
    /// Registry job id (`job_<ulid>`); matches `X-Job-Id`, `GET /v1/jobs`,
    /// and the cancel routes while the job is live.
    pub job_id: String,
    /// The request's idempotency key, verbatim.
    pub request_id: String,
    /// Bundle id (ULID) under `<data_dir>/jobs/{bundle}/` — the handle
    /// artifact inputs reference. Absent when the server runs without a
    /// data dir (bundles are disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    /// One entry per input, in order.
    pub outputs: Vec<ImageOpOutput>,
    /// Advisory notes; omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

/// Image-ops capability (`Capabilities::image_ops`): the supported op
/// vocabulary of `POST /v1/image-ops`. Present whenever the ops route is
/// compiled into this server (always today); `None` means the route is not
/// mounted and clients must not call it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ImageOpsCapability {
    /// Supported op wire names (the `ImageOpCommand` tag vocabulary).
    pub ops: Vec<String>,
}

/// `POST /v1/audio/transcriptions` plain shape (`response_format` absent
/// or `"json"`; REQ-9).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct TranscriptionText {
    /// The transcript text.
    pub text: String,
}

/// `POST /v1/audio/transcriptions` verbose shape
/// (`response_format = "verbose_json"`; REQ-9): the transcript plus the
/// STT worker's timestamped segments (millisecond integers, honkbox
/// segment shape).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct TranscriptionVerbose {
    /// The transcript text.
    pub text: String,
    /// Timestamped segments from the STT worker.
    pub segments: Vec<TranscriptSegment>,
}

/// Phase of a [`TranscriptionStreamEvent`] frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionStreamPhase {
    /// One bounded audio window was transcribed; more may follow.
    Partial,
    /// Terminal frame: all windows done, cancelled, or the deadline hit.
    Final,
}

/// One ordered frame of the SSE streaming-transcription feed
/// (`POST /v1/audio/transcriptions?stream=true`). Frames arrive in strict
/// `seq` order with no gaps; `final` is always last. Carried as SSE `data:`
/// on an existing route, so it needs no OpenAPI/TS schema entry of its own.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionStreamEvent {
    /// Monotonic, gap-free frame index (0-based).
    pub seq: u32,
    /// `partial` for a window result, `final` for the terminal frame.
    pub phase: TranscriptionStreamPhase,
    /// Transcript text for this window (empty for a silent window).
    pub text: String,
    /// Window start offset in milliseconds.
    pub start_ms: u32,
    /// Window end offset in milliseconds.
    pub end_ms: u32,
}

/// `POST /v1/audio/speech` request (REQ-9). `model` is optional (server
/// default TTS); `response_format` accepts `wav` (default, RIFF/WAVE
/// container), `pcm` (raw little-endian signed 16-bit mono samples,
/// matching the `OpenAI` pcm semantics — no container, no header), `flac`
/// (lossless FLAC), or `opus` (Ogg-encapsulated gapless Opus); anything
/// else is an explicit 400.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct SpeechRequest {
    /// Model id (alias or registry slug); omitted → server default TTS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Text to synthesize (required).
    pub input: String,
    /// Voice name; passed through to the TTS worker (Kokoro voice or
    /// backend default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Output format: `wav` (default), `pcm` (raw s16le, no container),
    /// `flac` (lossless), or `opus` (Ogg-encapsulated, gapless).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

/// One TTS voice entry of `GET /v1/audio/voices` (`data[]`). Entries describe
/// the loaded backend only: Kokoro serves its documented static set, while
/// backends without a voice-enumeration surface contribute no entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct Voice {
    /// Voice id to pass as `/v1/audio/speech` `voice` (e.g. `af_heart` for
    /// Kokoro).
    pub id: String,
    /// Always `"voice"` (OpenAI-style entry tag).
    pub object: String,
    /// Language/locale tag derived from the Kokoro voice prefix
    /// (e.g. `en-US`).
    pub language: String,
    /// `female` or `male`, from the Kokoro prefix.
    pub gender: String,
}

/// `GET /v1/audio/voices` response: OpenAI-style list envelope whose `data`
/// depends on the loaded TTS backend. `kokoro-cpu` serves the statically
/// documented Kokoro set (the backend cannot enumerate voices at runtime);
/// other backends, including `stub-tts`, return an empty `data` rather than
/// advertising another backend's ids. The `object: "list"` envelope is stable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct VoiceList {
    /// Always `"list"`.
    pub object: String,
    /// Voices supported by the loaded backend; empty when it cannot
    /// enumerate them.
    pub data: Vec<Voice>,
}

/// One transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct TranscriptSegment {
    /// Segment start, milliseconds from the audio start.
    pub start_ms: u32,
    /// Segment end, milliseconds from the audio start.
    pub end_ms: u32,
    /// Segment text.
    pub text: String,
}

/// Transcription result.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct Transcript {
    /// Which backend produced this.
    pub backend_id: String,
    /// Full transcription text.
    pub text: String,
    /// Timestamped segments, in order.
    pub segments: Vec<TranscriptSegment>,
    /// Wall-clock time in milliseconds.
    pub millis: u64,
}

/// Speech synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct SynthesizeRequest {
    /// Model id (informational; the backend is chosen by the server).
    pub model: String,
    /// Text to synthesize.
    pub text: String,
    /// Voice name; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Output sample rate in Hz; backend default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Per-request residency budget for this model's swap entry, in seconds
    /// (NEW-8; same contract as [`ChatRequest::ttl_seconds`]). Over-cap
    /// values are clamped at the route; audio pipelines are startup-loaded
    /// today, so the advice takes effect once per-kind swap lands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

/// Model modality as discovered on disk (`GET /v1/models` entries). Mirrors
/// the registry's on-disk kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Language-model weights (LLM root).
    Llm,
    /// Stable-Diffusion checkpoint / diffusion-model weights.
    Sd,
    /// `ControlNet` conditioning weights (`controlnet` category).
    ControlNet,
    /// `LoRA` adapter.
    Lora,
    /// Embedding / textual-inversion weights.
    Embedding,
    /// Standalone VAE weights.
    Vae,
    /// Speech-to-text weights (STT root).
    Stt,
    /// Text-to-speech weights (TTS root).
    Tts,
    /// OCR weights (OCR root).
    Ocr,
}

/// GGUF header summary (`GET /v1/models` entry `metadata`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct GgufMetadata {
    /// GGUF spec version.
    pub version: u32,
    /// Tensor count declared in the header.
    pub tensor_count: u64,
    /// `general.name` KV string, when present.
    pub name: Option<String>,
    /// `general.architecture` KV string, when present.
    pub architecture: Option<String>,
}

/// Safetensors header summary (`GET /v1/models` entry `metadata`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct SafetensorsMetadata {
    /// Tensor entries in the JSON header (`__metadata__` excluded).
    pub tensor_count: u64,
    /// Unique dtype names, sorted (`F32`, `F16`, `BF16`, …).
    pub dtypes: Vec<String>,
}

/// Optional on-disk header metadata, tagged by container format.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
#[serde(tag = "format", rename_all = "snake_case")]
pub enum ModelMetadata {
    /// `.gguf`: llama.cpp container header.
    Gguf(GgufMetadata),
    /// `.safetensors`: `HuggingFace` container header.
    Safetensors(SafetensorsMetadata),
}

/// Feature flags for one model (`GET /v1/models` entry `capabilities`).
///
/// Defaults derive from the entry's `kind` (see [`ModelCapabilities::for_kind`]);
/// an operator `[[models.entry]]` capabilities declaration overrides
/// individual flags. Emitted as a nested object for OpenAI/llama-swap-style
/// consumers — loop-lore reads the same values off the entry root as flat
/// fields, which are emitted alongside (`tool_calling`, `modalities`, …).
// The five-bool shape is the wire contract (llama-swap's nested capabilities
// map); a state machine would change what clients parse, not improve it.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ModelCapabilities {
    /// Chat-template conversation turns (an LLM pipeline can serve it).
    pub chat: bool,
    /// Raw completion; mirrors `chat` (honkbox loads whole turns either way).
    pub completion: bool,
    /// OpenAI-style tool/function calling.
    pub tool_calling: bool,
    /// Vision (`image` content parts).
    pub vision: bool,
    /// Embedding vectors (e.g. an operator-declared embedding GGUF).
    pub embeddings: bool,
}

impl ModelCapabilities {
    /// Inferred defaults for `kind`: LLM entries chat/completion-capable;
    /// on-disk `embedding` weights embedding-capable; everything else
    /// featureless until declared.
    #[must_use]
    pub fn for_kind(kind: ModelKind) -> Self {
        let (chat, embeddings) = match kind {
            ModelKind::Llm => (true, false),
            ModelKind::Embedding => (false, true),
            _ => (false, false),
        };
        Self {
            chat,
            completion: chat,
            tool_calling: false,
            vision: false,
            embeddings,
        }
    }
}

/// Input/output modality sets for one catalog entry. Additive to the legacy
/// flat `modalities` list so existing loop-lore consumers keep working.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ModelModalities {
    /// Accepted input content kinds (`text`, `image`, `audio`).
    pub input: Vec<String>,
    /// Produced output content kinds (`text`, `image`, `audio`).
    pub output: Vec<String>,
}

impl ModelModalities {
    /// Honest empty set for projections that do not derive modalities from
    /// an on-disk catalog entry (curated manifest rows and test fixtures).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            input: Vec::new(),
            output: Vec::new(),
        }
    }
}

/// One discovered model in the catalog (`GET /v1/models`).
///
/// `id` is the operator alias when one is configured for the file, else the
/// registry slug; requests accept either (the raw slug always works, via
/// `slug`). The loop-lore flat fields (`owned_by`, `context_length`,
/// `max_output`, `thinking`, `tool_calling`, `modalities`) mirror the nested
/// `capabilities` object so both consumer shapes work.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ModelCatalogEntry {
    /// Operator alias when configured, else the registry slug.
    pub id: String,
    /// Registry slug (`llm:…` / `comfy:…`), always present regardless of `id`.
    pub slug: String,
    /// Modality (capability overrides may reclassify a root, e.g. an
    /// operator-declared embedding GGUF under an LLM root).
    pub kind: ModelKind,
    /// Ownership label; local files are always `"local"`.
    pub owned_by: String,
    /// Training context length from GGUF metadata; `null` when unprobed.
    pub context_length: Option<u64>,
    /// Generation cap; `null` (honkbox has no per-model output ceiling).
    pub max_output: Option<u64>,
    /// Reasoning/thinking support (no honkbox model declares it today).
    pub thinking: bool,
    /// OpenAI-style tool calling; `capabilities.tool_calling` mirrored flat.
    pub tool_calling: bool,
    /// Content modalities (`["text"]`, plus `"vision"` when declared).
    pub modalities: Vec<String>,
    /// Input/output modality sets derived from the on-disk kind and
    /// capabilities; unknown kinds degrade to empty lists.
    pub modality: ModelModalities,
    /// Feature flags (also mirrored into the flat fields above).
    pub capabilities: ModelCapabilities,
    /// Directory category (`checkpoints`, `loras`, LLM parent dir, or `root`).
    pub category: String,
    /// Absolute path to the weight file.
    pub path: String,
    /// File size in bytes.
    pub bytes: u64,
    /// Header-derived metadata (GGUF/safetensors); absent when the file has
    /// no probeable header or the header is invalid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,
}

/// Model catalog: OpenAI-style list envelope (`GET /v1/models`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ModelCatalog {
    /// Always `"list"` (OpenAI-compatible envelope).
    pub object: String,
    /// Discovered models, sorted by id.
    pub data: Vec<ModelCatalogEntry>,
}

/// Explicit unload result (`POST /v1/models/{{slug}}/unload`,
/// `POST /api/models/unload/{{model}}`). `reason` is `"explicit"` when a
/// resident entry was evicted, `"not_loaded"` for an idempotent no-op (a
/// 200, never a 404 - the caller treats any 200 as an unconditional
/// continue-case).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct UnloadResult {
    /// `true` when a resident entry was evicted.
    pub unloaded: bool,
    /// The owning registry slug the request resolved to.
    pub slug: String,
    /// `"explicit"` or `"not_loaded"`.
    pub reason: String,
}

/// Explicit model deletion result (`DELETE /v1/models/{slug}`). Deletion
/// is **permanent**: the weight file(s) are removed from disk and the
/// catalog entry is dropped; recovery means downloading again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ModelDeleteResult {
    /// Always `true` on the 200 path.
    pub deleted: bool,
    /// Registry slug the request resolved to.
    pub slug: String,
    /// On-disk paths that were removed.
    pub removed: Vec<String>,
}

/// Bulk unload result (`POST /api/models/unload`): every non-pinned
/// resident was evicted; pinned residents are skipped, never refused here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct UnloadAllResult {
    /// Number of resident entries evicted.
    pub unloaded: u32,
    /// Always `"explicit"`.
    pub reason: String,
}

/// One resident entry of the swap manager (`GET /v1/memory`, `GET
/// /running`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ResidentModel {
    /// Owning registry slug.
    pub slug: String,
    /// Served alias when the operator bound one, else absent (the raw slug
    /// is always a valid request id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    /// Catalog kind string (currently always `"llm"` - the swap manager's
    /// only modality).
    pub kind: String,
    /// Load time, Unix seconds.
    pub loaded_at: u64,
    /// Idle-window end, Unix seconds; `null` for pinned entries (they have
    /// no TTL budget).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    /// `[swap].pin` member (never auto-evicts; explicit unload refuses).
    pub pinned: bool,
    /// Best-effort: another holder than the manager owns the pipeline `Arc`
    /// right now (an in-flight request).
    pub in_use: bool,
    /// Estimated resident weight size in bytes, taken from the registry
    /// scan's `bytes` for the slug (REQ-12). An **estimate** (on-disk
    /// weight file size), never a measurement; absent when the catalog no
    /// longer holds the slug (e.g. a re-scan removed it while resident).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
}

/// Configured warm-pipeline capacity (`GET /v1/memory`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct MemoryCapacity {
    /// LRU warm-set capacity (`[swap] llm_max_loaded`).
    pub llm_max_loaded: u32,
    /// Default idle window seconds (`[swap] llm_ttl_secs`).
    pub llm_ttl_secs: u64,
}

/// GPU slot readout (`GET /v1/memory`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct MemoryGpu {
    /// Another job currently holds the single inference slot.
    pub slot_busy: bool,
}

/// Residency readout (`GET /v1/memory`): exactly the models the swap
/// manager considers resident, plus configured capacity and GPU state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct MemoryInfo {
    /// Resident entries, sorted by slug.
    pub resident: Vec<ResidentModel>,
    /// Configured warm-set capacity.
    pub capacity: MemoryCapacity,
    /// GPU slot state.
    pub gpu: MemoryGpu,
    /// Sum of the resident entries' estimated bytes (REQ-12); an
    /// **estimate** (registry on-disk weight sizes), never a measurement.
    /// Absent when no resident entry has a known estimate (or the swap
    /// manager is off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_reserved: Option<u64>,
}

/// One llama-swap-compat running entry (`GET /running`). The envelope key
/// `running` and this entry's `model` key are load-bearing: loop-lore's
/// `readRunningModels` reads only `running[].model` and classifies any
/// other envelope as malformed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct RunningModel {
    /// Owning registry slug (llama-swap reports its model id here).
    pub model: String,
    /// Always `"running"` (llama-swap's healthy state string).
    pub state: String,
    /// Remaining idle-window seconds; `null` for pinned entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_secs: Option<u64>,
    /// `[swap].pin` member.
    pub pinned: bool,
}

/// llama-swap-compatible warm-set envelope (`GET /running`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct RunningList {
    /// Resident entries.
    pub running: Vec<RunningModel>,
}

/// Download modality target (`POST /v1/downloads`): picks the managed
/// `<data_dir>/models` subdir the completed file lands in (and labels the
/// download metrics). The on-disk `ModelKind` set is wider (`lora`,
/// `embedding`, `vae` come from `ComfyUI` category dirs); downloads only
/// target the four root modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum DownloadKind {
    /// Language-model weights (`<data_dir>/models/llm`).
    Llm,
    /// Stable-Diffusion weights (`<data_dir>/models/sd/checkpoints` - the
    /// registry classifies SD `.gguf` only under a checkpoints dir).
    Sd,
    /// Speech-to-text weights (`<data_dir>/models/stt`).
    Stt,
    /// Text-to-speech weights (`<data_dir>/models/tts`).
    Tts,
}

/// Download lifecycle state (`GET /v1/downloads` polling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    /// Accepted, waiting for a concurrency slot (max 2).
    Queued,
    /// Fetching bytes from scratch.
    Running,
    /// Appending to an existing sibling `.part` via a ranged `GET`.
    Resuming,
    /// File committed to its final path; a catalog rescan was triggered.
    Done,
    /// Terminal failure; the `.part` + meta sidecar are kept for a later
    /// resume (except checksum mismatch, which deletes them).
    Failed,
    /// Cancelled via `DELETE /v1/downloads/{id}`; `.part` + meta deleted.
    Cancelled,
}

/// Start one managed download (`POST /v1/downloads`).
#[derive(Debug, Clone, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadRequest {
    /// `HuggingFace` repo id (`owner/name`); together with `file` it derives
    /// the source URL `https://huggingface.co/{repo}/resolve/main/{file}`
    /// and enables the LFS sha256 checksum check when the API reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// File path inside `repo` (its last segment becomes the local file
    /// name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Direct download URL; the alternative to `repo` + `file`. Exactly one
    /// source form must be present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Target modality (see `DownloadKind`).
    pub kind: DownloadKind,
}

/// 202 acceptance body (`POST /v1/downloads`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadAccepted {
    /// Download id (`dl_<ulid>`); poll/cancel with it.
    pub id: String,
    /// State at acceptance: `queued` normally, `done` when the target file
    /// already existed (`already_present`), or the in-flight state when an
    /// identical download is already running (single-flight join).
    pub state: DownloadStatus,
}

/// One managed download's progress (`GET /v1/downloads[/id]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadState {
    /// Download id (`dl_<ulid>`).
    pub id: String,
    /// Lifecycle state.
    pub state: DownloadStatus,
    /// Bytes committed to the `.part` file so far.
    pub bytes_done: u64,
    /// Expected total size; absent until the server reports a length
    /// (and on terminal states where none was ever learned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
    /// Source URL the bytes are fetched from.
    pub source: String,
    /// Final file path under the managed models root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,
    /// Failure reason (only when `state` is `failed`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Ranged resume is possible for this source: the server supplied a
    /// strong `ETag` (usable as `If-Range` validator) and advertised
    /// `Accept-Ranges: bytes`. `false` means the next pull cold-restarts.
    pub resumable: bool,
    /// `done` because the target file already existed (no bytes fetched).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub already_present: bool,
}

/// Download list envelope (`GET /v1/downloads`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct DownloadList {
    /// Known downloads, sorted by id.
    pub downloads: Vec<DownloadState>,
}

/// One artifact emitted by a job, as reported by both `job_result` frames
/// and the live `GET /v1/jobs` snapshot. The five fields are the selection
/// metadata a consumer needs when a frame was dropped: identity, producer
/// name/kind, emission order and the single preferred artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct JobArtifactRef {
    /// Content-addressed artifact id (same value as `JobEvent::artifact_id`).
    pub artifact_id: String,
    /// Artifact name from the producer's own record (`final_00.png` for an
    /// image, the bundle file name for a Blender artifact).
    pub name: String,
    /// Artifact kind from the producer's own record (`image`, `model`,
    /// `map`, `preview`). This is NOT the job modality (`JobInfo::kind`).
    pub kind: String,
    /// Zero-based artifact position in emission order.
    pub index: u32,
    /// `true` for the one artifact a consumer should prefer; at most one
    /// entry of a job carries it.
    pub primary: bool,
}

/// One in-flight GPU job of `GET /v1/jobs` (NEW-3). Entries appear when a
/// GPU route admits a request and vanish when the request ends — this is a
/// liveness view, not history.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct JobInfo {
    /// Registry id (`job_<ulid>`); pass to `POST /v1/jobs/{id}/cancel`.
    /// Also stamped as the `X-Job-Id` response header of the owning request
    /// (on SSE routes it precedes the stream body).
    pub id: String,
    /// Modality: `chat`, `embeddings`, `images`, `transcribe`,
    /// `synthesize`, `translations`, or `image_ops`.
    pub kind: String,
    /// Model slug the job runs against, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Registration time, Unix milliseconds.
    pub started_at_ms: u64,
    /// Whether the job checks its cancel token between work units:
    /// `true` = a cancel stops it at the next checkpoint (multi-image
    /// generations between images, streaming chat at the next token);
    /// `false` = the single blocking FFI call cannot be interrupted and
    /// runs to completion (engine-level cancellation needs a per-backend
    /// spike, per the design record).
    pub interruptible: bool,
    /// Artifacts emitted so far by this live job, in emission order (bounded
    /// to 64). Absent until the first `job_result`; a dropped frame can be
    /// reconciled from this snapshot without a second request. The snapshot
    /// disappears with the live entry — this is not history or replay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Vec<JobArtifactRef>>,
}

/// `GET /v1/jobs` response: OpenAI-style list envelope over in-flight GPU
/// jobs, oldest first.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct JobList {
    /// Always `"list"`.
    pub object: String,
    /// The in-flight jobs.
    pub data: Vec<JobInfo>,
}

/// `POST /v1/jobs/{id}/cancel` response (NEW-3). `cancelled` is `true`
/// when a cancel was requested for a found id — the cooperative-stop token
/// was marked — and is never `false` for a found id; an unknown id
/// (finished or never registered) is a plain-text 404. Whether the work can
/// actually stop early is a SEPARATE fact: `interruptible` is reported per
/// job by [`JobInfo`] (`GET /v1/jobs`) and by the `job_cancelled`
/// [`JobEvent`] (`GET /v1/events`), not by this response.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct JobCancelResult {
    /// `true` = a cancel was requested and the token was marked (every
    /// found id); `false` is never returned here — an unknown id is a 404.
    /// Does NOT say whether the job can stop early; read `interruptible`
    /// from [`JobInfo`] or the `job_cancelled` [`JobEvent`].
    pub cancelled: bool,
}

/// Which lifecycle transition a [`JobEvent`] carries (`GET /v1/events`
/// SSE frames; the wire labels double as the SSE `event:` line).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS, ToSchema)]
pub enum JobEventType {
    /// The job entered the registry (route admitted the request).
    #[serde(rename = "job_registered")]
    Registered,
    /// A cancel was REQUESTED for the job (`POST /v1/jobs/{id}/cancel` or
    /// `DELETE /v1/jobs/{id}`). Not a completion event: whether the job
    /// actually stops is the `interruptible` contract.
    #[serde(rename = "job_cancelled")]
    Cancelled,
    /// The job left the registry (its request ended, any outcome).
    #[serde(rename = "job_finished")]
    Finished,
    /// Incremental work update: `step`/`total_steps` (and `pct` when the
    /// total is known) say how much has run. Not a lifecycle transition.
    #[serde(rename = "job_progress")]
    Progress,
    /// A produced artifact became available: `artifact_id`, `sha256`, and
    /// `size_bytes` identify it. A job may emit several; not a completion
    /// signal.
    #[serde(rename = "job_result")]
    Result,
}

impl JobEventType {
    /// SSE `event:` line label (mirrored in the JSON `event` field).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Registered => "job_registered",
            Self::Cancelled => "job_cancelled",
            Self::Finished => "job_finished",
            Self::Progress => "job_progress",
            Self::Result => "job_result",
        }
    }
}

/// One job lifecycle event streamed by `GET /v1/events`. The registry is a
/// liveness view, so events are transitional: no history is replayed on
/// (re)connect — a fresh stream only sees jobs registered after it opened.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct JobEvent {
    /// Lifecycle transition (also carried as the SSE `event:` line so
    /// JSON-only consumers do not need the envelope).
    pub event: JobEventType,
    /// Registry id (`job_<ulid>`); matches `GET /v1/jobs` and `X-Job-Id`.
    pub id: String,
    /// Modality label (`chat`, `embeddings`, `images`, `transcribe`,
    /// `synthesize`, `translations`, `image_ops`).
    pub kind: String,
    /// Model slug the job runs against, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Registration time, Unix milliseconds (same value `GET /v1/jobs`
    /// reports for the live entry).
    pub started_at_ms: u64,
    /// Whether the job checks its cancel token between work units (same
    /// contract as [`JobInfo::interruptible`]).
    pub interruptible: bool,
    /// Whether a cancel had been requested for the job at the event's
    /// point in time: always `true` on `job_cancelled`, `false` on
    /// `job_registered`, and on `job_finished` `true` means the token was
    /// marked before the job ended — for a non-interruptible job that
    /// still means the work ran to completion.
    pub cancelled: bool,
    /// Work units completed so far (`job_progress` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<u32>,
    /// Total work units (`job_progress` only); 0 when the total is unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<u32>,
    /// Percent complete, `0..=100` (`job_progress` only; `None` when
    /// `total_steps` is 0, i.e. no percentage can be computed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pct: Option<f64>,
    /// Content-addressed artifact id (`job_result` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    /// Sha256 hex digest of the artifact bytes (`job_result` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Artifact byte length (`job_result` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Artifact name from the producer's own record (`job_result` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_name: Option<String>,
    /// Artifact kind from the producer's own record (`job_result` only).
    /// NOT the job modality already carried by `kind`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_kind: Option<String>,
    /// Zero-based artifact position in emission order (`job_result` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_index: Option<u32>,
    /// `Some(true)` marks the one preferred artifact of a job (`job_result`
    /// only); `None` on older frames and on annotation re-emits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_primary: Option<bool>,
    /// NSFW annotations attached to the job's output, when a classifier
    /// ran. Absent when no annotation was produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<NsfwAnnotation>>,
}

/// One Blender asset job (`POST /v1/jobs/blender`).
///
/// The route admits the job asynchronously: it answers `202` with the
/// registry id, and the lifecycle runs through `GET /v1/jobs` and
/// `GET /v1/events` like every other job class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobRequest {
    /// The mesh the job works on.
    pub input: BlenderJobInput,
    /// The stages to run, in execution order.
    pub stages: Vec<BlenderStage>,
    /// Job-wide parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<BlenderJobParams>,
    /// The engine contract the caller expects; checked before Blender starts.
    pub engine: BlenderEngine,
    /// The output limits the caller expects.
    pub output: BlenderOutput,
    /// The manifest schema version the caller expects (currently `1`).
    pub manifest: u32,
}

/// The mesh a Blender job works on. The mesh is read from the artifact
/// store by id — the route never accepts a filesystem path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobInput {
    /// The stored artifact id: the 64-hex sha256 of the content.
    pub asset_id: String,
    /// The format the stored mesh is in.
    pub format: BlenderMeshFormat,
    /// The sha256 the caller expects; a mismatch is rejected before the
    /// Blender process starts.
    pub sha256: String,
}

/// The mesh formats a Blender job can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum BlenderMeshFormat {
    /// Binary glTF.
    Glb,
    /// Wavefront OBJ.
    Obj,
    /// Stanford PLY.
    Ply,
}

/// Job-wide parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobParams {
    /// The seed recorded in the manifest. The recipe is deterministic, so
    /// the seed is metadata, not entropy.
    #[serde(default)]
    pub seed: u64,
    /// The artifact name prefix; defaults to `output` (the artifacts land
    /// as `output.glb`, `output_color.png`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    /// The job's wall-clock bound in seconds (default 600, max 3600). The
    /// process group is killed when it expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

/// The engine contract a Blender job expects. Both fields are checked
/// against the installed engine before the process starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderEngine {
    /// The Blender version the caller expects (e.g. `"4.5.15"`).
    pub blender_version: String,
    /// The recipe content pin: the sha256 of the embedded recipe BYTES
    /// (`recipe_sha256()`, 64 lowercase hex). Absent = the caller pins no
    /// recipe (the server runs its current one); present = the digest must
    /// match, so a recipe change is a named 409
    /// (`blender_script_version_mismatch`) instead of a silently different
    /// result. This is a CONTENT digest only: a digest of a script path (or
    /// of any other string) can never match and is rejected the same way.
    /// A caller that cannot compute the byte digest omits the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_version: Option<String>,
}

/// The output limits a Blender job expects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderOutput {
    /// The output format the caller expects (currently `"glb"`).
    pub format: String,
    /// The byte cap for the produced artifacts.
    pub max_bytes: u64,
}

/// One stage of a Blender job, in execution order. The vocabulary is
/// closed: the recipe knows exactly these stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum BlenderStage {
    /// Smart UV project.
    UvUnpack {
        /// The island margin in Blender units.
        margin: f64,
    },
    /// Pack the file-backed images into the blend.
    UvPack,
    /// Voxel remesh.
    VoxelRemesh {
        /// The voxel size in Blender units.
        voxel_size: f64,
    },
    /// Quadriflow retopology.
    Quadriflow {
        /// The target face count.
        target_faces: u64,
    },
    /// Attach a texture through a `ShaderNodeTexImage`.
    AttachTexture {
        /// The stored artifact id of the image to attach.
        image_id: String,
        /// The node name to create.
        node: String,
    },
    /// Bake maps to PNG files.
    Bake {
        /// The maps to bake.
        maps: Vec<BlenderBakeMap>,
        /// The square resolution of the baked maps.
        resolution: u32,
    },
    /// Export the scene as a binary glTF.
    ExportGlb,
    /// Render a turntable preview: `frames` views around the object, tiled
    /// into one image (the workspace has no animated webp encoder, so the
    /// turntable ships as one still).
    Turntable {
        /// How many views to render (1..=8).
        frames: u32,
        /// The square resolution of each view (16..=1024).
        resolution: u32,
    },
}

/// The maps a bake can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum BlenderBakeMap {
    /// The diffuse colour map.
    Color,
    /// The tangent-space normal map.
    Normal,
    /// The roughness map.
    Roughness,
}

/// 202 acceptance body (`POST /v1/jobs/blender`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct BlenderJobAccepted {
    /// Registry job id (`job_<ulid>`); matches `GET /v1/jobs`, the
    /// `X-Job-Id` header and the cancel route while the job is live.
    pub job_id: String,
    /// The state at acceptance: always `"queued"`.
    pub status: String,
    /// Bundle id (ULID) under `<data_dir>/jobs/{bundle}/` — the manifest and
    /// the artifacts land there. Absent when the server runs without a data
    /// dir (bundles are disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
}

/// The typed completion record of a Blender asset job: the same shape the
/// bundle's `manifest.json` carries. The writer is
/// `routes/blender_exec.rs::manifest_record`, and a unit test there pins
/// this DTO to it — the file on disk and this contract cannot drift apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobCompletion {
    /// The manifest schema version (currently `1`).
    pub manifest: u32,
    /// The Blender version that ran the job.
    pub blender_version: String,
    /// The recipe version that ran the job.
    pub recipe_version: String,
    /// The sha256 of the recipe bytes that ran the job.
    pub recipe_sha256: String,
    /// The job, echoed back.
    pub job: BlenderJobCompletionJob,
    /// The produced artifacts, in recipe order.
    pub artifacts: Vec<BlenderJobCompletionArtifact>,
    /// Non-fatal notes from the recipe.
    pub warnings: Vec<String>,
    /// The progress events the recipe reported.
    pub progress: Vec<BlenderJobCompletionProgress>,
}

/// The echoed job inside a [`BlenderJobCompletion`]: what ran, without any
/// filesystem path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobCompletionJob {
    /// The mesh the job consumed.
    pub input: BlenderJobCompletionInput,
    /// The stages as requested (the request echo).
    pub stages: Vec<BlenderStage>,
    /// The artifact name prefix the job used.
    pub prefix: String,
    /// The seed recorded for the job (metadata, not entropy).
    pub seed: u64,
}

/// The mesh a [`BlenderJobCompletion`] records as consumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobCompletionInput {
    /// The artifact-store id of the mesh (its sha256).
    pub artifact_id: String,
    /// The mesh format (`glb`, `obj` or `ply`).
    pub format: String,
}

/// One produced artifact in a [`BlenderJobCompletion`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobCompletionArtifact {
    /// The bundle file name (never a staging path).
    pub name: String,
    /// Content-addressed id of the promoted artifact in the registry
    /// (`GET /v1/artifacts/{id}/bytes`); the digest recorded in the
    /// bundle's `manifest.json`.
    pub artifact_id: String,
    /// What it is (`map`, `model` or the `preview` strip).
    pub kind: String,
    /// Its size in bytes (post-conversion for the preview).
    pub bytes: u64,
    /// Its sha256 (of the bundle bytes).
    pub sha256: String,
}

/// One progress event a [`BlenderJobCompletion`] records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct BlenderJobCompletionProgress {
    /// The stage the event belongs to.
    pub stage: String,
    /// What happened: `start`, `done` or `cancelled`.
    pub event: String,
}

/// Fit-pill verdict for a curated catalog entry against the configured
/// `[models] vram_budget_bytes`. Estimates are advisory heuristics (per-kind
/// overhead factors over weight bytes), never a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Pill {
    /// Estimated VRAM fits the budget (`est <= budget`).
    Ok,
    /// Weights fit but the estimate does not (`bytes <= budget < est`):
    /// loads, likely offloads — slow.
    Slow,
    /// Weights alone exceed the budget (`bytes > budget`).
    Fail,
}

/// Local scan presence for one curated catalog entry (`GET /v1/catalog`);
/// absent when no scanned root holds a matching file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct CatalogLocal {
    /// Matched catalog slug (`llm:…` / `comfy:…` / `stt:…` / `tts:…`).
    pub slug: String,
    /// On-disk weight size in bytes (the pill's byte input).
    pub bytes: u64,
    /// Estimated VRAM need (advisory heuristic, not a measurement); absent
    /// without a configured `[models] vram_budget_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub est_vram_bytes: Option<u64>,
    /// Fit pill vs the configured budget; absent without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pill: Option<Pill>,
}

/// One curated catalog offering (`GET /v1/catalog`). The manifest is a
/// checked-in JSON list — review-by-PR is the signature; entries carry a
/// `verified_date` and rot without it (12 months ⇒ drop at release).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct CatalogEntry {
    /// Stable manifest id (e.g. `llama-3.2-1b-q4`); NOT a catalog slug.
    pub id: String,
    /// `HuggingFace` repo id the file downloads from (`owner/name`).
    pub repo: String,
    /// File path inside `repo` (last segment is the local file name).
    pub file: String,
    /// Modality (drives the download target dir and the pill factor).
    pub kind: ModelKind,
    /// Quantization/format tag (e.g. `Q4_K_M`, `fp16`).
    pub quant: String,
    /// Model license tag (e.g. `apache-2.0`, `llama3.2`).
    pub license: String,
    /// Manual verification month (`YYYY-MM`); entries unverified for 12
    /// months are dropped at release.
    pub verified_date: String,
    /// Short human note (why this entry is curated).
    pub notes: String,
    /// Local scan presence + fit pill; absent when not found under the
    /// scanned roots.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local: Option<CatalogLocal>,
}

/// Curated catalog envelope (`GET /v1/catalog`), OpenAI-style list shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct CatalogList {
    /// Always `"list"` (OpenAI-compatible envelope).
    pub object: String,
    /// Curated entries, sorted by id.
    pub data: Vec<CatalogEntry>,
}

/// One received chunk record in an artifact upload manifest
/// (`POST /v1/artifacts` chunked mode).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct ArtifactChunk {
    /// 0-based chunk index within the upload.
    pub index: u32,
    /// sha256 digest (lowercase hex) of this chunk's bytes.
    pub sha256: String,
    /// Chunk size in bytes.
    pub bytes: u64,
}

/// Completed artifact descriptor: `id` is the content sha256 hex digest and
/// doubles as the `{id}` of `GET /v1/artifacts/{id}/bytes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct ArtifactInfo {
    /// Content sha256 (lowercase hex) — the artifact id.
    pub id: String,
    /// Same digest, named explicitly for direct hash checks.
    pub sha256: String,
    /// Total artifact size in bytes.
    pub size_bytes: u64,
    /// Number of chunks the artifact was uploaded in (1 for single-shot).
    pub chunk_count: u32,
    /// Content type served by the bytes route (default
    /// `application/octet-stream`).
    pub content_type: String,
}

/// Upload manifest returned on every chunked `POST /v1/artifacts` request:
/// the registry of received chunks plus the completed artifact descriptor
/// once the final chunk lands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct ArtifactManifest {
    /// Server-assigned upload session id (`X-Upload-Id` on resume).
    pub upload_id: String,
    /// Declared total chunk count.
    pub chunk_count: u32,
    /// Received chunk indices, ascending (the resume worklist's complement).
    pub received: Vec<u32>,
    /// Per-chunk digests/sizes, ordered by index.
    pub chunks: Vec<ArtifactChunk>,
    /// `"uploading"` until the final chunk, then `"complete"`.
    pub status: String,
    /// Absent until the upload completes; present exactly when `status` is
    /// `"complete"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ArtifactInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regenerate the checked-in TS bindings (drift is gated by `git diff` in CI).
    /// Emit every wire DTO's TS declaration (the codegen source of truth).
    /// The declaration list grows one line per DTO by design; the length
    /// allowance tracks that.
    #[test]
    #[allow(clippy::too_many_lines)] // one decl line per wire DTO, by design
    fn export_ts() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/web-ui/src/gen");
        std::fs::create_dir_all(&dir).expect("create gen dir");
        let config = ts_rs::Config::new();
        // ts-rs v12 `decl()` never emits the `export` keyword (no config knob
        // for it), which would leave the bindings unimportable. Re-add it: the
        // keyword opens every decl string, and ts-rs never renders a line that
        // starts with `type ` inside doc comments or object bodies (fields end
        // in `:`), so a first-occurrence prefix rewrite is exact.
        let exported = |decl: String| decl.replacen("type ", "export type ", 1);
        let mut out =
            String::from("// AUTO-GENERATED by `cargo test -p honkbox-api-types` — do not edit.\n");
        for decl in [
            Health::decl(&config),
            Version::decl(&config),
            // `serde_json::Value` renders as `JsonValue` (ts-rs serde-json
            // impl); fields like `FunctionDefinition::parameters` reference
            // it, so its definition must ship alongside them.
            serde_json::Value::decl(&config),
            ChatRole::decl(&config),
            ContentPart::decl(&config),
            ImageUrl::decl(&config),
            ChatMessage::decl(&config),
            ToolCall::decl(&config),
            ToolCallType::decl(&config),
            FunctionCall::decl(&config),
            ChatTool::decl(&config),
            ToolType::decl(&config),
            FunctionDefinition::decl(&config),
            ToolChoiceSpec::decl(&config),
            NamedFunction::decl(&config),
            ResponseFormat::decl(&config),
            ResponseFormatType::decl(&config),
            JsonSchemaSpec::decl(&config),
            ChatRequest::decl(&config),
            ChatResponse::decl(&config),
            OpenAiChatRequest::decl(&config),
            FinishReason::decl(&config),
            OpenAiUsage::decl(&config),
            OpenAiChoice::decl(&config),
            OpenAiChatCompletion::decl(&config),
            OpenAiDelta::decl(&config),
            DeltaToolCall::decl(&config),
            DeltaFunction::decl(&config),
            OpenAiChunkChoice::decl(&config),
            OpenAiChunk::decl(&config),
            OpenAiErrorDetail::decl(&config),
            OpenAiError::decl(&config),
            EmbeddingsRequest::decl(&config),
            EmbeddingsInput::decl(&config),
            EmbeddingsData::decl(&config),
            EmbeddingsUsage::decl(&config),
            EmbeddingsResponse::decl(&config),
            ImagesGenerationRequest::decl(&config),
            ImagesLora::decl(&config),
            ImagesHiresFix::decl(&config),
            ImageB64::decl(&config),
            ImageArtifact::decl(&config),
            ImagesGenerationResponse::decl(&config),
            TranscriptionText::decl(&config),
            TranscriptionVerbose::decl(&config),
            ImageOpKind::decl(&config),
            ImageOpCommand::decl(&config),
            ImageOpResize::decl(&config),
            ImageOpCrop::decl(&config),
            ImageOpRotate::decl(&config),
            ImageOpInput::decl(&config),
            ImageOpArtifact::decl(&config),
            ImageOpRequest::decl(&config),
            ImageOpOutput::decl(&config),
            ImageOpResponse::decl(&config),
            ImageOpsCapability::decl(&config),
            SpeechRequest::decl(&config),
            TranscriptSegment::decl(&config),
            Transcript::decl(&config),
            SynthesizeRequest::decl(&config),
            Voice::decl(&config),
            VoiceList::decl(&config),
            ModelDeleteResult::decl(&config),
            ModelKind::decl(&config),
            ModelCapabilities::decl(&config),
            ModelModalities::decl(&config),
            GgufMetadata::decl(&config),
            SafetensorsMetadata::decl(&config),
            ModelMetadata::decl(&config),
            ModelCatalogEntry::decl(&config),
            ModelCatalog::decl(&config),
            UnloadResult::decl(&config),
            UnloadAllResult::decl(&config),
            ResidentModel::decl(&config),
            MemoryCapacity::decl(&config),
            MemoryGpu::decl(&config),
            MemoryInfo::decl(&config),
            RunningModel::decl(&config),
            RunningList::decl(&config),
            DownloadKind::decl(&config),
            DownloadStatus::decl(&config),
            DownloadRequest::decl(&config),
            DownloadAccepted::decl(&config),
            DownloadState::decl(&config),
            DownloadList::decl(&config),
            JobArtifactRef::decl(&config),
            JobInfo::decl(&config),
            JobList::decl(&config),
            JobCancelResult::decl(&config),
            JobEventType::decl(&config),
            JobEvent::decl(&config),
            Pill::decl(&config),
            CatalogLocal::decl(&config),
            CatalogEntry::decl(&config),
            CatalogList::decl(&config),
            ArtifactChunk::decl(&config),
            ArtifactInfo::decl(&config),
            ArtifactManifest::decl(&config),
            Capabilities::decl(&config),
            LicenseCapability::decl(&config),
            LicenseSummaryRow::decl(&config),
            StorageCapability::decl(&config),
            ServerInfo::decl(&config),
            AudioCppCapabilities::decl(&config),
            FeatureFlags::decl(&config),
            VisionFeature::decl(&config),
            ControlNetFeature::decl(&config),
            Formats::decl(&config),
            OutputFormatCapability::decl(&config),
            FormatLimits::decl(&config),
            CompressionRange::decl(&config),
            AuthCapabilities::decl(&config),
            QueueCapabilities::decl(&config),
            QueueKindCapacity::decl(&config),
            ToolCallingCapability::decl(&config),
            ExternalTool::decl(&config),
            ExternalToolsCapability::decl(&config),
            RecKind::decl(&config),
            RecognitionFeature::decl(&config),
            RecognizeRequest::decl(&config),
            RecognitionRegion::decl(&config),
            RecognitionPage::decl(&config),
            RecognitionEngineMeta::decl(&config),
            RecognizeResult::decl(&config),
            BlenderJobRequest::decl(&config),
            BlenderJobInput::decl(&config),
            BlenderMeshFormat::decl(&config),
            BlenderJobParams::decl(&config),
            BlenderEngine::decl(&config),
            BlenderOutput::decl(&config),
            BlenderStage::decl(&config),
            BlenderBakeMap::decl(&config),
            BlenderJobAccepted::decl(&config),
            BlenderJobCompletion::decl(&config),
            BlenderJobCompletionJob::decl(&config),
            BlenderJobCompletionInput::decl(&config),
            BlenderJobCompletionArtifact::decl(&config),
            BlenderJobCompletionProgress::decl(&config),
            NsfwCapability::decl(&config),
            MeshExternalCapability::decl(&config),
            SignedUrlCapability::decl(&config),
            ScopeCapability::decl(&config),
            NsfwAnnotation::decl(&config),
            QueueJob::decl(&config),
            QueueSnapshot::decl(&config),
            RerankRequest::decl(&config),
            RerankItem::decl(&config),
            RerankResult::decl(&config),
            FizgigTrainRequest::decl(&config),
            FizgigTrainAccepted::decl(&config),
            FizgigTrainStatus::decl(&config),
            TexturePostprocessCapability::decl(&config),
            OcrIngestMode::decl(&config),
            OcrIngestRequest::decl(&config),
            OcrRegion::decl(&config),
            OcrPage::decl(&config),
            OcrEngineMeta::decl(&config),
            OcrResult::decl(&config),
            OcrJobAccepted::decl(&config),
            ManagedApp::decl(&config),
            ManagedAppList::decl(&config),
            ComfyUiStatus::decl(&config),
            ComfyUiQueue::decl(&config),
            ComfyUiProgress::decl(&config),
            ComfyUiHistory::decl(&config),
        ] {
            out.push_str(&exported(decl));
            out.push('\n');
        }
        std::fs::write(dir.join("api-types.ts"), out).expect("write api-types.ts");
    }

    /// Inferred defaults: LLM chats, on-disk embeddings embed, the rest are
    /// featureless until an operator declaration overrides them.
    #[test]
    fn capabilities_default_from_kind() {
        let llm = ModelCapabilities::for_kind(ModelKind::Llm);
        assert!(llm.chat && llm.completion);
        assert!(!llm.tool_calling && !llm.vision && !llm.embeddings);

        let embedding = ModelCapabilities::for_kind(ModelKind::Embedding);
        assert!(embedding.embeddings);
        assert!(!embedding.chat && !embedding.completion);

        for kind in [
            ModelKind::Sd,
            ModelKind::ControlNet,
            ModelKind::Lora,
            ModelKind::Vae,
            ModelKind::Stt,
            ModelKind::Tts,
            ModelKind::Ocr,
        ] {
            let caps = ModelCapabilities::for_kind(kind);
            assert!(!caps.chat && !caps.completion && !caps.embeddings);
        }
    }

    /// `skip_serializing_if` omits these keys on the wire, so their TS
    /// declarations must be optional (`?:`) — a required `T | null` field
    /// makes a client's `!== null` check trip over `undefined`.
    #[test]
    fn omitted_wire_fields_are_optional_in_ts() {
        let config = ts_rs::Config::new();
        let region = RecognitionRegion::decl(&config);
        assert!(
            region.contains("confidence?: number | null"),
            "confidence must be optional in TS: {region}"
        );
        let engine = RecognitionEngineMeta::decl(&config);
        assert!(
            engine.contains("version?: string | null"),
            "version must be optional in TS: {engine}"
        );
        let manifest = ArtifactManifest::decl(&config);
        assert!(
            manifest.contains("artifact?: ArtifactInfo | null"),
            "artifact must be optional in TS: {manifest}"
        );
    }
}

/// Property-style hardening of the widened `ChatMessage.content` contract
/// (REQ-8). proptest is not a dependency of this crate, so these tests use
/// exhaustive enumeration over the malformed-grammar variants plus a tiny
/// seeded LCG driving randomized strings and part sequences (deterministic,/// hence reproducible from the seed). Invariants:
/// - the string form round-trips byte-identically (serialization is exactly
///   the JSON encoding of the plain string, for EVERY input);
/// - serialization is ALWAYS `string | null` — never the parts array, and
///   image payloads never echo back;
/// - an unknown part `type` is a named rejection (never a silent accept);
/// - every malformed variant (missing/wrong-typed fields, non-object parts,
///   nested arrays, scalar content) maps to a NAMED error, never a panic;
/// - text parts flatten in order with no separator, and each `image_url`
///   part records its byte offset in the flattened text, so media markers
///   splice back at exactly the right positions.
#[cfg(test)]
mod content_parts_properties {
    use super::*;

    /// Tiny deterministic LCG (Knuth's MMIX constants): seeded, so a
    /// failure reproduces from the reported seed alone.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed ^ 0xC0BB_AB1E_C0DE)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn index(&mut self, len: usize) -> usize {
            if len == 0 {
                0
            } else {
                usize::try_from(self.next_u64() % u64::try_from(len).unwrap_or(u64::MAX))
                    .unwrap_or(0)
            }
        }
    }

    /// String atoms with distinct JSON-escaping and byte-length behavior:
    /// ASCII, quotes/backslashes, control chars, multi-byte UTF-8, emoji,
    /// and the media marker itself (for collision coverage).
    const ATOMS: [&str; 14] = [
        "",
        "a",
        "hello world",
        "\n",
        "\t",
        "\"",
        "\\",
        "\u{0}",
        "\u{7f}",
        "é",
        "🎉",
        "語",
        "data:image/png;base64,AAAA",
        "<__media__>",
    ];

    /// Atoms that can never contain the media marker (marker-free alphabet
    /// for the exact-splice-position property).
    const MARKER_FREE_ATOMS: [&str; 11] = [
        "",
        "a",
        "hello world",
        "\n",
        "\t",
        "\"",
        "\\",
        "\u{0}",
        "é",
        "🎉",
        "語",
    ];

    /// A seeded random string over `atoms` (0..=6 atoms concatenated).
    fn random_string(rng: &mut Lcg, atoms: &[&str]) -> String {
        let mut out = String::new();
        for _ in 0..rng.index(7) {
            out.push_str(atoms[rng.index(atoms.len())]);
        }
        out
    }

    /// The string form is byte-identical to a plain JSON string, in both
    /// directions, for EVERY string — including escaping-heavy ones.
    #[test]
    fn string_form_round_trips_byte_identically() {
        for seed in 0..256u64 {
            let mut rng = Lcg::new(seed);
            let text = random_string(&mut rng, &ATOMS);
            let content = ChatContent::text(text.clone());

            // Serializing the content is byte-identical to serializing the
            // bare string (the wire shape is exactly a JSON string).
            let as_content = serde_json::to_string(&content).expect("serialize content");
            let as_string = serde_json::to_string(&text).expect("serialize string");
            assert_eq!(
                as_content, as_string,
                "seed {seed}: wire form must be the plain string"
            );

            // Parse(serialize) reproduces the exact text.
            let reparsed =
                ChatContent::parse(&serde_json::Value::String(text.clone())).expect("parses");
            assert_eq!(reparsed.text.as_deref(), Some(text.as_str()));
            assert!(!reparsed.has_image);
            assert_eq!(reparsed.images, [] as [ChatImageRef; 0]);

            // Via the serde entry point too (the routes' deserialization).
            let via_serde: ChatContent =
                serde_json::from_value(serde_json::Value::String(text.clone()))
                    .expect("string form deserializes");
            assert_eq!(via_serde, content);

            // A ChatMessage serializes its content as that exact string.
            let message = ChatMessage {
                role: ChatRole::User,
                content: content.clone(),
                tool_calls: None,
                tool_call_id: None,
            };
            let rendered = serde_json::to_string(&message).expect("serialize message");
            let expected = format!("{{\"role\":\"user\",\"content\":{as_content}}}");
            assert_eq!(rendered, expected, "seed {seed}: message content wire form");
        }
    }

    /// Serialization NEVER emits the parts array and never echoes image
    /// payloads: parsed parts arrays serialize as the flattened string (or
    /// null), with no image URL anywhere in the output.
    #[test]
    fn serialization_is_never_an_array_and_images_never_echo() {
        for seed in 0..256u64 {
            let image_url = format!("data:image/png;base64,seed{seed}");
            let parts = vec![
                serde_json::json!({"type": "text", "text": "before "}),
                serde_json::json!({"type": "image_url", "image_url": {"url": image_url}}),
                serde_json::json!({"type": "text", "text": "after"}),
            ];
            let content: ChatContent =
                serde_json::from_value(serde_json::Value::Array(parts)).expect("parts parse");
            assert!(content.has_image);
            let rendered = serde_json::to_string(&content).expect("serialize");
            assert_eq!(rendered, "\"before after\"", "seed {seed}");
            assert!(
                !rendered.contains(&image_url),
                "image payloads never echo back"
            );
            assert!(!rendered.starts_with('['), "never the parts array");

            // Null content serializes as null.
            let null_content: ChatContent =
                serde_json::from_value(serde_json::Value::Null).expect("null parses");
            assert_eq!(
                serde_json::to_string(&null_content).expect("serialize null"),
                "null"
            );
        }
    }

    /// An unknown part `type` is rejected BY NAME — the exact documented
    /// message, quoting the offending type — for arbitrary type strings.
    #[test]
    fn unknown_part_type_is_rejected_by_name() {
        for seed in 0..256u64 {
            let mut rng = Lcg::new(seed);
            let unknown = random_string(&mut rng, &ATOMS);
            if unknown == "text" || unknown == "image_url" {
                continue;
            }
            let body = serde_json::json!({"type": unknown, "text": "x"});
            let err = ChatContent::parse(&serde_json::Value::Array(vec![body]))
                .expect_err("unknown type must be rejected");
            assert_eq!(
                err,
                format!(
                    "unsupported content part type {unknown:?}: supported types are `text` and `image_url`"
                ),
                "seed {seed}: the rejection must name the offending type"
            );
        }
    }

    /// Every malformed variant of the parts grammar maps to its named
    /// error — never a panic, never a silent accept.
    #[test]
    #[allow(clippy::too_many_lines)] // the grammar's variants are enumerated flat
    fn malformed_variants_map_to_named_errors() {
        let scalar_parts: Vec<serde_json::Value> = vec![
            serde_json::json!("bare string"),
            serde_json::json!(42),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!(true),
            serde_json::json!(null),
            serde_json::json!([{"type": "text", "text": "nested"}]),
        ];
        for part in &scalar_parts {
            let err = ChatContent::parse(&serde_json::Value::Array(vec![part.clone()]))
                .expect_err("non-object part must be rejected");
            assert!(
                err.contains("content part must be an object with a `type` field"),
                "unexpected error for {part}: {err}"
            );
        }

        // A `type` field that is missing or not a string.
        for type_field in [
            serde_json::json!({"text": "no type"}),
            serde_json::json!({"type": 42, "text": "x"}),
            serde_json::json!({"type": true, "text": "x"}),
            serde_json::json!({"type": null, "text": "x"}),
            serde_json::json!({"type": ["text"], "text": "x"}),
            serde_json::json!({"type": {"t": "text"}, "text": "x"}),
        ] {
            let err = ChatContent::parse(&serde_json::Value::Array(vec![type_field.clone()]))
                .expect_err("missing/non-string type must be rejected");
            assert_eq!(
                err, "content part requires a string `type` field",
                "unexpected error for {type_field}"
            );
        }

        // A `text` part without a string `text` field.
        for text_field in [
            serde_json::json!({"type": "text"}),
            serde_json::json!({"type": "text", "text": 42}),
            serde_json::json!({"type": "text", "text": null}),
            serde_json::json!({"type": "text", "text": ["x"]}),
            serde_json::json!({"type": "text", "text": {"t": "x"}}),
        ] {
            let err = ChatContent::parse(&serde_json::Value::Array(vec![text_field.clone()]))
                .expect_err("text part without string text must be rejected");
            assert_eq!(
                err, "text part requires a string `text` field",
                "unexpected error for {text_field}"
            );
        }

        // An `image_url` part without a string `image_url.url`.
        for image_field in [
            serde_json::json!({"type": "image_url"}),
            serde_json::json!({"type": "image_url", "image_url": null}),
            serde_json::json!({"type": "image_url", "image_url": "data:image/png;base64,x"}),
            serde_json::json!({"type": "image_url", "image_url": {}}),
            serde_json::json!({"type": "image_url", "image_url": {"url": 42}}),
            serde_json::json!({"type": "image_url", "image_url": {"url": null}}),
        ] {
            let err = ChatContent::parse(&serde_json::Value::Array(vec![image_field.clone()]))
                .expect_err("image_url part without url must be rejected");
            assert_eq!(
                err, "image_url part requires a string `image_url.url` field",
                "unexpected error for {image_field}"
            );
        }

        // Content itself is neither string, null, nor array.
        for content in [
            serde_json::json!(42),
            serde_json::json!(true),
            serde_json::json!(1.5),
            serde_json::json!({"type": "text", "text": "x"}),
        ] {
            let err =
                ChatContent::parse(&content).expect_err("scalar/object content must be rejected");
            assert!(
                err.contains("content must be a string, null, or an array of content parts"),
                "unexpected error for {content}: {err}"
            );
            let via_serde = serde_json::from_value::<ChatContent>(content.clone());
            assert!(via_serde.is_err(), "serde path must reject {content}");
        }

        // Deeply nested arrays are still just non-object parts.
        let nested = serde_json::json!([[{"type": "text", "text": "x"}]]);
        let err = ChatContent::parse(&nested).expect_err("nested array must be rejected");
        assert!(err.contains("content part must be an object"));
    }

    /// The degenerate forms parse to the documented defaults: `null` and
    /// `[]` carry no text and no images (an empty flattening is `None`, not
    /// the empty string); missing content (via `ChatMessage`) equals null.
    #[test]
    fn degenerate_forms_parse_to_defaults() {
        let null_content: ChatContent =
            serde_json::from_value(serde_json::Value::Null).expect("null");
        assert_eq!(null_content.text, None);
        assert!(!null_content.has_image);
        assert_eq!(null_content.images, [] as [ChatImageRef; 0]);

        let empty: ChatContent =
            serde_json::from_value(serde_json::Value::Array(Vec::new())).expect("empty array");
        assert_eq!(empty.text, None, "an empty flattening is None");
        assert!(!empty.has_image);
        assert_eq!(empty.images, [] as [ChatImageRef; 0]);

        let message: ChatMessage =
            serde_json::from_str("{\"role\":\"assistant\"}").expect("missing content");
        assert_eq!(message.content.text, None);
        assert!(!message.content.has_image);
    }

    /// Random text/image part sequences: text parts flatten in order with
    /// no separator; `has_image` is exactly "at least one image part"; each
    /// image ref keeps its URL in request order and its BYTE OFFSET in the
    /// flattened text; markers spliced at those offsets land at
    /// `offset + i * marker_len` (marker-free alphabet: offsets are exact).
    #[test]
    fn part_sequences_flatten_in_order_with_exact_offsets() {
        const MARKER: &str = "<__media__>";
        for seed in 0..256u64 {
            let mut rng = Lcg::new(seed);
            let mut parts = Vec::new();
            let mut expected_text = String::new();
            let mut expected_images: Vec<(String, usize)> = Vec::new();
            for _ in 0..=rng.index(8) {
                if rng.index(3) == 0 {
                    let url = format!("data:image/png;base64,s{seed}i{}", expected_images.len());
                    expected_images.push((url.clone(), expected_text.len()));
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": {"url": url}
                    }));
                } else {
                    let text = random_string(&mut rng, &MARKER_FREE_ATOMS);
                    expected_text.push_str(&text);
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
            }

            let content: ChatContent =
                serde_json::from_value(serde_json::Value::Array(parts)).expect("parts parse");
            assert_eq!(content.has_image, !expected_images.is_empty());
            let flattened = content.text.clone().unwrap_or_default();
            if expected_text.is_empty() {
                assert_eq!(content.text, None);
            } else {
                assert_eq!(flattened, expected_text, "seed {seed}: flattening order");
            }
            assert_eq!(content.images.len(), expected_images.len());
            for (image, (url, offset)) in content.images.iter().zip(expected_images.iter()) {
                assert_eq!(&image.url, url);
                assert_eq!(
                    image.text_offset, *offset,
                    "byte offset of part in flat text"
                );
            }

            // Markers splice exactly at the recorded offsets (inserting in
            // reverse order shifts image i's marker by i marker lengths).
            let spliced = content.text_with_media_markers(MARKER);
            let marker_len = MARKER.len();
            assert_eq!(
                spliced.len(),
                flattened.len() + expected_images.len() * marker_len,
                "seed {seed}: exactly one marker per image"
            );
            for (i, (_, offset)) in expected_images.iter().enumerate() {
                let at = offset + i * marker_len;
                assert_eq!(
                    &spliced[at..at + marker_len],
                    MARKER,
                    "seed {seed}: marker {i} at its part offset"
                );
            }
        }
    }

    /// Marker-collision robustness: texts CONTAINING the media marker never
    /// panic the splice and always add exactly one marker per image to the
    /// byte length.
    #[test]
    fn media_marker_inside_text_never_panics() {
        const MARKER: &str = "<__media__>";
        for seed in 0..128u64 {
            let mut rng = Lcg::new(seed);
            let mut parts = Vec::new();
            let mut flattened_len = 0usize;
            let mut images = 0usize;
            for i in 0..=rng.index(6) {
                if i % 2 == 0 {
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": {"url": format!("data:x,s{seed}i{i}")}
                    }));
                    images += 1;
                } else {
                    let text = random_string(&mut rng, &ATOMS);
                    flattened_len += text.len();
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
            }
            let content: ChatContent =
                serde_json::from_value(serde_json::Value::Array(parts)).expect("parts parse");
            let spliced = content.text_with_media_markers(MARKER);
            assert_eq!(
                spliced.len(),
                flattened_len + images * MARKER.len(),
                "seed {seed}: splice length is exact even with marker-bearing text"
            );
        }
    }
}

/// Recognition surface kind (`POST /v1/recognize`, recognition tickets).
/// Snake-case wire names are the worker crate's `RecKind::as_str` spellings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum RecKind {
    /// Barcode/QR decode (symbology + payload + corner-derived bbox).
    Barcode,
    /// Document/scene OCR, served by the in-process `ocrs`/`rten` backend
    /// when weights are present; otherwise a named 501.
    Ocr,
}

/// One externally discovered engine (TASK-external-tool-registry). Every
/// field is derived from the startup/reload probe; nothing is user-supplied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ExternalTool {
    /// Stable lowercase tool id (`"blender"` / `"comfyui"`).
    pub id: String,
    /// Version the tool reported; absent when it never reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Capabilities the probe actually proved: the tool capabilities plus the
    /// probed add-on (`addon:<id>`) and headless operator (`op:<id>`) ids.
    pub capabilities: Vec<String>,
    /// Probed device names; never credentials or paths.
    pub devices: Vec<String>,
    /// Whether the tool can serve work right now.
    pub available: bool,
    /// Why it cannot serve work; absent when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable: Option<String>,
}

/// Externally discovered engines reported by `GET /v1/external-tools` and
/// the top-level capabilities document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ExternalToolsCapability {
    /// Every known tool, in the registry's stable order, available or not.
    pub tools: Vec<ExternalTool>,
}

/// Recognition capability (`Capabilities::recognition`): probe-driven flags
/// per kind — a `true` here is the same compile probe the route's 501/200
/// behavior keys on, so the document cannot drift from the binary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct RecognitionFeature {
    /// Barcode/QR decode is compiled and served by `POST /v1/recognize`.
    pub barcode: bool,
    /// OCR is compiled and its weights actually loaded (`serves(Ocr)`).
    /// Both OCR routes key their 501 on this same predicate — honest
    /// absence, never a half seam. The `ocrs` backend emits no confidence;
    /// see `OcrEngineMeta`.
    pub ocr: bool,
}

/// One recognition request over one image (`POST /v1/recognize`).
///
/// Stage 1 is synchronous: barcode-class latency is sub-millisecond.
/// Document ingest is the job-based `POST /v1/ocr/ingest`; this request
/// shape is its already-compatible single-image prefix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RecognizeRequest {
    /// Which recognition work to run.
    pub kind: RecKind,
    /// One image as a data URI (`data:image/png;base64,…`) — no paths, no
    /// URLs, nothing is ever fetched (no SSRF surface, same policy as
    /// `/v1/image-ops`).
    pub image: String,
    /// Language hint (BCP-47); barcode decode ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Mode hint (`text_layer` | `layout_regions` | `full_page` |
    /// `replace_text_layer`); barcode decode ignores it. Unparsed beyond
    /// presence in stage 1 — the OCR ticket owns the vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Optional engine override (`"rxing"`, `"vlm-mtmd"`, …); `None` = the
    /// server's default backend for the kind. Unknown engines answer the
    /// named 501 naming the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
}

/// One located region: axis-aligned pixel bbox + text. `confidence` is
/// absent when the engine produces none (rxing does not).
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct RecognitionRegion {
    /// Pixel bbox `[x_min, y_min, x_max, y_max]` in source-image
    /// coordinates; meaningful only when the engine's `bbox_support` is
    /// `true`.
    pub bbox: [f32; 4],
    /// Recognized text for this region (barcode: the full payload).
    pub text: String,
    /// Engine confidence in `0.0..=1.0`, absent when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// One page of recognition output; barcode decode yields exactly one.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct RecognitionPage {
    /// Full recognized text.
    pub text: String,
    /// Located regions; empty when the engine cannot localize or nothing
    /// was found.
    pub regions: Vec<RecognitionRegion>,
}

/// Which engine produced a result, and what its output honestly contains.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct RecognitionEngineMeta {
    /// Stable backend id (`"rxing"`, `"stub-recognition"`, …).
    pub id: String,
    /// Engine dependency version this build linked; absent when unknown
    /// (the stub has none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// `true` only when `regions[].bbox` carries real coordinates — a
    /// client must not treat regions as located otherwise (VLM rule).
    pub bbox_support: bool,
}

/// Recognition result. `pages: []` is the honest "processed, found
/// nothing" answer — never an error for a valid image.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct RecognizeResult {
    /// The kind that was served.
    pub kind: RecKind,
    /// Producing engine metadata.
    pub engine: RecognitionEngineMeta,
    /// One entry per processed page.
    pub pages: Vec<RecognitionPage>,
}

/// Ingest mode (`POST /v1/ocr/ingest`). The vocabulary is closed; an
/// unknown name is a typed deserialization 400.
///
/// - `text_layer`: extract the existing PDF text layer (no rendering, no
///   OCR engine inference), one region per non-empty text line, no boxes.
/// - `replace_text_layer`: same extraction, for clients that overlay the
///   recovered text onto a broken layer; honkbox never rewrites the PDF.
/// - `layout_regions` / `full_page`: need rasterization. PDF inputs answer
///   the named 501 `ocr_pdf_raster_unavailable` — this pure-Rust build has
///   no MIT/Apache PDF rasterizer. Image inputs run `ocrs` in-process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum OcrIngestMode {
    /// Extract the existing PDF text layer; no rendering and no OCR.
    TextLayer,
    /// Raster-based layout analysis (image inputs only in this build).
    LayoutRegions,
    /// Whole-page OCR with word boxes (image inputs only in this build).
    FullPage,
    /// Recover text for clients to overlay onto a broken PDF layer.
    ReplaceTextLayer,
}

impl OcrIngestMode {
    /// Stable wire spelling, also used in the cache key and job metadata.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextLayer => "text_layer",
            Self::LayoutRegions => "layout_regions",
            Self::FullPage => "full_page",
            Self::ReplaceTextLayer => "replace_text_layer",
        }
    }

    /// Whether this mode needs a PDF rasterizer (unavailable in this build).
    #[must_use]
    pub fn needs_raster(self) -> bool {
        matches!(self, Self::LayoutRegions | Self::FullPage)
    }
}

/// Start one OCR ingest job (`POST /v1/ocr/ingest`).
///
/// Input is PATH-FREE by design: a single `data:` URI — either
/// `data:application/pdf;base64,…` or `data:image/<container>;base64,…`.
/// Nothing is fetched from a path, URL or artifact store, so there is no
/// SSRF or host-path surface. The decoded bytes are capped by the server;
/// `dpi` is bounded and recorded in the cache key even when the current
/// backend does not need it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct OcrIngestRequest {
    /// Path-free input: `data:application/pdf;base64,…` or an image data URI.
    pub data_uri: String,
    /// Which extraction/recognition mode to run.
    pub mode: OcrIngestMode,
    /// BCP-47 language hint; `None` = backend default (Latin for `ocrs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Requested render DPI, bounded by the server; unused by the PDF text
    /// layer and by `ocrs` (which reads native pixels) but part of the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u32>,
}

/// One OCR region. A PDF text layer has no boxes, so `bbox` is optional;
/// `confidence` is absent because `ocrs` 0.14 emits no score (never guessed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
pub struct OcrRegion {
    /// `[x_min, y_min, x_max, y_max]` in page pixels, or `None` when the
    /// source has no geometry (PDF text layer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f32; 4]>,
    /// Recognized text for this region (one text line for a PDF layer).
    pub text: String,
    /// Always absent in this build: `ocrs` does not expose a per-word score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// One processed page of an OCR ingest result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
pub struct OcrPage {
    /// Zero-based page index in source order.
    pub index: u32,
    /// Full page text (regions joined with `\n`, in deterministic order).
    pub text: String,
    /// Located regions, in reading order; `bbox`/`confidence` may be absent.
    pub regions: Vec<OcrRegion>,
}

/// The engine that produced an [`OcrResult`], with an honest account of
/// what its output contains.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct OcrEngineMeta {
    /// Stable backend id (`"ocrs-rten"` when OCR weights are loaded).
    pub id: String,
    /// Dependency/model version pin, absent when the engine reports none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Whether region boxes are real coordinates (false for PDF text layer).
    pub bbox_support: bool,
    /// Always `false`: `ocrs` 0.14 has no per-word confidence API; a client
    /// must not interpret missing scores as zero or full confidence.
    pub confidence_support: bool,
}

/// Result of one OCR ingest job, also stored in the job bundle as
/// `ocr_result.json`. Replaying an unchanged request sets `cache_hit` and
/// consumes no admission lease.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS, ToSchema)]
pub struct OcrResult {
    /// Producing engine and the honesty flags for its output.
    pub engine: OcrEngineMeta,
    /// Mode that actually ran.
    pub mode: OcrIngestMode,
    /// Number of pages processed before completion or cancellation.
    pub page_count: u32,
    /// Processed pages, in source order.
    pub pages: Vec<OcrPage>,
    /// `true` when a bounded TTL cache replayed a previous identical job.
    pub cache_hit: bool,
    /// `true` when the job's cancel token stopped it between pages.
    pub cancelled: bool,
}

/// `202` acceptance body (`POST /v1/ocr/ingest`). The job id is also sent
/// as `X-Job-Id`, so it can be cancelled through `POST /v1/jobs/{id}/cancel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
pub struct OcrJobAccepted {
    /// Registry id of the accepted job.
    pub job_id: String,
    /// Fixed `"queued"` acceptance state.
    pub status: String,
}

/// Texture post-processing capability (`Capabilities::texture_postprocess`):
/// whether the server can run texture conversion passes, and the tool
/// versions it found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct TexturePostprocessCapability {
    /// `true` when the texture post-processing tools are present and
    /// usable.
    pub available: bool,
    /// Version of the `oiiotool` binary, when found.
    pub oiiotool_version: Option<String>,
    /// Version of the `maketx` binary, when found.
    pub maketx_version: Option<String>,
}

/// NSFW classification capability (`Capabilities::nsfw`): the policy in
/// effect, the classifier model version, and the score threshold above
/// which content is flagged.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct NsfwCapability {
    /// `true` when NSFW classification is available in this build.
    pub available: bool,
    /// Policy applied to scores (for example `"flag"` or `"block"`).
    pub policy: String,
    /// Version of the classifier model in use.
    pub model_version: String,
    /// Score threshold above which content is flagged.
    pub threshold: f64,
}

/// External mesh capability (`Capabilities::mesh`): the external mesh
/// generators/upstreams this server knows about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct MeshExternalCapability {
    /// `true` when the external mesh surface is enabled.
    pub enabled: bool,
    /// Number of reachable upstreams.
    pub upstreams: usize,
    /// Known external mesh model ids.
    pub models: Vec<String>,
    /// Age of the last successful observation, Unix milliseconds; absent
    /// when no observation has been made yet.
    pub last_observation_ms: Option<u64>,
    /// `true` when the last observation is older than the freshness
    /// bound, i.e. the upstream view may be out of date.
    pub stale: bool,
}

/// Signed-URL capability (`Capabilities::signed_urls`): whether artifact
/// URLs can be signed, and how long a signature stays valid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct SignedUrlCapability {
    /// `true` when signed URLs are enabled.
    pub enabled: bool,
    /// Signature time-to-live in seconds.
    pub ttl_secs: u64,
}

/// Scope capability (`Capabilities::scopes`): whether scoped credentials
/// are enabled, and which scopes the server recognizes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ScopeCapability {
    /// `true` when scoped credentials are enabled.
    pub enabled: bool,
    /// Recognized scope names.
    pub scopes: Vec<String>,
}

/// One NSFW annotation attached to a job's output (`JobEvent::annotations`).
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct NsfwAnnotation {
    /// Classifier score for the annotated content (higher is more likely
    /// NSFW).
    pub score: f64,
    /// Classes the classifier assigned.
    pub classes: Vec<String>,
    /// Version of the classifier model that produced the annotation.
    pub model_version: String,
}

/// One queued job (ledger ticket) in a [`QueueSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct QueueJob {
    /// Public job id, when the ticket has one; absent for queued ledger
    /// tickets that have not been admitted yet.
    pub id: Option<String>,
    /// Job kind (for example `"chat"` or `"images"`).
    pub kind: String,
    /// Model slug the job runs against, when known.
    pub model: Option<String>,
    /// Current lifecycle state of the ticket.
    pub state: String,
    /// Zero-based position in the queue.
    pub position: usize,
    /// Time-to-live of the queued ticket in seconds.
    pub ttl_secs: u64,
}

/// Queue snapshot: the currently queued tickets plus the aggregate queue
/// depth and GPU slot count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct QueueSnapshot {
    /// Queued tickets, in queue order.
    pub jobs: Vec<QueueJob>,
    /// Number of GPU slots the server has.
    pub gpu_slots: u64,
    /// Aggregate pending count from the ledger queue depth.
    pub queued: u64,
    /// Current queue wait bound in seconds.
    pub queue_wait_secs: u64,
}

/// Rerank request: a query scored against a set of texts or documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct RerankRequest {
    /// The query the candidates are scored against.
    pub query: String,
    /// Candidate texts to rerank.
    pub texts: Option<Vec<String>>,
    /// Candidate documents to rerank.
    pub documents: Option<Vec<String>>,
    /// How many results to keep.
    pub top_n: Option<usize>,
    /// Rerank model slug, when the caller wants a specific one.
    pub model: Option<String>,
    /// Whether each result carries the original text back.
    pub return_text: Option<bool>,
}

/// One reranked candidate.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct RerankItem {
    /// Index of the candidate in the request's input order.
    pub index: usize,
    /// Relevance score for the candidate.
    pub score: f64,
    /// The candidate's text, present only when `return_text` was
    /// requested.
    pub text: Option<String>,
}

/// Rerank result: the scored candidates plus the model that produced
/// them.
#[derive(Debug, Clone, PartialEq, Serialize, TS, ToSchema)]
pub struct RerankResult {
    /// Scored candidates, most relevant first.
    pub results: Vec<RerankItem>,
    /// The rerank model that produced the scores.
    pub model: String,
}

/// One Fizgig `LoRA` training request (`POST /v1/fizgig/train`).
///
/// All paths are operator-confined: the server resolves and allowlists them
/// before any subprocess starts. The route admits the job asynchronously and
/// answers `202` with a job id; lifecycle is polled at
/// `GET /v1/fizgig/jobs/{id}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct FizgigTrainRequest {
    /// Fizgig family id (for example `sdxl`); must be allowlisted.
    pub family: String,
    /// Directory of images plus `.txt` caption sidecars.
    pub dataset_dir: String,
    /// Base checkpoint used as `--dit`.
    pub dit: String,
    /// Optional VAE path; SDXL may use the checkpoint's own VAE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vae: Option<String>,
    /// Optional text-encoder path; SDXL reads it from the checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_encoder: Option<String>,
    /// Output basename (letters, digits, `_` and `-` only).
    pub output_name: String,
    /// Training epochs (1..=1000).
    pub epochs: u32,
    /// `LoRA` rank (1..=256), used for `--network_dim` and `--network_alpha`.
    pub rank: u32,
    /// Seed shared by training and previews.
    #[serde(default)]
    pub seed: u64,
}

/// Acknowledgement for an admitted Fizgig training job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct FizgigTrainAccepted {
    /// Id used by the status and cancel routes.
    pub job_id: String,
    /// Lifecycle state at acknowledgement, always `queued`.
    pub state: String,
}

/// Live status of one Fizgig training job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct FizgigTrainStatus {
    /// The job id from [`FizgigTrainAccepted`].
    pub job_id: String,
    /// `queued`, `running`, `completed`, `failed` or `cancelled`.
    pub state: String,
    /// Current stage: `latents`, `text`, `train`, `done` or `none`.
    pub stage: String,
    /// Bundle directory relative to the data dir, once artifacts land.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    /// Bounded failure reason, when the run failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One operator-declared managed CLI app's live state (`GET /v1/apps`).
/// Every field is derived from config plus the manager's observed process
/// state; no path or binary from an HTTP request can appear here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ManagedApp {
    /// Stable app id (`llama_server` / `comfyui`).
    pub id: String,
    /// `stopped`, `starting`, `ready`, `failed` or `stopping`.
    pub state: String,
    /// Launch mode: `path`, `installation` or `script`.
    pub launch: String,
    /// Ledger kind the held lease reserves: `llm` or `comfy`.
    pub resource_kind: String,
    /// Loopback URL or `unix:` socket path while the app can serve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Version proved by the start-time `--version` probe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Capped failure reason (including a stderr tail), absent when healthy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Declared managed apps and their live state (`GET /v1/apps`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ManagedAppList {
    /// All declared apps, including disabled ones (reported as `stopped`).
    pub apps: Vec<ManagedApp>,
}

/// Honest `ComfyUI` reachability (`GET /comfyui/status`). `true` requires a
/// passing probe and a live process for a managed app.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ComfyUiStatus {
    /// Whether the typed `/comfyui` surface can serve right now.
    pub available: bool,
    /// Managed state (`stopped`/`starting`/`ready`/`failed`/`stopping`) or
    /// `detached` when no managed app is declared.
    pub state: String,
    /// Truth source: `managed` (health probe at start) or `external-tools`
    /// (the startup registry probe).
    pub source: String,
    /// Version from the same probe that produced `available`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Capabilities proved by that probe (never a hand-written claim).
    pub capabilities: Vec<String>,
    /// Always `true`: the upstream is validated loopback-only at config load.
    pub loopback_only: bool,
}

/// Typed queue summary read from `ComfyUI`'s internal `/queue` (`GET
/// /comfyui/queue`). No node graph is ever returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ComfyUiQueue {
    /// Number of currently executing prompts.
    pub running: usize,
    /// Number of queued prompts.
    pub pending: usize,
}

/// Progress extracted from `ComfyUI`'s internal history response. No raw
/// workflow JSON crosses the bearer boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ComfyUiProgress {
    /// Last executing node, when the history record names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// Completed step count, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<usize>,
    /// Total step count, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
}

/// Typed history summary for a prompt id (`GET /comfyui/history/{id}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct ComfyUiHistory {
    /// The validated prompt id from the path.
    pub prompt_id: String,
    /// `queued`, `running` or `completed` (from history status).
    pub status: String,
    /// Output node ids present in the history record.
    pub outputs: Vec<String>,
    /// Best-effort progress, absent when the record carries none.
    pub progress: ComfyUiProgress,
}
