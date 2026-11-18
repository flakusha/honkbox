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
/// Everything except `"none"` is clamped to `auto` behaviour (tools are
/// offered; the model decides) — no rejection.
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
    /// True when tool calls must not be offered (`"none"`). Any other
    /// value — `auto`, `required`, `any`, named — samples with tools
    /// available (clamped to `auto`).
    #[must_use]
    pub fn suppresses_tools(&self) -> bool {
        matches!(self, Self::Mode(mode) if mode == "none")
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
/// - **Clamped**: `tool_choice` `"required"`/`"any"`/named function all
///   behave as `"auto"` (tools offered, model decides).
/// - **Accepted and ignored**: `parallel_tool_calls` — one tool call is
///   produced per turn regardless; the field exists so clients that always
///   send it are not rejected.
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
    /// behavior; everything else is clamped to `auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceSpec>,
    /// Accepted and ignored: honkbox produces at most one tool call per
    /// turn. Present so parallel-`true` clients are not rejected.
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

/// `POST /v1/images/generations` response: Unix `created` seconds plus one
/// `data` entry per requested image (`n`).
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct ImagesGenerationResponse {
    /// Creation time, Unix seconds.
    pub created: u64,
    /// One entry per requested image, in order.
    pub data: Vec<ImageB64>,
    /// Advisory notes (`LoRA` license surfacing); omitted when empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
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

/// `POST /v1/audio/speech` request (REQ-9). `model` is optional (server
/// default TTS); `response_format` accepts `wav` (default, RIFF/WAVE
/// container) or `pcm` (raw little-endian signed 16-bit mono samples,
/// matching the `OpenAI` pcm semantics — no container, no header); anything
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
    /// Output format: `wav` (default) or `pcm` (raw s16le, no container).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

/// One TTS voice entry of `GET /v1/audio/voices` (`data[]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct Voice {
    /// Voice id to pass as `/v1/audio/speech` `voice` (e.g. `af_heart`).
    pub id: String,
    /// Always `"voice"` (OpenAI-style entry tag).
    pub object: String,
    /// Language/locale tag derived from the Kokoro voice prefix
    /// (e.g. `en-US`).
    pub language: String,
    /// `female` or `male`, from the Kokoro prefix.
    pub gender: String,
}

/// `GET /v1/audio/voices` response: OpenAI-style list envelope over the
/// statically documented Kokoro voice set (the backend cannot enumerate
/// voices at runtime, so this is the *supported* set — served without any
/// model loaded).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS, ToSchema)]
pub struct VoiceList {
    /// Always `"list"`.
    pub object: String,
    /// The supported voices.
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
    /// `synthesize`, or `translations`.
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

/// `POST /v1/jobs/{id}/cancel` response (NEW-3). `cancelled` mirrors the
/// job's `interruptible` flag: `true` = the job will actually stop at its
/// next checkpoint; `false` = the job was found and marked, but its single
/// blocking inference call is not interruptible mid-flight. An unknown id
/// (finished or never registered) is a plain-text 404.
#[derive(Debug, Clone, Serialize, TS, ToSchema)]
pub struct JobCancelResult {
    /// Whether the cancel will stop the job at its next work-unit boundary.
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
}

impl JobEventType {
    /// SSE `event:` line label (mirrored in the JSON `event` field).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Registered => "job_registered",
            Self::Cancelled => "job_cancelled",
            Self::Finished => "job_finished",
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
    /// `synthesize`, `translations`).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regenerate the checked-in TS bindings (drift is gated by `git diff` in CI).
    #[test]
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
            ImagesGenerationResponse::decl(&config),
            TranscriptionText::decl(&config),
            TranscriptionVerbose::decl(&config),
            SpeechRequest::decl(&config),
            TranscriptSegment::decl(&config),
            Transcript::decl(&config),
            SynthesizeRequest::decl(&config),
            Voice::decl(&config),
            VoiceList::decl(&config),
            ModelDeleteResult::decl(&config),
            ModelKind::decl(&config),
            ModelCapabilities::decl(&config),
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
            JobInfo::decl(&config),
            JobList::decl(&config),
            JobCancelResult::decl(&config),
            JobEventType::decl(&config),
            JobEvent::decl(&config),
            Pill::decl(&config),
            CatalogLocal::decl(&config),
            CatalogEntry::decl(&config),
            CatalogList::decl(&config),
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
        ] {
            let caps = ModelCapabilities::for_kind(kind);
            assert!(!caps.chat && !caps.completion && !caps.embeddings);
        }
    }
}
