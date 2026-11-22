//! OpenAI-shaped error envelope and `REQ-3` constraint resolution,
//! extracted from `routes.rs`.
//!
//! The router registrations stay in `routes.rs`; only the error helper and
//! the constraint resolver live here.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use honkbox_api_types::{
    OpenAiChatRequest, OpenAiError, ResponseFormat, ResponseFormatType, ToolChoiceSpec,
};
use honkbox_llm_worker::{LlmError, SamplerParams};

/// OpenAI-shaped error body → JSON response (`/v1/chat/completions` only;
/// the native route keeps its plain-text errors).
pub fn openai_error_response(status: StatusCode, message: String, code: Option<&str>) -> Response {
    let body = OpenAiError::new(status.as_u16(), message, code.map(str::to_owned));
    (status, Json(body)).into_response()
}

/// Classify one worker failure the same way on the sync and SSE paths.
/// `Context` is a client input failure (prompt does not fit `n_ctx`), so a
/// loop-lore SDK sees 400 `context_length_exceeded`, never a misleading 500.
pub(super) fn llm_error_class(err: &LlmError) -> (StatusCode, &'static str, String) {
    match err {
        LlmError::InvalidRequest(_) => (
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("chat failed: {err}"),
        ),
        LlmError::Vision(_) => (
            StatusCode::BAD_REQUEST,
            "vision_unsupported",
            format!("chat failed: {err}"),
        ),
        LlmError::Context(_) => (
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            format!("chat failed: {err}"),
        ),
        LlmError::Upstream { status, .. } => (
            if *status == 503 {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_GATEWAY
            },
            if *status == 503 {
                "upstream_unavailable"
            } else {
                "upstream_error"
            },
            format!("chat failed: {err}"),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            format!("chat failed: {err}"),
        ),
    }
}

/// Resolved `REQ-3` constraint fields for one completions request.
pub(super) struct ChatConstraints {
    /// Raw GBNF grammar, honored verbatim.
    pub(super) grammar: Option<String>,
    /// JSON Schema text (`response_format: json_schema`); the worker
    /// converts it to GBNF (unconvertible schema = worker `InvalidRequest`
    /// = honest 400).
    pub(super) json_schema: Option<String>,
    /// `response_format: json_object` → generic any-JSON grammar.
    pub(super) json_object: bool,
    /// `OpenAI` tools array JSON; `None` = no tools offered.
    pub(super) tools_json: Option<String>,
    /// `tool_choice` clamped: only `"none"` survives (everything else
    /// behaves as `auto`). The clamp is disclosed machine-readably on
    /// `GET /v1/capabilities` (`tools`) — never a silent downgrade.
    pub(super) tool_choice_none: bool,
    /// Extended sampler knobs, wired when present.
    pub(super) sampler: SamplerParams,
}

/// Resolve the `REQ-3` request fields into worker-protocol constraints.
/// Every field is enforced, clamped, or rejected with an `OpenAI`
/// envelope — never silently ignored: output constraints (`grammar`,
/// `response_format`) and `tools` are mutually exclusive (llama.cpp
/// itself rejects the combination; dropping one side would be fake
/// support), and so are `grammar` and `response_format` with each other.
/// An empty conversation is rejected here too: the llama backend refuses
/// it (`InvalidRequest`), so the route answers 400 up front rather than
/// surfacing a backend-specific 500 later.
///
/// Accepted `tool_choice` vocabulary (`ToolChoiceSpec::HONORED_MODES` /
/// `CLAMPED_MODES`): `"none"` and `"auto"` are honored verbatim;
/// `"required"`, `"any"`, and the named-function form are accepted and
/// clamped to `"auto"` — disclosed on `GET /v1/capabilities` (`tools`),
/// never a 400. The vendored `llama-cpp-4` binding also exposes a
/// grammar-enforced `ToolChoice::Required`; it stays clamped here pending
/// an exact-choice consumer (TASK-llm-tool-choice-contract follow-up).
pub(super) fn resolve_constraints(
    req: &OpenAiChatRequest,
) -> Result<ChatConstraints, (&'static str, String)> {
    if req.messages.is_empty() {
        return Err((
            "messages_required",
            "messages is required: pass at least one conversation turn".to_owned(),
        ));
    }
    let tools_json = match req.tools.as_deref() {
        None | Some([]) => None,
        Some(tools) => Some(
            serde_json::to_string(tools)
                .map_err(|err| ("unsupported_constraint", format!("tools: {err}")))?,
        ),
    };
    let mut json_schema = None;
    let mut json_object = false;
    match &req.response_format {
        Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: Some(spec),
        }) => {
            if spec.schema.is_null() {
                return Err((
                    "unsupported_constraint",
                    "response_format.json_schema.schema is required".to_owned(),
                ));
            }
            json_schema = Some(spec.schema.to_string());
        }
        Some(ResponseFormat {
            format_type: ResponseFormatType::JsonSchema,
            json_schema: None,
        }) => {
            return Err((
                "unsupported_constraint",
                "response_format.json_schema is required for type json_schema".to_owned(),
            ));
        }
        Some(ResponseFormat {
            format_type: ResponseFormatType::JsonObject,
            ..
        }) => json_object = true,
        _ => {}
    }
    if tools_json.is_some() && (req.grammar.is_some() || json_schema.is_some() || json_object) {
        return Err((
            "unsupported_constraint",
            "tools combined with grammar or response_format is not supported: the tool-call constraint and the output constraint are mutually exclusive"
                .to_owned(),
        ));
    }
    if req.grammar.is_some() && (json_schema.is_some() || json_object) {
        return Err((
            "unsupported_constraint",
            "grammar and response_format are mutually exclusive: pass one output constraint"
                .to_owned(),
        ));
    }
    Ok(ChatConstraints {
        grammar: req.grammar.clone(),
        json_schema,
        json_object,
        tools_json,
        tool_choice_none: req
            .tool_choice
            .as_ref()
            .is_some_and(ToolChoiceSpec::suppresses_tools),
        sampler: SamplerParams {
            top_k: req.top_k,
            min_p: req.min_p,
            typical_p: req.typical_p,
            repeat_penalty: req.repeat_penalty,
            xtc_probability: req.xtc_probability,
            dry_multiplier: req.dry_multiplier,
            dry_base: req.dry_base,
            dry_allowed_length: req.dry_allowed_length,
            dynatemp_range: req.dynatemp_range,
            dynatemp_exponent: req.dynatemp_exponent,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_status_maps_to_gateway_errors() {
        let loading = LlmError::Upstream {
            status: 503,
            message: "loading".to_owned(),
        };
        assert_eq!(llm_error_class(&loading).0, StatusCode::SERVICE_UNAVAILABLE);
        let gone = LlmError::Upstream {
            status: 404,
            message: "gone".to_owned(),
        };
        assert_eq!(llm_error_class(&gone).0, StatusCode::BAD_GATEWAY);
    }
}
