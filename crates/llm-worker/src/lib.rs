//! LLM sessions (llama-cpp flavors, token streaming); sync only, no HTTP/DB.

mod backend;

pub use backend::LlamaCppLlm;

use std::fmt;
use std::time::Instant;

/// Role of a single chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Human / caller turn.
    User,
    /// Model turn.
    Assistant,
    /// System instructions.
    System,
}

/// One chat message: a role plus its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Who said it.
    pub role: Role,
    /// Message text.
    pub content: String,
}

/// Chat completion request (the server converts its own DTOs into this).
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Model id (informational; [`LlamaCppLlm`] already holds its loaded file).
    pub model: String,
    /// Conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate (default 512).
    pub max_tokens: Option<u32>,
    /// Sampling temperature (default 0.7).
    pub temperature: Option<f32>,
}

/// Chat completion response.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Which backend produced this (`stub-llm` | `llama-cpp-vulkan` | `llama-cpp-cuda`).
    pub backend_id: String,
    /// Generated text.
    pub text: String,
    /// Number of generated tokens.
    pub tokens: u32,
    /// Wall-clock time in milliseconds.
    pub millis: u64,
}

/// Options for [`LlamaCppLlm::load`].
#[derive(Debug, Clone)]
pub struct LlmOptions {
    /// Context size in tokens.
    pub n_ctx: u32,
    /// GPU layers to offload (`u32::MAX` = all).
    pub n_gpu_layers: u32,
    /// CPU threads.
    pub threads: u32,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self {
            n_ctx: 8192,
            n_gpu_layers: u32::MAX,
            threads: std::thread::available_parallelism()
                .map_or(4, |n| u32::try_from(n.get()).unwrap_or(u32::MAX)),
        }
    }
}

/// Errors from [`LlmPipeline`] backends (hand-rolled, no thiserror).
#[derive(Debug)]
pub enum LlmError {
    /// Model file missing, backend init failed, or load failed.
    Load(String),
    /// Chat template rendering failed.
    ChatTemplate(String),
    /// Tokenization failed.
    Tokenizer(String),
    /// Context creation failed, or the prompt does not fit `n_ctx`.
    Context(String),
    /// Prompt decode or token generation failed.
    Decode(String),
    /// Request is unusable (e.g. no messages).
    InvalidRequest(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(e) => write!(f, "failed to load model: {e}"),
            Self::ChatTemplate(e) => write!(f, "chat template failed: {e}"),
            Self::Tokenizer(e) => write!(f, "tokenizer failed: {e}"),
            Self::Context(e) => write!(f, "context failed: {e}"),
            Self::Decode(e) => write!(f, "decode failed: {e}"),
            Self::InvalidRequest(e) => write!(f, "invalid request: {e}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Sync LLM pipeline: full completion plus token-streaming completion.
pub trait LlmPipeline: Send + Sync {
    /// Run one chat completion.
    fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
    /// Run one chat completion, pushing each decoded piece to `sink` as it arrives.
    fn complete_stream(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError>;
    /// Backend id (`stub-llm` | `llama-cpp-vulkan` | `llama-cpp-cuda`).
    fn backend_id(&self) -> &'static str;
}

/// Deterministic stub backend: canned text, no model, no GPU.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubLlm;

impl LlmPipeline for StubLlm {
    fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let start = Instant::now();
        let text = match req.messages.last() {
            Some(m) => format!("stub reply to: {}", m.content),
            None => "stub reply".to_string(),
        };
        Ok(ChatResponse {
            backend_id: self.backend_id().to_string(),
            tokens: u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX),
            millis: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            text,
        })
    }

    fn complete_stream(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        let resp = self.complete(req)?;
        sink(&resp.text);
        Ok(resp)
    }

    fn backend_id(&self) -> &'static str {
        "stub-llm"
    }
}

const _: fn() = || {
    fn assert_pipeline<T: LlmPipeline>() {}
    assert_pipeline::<StubLlm>();
    assert_pipeline::<LlamaCppLlm>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn req() -> ChatRequest {
        ChatRequest {
            model: "stub".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".to_string(),
            }],
            max_tokens: None,
            temperature: None,
        }
    }

    #[test]
    fn stub_round_trip() {
        let llm = StubLlm;
        assert_eq!(llm.backend_id(), "stub-llm");
        let resp = llm.complete(&req()).unwrap();
        assert_eq!(resp.backend_id, "stub-llm");
        assert!(resp.text.contains("hello"));

        let mut pieces = Vec::new();
        let streamed = llm
            .complete_stream(&req(), &mut |s: &str| pieces.push(s.to_string()))
            .unwrap();
        assert_eq!(streamed.text, resp.text);
        assert_eq!(streamed.text, pieces.concat());
    }

    #[test]
    fn load_missing_file_errors() {
        let err = LlamaCppLlm::load(Path::new("/nonexistent/model.gguf"), &LlmOptions::default())
            .unwrap_err();
        assert!(matches!(err, LlmError::Load(_)));
    }
}
