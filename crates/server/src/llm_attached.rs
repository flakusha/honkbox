//! Attached `llama-server` LLM backend (FEAT-llama-server-client-backend).
//!
//! The operator owns the process: this backend only talks to an already
//! running, health-proved server over loopback HTTP. It implements the sync
//! `LlmPipeline` seam with reqwest's blocking client (which runs on its own
//! runtime thread, so construction inside axum is safe).
//!
//! A generated per-app API key lives only in the client's default headers;
//! honkbox's bearer never crosses this boundary.

use std::io::Read;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use honkbox_llm_worker::{
    ChatRequest, ChatResponse, LlmError, LlmPipeline, RequestCancel, Role, ToolCallOut,
};
use serde_json::{Value, json};

/// Backend id reported to callers and capabilities.
pub const BACKEND_ID: &str = "llama-server-http";

const MAX_RESPONSE_BYTES: u64 = 1 << 20;
const MAX_SSE_LINE: usize = 64 * 1024;
const TIMEOUT: Duration = Duration::from_secs(300);
/// How often a stalled SSE read yields to the per-request cancel check.
const CANCEL_POLL: Duration = Duration::from_millis(25);

/// A proved, loopback-only `llama-server` endpoint.
pub struct AttachedLlm {
    base: String,
    model: String,
    client: reqwest::blocking::Client,
    /// Async twin used only for SSE: its read future can be dropped on
    /// cancellation, which closes the upstream socket immediately.
    async_client: reqwest::Client,
}

impl std::fmt::Debug for AttachedLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttachedLlm")
            .field("base", &self.base)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl AttachedLlm {
    /// Probe `/health` and `/v1/models`, then build the client.
    ///
    /// A failing probe is a loud `Upstream` error so startup can exit before
    /// advertising a backend that cannot serve. Non-loopback URLs are refused:
    /// the app key must never travel over an untrusted network.
    pub fn new(base: &str, model: Option<&str>) -> Result<Self, LlmError> {
        let parsed = reqwest::Url::parse(base)
            .map_err(|err| upstream(502, format!("invalid llama-server URL: {err}")))?;
        let loopback = matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
        if parsed.scheme() != "http"
            || !loopback
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some_and(|query| !query.is_empty())
            || parsed
                .fragment()
                .is_some_and(|fragment| !fragment.is_empty())
        {
            return Err(upstream(
                502,
                "attached llama-server URL must be credential-free loopback HTTP without query or fragment",
            ));
        }
        let base = parsed.to_string().trim_end_matches('/').to_owned();
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(key) = std::env::var("LLAMA_API_KEY")
            && !key.is_empty()
        {
            let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|err| upstream(502, err.to_string()))?;
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
        // No redirects: a loopback 3xx must never move a probe or a prompt
        // to another host, and readiness must answer from this listener.
        let client = reqwest::blocking::Client::builder()
            .timeout(TIMEOUT)
            .connect_timeout(Duration::from_secs(3))
            .redirect(reqwest::redirect::Policy::none())
            .default_headers(headers.clone())
            .build()
            .map_err(|err| upstream(502, err.to_string()))?;
        let async_client = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .connect_timeout(Duration::from_secs(3))
            .redirect(reqwest::redirect::Policy::none())
            .default_headers(headers)
            .build()
            .map_err(|err| upstream(502, err.to_string()))?;
        let ready = client
            .get(format!("{base}/health"))
            .send()
            .map_err(|err| upstream(503, err.to_string()))?;
        if ready.status().as_u16() != 200 {
            return Err(upstream(503, "attached llama-server is not ready"));
        }
        let listed = client
            .get(format!("{base}/v1/models"))
            .send()
            .map_err(|err| upstream(503, err.to_string()))?;
        if listed.status().as_u16() != 200 {
            return Err(upstream(
                503,
                "attached llama-server returned no model list",
            ));
        }
        let bytes = bounded_body(listed).map_err(|err| upstream(503, err.to_string()))?;
        let models: Value =
            serde_json::from_slice(&bytes).map_err(|err| upstream(503, err.to_string()))?;
        if !models.get("data").is_some_and(Value::is_array) {
            return Err(upstream(
                503,
                "attached llama-server returned no model list",
            ));
        }
        Ok(Self {
            base,
            model: model.unwrap_or_default().to_owned(),
            client,
            async_client,
        })
    }

    /// Map the worker request onto the upstream OpenAI-compatible body.
    fn body(&self, req: &ChatRequest, stream: bool) -> Result<Value, LlmError> {
        let mut messages = Vec::with_capacity(req.messages.len());
        for message in &req.messages {
            // Marker-based vision is llama.cpp-native; this backend has no
            // way to carry decoded attachments, so refuse instead of dropping.
            if !message.images.is_empty() {
                return Err(LlmError::Vision(
                    "image parts cannot cross the attached llama-server backend".to_owned(),
                ));
            }
            let mut entry = json!({
                "role": match message.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                },
                "content": message.content,
            });
            if let Some(id) = &message.tool_call_id {
                entry["tool_call_id"] = json!(id);
            }
            if !message.tool_calls.is_empty() {
                entry["tool_calls"] = Value::Array(
                    message
                        .tool_calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            })
                        })
                        .collect(),
                );
            }
            messages.push(entry);
        }
        let model = if req.model.is_empty() {
            &self.model
        } else {
            &req.model
        };
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": stream,
        });
        set(&mut body, "max_tokens", req.max_tokens);
        set(&mut body, "temperature", req.temperature);
        set(&mut body, "seed", req.seed);
        set(&mut body, "grammar", req.grammar.as_deref());
        let s = &req.sampler;
        set(&mut body, "top_k", s.top_k);
        set(&mut body, "min_p", s.min_p);
        set(&mut body, "typical_p", s.typical_p);
        set(&mut body, "repeat_penalty", s.repeat_penalty);
        set(&mut body, "xtc_probability", s.xtc_probability);
        set(&mut body, "dry_multiplier", s.dry_multiplier);
        set(&mut body, "dry_base", s.dry_base);
        set(&mut body, "dry_allowed_length", s.dry_allowed_length);
        set(&mut body, "dynatemp_range", s.dynatemp_range);
        set(&mut body, "dynatemp_exponent", s.dynatemp_exponent);
        if req.json_object {
            body["response_format"] = json!({"type": "json_object"});
        } else if let Some(schema) = &req.json_schema {
            let schema: Value = serde_json::from_str(schema)
                .map_err(|err| LlmError::InvalidRequest(format!("invalid JSON schema: {err}")))?;
            body["response_format"] = json!({"type": "json_schema", "json_schema": schema});
        }
        if let Some(tools) = &req.tools_json {
            let tools: Value = serde_json::from_str(tools)
                .map_err(|err| LlmError::InvalidRequest(format!("invalid tools JSON: {err}")))?;
            body["tools"] = tools;
            if req.tool_choice_none {
                body["tool_choice"] = json!("none");
            }
        }
        Ok(body)
    }

    /// One POST, with the upstream status classified before the caller reads.
    fn post(&self, body: &Value) -> Result<reqwest::blocking::Response, LlmError> {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base))
            .json(body)
            .send()
            .map_err(|err| upstream(502, err.to_string()))?;
        classify(response)
    }
}

impl LlmPipeline for AttachedLlm {
    fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let started = Instant::now();
        let mut body = self.body(req, false)?;
        body.as_object_mut().and_then(|body| body.remove("stream"));
        let bytes = bounded_body(self.post(&body)?)?;
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|err| upstream(502, err.to_string()))?;
        reply(BACKEND_ID, &value, String::new(), &[], started)
    }

    fn complete_stream(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<ChatResponse, LlmError> {
        self.complete_stream_cancellable(req, sink, &RequestCancel::default())
    }

    fn complete_stream_cancellable(
        &self,
        req: &ChatRequest,
        sink: &mut dyn FnMut(&str),
        cancel: &RequestCancel,
    ) -> Result<ChatResponse, LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        let started = Instant::now();
        let body = self.body(req, true)?;
        let url = format!("{}/v1/chat/completions", self.base);
        let client = self.async_client.clone();
        let task_cancel = cancel.clone();
        let (tx, rx) = mpsc::channel::<StreamMsg>();
        let done_tx = tx.clone();
        // One short-lived runtime per stream: dropping the async read future
        // on cancel closes the upstream socket instead of parking in a
        // blocking `read_line` until the 300 s client timeout. Built here,
        // never stored, so it is always dropped from a sync context
        // (spawn_blocking worker or CLI) without a runtime drop panic.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|err| upstream(502, format!("stream runtime: {err}")))?;
        runtime.spawn(async move {
            let result = stream_sse(client, url, body, task_cancel, tx, started).await;
            let _ = done_tx.send(StreamMsg::Done(result));
        });
        loop {
            match rx.recv_timeout(CANCEL_POLL) {
                Ok(StreamMsg::Token(token)) => sink(&token),
                Ok(StreamMsg::Done(result)) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if cancel.is_cancelled() {
                        return Err(LlmError::Cancelled);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(upstream(502, "upstream stream stopped"));
                }
            }
        }
    }

    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }
}

/// One message from the async SSE reader to the sync caller.
/// Tokens are forwarded through the caller's `sink` on its own thread.
enum StreamMsg {
    Token(String),
    Done(Result<ChatResponse, LlmError>),
}

/// Turn the per-request atomic flag into a one-shot signal `select!` can
/// await without polling (and cancelling) the in-flight network future.
fn cancel_signal(cancel: &RequestCancel) -> tokio::sync::oneshot::Receiver<()> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let watch = cancel.clone();
    tokio::spawn(async move {
        loop {
            if watch.is_cancelled() {
                let _ = tx.send(());
                return;
            }
            tokio::time::sleep(CANCEL_POLL).await;
        }
    });
    rx
}

/// Bounded async SSE read. Cancellation drops both the request/body future
/// and the response, closing the upstream socket immediately.
async fn stream_sse(
    client: reqwest::Client,
    url: String,
    body: Value,
    cancel: RequestCancel,
    tx: mpsc::Sender<StreamMsg>,
    started: Instant,
) -> Result<ChatResponse, LlmError> {
    let mut cancel_rx = cancel_signal(&cancel);
    let response = tokio::select! {
        _ = &mut cancel_rx => return Err(LlmError::Cancelled),
        sent = client.post(url).json(&body).send() => {
            sent.map_err(|err| upstream(502, err.to_string()))?
        }
    };
    let status = response.status().as_u16();
    if status != 200 {
        let bytes = tokio::select! {
            _ = &mut cancel_rx => return Err(LlmError::Cancelled),
            bytes = bounded_async_body(response) => bytes.unwrap_or_default(),
        };
        let message = bounded_text(&String::from_utf8_lossy(&bytes));
        return Err(classify_status(status, message));
    }
    let mut response = response;
    let mut text = String::new();
    let mut tool_deltas: Vec<(usize, String, String, String)> = Vec::new();
    let mut usage = Usage::default();
    let mut pending: Vec<u8> = Vec::new();
    let mut total = 0usize;
    let mut sink = |token: &str| {
        let _ = tx.send(StreamMsg::Token(token.to_owned()));
    };
    let mut stopped = false;
    loop {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        tokio::select! {
            _ = &mut cancel_rx => return Err(LlmError::Cancelled),
            chunk = response.chunk() => {
                let Some(chunk) = chunk.map_err(|err| upstream(502, err.to_string()))? else {
                    break;
                };
                total = total.saturating_add(chunk.len());
                if u64::try_from(total).unwrap_or(u64::MAX) > MAX_RESPONSE_BYTES {
                    return Err(upstream(502, "upstream response too large"));
                }
                pending.extend_from_slice(&chunk);
                while let Some(pos) = pending.iter().position(|byte| *byte == b'\n') {
                    let bytes: Vec<u8> = pending.drain(..=pos).collect();
                    let line = String::from_utf8(bytes)
                        .map_err(|err| upstream(502, format!("invalid upstream SSE: {err}")))?;
                    if !sse_line(&line, &mut text, &mut tool_deltas, &mut usage, &mut sink)? {
                        stopped = true;
                        break;
                    }
                }
                if stopped {
                    break;
                }
            }
        }
    }
    if !stopped && !pending.is_empty() {
        let line = String::from_utf8(pending)
            .map_err(|err| upstream(502, format!("invalid upstream SSE: {err}")))?;
        let _ = sse_line(&line, &mut text, &mut tool_deltas, &mut usage, &mut sink)?;
    }
    let tools: Vec<ToolCallOut> = tool_deltas
        .into_iter()
        .filter(|call| !call.2.is_empty())
        .map(|(_, _, name, arguments)| ToolCallOut { name, arguments })
        .collect();
    reply(
        BACKEND_ID,
        &json!({
            "usage": usage.to_json(),
            "choices": [{"message": {"content": text}}],
        }),
        String::new(),
        &tools,
        started,
    )
}

/// Decode one complete SSE line. Returns `false` on `[DONE]`.
fn sse_line(
    line: &str,
    text: &mut String,
    tool_deltas: &mut Vec<(usize, String, String, String)>,
    usage: &mut Usage,
    sink: &mut dyn FnMut(&str),
) -> Result<bool, LlmError> {
    if line.len() > MAX_SSE_LINE {
        return Err(upstream(502, "upstream SSE line too long"));
    }
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(true);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(false);
    }
    let event: Value = serde_json::from_str(data)
        .map_err(|err| upstream(502, format!("invalid upstream SSE: {err}")))?;
    if let Some(error) = event.get("error") {
        return Err(upstream(502, bounded_message(error)));
    }
    usage.read(&event);
    let Some(choice) = event
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return Ok(true);
    };
    let delta = choice.get("delta").or_else(|| choice.get("message"));
    if let Some(content) = delta.and_then(|d| d.get("content")).and_then(Value::as_str) {
        text.push_str(content);
        sink(content);
    }
    if let Some(calls) = delta
        .and_then(|d| d.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for call in calls {
            apply_tool_call(call, tool_deltas)?;
        }
    }
    Ok(true)
}

/// Accumulate one streamed tool-call delta into its indexed slot.
fn apply_tool_call(
    call: &Value,
    tool_deltas: &mut Vec<(usize, String, String, String)>,
) -> Result<(), LlmError> {
    let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
    if index > 64 {
        return Err(upstream(502, "upstream tool-call index out of range"));
    }
    let index = usize::try_from(index).unwrap_or(0);
    if tool_deltas.len() <= index {
        tool_deltas.resize_with(index + 1, || {
            (0, String::new(), String::new(), String::new())
        });
    }
    let slot = &mut tool_deltas[index];
    if let Some(id) = call.get("id").and_then(Value::as_str) {
        slot.1.push_str(id);
    }
    let function = call.get("function");
    if let Some(name) = function.and_then(|f| f.get("name")).and_then(Value::as_str) {
        slot.2.push_str(name);
    }
    if let Some(args) = function
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
    {
        slot.3.push_str(args);
    }
    Ok(())
}

#[derive(Default)]
struct Usage {
    prompt: u32,
    completion: u32,
}

impl Usage {
    fn read(&mut self, event: &Value) {
        let Some(usage) = event.get("usage") else {
            return;
        };
        if let Some(n) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            self.prompt = u32::try_from(n).unwrap_or(u32::MAX);
        }
        if let Some(n) = usage.get("completion_tokens").and_then(Value::as_u64) {
            self.completion = u32::try_from(n).unwrap_or(u32::MAX);
        }
    }

    fn to_json(&self) -> Value {
        json!({"prompt_tokens": self.prompt, "completion_tokens": self.completion})
    }
}

fn set<T: serde::Serialize>(body: &mut Value, key: &str, value: Option<T>) {
    if let Some(value) = value
        && let Some(object) = body.as_object_mut()
    {
        object.insert(key.to_owned(), json!(value));
    }
}

fn classify_status(status: u16, message: String) -> LlmError {
    if status == 400 {
        LlmError::InvalidRequest(message)
    } else if status == 503 {
        upstream(503, message)
    } else {
        upstream(502, message)
    }
}

fn classify(
    response: reqwest::blocking::Response,
) -> Result<reqwest::blocking::Response, LlmError> {
    let status = response.status().as_u16();
    if status == 200 {
        return Ok(response);
    }
    let bytes = bounded_body(response).unwrap_or_default();
    let message = bounded_text(&String::from_utf8_lossy(&bytes));
    Err(classify_status(status, message))
}

/// Read at most `MAX_RESPONSE_BYTES` from an upstream body: no raw
/// `.text()`/`.json()` on a local server we do not control.
fn bounded_body(response: reqwest::blocking::Response) -> Result<Vec<u8>, LlmError> {
    let mut bytes = Vec::new();
    response
        .take(MAX_RESPONSE_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|err| upstream(502, err.to_string()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) >= MAX_RESPONSE_BYTES {
        return Err(upstream(502, "upstream response too large"));
    }
    Ok(bytes)
}

async fn bounded_async_body(mut response: reqwest::Response) -> Result<Vec<u8>, LlmError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| upstream(502, err.to_string()))?
    {
        bytes.extend_from_slice(&chunk);
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) >= MAX_RESPONSE_BYTES {
            return Err(upstream(502, "upstream response too large"));
        }
    }
    Ok(bytes)
}

fn bounded_message(value: &Value) -> String {
    let text = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| value.as_str().unwrap_or("upstream error"));
    bounded_text(text)
}

fn bounded_text(text: &str) -> String {
    text.chars().take(512).collect()
}

fn upstream(status: u16, message: impl Into<String>) -> LlmError {
    LlmError::Upstream {
        status,
        message: message.into().chars().take(512).collect(),
    }
}

fn reply(
    backend_id: &str,
    value: &Value,
    text: String,
    tool_calls: &[ToolCallOut],
    started: Instant,
) -> Result<ChatResponse, LlmError> {
    let mut text = text;
    let mut tools = tool_calls.to_vec();
    let usage = value.get("usage");
    if text.is_empty() && tools.is_empty() {
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .ok_or_else(|| upstream(502, "upstream returned no choices"))?;
        let message = choice
            .get("message")
            .ok_or_else(|| upstream(502, "upstream returned no message"))?;
        message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .clone_into(&mut text);
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            tools = calls
                .iter()
                .filter_map(|call| {
                    let function = call.get("function")?;
                    Some(ToolCallOut {
                        name: function.get("name")?.as_str()?.to_owned(),
                        arguments: function
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}")
                            .to_owned(),
                    })
                })
                .collect();
        }
    }
    let count = |key: &str| {
        usage
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
    };
    Ok(ChatResponse {
        backend_id: backend_id.to_owned(),
        text,
        tokens: count("completion_tokens"),
        prompt_tokens: count("prompt_tokens"),
        tool_calls: tools,
        millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use honkbox_llm_worker::ChatMessage;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::mpsc;

    fn fake_server(body: &'static str, sse: bool) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake");
        let port = listener.local_addr().expect("addr").port();
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming().take(3) {
                let Ok(mut stream) = stream else { return };
                let mut request = Vec::new();
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    request.extend_from_slice(&chunk[..n]);
                    if request.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&request);
                let (status, payload, content_type) = if text.contains("/health") {
                    ("200 OK", r#"{"status":"ok"}"#, "application/json")
                } else if text.contains("/v1/models") {
                    ("200 OK", r#"{"data":[{"id":"local"}]}"#, "application/json")
                } else if sse && text.contains("\"stream\":true") {
                    ("200 OK", body, "text/event-stream")
                } else {
                    ("200 OK", body, "application/json")
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (format!("http://127.0.0.1:{port}"), handle)
    }

    #[test]
    fn stream_emits_deltas_and_usage() {
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"!\"}}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
        let (url, server) = fake_server(SSE, true);
        let llm = AttachedLlm::new(&url, Some("local")).expect("probe");
        let mut tokens = Vec::new();
        let reply = llm
            .complete_stream(&ChatRequest::default(), &mut |token| {
                tokens.push(token.to_owned());
            })
            .expect("stream");
        assert_eq!(tokens, ["hi", "!"]);
        assert_eq!(reply.text, "hi!");
        assert_eq!(reply.tokens, 2);
        assert_eq!(reply.prompt_tokens, 3);
        assert_eq!(reply.backend_id, BACKEND_ID);
        server.join().expect("server");
    }

    #[test]
    fn non_loopback_url_refused() {
        let err = AttachedLlm::new("https://example.invalid", None).expect_err("refused");
        assert!(matches!(err, LlmError::Upstream { status: 502, .. }));
    }

    #[test]
    fn userinfo_cannot_spoof_loopback() {
        for url in [
            "http://127.0.0.1:80@evil.invalid",
            "http://[::1]:80@evil.invalid",
            "http://user@127.0.0.1:1",
        ] {
            let err = AttachedLlm::new(url, None).expect_err("refused");
            assert!(
                matches!(err, LlmError::Upstream { status: 502, .. }),
                "{url} was accepted"
            );
        }
    }

    #[test]
    fn probe_failure_is_named_503() {
        let err = AttachedLlm::new("http://127.0.0.1:1", None).expect_err("refused");
        assert!(matches!(err, LlmError::Upstream { status: 503, .. }));
    }
    fn offline() -> AttachedLlm {
        AttachedLlm {
            base: "http://127.0.0.1:1".to_owned(),
            model: "local".to_owned(),
            client: reqwest::blocking::Client::new(),
            async_client: reqwest::Client::new(),
        }
    }

    /// SSE server: health + models, then `chats` streams that emit one
    /// token, pause long enough for a cancel to be observed, then `[DONE]`.
    fn stalling_server(chats: usize) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake");
        let port = listener.local_addr().expect("addr").port();
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming().take(2 + chats) {
                let Ok(mut stream) = stream else { return };
                let mut request = Vec::new();
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    request.extend_from_slice(&chunk[..n]);
                    if request.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&request);
                if text.contains("/health") {
                    let payload = r#"{"status":"ok"}"#;
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    let _ = stream.write_all(reply.as_bytes());
                } else if text.contains("/v1/models") {
                    let payload = r#"{"data":[{"id":"local"}]}"#;
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    let _ = stream.write_all(reply.as_bytes());
                } else {
                    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.flush();
                    std::thread::sleep(Duration::from_millis(100));
                    let _ = stream
                        .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
                    let _ = stream.flush();
                    std::thread::sleep(Duration::from_secs(2));
                    let _ = stream.write_all(b"data: [DONE]\n\n");
                }
            }
        });
        (format!("http://127.0.0.1:{port}"), handle)
    }

    #[test]
    fn cancel_is_request_scoped() {
        let (url, server) = stalling_server(2);
        let llm = Arc::new(AttachedLlm::new(&url, Some("local")).expect("probe"));
        let token_a = RequestCancel::new();
        let token_b = RequestCancel::new();
        let (tx, rx) = mpsc::channel::<String>();
        let a = {
            let llm = Arc::clone(&llm);
            let token = token_a.clone();
            std::thread::spawn(move || {
                llm.complete_stream_cancellable(
                    &ChatRequest::default(),
                    &mut |piece| {
                        let _ = tx.send(piece.to_owned());
                    },
                    &token,
                )
            })
        };
        let b = {
            let llm = Arc::clone(&llm);
            let token = token_b.clone();
            std::thread::spawn(move || {
                llm.complete_stream_cancellable(&ChatRequest::default(), &mut |_| {}, &token)
            })
        };
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(3)).expect("token a"),
            "hi"
        );
        token_a.cancel();
        let err = a.join().expect("thread a").expect_err("cancelled a");
        assert!(matches!(err, LlmError::Cancelled), "{err:?}");
        assert!(!token_b.is_cancelled(), "sibling request was cancelled");
        let reply = b.join().expect("thread b").expect("b completes");
        assert_eq!(reply.text, "hi");
        server.join().expect("server");
    }

    #[test]
    fn cancel_interrupts_stalled_stream() {
        let (url, server) = stalling_server(1);
        let llm = AttachedLlm::new(&url, Some("local")).expect("probe");
        let token = RequestCancel::new();
        let (tx, rx) = mpsc::channel::<String>();
        let started = Instant::now();
        let worker = {
            let token = token.clone();
            std::thread::spawn(move || {
                llm.complete_stream_cancellable(
                    &ChatRequest::default(),
                    &mut |piece| {
                        let _ = tx.send(piece.to_owned());
                    },
                    &token,
                )
            })
        };
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(3))
                .expect("first token"),
            "hi"
        );
        token.cancel();
        let err = worker.join().expect("worker").expect_err("cancelled");
        assert!(matches!(err, LlmError::Cancelled), "{err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "cancel took {:?}",
            started.elapsed()
        );
        server.join().expect("server");
    }

    #[test]
    fn query_and_fragment_refused() {
        for url in [
            "http://127.0.0.1:8080/?x=1",
            "http://127.0.0.1:8080/#part",
            "http://127.0.0.1:8080/prefix?x=1",
        ] {
            let err = AttachedLlm::new(url, None).expect_err("refused");
            assert!(
                matches!(err, LlmError::Upstream { status: 502, .. }),
                "{url} was accepted: {err:?}"
            );
        }
    }

    #[test]
    fn probe_does_not_follow_redirects() {
        let target = TcpListener::bind("127.0.0.1:0").expect("bind target");
        target.set_nonblocking(true).expect("nonblocking");
        let target_port = target.local_addr().expect("addr").port();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake");
        let port = listener.local_addr().expect("addr").port();
        let server = std::thread::spawn(move || {
            for stream in listener.incoming().take(1) {
                let Ok(mut stream) = stream else { return };
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let reply = format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{target_port}/health\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        let err = AttachedLlm::new(&format!("http://127.0.0.1:{port}"), None).expect_err("refused");
        assert!(
            matches!(err, LlmError::Upstream { status: 503, .. }),
            "{err:?}"
        );
        match target.accept() {
            Ok((_, _)) => panic!("redirect was followed"),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(err) => panic!("unexpected target accept error: {err}"),
        }
        server.join().expect("server");
    }

    #[test]
    fn constraints_are_mapped_not_dropped() {
        let llm = offline();
        let req = ChatRequest {
            model: String::new(),
            messages: vec![ChatMessage {
                content: "hi".to_owned(),
                ..ChatMessage::default()
            }],
            json_schema: Some(r#"{"type":"object"}"#.to_owned()),
            tools_json: Some(r#"[{"type":"function"}]"#.to_owned()),
            tool_choice_none: true,
            ..ChatRequest::default()
        };
        let body = llm.body(&req, false).expect("body");
        assert_eq!(body["model"], "local");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tool_choice"], "none");
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn images_refused_with_named_vision_error() {
        let llm = offline();
        let req = ChatRequest {
            messages: vec![ChatMessage {
                images: vec![honkbox_llm_worker::ChatImage::new(vec![1])],
                ..ChatMessage::default()
            }],
            ..ChatRequest::default()
        };
        assert!(matches!(llm.body(&req, false), Err(LlmError::Vision(_))));
    }

    #[test]
    fn malformed_constraints_are_named_400() {
        let llm = offline();
        let req = ChatRequest {
            tools_json: Some("not-json".to_owned()),
            ..ChatRequest::default()
        };
        assert!(matches!(
            llm.body(&req, false),
            Err(LlmError::InvalidRequest(_))
        ));
    }
}
