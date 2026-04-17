//! HTTP client for Anthropic / OpenAI-compatible chat completions APIs.
//!
//! Implements:
//!   * SSE streaming with `text_delta`, `thinking_delta`, `input_json_delta`
//!   * Usage accounting (input / output / cache tokens)
//!   * `withRetry`-style retry policy with exponential backoff + jitter
//!   * Classification of API errors (timeout / 429 / 529 / auth / 5xx / other)
//!   * Extended-thinking parameter handling
//!
//! Mirrors the behavior of `src/services/api/claude.ts` and
//! `src/services/api/withRetry.ts` from Claude Code.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use crate::config::Config;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// Extended thinking block. We keep the signature so we can replay it
    /// back to the API on subsequent turns (required by Anthropic).
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }
    pub fn tool_result(id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: if is_error { Some(true) } else { None },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

impl ApiMessage {
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self { role: "user".into(), content }
    }
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::text(text)])
    }
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self { role: "assistant".into(), content }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
    }
}

/// Result of one streaming completion turn.
#[allow(dead_code)]
pub struct StreamResult {
    pub blocks: Vec<ContentBlock>,
    pub usage: Usage,
    pub stop_reason: String,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum StreamEvent {
    TextDelta { text: String },
    ThinkingDelta { text: String },
    ToolUseStart { index: usize, id: String, name: String },
    InputJsonDelta { index: usize, partial_json: String },
    BlockStop { index: usize },
    MessageStop { stop_reason: String },
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Network / connect / read / SDK timeout.
    Timeout,
    /// 429 rate-limit.
    RateLimit,
    /// 529 overloaded.
    Overloaded,
    /// 401 / 403.
    Auth,
    /// 5xx (other than 529).
    Server,
    /// 4xx (other than 401/403/408/409/429).
    BadRequest,
    /// Anything else.
    Unknown,
}

impl ErrorClass {
    pub fn is_retriable(self) -> bool {
        matches!(
            self,
            Self::Timeout | Self::RateLimit | Self::Overloaded | Self::Server
        )
    }
}

#[derive(Debug)]
pub struct ApiError {
    pub class: ErrorClass,
    pub status: Option<u16>,
    pub retry_after: Option<Duration>,
    pub message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(s) => write!(f, "API error {s} ({:?}): {}", self.class, self.message),
            None => write!(f, "API error ({:?}): {}", self.class, self.message),
        }
    }
}

impl std::error::Error for ApiError {}

fn classify_status(status: u16, body: &str, retry_after: Option<Duration>) -> ApiError {
    let class = match status {
        401 | 403 => ErrorClass::Auth,
        408 | 409 => ErrorClass::Timeout, // treated as retriable in withRetry
        429 => ErrorClass::RateLimit,
        529 => ErrorClass::Overloaded,
        s if s >= 500 => ErrorClass::Server,
        s if (400..500).contains(&s) => {
            // overloaded_error sometimes leaks into 4xx text
            if body.contains("overloaded_error") {
                ErrorClass::Overloaded
            } else {
                ErrorClass::BadRequest
            }
        }
        _ => ErrorClass::Unknown,
    };
    ApiError {
        class,
        status: Some(status),
        retry_after,
        message: body.chars().take(2000).collect(),
    }
}

fn classify_transport(err: &reqwest::Error) -> ApiError {
    let class = if err.is_timeout() {
        ErrorClass::Timeout
    } else if err.is_connect() {
        ErrorClass::Timeout
    } else {
        ErrorClass::Unknown
    };
    ApiError {
        class,
        status: None,
        retry_after: None,
        message: err.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Retry policy (mirrors withRetry.ts)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub max_529_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: parse_env_usize("CLAUDE_RS_MAX_RETRIES", 10),
            max_529_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(32),
        }
    }
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Compute the next backoff delay for an attempt (1-indexed) and a possible
/// `Retry-After` hint from the server.
fn backoff_delay(cfg: &RetryConfig, attempt: usize, retry_after: Option<Duration>) -> Duration {
    if let Some(d) = retry_after {
        return d.min(Duration::from_secs(60 * 5));
    }
    let exp = 1u64.checked_shl(attempt.saturating_sub(1) as u32).unwrap_or(u64::MAX);
    let base = cfg.base_delay.saturating_mul(exp.min(64) as u32);
    let capped = base.min(cfg.max_delay);
    // 0..25% jitter
    let jitter_ms = {
        // tiny PRNG: stir from current nanos
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        (n % (capped.as_millis() as u64 / 4 + 1)) as u32
    };
    capped + Duration::from_millis(jitter_ms as u64)
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ApiClient {
    client: reqwest::Client,
    config: Config,
    retry: RetryConfig,
}

impl ApiClient {
    pub fn new(config: Config) -> Result<Self> {
        let timeout = Duration::from_secs(parse_env_usize("CLAUDE_RS_TIMEOUT_SECS", 600) as u64);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .context("Failed to build HTTP client")?;
        Ok(Self {
            client,
            config,
            retry: RetryConfig::default(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Stream a message with retries, calling `on_event` for each event.
    pub async fn stream_message<F>(
        &self,
        messages: &[ApiMessage],
        tools: &[Tool],
        system_prompt: &str,
        mut on_event: F,
    ) -> Result<StreamResult>
    where
        F: FnMut(StreamEvent),
    {
        let mut attempt: usize = 0;
        let mut overload_attempts: usize = 0;

        loop {
            attempt += 1;
            let result = self
                .stream_once(messages, tools, system_prompt, &mut on_event)
                .await;

            match result {
                Ok(r) => return Ok(r),
                Err(api_err) => {
                    let stop = !api_err.class.is_retriable()
                        || attempt > self.retry.max_retries
                        || (api_err.class == ErrorClass::Overloaded
                            && overload_attempts >= self.retry.max_529_retries);

                    if stop {
                        return Err(anyhow::Error::new(api_err));
                    }

                    if api_err.class == ErrorClass::Overloaded {
                        overload_attempts += 1;
                    }

                    let delay = backoff_delay(&self.retry, attempt, api_err.retry_after);
                    eprintln!(
                        "\n[api] {} — retry {}/{} in {:.1}s",
                        api_err,
                        attempt,
                        self.retry.max_retries,
                        delay.as_secs_f64()
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn stream_once<F>(
        &self,
        messages: &[ApiMessage],
        tools: &[Tool],
        system_prompt: &str,
        on_event: &mut F,
    ) -> std::result::Result<StreamResult, ApiError>
    where
        F: FnMut(StreamEvent),
    {
        let body = self.build_request(messages, tools, system_prompt, true);

        let url = self.endpoint();
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");

        req = self.apply_auth(req);
        let req = req.json(&body);

        let response = req.send().await.map_err(|e| classify_transport(&e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs);
            let text = response.text().await.unwrap_or_default();
            return Err(classify_status(status, &text, retry_after));
        }

        if self.config.is_anthropic() {
            self.parse_anthropic_stream(response, on_event).await
        } else {
            self.parse_openai_stream(response, on_event).await
        }
    }

    fn endpoint(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        if self.config.is_anthropic() {
            format!("{base}/v1/messages")
        } else {
            format!("{base}/chat/completions")
        }
    }

    fn apply_auth(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.config.is_anthropic() {
            req = req
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", "2023-06-01");
            if self.config.thinking_enabled() {
                req = req.header(
                    "anthropic-beta",
                    "interleaved-thinking-2025-05-14",
                );
            }
        } else {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }
        req
    }

    fn build_request(
        &self,
        messages: &[ApiMessage],
        tools: &[Tool],
        system_prompt: &str,
        stream: bool,
    ) -> Value {
        if self.config.is_anthropic() {
            let mut body = json!({
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
                "tools": tools,
                "stream": stream,
            });
            if !system_prompt.is_empty() {
                body["system"] = json!(system_prompt);
            }
            if let Some(t) = self.config.thinking_param() {
                body["thinking"] = t;
                // When thinking is on, temperature must not be set (must be default 1).
            } else if let Some(temp) = self.config.temperature {
                body["temperature"] = json!(temp);
            }
            body
        } else {
            // OpenAI-compatible body. Does *not* support extended thinking.
            let oai_messages = anthropic_to_openai_messages(messages, system_prompt);
            let oai_tools = anthropic_tools_to_openai(tools);
            let mut body = json!({
                "model": self.config.model,
                "messages": oai_messages,
                "stream": stream,
                "max_tokens": self.config.max_tokens,
            });
            if !oai_tools.is_empty() {
                body["tools"] = json!(oai_tools);
            }
            if let Some(temp) = self.config.temperature {
                body["temperature"] = json!(temp);
            }
            body
        }
    }

    // -----------------------------------------------------------------------
    // Anthropic SSE parsing
    // -----------------------------------------------------------------------

    async fn parse_anthropic_stream<F>(
        &self,
        response: reqwest::Response,
        on_event: &mut F,
    ) -> std::result::Result<StreamResult, ApiError>
    where
        F: FnMut(StreamEvent),
    {
        let mut text_blocks: Vec<(usize, String)> = Vec::new();
        let mut thinking_blocks: Vec<(usize, String, Option<String>)> = Vec::new();
        let mut tool_blocks: Vec<(usize, String, String, String)> = Vec::new();
        let mut stop_reason = String::from("end_turn");
        let mut usage = Usage::default();

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| classify_transport(&e))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer.drain(..=newline_pos);

                let Some(data) = line.strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
                let ty = event["type"].as_str().unwrap_or("");

                match ty {
                    "message_start" => {
                        if let Some(u) = event["message"]["usage"].as_object() {
                            apply_usage(&mut usage, &Value::Object(u.clone()));
                        }
                    }
                    "content_block_start" => {
                        let index = event["index"].as_u64().unwrap_or(0) as usize;
                        let block = &event["content_block"];
                        match block["type"].as_str().unwrap_or("") {
                            "text" => text_blocks.push((index, String::new())),
                            "thinking" => {
                                thinking_blocks.push((index, String::new(), None));
                            }
                            "tool_use" => {
                                let id = block["id"].as_str().unwrap_or("").to_string();
                                let name = block["name"].as_str().unwrap_or("").to_string();
                                on_event(StreamEvent::ToolUseStart {
                                    index,
                                    id: id.clone(),
                                    name: name.clone(),
                                });
                                tool_blocks.push((index, id, name, String::new()));
                            }
                            _ => {}
                        }
                    }
                    "content_block_delta" => {
                        let index = event["index"].as_u64().unwrap_or(0) as usize;
                        let delta = &event["delta"];
                        match delta["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                if let Some(t) = delta["text"].as_str() {
                                    on_event(StreamEvent::TextDelta { text: t.to_string() });
                                    if let Some(b) = text_blocks.iter_mut().find(|(i, _)| *i == index) {
                                        b.1.push_str(t);
                                    }
                                }
                            }
                            "thinking_delta" => {
                                if let Some(t) = delta["thinking"].as_str() {
                                    on_event(StreamEvent::ThinkingDelta { text: t.to_string() });
                                    if let Some(b) = thinking_blocks.iter_mut().find(|(i, ..)| *i == index) {
                                        b.1.push_str(t);
                                    }
                                }
                            }
                            "signature_delta" => {
                                if let Some(s) = delta["signature"].as_str() {
                                    if let Some(b) = thinking_blocks.iter_mut().find(|(i, ..)| *i == index) {
                                        b.2 = Some(b.2.take().unwrap_or_default() + s);
                                    }
                                }
                            }
                            "input_json_delta" => {
                                if let Some(j) = delta["partial_json"].as_str() {
                                    on_event(StreamEvent::InputJsonDelta {
                                        index,
                                        partial_json: j.to_string(),
                                    });
                                    if let Some(b) = tool_blocks.iter_mut().find(|(i, ..)| *i == index) {
                                        b.3.push_str(j);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        let index = event["index"].as_u64().unwrap_or(0) as usize;
                        on_event(StreamEvent::BlockStop { index });
                    }
                    "message_delta" => {
                        if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                            stop_reason = reason.to_string();
                            on_event(StreamEvent::MessageStop { stop_reason: reason.to_string() });
                        }
                        if let Some(u) = event.get("usage") {
                            apply_usage(&mut usage, u);
                        }
                    }
                    "message_stop" => {}
                    "error" => {
                        let msg = event["error"].to_string();
                        return Err(ApiError {
                            class: if msg.contains("overloaded_error") {
                                ErrorClass::Overloaded
                            } else if msg.contains("rate_limit") {
                                ErrorClass::RateLimit
                            } else {
                                ErrorClass::Unknown
                            },
                            status: None,
                            retry_after: None,
                            message: msg,
                        });
                    }
                    _ => {}
                }
            }
        }

        let blocks = assemble_blocks(text_blocks, thinking_blocks, tool_blocks);
        Ok(StreamResult { blocks, usage, stop_reason })
    }

    // -----------------------------------------------------------------------
    // OpenAI SSE parsing
    // -----------------------------------------------------------------------

    async fn parse_openai_stream<F>(
        &self,
        response: reqwest::Response,
        on_event: &mut F,
    ) -> std::result::Result<StreamResult, ApiError>
    where
        F: FnMut(StreamEvent),
    {
        let mut text = String::new();
        let mut tool_calls: Vec<(usize, String, String, String)> = Vec::new(); // index, id, name, json
        let mut stop_reason = String::from("stop");
        let mut usage = Usage::default();

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        let mut text_started = false;
        let text_index: usize = 0;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| classify_transport(&e))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer.drain(..=newline_pos);

                let Some(data) = line.strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<Value>(data) else { continue };

                if let Some(u) = event.get("usage") {
                    if let Some(p) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                        usage.input_tokens = p;
                    }
                    if let Some(c) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                        usage.output_tokens = c;
                    }
                }

                let Some(choice) = event["choices"].get(0) else { continue };
                let delta = &choice["delta"];

                // Kimi / some reasoning-model providers send thinking tokens in
                // `delta.reasoning_content` (non-standard OpenAI extension).
                if let Some(t) = delta["reasoning_content"].as_str() {
                    if !t.is_empty() {
                        on_event(StreamEvent::ThinkingDelta { text: t.to_string() });
                    }
                }

                if let Some(t) = delta["content"].as_str() {
                    if !text_started {
                        text_started = true;
                    }
                    on_event(StreamEvent::TextDelta { text: t.to_string() });
                    text.push_str(t);
                }

                if let Some(tcs) = delta["tool_calls"].as_array() {
                    for tc in tcs {
                        let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                        // ensure slot
                        while tool_calls.len() <= idx {
                            tool_calls.push((tool_calls.len(), String::new(), String::new(), String::new()));
                        }
                        let slot = &mut tool_calls[idx];
                        if let Some(id) = tc["id"].as_str() {
                            if slot.1.is_empty() {
                                slot.1 = id.to_string();
                            }
                        }
                        if let Some(name) = tc["function"]["name"].as_str() {
                            if slot.2.is_empty() {
                                slot.2 = name.to_string();
                                on_event(StreamEvent::ToolUseStart {
                                    index: idx + 100, // distinct from text index
                                    id: slot.1.clone(),
                                    name: name.to_string(),
                                });
                            }
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            slot.3.push_str(args);
                            on_event(StreamEvent::InputJsonDelta {
                                index: idx + 100,
                                partial_json: args.to_string(),
                            });
                        }
                    }
                }

                if let Some(reason) = choice["finish_reason"].as_str() {
                    stop_reason = match reason {
                        "tool_calls" => "tool_use".to_string(),
                        "length" => "max_tokens".to_string(),
                        "stop" => "end_turn".to_string(),
                        other => other.to_string(),
                    };
                    on_event(StreamEvent::MessageStop { stop_reason: stop_reason.clone() });
                }
            }
        }

        let mut blocks: Vec<ContentBlock> = Vec::new();
        if !text.is_empty() {
            blocks.push(ContentBlock::text(text));
        }
        for (_, id, name, json) in tool_calls {
            if name.is_empty() && id.is_empty() {
                continue;
            }
            let input: Value = serde_json::from_str(&json).unwrap_or(Value::Object(Default::default()));
            blocks.push(ContentBlock::ToolUse { id, name, input });
        }

        let _ = text_index; // silence warning if text not used
        Ok(StreamResult { blocks, usage, stop_reason })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn apply_usage(dst: &mut Usage, raw: &Value) {
    if let Some(v) = raw.get("input_tokens").and_then(|v| v.as_u64()) {
        dst.input_tokens += v;
    }
    if let Some(v) = raw.get("output_tokens").and_then(|v| v.as_u64()) {
        dst.output_tokens += v;
    }
    if let Some(v) = raw.get("cache_creation_input_tokens").and_then(|v| v.as_u64()) {
        dst.cache_creation_input_tokens += v;
    }
    if let Some(v) = raw.get("cache_read_input_tokens").and_then(|v| v.as_u64()) {
        dst.cache_read_input_tokens += v;
    }
}

fn assemble_blocks(
    text_blocks: Vec<(usize, String)>,
    thinking_blocks: Vec<(usize, String, Option<String>)>,
    tool_blocks: Vec<(usize, String, String, String)>,
) -> Vec<ContentBlock> {
    let mut all: Vec<(usize, ContentBlock)> = Vec::new();

    for (idx, content, sig) in thinking_blocks {
        if !content.is_empty() {
            all.push((idx, ContentBlock::Thinking { thinking: content, signature: sig }));
        }
    }
    for (idx, text) in text_blocks {
        if !text.is_empty() {
            all.push((idx, ContentBlock::text(text)));
        }
    }
    for (idx, id, name, json) in tool_blocks {
        let input: Value = serde_json::from_str(&json).unwrap_or(Value::Object(Default::default()));
        all.push((idx, ContentBlock::ToolUse { id, name, input }));
    }
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, b)| b).collect()
}

// ---------------------------------------------------------------------------
// Anthropic ↔ OpenAI translation
// ---------------------------------------------------------------------------

fn anthropic_to_openai_messages(messages: &[ApiMessage], system_prompt: &str) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();

    if !system_prompt.is_empty() {
        out.push(json!({ "role": "system", "content": system_prompt }));
    }

    for m in messages {
        // Pull tool_results out into separate "tool" messages (OpenAI requires).
        let mut user_text = String::new();
        let mut tool_results: Vec<(String, String, bool)> = Vec::new();

        // For assistants we need to re-emit text + tool_calls together.
        if m.role == "assistant" {
            let mut text = String::new();
            let mut calls: Vec<Value> = Vec::new();
            for b in &m.content {
                match b {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::ToolUse { id, name, input } => {
                        calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": input.to_string(),
                            }
                        }));
                    }
                    ContentBlock::Thinking { .. } => {}
                    ContentBlock::ToolResult { .. } => {}
                }
            }
            let mut msg = json!({ "role": "assistant" });
            if !text.is_empty() {
                msg["content"] = json!(text);
            } else {
                msg["content"] = Value::Null;
            }
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            out.push(msg);
            continue;
        }

        // role == user
        for b in &m.content {
            match b {
                ContentBlock::Text { text } => {
                    if !user_text.is_empty() {
                        user_text.push('\n');
                    }
                    user_text.push_str(text);
                }
                ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                    tool_results.push((
                        tool_use_id.clone(),
                        content.clone(),
                        matches!(is_error, Some(true)),
                    ));
                }
                _ => {}
            }
        }

        if !user_text.is_empty() {
            out.push(json!({ "role": "user", "content": user_text }));
        }
        for (id, content, _is_err) in tool_results {
            out.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": content,
            }));
        }
    }

    out
}

fn anthropic_tools_to_openai(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            })
        })
        .collect()
}
