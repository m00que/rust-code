//! Layered configuration: file (`~/.claude-rs/config.json`) overridden by
//! environment variables, overridden again by CLI flags.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,

    /// Optional system prompt fragment to *append* to the auto-generated one.
    /// Mirrors Claude Code's `appendSystemPrompt`.
    #[serde(default)]
    pub append_system_prompt: String,

    /// `None` => use API default. Forced to `None` when thinking is enabled.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Extended thinking configuration.
    #[serde(default)]
    pub thinking: ThinkingMode,

    /// Approximate context window size (tokens). Used for auto-compact.
    #[serde(default = "default_context_window")]
    pub context_window: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ThinkingMode {
    Disabled,
    Enabled { budget_tokens: u32 },
}

impl Default for ThinkingMode {
    fn default() -> Self {
        Self::Disabled
    }
}

fn default_context_window() -> u64 {
    200_000
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-5".to_string(),
            max_tokens: 8192,
            append_system_prompt: String::new(),
            temperature: None,
            thinking: ThinkingMode::Disabled,
            context_window: default_context_window(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let mut config = Self::load_from_file().unwrap_or_default();

        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            if !key.is_empty() {
                config.api_key = key;
            }
        }
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            if !url.is_empty() {
                config.base_url = url;
            }
        }
        if let Ok(model) = std::env::var("CLAUDE_RS_MODEL") {
            if !model.is_empty() {
                config.model = model;
            }
        }

        if env_truthy("CLAUDE_CODE_USE_OPENAI") {
            if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                config.api_key = key;
            }
            if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
                if !url.is_empty() {
                    config.base_url = url;
                } else {
                    config.base_url = "https://api.openai.com/v1".to_string();
                }
            } else {
                config.base_url = "https://api.openai.com/v1".to_string();
            }
            if let Ok(model) = std::env::var("OPENAI_MODEL") {
                if !model.is_empty() {
                    config.model = model;
                }
            }
        }

        if env_truthy("CLAUDE_RS_DISABLE_THINKING") || env_truthy("CLAUDE_CODE_DISABLE_THINKING") {
            config.thinking = ThinkingMode::Disabled;
        }
        if let Ok(b) = std::env::var("CLAUDE_RS_THINKING_BUDGET") {
            if let Ok(n) = b.parse::<u32>() {
                config.thinking = ThinkingMode::Enabled { budget_tokens: n };
            }
        }

        if let Ok(t) = std::env::var("CLAUDE_RS_TEMPERATURE") {
            if let Ok(v) = t.parse::<f32>() {
                config.temperature = Some(v);
            }
        }

        if let Ok(m) = std::env::var("CLAUDE_RS_MAX_TOKENS") {
            if let Ok(n) = m.parse::<u32>() {
                config.max_tokens = n;
            }
        }

        // If the api_key is still the placeholder written by the generator, try
        // provider-specific env vars based on the configured base_url.
        if config.api_key.is_empty() || config.api_key.starts_with("PLACEHOLDER_") {
            config.api_key = resolve_api_key_from_env(&config.base_url);
        }

        if config.api_key.is_empty() || config.api_key.starts_with("PLACEHOLDER_") {
            let hint = api_key_hint(&config.base_url);
            anyhow::bail!(
                "API key not set.\n\
                 {hint}\n\
                 Or: claude-rs config --api-key <KEY>\n\
                 Config file: {}",
                config_file_path().display()
            );
        }

        Ok(config)
    }

    fn load_from_file() -> Option<Self> {
        let path = config_file_path();
        let content = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    pub fn save(&self) -> Result<()> {
        let path = config_file_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config dir {parent:?}"))?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write config to {path:?}"))?;
        Ok(())
    }

    pub fn is_anthropic(&self) -> bool {
        let url = self.base_url.to_ascii_lowercase();
        // Heuristic: Anthropic-style if hostname contains "anthropic"
        // or the user explicitly opted in via env var.
        if env_truthy("CLAUDE_CODE_USE_OPENAI") {
            return false;
        }
        url.contains("anthropic.com")
    }

    pub fn thinking_enabled(&self) -> bool {
        matches!(self.thinking, ThinkingMode::Enabled { .. })
    }

    /// Build the `thinking` API parameter (Anthropic only).
    pub fn thinking_param(&self) -> Option<Value> {
        match self.thinking {
            ThinkingMode::Disabled => None,
            ThinkingMode::Enabled { budget_tokens } => {
                // Per Anthropic: budget_tokens must be < max_tokens
                let budget = budget_tokens.min(self.max_tokens.saturating_sub(1));
                Some(json!({ "type": "enabled", "budget_tokens": budget }))
            }
        }
    }
}

/// Build a user-facing hint for which env var to set based on the base URL.
fn api_key_hint(base_url: &str) -> String {
    let url_lower = base_url.to_ascii_lowercase();
    let map: &[(&str, &str, &str)] = &[
        ("moonshot.cn",    "MOONSHOT_API_KEY",   "Moonshot (Kimi)"),
        ("deepseek.com",   "DEEPSEEK_API_KEY",   "DeepSeek"),
        ("groq.com",       "GROQ_API_KEY",        "Groq"),
        ("together.xyz",   "TOGETHER_API_KEY",    "Together AI"),
        ("mistral.ai",     "MISTRAL_API_KEY",     "Mistral"),
        ("openrouter.ai",  "OPENROUTER_API_KEY",  "OpenRouter"),
        ("openai.com",     "OPENAI_API_KEY",      "OpenAI"),
    ];
    for (host, env_key, name) in map {
        if url_lower.contains(host) {
            return format!("Set {env_key}=<your {name} key>");
        }
    }
    "Set ANTHROPIC_API_KEY (Anthropic) or OPENAI_API_KEY (OpenAI-compatible)".into()
}

/// Pick an API key from well-known provider env vars based on `base_url`.
///
/// Priority:
///   1. `OPENAI_API_KEY` — generic override for any OpenAI-compat endpoint
///   2. Provider-specific key derived from the URL hostname:
///      * moonshot.cn    → MOONSHOT_API_KEY
///      * deepseek.com   → DEEPSEEK_API_KEY
///      * groq.com       → GROQ_API_KEY
///      * together.xyz   → TOGETHER_API_KEY
///      * mistral.ai     → MISTRAL_API_KEY
///      * openrouter.ai  → OPENROUTER_API_KEY
///      * cohere.com     → CO_API_KEY
fn resolve_api_key_from_env(base_url: &str) -> String {
    // 1. Generic override
    if let Ok(k) = std::env::var("OPENAI_API_KEY") {
        if !k.is_empty() {
            return k;
        }
    }
    // 2. Provider-specific
    let url_lower = base_url.to_ascii_lowercase();
    let candidates: &[(&str, &str)] = &[
        ("moonshot.cn",     "MOONSHOT_API_KEY"),
        ("deepseek.com",    "DEEPSEEK_API_KEY"),
        ("groq.com",        "GROQ_API_KEY"),
        ("together.xyz",    "TOGETHER_API_KEY"),
        ("mistral.ai",      "MISTRAL_API_KEY"),
        ("openrouter.ai",   "OPENROUTER_API_KEY"),
        ("cohere.com",      "CO_API_KEY"),
        ("01.ai",           "YI_API_KEY"),
        ("baichuan",        "BAICHUAN_API_KEY"),
        ("zhipuai.cn",      "ZHIPUAI_API_KEY"),
        ("dashscope",       "DASHSCOPE_API_KEY"),
        ("ark.cn-beijing",  "ARK_API_KEY"),
    ];
    for (host_fragment, env_key) in candidates {
        if url_lower.contains(host_fragment) {
            if let Ok(k) = std::env::var(env_key) {
                if !k.is_empty() {
                    return k;
                }
            }
        }
    }
    String::new()
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True" | "yes"))
        .unwrap_or(false)
}

pub fn config_file_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claude-rs")
        .join("config.json")
}
