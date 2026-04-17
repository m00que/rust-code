mod api;
mod config;
mod messages;
mod permissions;
mod repl;
mod system_prompt;
mod tools;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

#[derive(Parser)]
#[command(
    name = "claude-rs",
    about = "Claude Code CLI — rewritten in Rust",
    version,
    author
)]
struct Cli {
    /// Send a single prompt non-interactively (auto-approves all tools).
    #[arg(short, long)]
    prompt: Option<String>,

    /// Model to use (overrides config and env).
    #[arg(short, long)]
    model: Option<String>,

    /// API key (overrides ANTHROPIC_API_KEY / OPENAI_API_KEY).
    #[arg(long)]
    api_key: Option<String>,

    /// API base URL.
    #[arg(long)]
    base_url: Option<String>,

    /// Max output tokens.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Permission mode for the session: default | accept-edits | bypass | plan
    #[arg(long, value_name = "MODE")]
    mode: Option<String>,

    /// Enable extended thinking with the given budget (Anthropic only).
    #[arg(long, value_name = "TOKENS")]
    thinking: Option<u32>,

    /// Sampling temperature (ignored when thinking is enabled).
    #[arg(long)]
    temperature: Option<f32>,

    /// Append text to the auto-generated system prompt.
    #[arg(long, value_name = "TEXT")]
    append_system_prompt: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Save flags to ~/.claude-rs/config.json.
    Config {
        #[arg(long)] api_key: Option<String>,
        #[arg(long)] model: Option<String>,
        #[arg(long)] base_url: Option<String>,
        #[arg(long)] max_tokens: Option<u32>,
    },
    /// Print effective configuration and exit.
    ShowConfig,
    /// List available tools and exit.
    Tools,
    /// Print the assembled system prompt and exit.
    SystemPrompt,
}

/// Extract the first sentence of a tool description, treating a "." as a
/// boundary only when followed by whitespace (so backtick-quoted patterns
/// like `**/*.rs` aren't split).
fn first_sentence(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '.' {
            if matches!(chars.peek(), Some(' ' | '\n') | None) {
                break;
            }
        }
    }
    out
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("{}", format!("Error: {e}").red().bold());
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Subcommands that don't need full config (no API key required).
    match &cli.command {
        Some(Commands::SystemPrompt) => {
            let extra = cli.append_system_prompt.as_deref();
            let sp = system_prompt::build(&tools::all_tools(), extra);
            println!("{sp}");
            return Ok(());
        }
        Some(Commands::Tools) => {
            println!("{}", "Available tools:".bold());
            for tool in tools::all_tools() {
                let first = first_sentence(&tool.description);
                println!("  {} — {}", tool.name.cyan().bold(), first);
            }
            return Ok(());
        }
        Some(Commands::Config { api_key, model, base_url, max_tokens }) => {
            let mut cfg = config::Config::load().unwrap_or_default();
            if let Some(k) = api_key { cfg.api_key = k.clone(); }
            if let Some(m) = model { cfg.model = m.clone(); }
            if let Some(u) = base_url { cfg.base_url = u.clone(); }
            if let Some(t) = max_tokens { cfg.max_tokens = *t; }
            cfg.save()?;
            println!(
                "{}  → {}",
                "Configuration saved.".green(),
                config::config_file_path().display()
            );
            return Ok(());
        }
        _ => {}
    }

    // Full config + CLI overrides.
    let mut config = config::Config::load()?;
    if let Some(k) = cli.api_key { config.api_key = k; }
    if let Some(m) = cli.model { config.model = m; }
    if let Some(u) = cli.base_url { config.base_url = u; }
    if let Some(t) = cli.max_tokens { config.max_tokens = t; }
    if let Some(t) = cli.temperature { config.temperature = Some(t); }
    if let Some(s) = cli.append_system_prompt { config.append_system_prompt = s; }
    if let Some(b) = cli.thinking {
        config.thinking = config::ThinkingMode::Enabled { budget_tokens: b };
    }

    match &cli.command {
        Some(Commands::ShowConfig) => {
            println!("{}", "Effective configuration:".bold());
            println!("  model:               {}", config.model.cyan());
            println!("  base_url:            {}", config.base_url.cyan());
            let preview = if config.api_key.is_empty() {
                "(none)".to_string()
            } else {
                format!("{}***", &config.api_key[..config.api_key.len().min(8)])
            };
            println!("  api_key:             {}", preview);
            println!("  max_tokens:          {}", config.max_tokens);
            println!("  context_window:      {}", config.context_window);
            println!("  temperature:         {:?}", config.temperature);
            println!("  thinking:            {:?}", config.thinking);
            println!("  is_anthropic:        {}", config.is_anthropic());
            return Ok(());
        }
        _ => {}
    }

    // Mode override (only meaningful for interactive runs; bypass is forced
    // for one-shot mode anyway).
    let mode_override = cli.mode.as_deref().and_then(permissions::Mode::parse);
    if cli.mode.is_some() && mode_override.is_none() {
        eprintln!(
            "{}",
            "warning: --mode value not recognized; using default".yellow()
        );
    }

    if let Some(prompt) = cli.prompt {
        repl::run_once(&config, &prompt).await
    } else {
        repl::run_interactive(&config, mode_override).await
    }
}
