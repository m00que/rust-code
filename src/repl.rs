//! Interactive REPL and one-shot runner.
//!
//! Responsibilities:
//!   * pretty-print streaming assistant output (text, thinking, tool calls)
//!   * prompt the user for permission before running mutating tools
//!   * show a diff before `write_file` overwrites an existing file
//!   * track cumulative usage (tokens) and surface it via `/cost`
//!   * support Ctrl-C to abort an in-flight API request without exiting
//!   * support a rich set of slash commands (`/help`, `/clear`, `/model`,
//!     `/mode`, `/compact`, `/system`, `/cost`, `/todo`, …)

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::api::{ApiClient, ApiMessage, ContentBlock, StreamEvent, Usage};
use crate::config::Config;
use crate::messages;
use crate::permissions::{Decision, Mode, PermissionManager, UserChoice};
use crate::system_prompt;
use crate::tools;

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

struct Session {
    client: ApiClient,
    tool_defs: Vec<crate::api::Tool>,
    system_prompt: String,
    messages: Vec<ApiMessage>,
    permissions: PermissionManager,
    usage_total: Usage,
    interrupt_flag: Arc<AtomicBool>,
}

impl Session {
    fn new(config: Config) -> Result<Self> {
        let client = ApiClient::new(config.clone())?;
        let tool_defs = tools::all_tools();
        let extra = if config.append_system_prompt.is_empty() {
            None
        } else {
            Some(config.append_system_prompt.as_str())
        };
        let system_prompt = system_prompt::build(&tool_defs, extra);
        Ok(Self {
            client,
            tool_defs,
            system_prompt,
            messages: Vec::new(),
            permissions: PermissionManager::new(Mode::Default),
            usage_total: Usage::default(),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
        })
    }

    fn install_signal_handler(&self) {
        let flag = self.interrupt_flag.clone();
        // SIGINT during an active turn flips the flag; rustyline already
        // intercepts SIGINT at the prompt itself.
        ctrlc_set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        });
    }

    fn clear_interrupt(&self) {
        self.interrupt_flag.store(false, Ordering::SeqCst);
    }

    fn was_interrupted(&self) -> bool {
        self.interrupt_flag.load(Ordering::SeqCst)
    }
}

// We rely on signal-hook style behavior via ctrlc-style closure. To keep the
// dep surface small, install a one-shot tokio task that listens for SIGINT.
fn ctrlc_set_handler<F: Fn() + Send + Sync + 'static>(f: F) {
    use tokio::signal;
    tokio::spawn(async move {
        loop {
            if signal::ctrl_c().await.is_ok() {
                f();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

pub async fn run_interactive(config: &Config, mode: Option<Mode>) -> Result<()> {
    let mut session = Session::new(config.clone())?;
    if let Some(m) = mode {
        session.permissions.set_mode(m);
    }
    session.install_signal_handler();

    print_banner(&session);

    let mut rl = DefaultEditor::new()?;
    let history_path = dirs::home_dir()
        .map(|h| h.join(".claude-rs").join("history"))
        .unwrap_or_else(|| ".claude_rs_history".into());
    rl.load_history(&history_path).ok();

    loop {
        let prompt = format!("{} ", "❯".cyan().bold());
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                rl.add_history_entry(&line)?;

                if line.starts_with('/') {
                    match handle_slash_command(&line, &mut session).await {
                        SlashResult::Handled => continue,
                        SlashResult::Exit => break,
                        SlashResult::NotACommand => {}
                    }
                }

                session.clear_interrupt();
                if let Err(e) = process_turn(&mut session, &line).await {
                    eprintln!("{}", format!("\nError: {e}").red());
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C at empty prompt → ignore (let user use /exit)
                println!("{}", "(use /exit or Ctrl-D to quit)".dimmed());
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("{}", "Bye!".dimmed());
                break;
            }
            Err(e) => {
                eprintln!("{}", format!("Readline error: {e}").red());
                break;
            }
        }
    }

    rl.save_history(&history_path).ok();
    Ok(())
}

pub async fn run_once(config: &Config, prompt: &str) -> Result<()> {
    let mut session = Session::new(config.clone())?;
    session.install_signal_handler();
    // In one-shot mode, auto-accept everything (matches Claude Code's `-p`).
    session.permissions.set_mode(Mode::BypassPermissions);
    process_turn(&mut session, prompt).await
}

fn print_banner(s: &Session) {
    println!(
        "{}",
        format!(
            "claude-rs · model={} · mode={} · tools={}",
            s.client.config().model,
            s.permissions.mode.as_str(),
            s.tool_defs.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(",")
        )
        .dimmed()
    );
    println!(
        "{}\n",
        "Type a message, or /help for commands. Ctrl-D to quit.".dimmed()
    );
}

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

enum SlashResult {
    Handled,
    Exit,
    #[allow(dead_code)]
    NotACommand,
}

async fn handle_slash_command(line: &str, session: &mut Session) -> SlashResult {
    let mut parts = line.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();

    match cmd {
        "/help" | "/h" | "/?" => {
            println!(
                "{}",
                r#"
  /help           show this help
  /clear          reset conversation
  /history        show truncated message history
  /model <name>   switch model
  /mode <m>       permission mode: default | accept-edits | bypass | plan
  /system         print effective system prompt
  /cost           show cumulative token usage
  /tokens         estimated tokens currently in context
  /compact [n]    drop oldest turns, keep last n (default 6)
  /todo           print current TODO list
  /allowed        list always-allowed tools / bash prefixes
  /save <file>    save conversation to a JSON file
  /load <file>    load conversation from a JSON file
  /exit           quit
"#
                .cyan()
            );
            SlashResult::Handled
        }
        "/clear" => {
            session.messages.clear();
            session.usage_total = Usage::default();
            println!("{}", "Conversation cleared.".dimmed());
            SlashResult::Handled
        }
        "/history" => {
            print_history(&session.messages);
            SlashResult::Handled
        }
        "/model" => {
            if arg.is_empty() {
                println!("current model: {}", session.client.config().model);
            } else {
                let mut new_cfg = session.client.config().clone();
                new_cfg.model = arg.to_string();
                match ApiClient::new(new_cfg.clone()) {
                    Ok(c) => {
                        session.client = c;
                        println!("{}", format!("model → {}", arg).dimmed());
                    }
                    Err(e) => eprintln!("{}", format!("failed: {e}").red()),
                }
            }
            SlashResult::Handled
        }
        "/mode" => {
            if arg.is_empty() {
                println!("current mode: {}", session.permissions.mode.as_str());
            } else if let Some(m) = Mode::parse(arg) {
                session.permissions.set_mode(m);
                println!("{}", format!("mode → {}", m.as_str()).dimmed());
            } else {
                println!(
                    "{}",
                    "valid modes: default | accept-edits | bypass | plan".yellow()
                );
            }
            SlashResult::Handled
        }
        "/system" => {
            println!("{}", "─── system prompt ───".dimmed());
            println!("{}", session.system_prompt);
            println!("{}", "─── end ───".dimmed());
            SlashResult::Handled
        }
        "/cost" => {
            let u = &session.usage_total;
            let cached = u.cache_creation_input_tokens + u.cache_read_input_tokens;
            println!(
                "{}",
                format!(
                    "tokens used:   input={}  output={}  cache={}  total={}",
                    u.input_tokens,
                    u.output_tokens,
                    cached,
                    u.input_tokens + u.output_tokens + cached
                )
                .dimmed()
            );
            SlashResult::Handled
        }
        "/tokens" => {
            let est = messages::estimate_tokens(&session.messages);
            let limit = session.client.config().context_window;
            println!(
                "{}",
                format!(
                    "context: ~{est} tokens / {limit} window  ({:.1}%)",
                    (est as f64 / limit as f64) * 100.0
                )
                .dimmed()
            );
            SlashResult::Handled
        }
        "/compact" => {
            let keep = arg.parse::<usize>().unwrap_or(6);
            let s = messages::compact_in_place(&mut session.messages, keep);
            println!(
                "{}",
                format!(
                    "compacted: dropped {} turns, kept {}",
                    s.dropped, s.kept
                )
                .dimmed()
            );
            SlashResult::Handled
        }
        "/todo" => {
            println!("{}", tools::current_todos_pretty().trim_end());
            SlashResult::Handled
        }
        "/allowed" => {
            println!("{}", session.permissions.allowlist_summary().dimmed());
            SlashResult::Handled
        }
        "/save" => {
            if arg.is_empty() {
                eprintln!("{}", "usage: /save <file>".yellow());
            } else {
                match serde_json::to_string_pretty(&session.messages) {
                    Ok(s) => match std::fs::write(arg, s) {
                        Ok(()) => println!("{}", format!("saved → {arg}").dimmed()),
                        Err(e) => eprintln!("{}", format!("failed: {e}").red()),
                    },
                    Err(e) => eprintln!("{}", format!("serialize failed: {e}").red()),
                }
            }
            SlashResult::Handled
        }
        "/load" => {
            if arg.is_empty() {
                eprintln!("{}", "usage: /load <file>".yellow());
            } else {
                match std::fs::read_to_string(arg)
                    .map_err(|e| e.to_string())
                    .and_then(|s| serde_json::from_str::<Vec<ApiMessage>>(&s).map_err(|e| e.to_string()))
                {
                    Ok(msgs) => {
                        session.messages = msgs;
                        println!(
                            "{}",
                            format!(
                                "loaded {} messages from {arg}",
                                session.messages.len()
                            )
                            .dimmed()
                        );
                    }
                    Err(e) => eprintln!("{}", format!("failed: {e}").red()),
                }
            }
            SlashResult::Handled
        }
        "/exit" | "/quit" | "/q" => SlashResult::Exit,
        _ => {
            println!(
                "{}",
                format!("Unknown command: {cmd}  (type /help)").yellow()
            );
            SlashResult::Handled
        }
    }
}

fn print_history(messages: &[ApiMessage]) {
    if messages.is_empty() {
        println!("{}", "(empty)".dimmed());
        return;
    }
    for (i, m) in messages.iter().enumerate() {
        let label = if m.role == "user" {
            "user".cyan().bold()
        } else {
            "assistant".green().bold()
        };
        println!("[{i}] {label}");
        for b in &m.content {
            match b {
                ContentBlock::Text { text } => {
                    let preview = preview(text, 200);
                    println!("    {preview}");
                }
                ContentBlock::Thinking { thinking, .. } => {
                    let preview = preview(thinking, 80);
                    println!("    {} {}", "[think]".magenta(), preview.dimmed());
                }
                ContentBlock::ToolUse { name, .. } => {
                    println!("    {} {name}(...)", "[tool]".yellow());
                }
                ContentBlock::ToolResult { content, is_error, .. } => {
                    let tag = if matches!(is_error, Some(true)) {
                        "[error]".red()
                    } else {
                        "[result]".dimmed().clear()
                    };
                    println!("    {} {}", tag, preview(content, 120).dimmed());
                }
            }
        }
    }
}

fn preview(s: &str, max: usize) -> String {
    let mut t: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        t.push('…');
    }
    t.replace('\n', " ⏎ ")
}

// ---------------------------------------------------------------------------
// Core agentic loop
// ---------------------------------------------------------------------------

const MAX_AGENT_STEPS: usize = 50;

async fn process_turn(session: &mut Session, user_input: &str) -> Result<()> {
    session.messages.push(ApiMessage::user_text(user_input));

    // Auto-compact check
    let est = messages::estimate_tokens(&session.messages);
    let threshold = messages::auto_compact_threshold(session.client.config().context_window);
    if est > threshold {
        let s = messages::compact_in_place(&mut session.messages, 8);
        eprintln!(
            "{}",
            format!(
                "[auto-compact: {} tokens > {} threshold; dropped {} turns]",
                est, threshold, s.dropped
            )
            .dimmed()
        );
    }

    for step in 0..MAX_AGENT_STEPS {
        let _ = step;
        if session.was_interrupted() {
            println!("{}", "[interrupted]".yellow());
            session.clear_interrupt();
            return Ok(());
        }

        let normalized = messages::normalize_for_api(&session.messages);

        print!("\n{}", "Claude  ".green().bold());
        std::io::stdout().flush().ok();

        let mut printer = StreamPrinter::default();

        let api_call = session.client.stream_message(
            &normalized,
            &session.tool_defs,
            &session.system_prompt,
            |evt| printer.handle(evt),
        );

        let interrupt_flag = session.interrupt_flag.clone();
        let result = tokio::select! {
            r = api_call => r,
            _ = wait_for_interrupt(interrupt_flag) => {
                println!("\n{}", "[aborted by user]".yellow());
                session.clear_interrupt();
                return Ok(());
            }
        };

        let stream = match result {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        printer.finish();
        session.usage_total.add(&stream.usage);

        // Collect tool calls
        let tool_calls: Vec<(String, String, serde_json::Value)> = stream
            .blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();

        session.messages.push(ApiMessage::assistant(stream.blocks));

        if tool_calls.is_empty() {
            println!(
                "{}",
                format!(
                    "  · {} in / {} out tokens (this turn)",
                    stream.usage.input_tokens, stream.usage.output_tokens
                )
                .dimmed()
            );
            return Ok(());
        }

        // Execute tools
        let mut tool_results: Vec<ContentBlock> = Vec::with_capacity(tool_calls.len());
        for (id, name, input) in tool_calls {
            let (output, is_error) = run_tool_with_permission(session, &name, &input).await;
            print_tool_outcome(&name, &output, is_error);
            tool_results.push(ContentBlock::tool_result(id, output, is_error));
        }
        session.messages.push(ApiMessage::user(tool_results));
        println!();
    }

    eprintln!(
        "{}",
        format!("[agent halted after {MAX_AGENT_STEPS} steps]").yellow()
    );
    Ok(())
}

async fn wait_for_interrupt(flag: Arc<AtomicBool>) {
    loop {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ---------------------------------------------------------------------------
// Permission UI + tool execution wrapper
// ---------------------------------------------------------------------------

async fn run_tool_with_permission(
    session: &mut Session,
    name: &str,
    input: &serde_json::Value,
) -> (String, bool) {
    print_tool_call_intro(name, input);

    match session.permissions.check(name, input) {
        Decision::Allow => tools::execute(name, input).await,
        Decision::Deny(reason) => (format!("Tool denied by permission policy: {reason}"), true),
        Decision::AskUser => {
            // For write_file, show a diff first.
            if name == "write_file" {
                if let Some(diff) = build_write_diff(input) {
                    println!("{}", "  proposed change:".dimmed());
                    println!("{diff}");
                }
            }
            match prompt_user_choice(name) {
                UserChoice::AllowOnce => {
                    let r = tools::execute(name, input).await;
                    r
                }
                UserChoice::AllowAlways => {
                    session.permissions.record(name, input, UserChoice::AllowAlways);
                    let r = tools::execute(name, input).await;
                    r
                }
                UserChoice::Deny => (
                    "User denied this tool call. Rethink your approach or ask for guidance.".into(),
                    true,
                ),
            }
        }
    }
}

fn print_tool_call_intro(name: &str, input: &serde_json::Value) {
    let summary = summarize_tool_input(name, input);
    println!(
        "  {} {} {}",
        "→".dimmed(),
        name.yellow().bold(),
        summary.dimmed()
    );
}

fn summarize_tool_input(name: &str, input: &serde_json::Value) -> String {
    match name {
        "bash" => input["command"].as_str().unwrap_or("").to_string(),
        "read_file" | "write_file" | "str_replace" => {
            input["path"].as_str().unwrap_or("").to_string()
        }
        "glob" => input["pattern"].as_str().unwrap_or("").to_string(),
        "grep" => format!(
            "{} in {}",
            input["pattern"].as_str().unwrap_or(""),
            input["path"].as_str().unwrap_or(".")
        ),
        "todo_write" => format!(
            "{} todos",
            input["todos"].as_array().map(|a| a.len()).unwrap_or(0)
        ),
        _ => {
            let s = input.to_string();
            if s.len() > 80 { format!("{}…", &s[..80]) } else { s }
        }
    }
}

fn print_tool_outcome(name: &str, output: &str, is_error: bool) {
    let total_lines = output.lines().count();
    let preview_lines: Vec<&str> = output.lines().take(8).collect();
    for l in &preview_lines {
        println!("    {}", l.dimmed());
    }
    if total_lines > preview_lines.len() {
        println!("    {}", format!("… ({total_lines} lines)").dimmed());
    }
    let badge = if is_error {
        format!("  ✗ {name}").red().bold().to_string()
    } else {
        format!("  ✓ {name}").green().bold().to_string()
    };
    println!("{badge}");
}

fn prompt_user_choice(name: &str) -> UserChoice {
    use std::io::{stdin, BufRead};
    println!(
        "{}",
        format!(
            "    Allow {name}?  [y]es once · [a]lways · [n]o (default: no)"
        )
        .yellow()
    );
    print!("    > ");
    std::io::stdout().flush().ok();
    let mut buf = String::new();
    let stdin = stdin();
    let mut lock = stdin.lock();
    if lock.read_line(&mut buf).is_err() {
        return UserChoice::Deny;
    }
    match buf.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => UserChoice::AllowOnce,
        "a" | "always" => UserChoice::AllowAlways,
        _ => UserChoice::Deny,
    }
}

fn build_write_diff(input: &serde_json::Value) -> Option<String> {
    let path = input["path"].as_str()?;
    let new_content = input["content"].as_str()?;
    let old_content = std::fs::read_to_string(path).ok()?;

    if old_content == new_content {
        return Some("    (no changes)".dimmed().to_string());
    }

    let old_lines: Vec<&str> = old_content.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    let mut out = String::new();
    // Tiny line-by-line diff: just print up to 20 changed lines around the
    // first divergence. Sufficient for human spot-checking.
    let mut i = 0;
    let mut changed = 0;
    while i < old_lines.len().max(new_lines.len()) && changed < 20 {
        let a = old_lines.get(i).copied();
        let b = new_lines.get(i).copied();
        match (a, b) {
            (Some(x), Some(y)) if x == y => {}
            (Some(x), Some(y)) => {
                out.push_str(&format!("    {} {x}\n", "-".red()));
                out.push_str(&format!("    {} {y}\n", "+".green()));
                changed += 1;
            }
            (Some(x), None) => {
                out.push_str(&format!("    {} {x}\n", "-".red()));
                changed += 1;
            }
            (None, Some(y)) => {
                out.push_str(&format!("    {} {y}\n", "+".green()));
                changed += 1;
            }
            (None, None) => break,
        }
        i += 1;
    }
    if changed == 0 {
        out.push_str("    (whitespace-only changes)\n");
    } else if changed >= 20 {
        out.push_str("    … (more changes truncated)\n");
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Stream printer — keeps text / thinking / tool layouts tidy
// ---------------------------------------------------------------------------

#[derive(Default)]
struct StreamPrinter {
    in_text: bool,
    in_thinking: bool,
    in_tool_args: bool,
    last_text_no_newline: bool,
}

impl StreamPrinter {
    fn handle(&mut self, evt: StreamEvent) {
        match evt {
            StreamEvent::TextDelta { text } => {
                if self.in_thinking {
                    println!();
                    self.in_thinking = false;
                }
                if self.in_tool_args {
                    println!();
                    self.in_tool_args = false;
                }
                if !self.in_text {
                    self.in_text = true;
                }
                print!("{text}");
                std::io::stdout().flush().ok();
                self.last_text_no_newline = !text.ends_with('\n');
            }
            StreamEvent::ThinkingDelta { text } => {
                if self.in_text {
                    println!();
                    self.in_text = false;
                }
                if !self.in_thinking {
                    print!("{}", "thinking ".magenta().italic());
                    self.in_thinking = true;
                }
                print!("{}", text.magenta().italic());
                std::io::stdout().flush().ok();
            }
            StreamEvent::ToolUseStart { name, .. } => {
                if self.in_text || self.in_thinking {
                    if self.last_text_no_newline {
                        println!();
                    }
                    self.in_text = false;
                    self.in_thinking = false;
                }
                print!("\n  {} {}(", "⚙".yellow(), name.yellow().bold());
                std::io::stdout().flush().ok();
                self.in_tool_args = true;
            }
            StreamEvent::InputJsonDelta { partial_json, .. } => {
                if self.in_tool_args {
                    print!("{}", partial_json.dimmed());
                    std::io::stdout().flush().ok();
                }
            }
            StreamEvent::BlockStop { .. } => {
                if self.in_tool_args {
                    println!("{}", ")".yellow());
                    self.in_tool_args = false;
                }
            }
            StreamEvent::MessageStop { .. } => {}
        }
    }

    fn finish(&mut self) {
        if (self.in_text && self.last_text_no_newline) || self.in_thinking {
            println!();
        }
    }
}
