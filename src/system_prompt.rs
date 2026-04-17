//! System prompt assembly — modeled after Claude Code's `getSystemPrompt`.
//!
//! Builds a prompt containing:
//!   * core identity & guidelines
//!   * tool catalog (auto-derived from `tools::all_tools`)
//!   * environment block (cwd, platform, date, git status)
//!   * project memory: any `CLAUDE.md` discovered in cwd and ancestors,
//!     plus `~/.claude/CLAUDE.md` if present.

use std::path::PathBuf;
use std::process::Command;

use crate::api::Tool;

/// Build the full system prompt for a session.
///
/// `extra` lets the caller append a user-supplied prompt fragment (mirrors
/// Claude Code's `appendSystemPrompt`).
pub fn build(tools: &[Tool], extra: Option<&str>) -> String {
    let mut sections: Vec<String> = Vec::new();

    sections.push(core_identity());
    sections.push(tool_catalog(tools));
    sections.push(environment_block());

    if let Some(memory) = load_memory_files() {
        sections.push(format!("# Project memory (CLAUDE.md)\n\n{memory}"));
    }

    if let Some(s) = extra {
        if !s.trim().is_empty() {
            sections.push(s.trim().to_string());
        }
    }

    sections.join("\n\n---\n\n")
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

fn core_identity() -> String {
    r#"You are claude-rs, an interactive CLI coding assistant. You operate inside the
user's terminal and help them read, modify, and reason about their codebase.

## Behavior
- Be concise and direct. Match the user's language. Prefer short answers; expand only when asked.
- Before editing files, briefly explain what you intend to do and why.
- Prefer targeted edits (`str_replace`) over rewriting whole files.
- After modifying code, run any obvious verification (build, tests, lint) using `bash` if available.
- Don't fabricate file paths, APIs, or behavior — read or search first.
- If a request is ambiguous, ask one focused clarifying question instead of guessing.

## Tool use
- Issue at most one tool call per message that needs a result, then react to the output.
- Quote file paths with backticks. Reference search results by `path:line`.
- Never log secrets or environment-variable values back to the user."#
        .to_string()
}

fn tool_catalog(tools: &[Tool]) -> String {
    let mut s = String::from("# Available tools\n\n");
    for t in tools {
        s.push_str(&format!("- `{}` — {}\n", t.name, first_sentence(&t.description)));
    }
    s
}

/// Split a description on the first `.` that's followed by whitespace
/// (so backtick-quoted patterns like `**/*.rs` aren't truncated).
fn first_sentence(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '.' && matches!(chars.peek(), Some(' ' | '\n') | None) {
            break;
        }
    }
    out.trim().to_string()
}

fn environment_block() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "?".to_string());
    let platform = format!("{} ({})", std::env::consts::OS, std::env::consts::ARCH);
    let date = chrono_today();

    let mut s = format!(
        "# Environment\n\n- cwd: `{cwd}`\n- platform: {platform}\n- date: {date}\n"
    );

    if let Some(git) = git_summary() {
        s.push_str(&format!("- git: {git}\n"));
    }
    s
}

fn chrono_today() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

fn git_summary() -> Option<String> {
    let in_repo = Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .ok()?;
    if !in_repo.status.success() {
        return None;
    }

    let branch = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let state = if dirty { "dirty" } else { "clean" };
    Some(format!("repo on `{branch}` ({state})"))
}

// ---------------------------------------------------------------------------
// CLAUDE.md discovery
// ---------------------------------------------------------------------------

const MEMORY_FILE_NAMES: &[&str] = &["CLAUDE.md", "AGENTS.md"];

fn load_memory_files() -> Option<String> {
    let mut chunks: Vec<(PathBuf, String)> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        let mut current: Option<PathBuf> = Some(cwd);
        let mut hops = 0_usize;

        while let Some(dir) = current {
            for name in MEMORY_FILE_NAMES {
                let p = dir.join(name);
                if let Ok(content) = std::fs::read_to_string(&p) {
                    chunks.push((p, content));
                }
            }
            let dot = dir.join(".claude").join("CLAUDE.md");
            if let Ok(content) = std::fs::read_to_string(&dot) {
                chunks.push((dot, content));
            }

            // `parent()` returns `None` for both `/` (Unix) and `C:\` (Windows),
            // which is the cleanest cross-platform "we hit the root" check.
            current = dir.parent().map(|p| p.to_path_buf());
            hops += 1;
            if hops > 32 {
                break;
            }
        }
    }

    // User-level memory
    if let Some(home) = dirs::home_dir() {
        let user_md = home.join(".claude").join("CLAUDE.md");
        if let Ok(content) = std::fs::read_to_string(&user_md) {
            chunks.push((user_md, content));
        }
    }

    if chunks.is_empty() {
        return None;
    }

    // Newest (closest to cwd) first; cap total size
    let mut out = String::new();
    let mut budget: usize = 16_000;
    for (path, content) in chunks {
        let header = format!("## {}\n\n", path.display());
        let chunk = if content.len() > budget.saturating_sub(header.len()) {
            let cut = budget.saturating_sub(header.len()).min(content.len());
            format!("{header}{}\n\n[... truncated]\n", &content[..cut])
        } else {
            format!("{header}{content}\n")
        };
        if chunk.len() >= budget {
            out.push_str(&chunk[..budget]);
            break;
        }
        budget -= chunk.len();
        out.push_str(&chunk);
        if budget < 256 {
            break;
        }
    }
    Some(out)
}
