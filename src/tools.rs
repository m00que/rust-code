//! Tool implementations.
//!
//! Mirrors the most-used tools from Claude Code:
//!   * `bash`        — shell execution (timeout, output truncation, danger checks live in `permissions`)
//!   * `read_file`   — file read with byte/line limits and offset/limit
//!   * `write_file`  — atomic write with parent-dir creation
//!   * `str_replace` — targeted edit (CRLF / quote-style normalization, `replace_all`)
//!   * `glob`        — file lookup (uses `rg --files` when available)
//!   * `grep`        — regex search (uses `rg` when available, falls back to walkdir)
//!   * `todo_write`  — durable TODO list for the agent

use anyhow::Result;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use crate::api::Tool;

// Resource limits (mirror Claude Code's defaults where reasonable)
const BASH_DEFAULT_TIMEOUT_SECS: u64 = 120;
const BASH_MAX_TIMEOUT_SECS: u64 = 600;
const BASH_MAX_OUTPUT_CHARS: usize = 30_000;

const FILE_READ_DEFAULT_LIMIT_LINES: usize = 2_000;
const FILE_READ_MAX_BYTES: usize = 256 * 1024;

const GLOB_RESULT_LIMIT: usize = 100;
const GREP_RESULT_LIMIT: usize = 250;

// Directories ignored by Glob/Grep regardless of .gitignore presence.
const ALWAYS_IGNORE_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    ".idea",
    ".vscode",
];

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

pub fn all_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "bash".into(),
            description: format!(
                "Execute a shell command via /bin/bash -c. Returns combined stdout+stderr. \
                 Default timeout {BASH_DEFAULT_TIMEOUT_SECS}s, max {BASH_MAX_TIMEOUT_SECS}s. \
                 Output truncated at {BASH_MAX_OUTPUT_CHARS} chars. Use for tests, builds, git, ls, etc."
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Shell command to run." },
                    "timeout_secs": {
                        "type": "integer",
                        "description": format!("Override timeout (max {BASH_MAX_TIMEOUT_SECS}s)."),
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line human description of what this command does (optional, for the user).",
                    }
                },
                "required": ["command"]
            }),
        },
        Tool {
            name: "read_file".into(),
            description: format!(
                "Read a UTF-8 text file. Adds line numbers. Hard limits: {FILE_READ_MAX_BYTES} bytes \
                 read by default, {FILE_READ_DEFAULT_LIMIT_LINES} lines returned. \
                 Use offset/limit for large files. Binary files return a hex preview."
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path (absolute or relative)." },
                    "offset": { "type": "integer", "description": "1-indexed first line to return." },
                    "limit": { "type": "integer", "description": format!("Max lines to return (default {FILE_READ_DEFAULT_LIMIT_LINES}).") }
                },
                "required": ["path"]
            }),
        },
        Tool {
            name: "write_file".into(),
            description: "Write (overwrite) a file. Creates parent dirs. \
                          Always show a brief description before using this on existing files.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        },
        Tool {
            name: "str_replace".into(),
            description: "Replace exact text in a file. `old_str` must match exactly (whitespace counts). \
                          Files are normalized to LF before matching. \
                          By default the match must be unique; pass `replace_all` to substitute every occurrence."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_str": { "type": "string" },
                    "new_str": { "type": "string" },
                    "replace_all": { "type": "boolean", "description": "Replace every occurrence." }
                },
                "required": ["path", "old_str", "new_str"]
            }),
        },
        Tool {
            name: "glob".into(),
            description: format!(
                "Find files by glob (`**/*.rs`, `src/**/*.{{ts,tsx}}`). \
                 Skips {} and dotfiles. Returns up to {GLOB_RESULT_LIMIT} results.",
                ALWAYS_IGNORE_DIRS.join(", ")
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "base_dir": { "type": "string", "description": "Default: cwd." }
                },
                "required": ["pattern"]
            }),
        },
        Tool {
            name: "grep".into(),
            description: format!(
                "Regex search. Uses ripgrep (`rg`) when available. \
                 Skips binaries and {}. Returns up to {GREP_RESULT_LIMIT} matches.",
                ALWAYS_IGNORE_DIRS.join(", ")
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string", "description": "File or directory; default cwd." },
                    "file_glob": { "type": "string", "description": "e.g. `*.rs` — restrict files." },
                    "case_sensitive": { "type": "boolean", "description": "Default true." },
                    "max_results": { "type": "integer", "description": format!("Default {GREP_RESULT_LIMIT}.") }
                },
                "required": ["pattern"]
            }),
        },
        Tool {
            name: "todo_write".into(),
            description: "Maintain a session-level TODO list. Pass the *complete* list every call. \
                          Each item: { content: string, status: pending|in_progress|completed }. \
                          Use to plan multi-step work and track progress."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] }
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }),
        },
    ]
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

pub async fn execute(name: &str, input: &Value) -> (String, bool) {
    match run_tool(name, input).await {
        Ok(output) => (output, false),
        Err(e) => (format!("Error: {e}"), true),
    }
}

async fn run_tool(name: &str, input: &Value) -> Result<String> {
    match name {
        "bash" => tool_bash(input).await,
        "read_file" => tool_read_file(input),
        "write_file" => tool_write_file(input),
        "str_replace" => tool_str_replace(input),
        "glob" => tool_glob(input),
        "grep" => tool_grep(input),
        "todo_write" => tool_todo_write(input),
        other => anyhow::bail!("Unknown tool: {other}"),
    }
}

// ---------------------------------------------------------------------------
// bash
// ---------------------------------------------------------------------------

async fn tool_bash(input: &Value) -> Result<String> {
    let command = input["command"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'command'"))?;

    let timeout_secs = input["timeout_secs"]
        .as_u64()
        .unwrap_or(BASH_DEFAULT_TIMEOUT_SECS)
        .min(BASH_MAX_TIMEOUT_SECS);

    let mut cmd = build_shell_command(command);
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = cmd.spawn().map_err(|e| anyhow::anyhow!("spawn failed: {e}"))?;

    let out = match tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output())
        .await
    {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => anyhow::bail!("bash error: {e}"),
        Err(_) => anyhow::bail!("Command timed out after {timeout_secs}s"),
    };

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let code = out.status.code().unwrap_or(-1);

    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n--- stderr ---\n");
        }
        result.push_str(&stderr);
    }
    if !out.status.success() {
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!("(exit code {code})"));
    } else if result.is_empty() {
        result = "(no output)".to_string();
    }

    if result.len() > BASH_MAX_OUTPUT_CHARS {
        let head = result.chars().take(BASH_MAX_OUTPUT_CHARS).collect::<String>();
        result = format!(
            "{head}\n... (output truncated; was {} chars total)",
            result.len()
        );
    }

    Ok(result)
}

/// Build a `Command` that runs `command_str` through the platform's shell.
///
/// * Unix: prefer `$SHELL` if it exists, otherwise `/bin/bash`, otherwise
///   `/bin/sh`. Always invoked with `-c <cmd>`.
/// * Windows: prefer `$COMSPEC` (cmd.exe) with `/C <cmd>`. As a fallback try
///   `bash.exe` (Git Bash / WSL) with `-c`.
fn build_shell_command(command_str: &str) -> tokio::process::Command {
    #[cfg(unix)]
    {
        let shell = std::env::var("SHELL").ok().filter(|s| !s.is_empty());
        let exe = shell
            .as_deref()
            .filter(|p| std::path::Path::new(p).is_file())
            .unwrap_or_else(|| {
                if std::path::Path::new("/bin/bash").is_file() {
                    "/bin/bash"
                } else {
                    "/bin/sh"
                }
            });
        let mut cmd = tokio::process::Command::new(exe);
        cmd.arg("-c").arg(command_str);
        cmd
    }
    #[cfg(windows)]
    {
        let comspec = std::env::var("ComSpec")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "cmd.exe".into());
        let mut cmd = tokio::process::Command::new(comspec);
        cmd.arg("/C").arg(command_str);
        cmd
    }
}

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

fn tool_read_file(input: &Value) -> Result<String> {
    let path = input["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'path'"))?;

    let meta = std::fs::metadata(path).map_err(|e| anyhow::anyhow!("Cannot stat {path}: {e}"))?;
    if meta.is_dir() {
        anyhow::bail!("{path} is a directory; use `glob` or `bash ls` to list it");
    }

    // Read up to FILE_READ_MAX_BYTES. Use `take(...).read_to_end(...)` so we
    // don't panic on size mismatches (the file may grow/shrink between the
    // metadata call and the read; e.g. /proc files on Linux report 0 size).
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("Cannot open {path}: {e}"))?;
    let mut content = Vec::with_capacity((meta.len() as usize).min(FILE_READ_MAX_BYTES));
    (&mut file)
        .take(FILE_READ_MAX_BYTES as u64)
        .read_to_end(&mut content)
        .map_err(|e| anyhow::anyhow!("Cannot read {path}: {e}"))?;

    let truncated_bytes = content.len() == FILE_READ_MAX_BYTES
        && (meta.len() as usize) > FILE_READ_MAX_BYTES;

    match std::str::from_utf8(&content) {
        Ok(text) => {
            let offset = input["offset"]
                .as_u64()
                .map(|n| n.saturating_sub(1) as usize)
                .unwrap_or(0);
            let limit = input["limit"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(FILE_READ_DEFAULT_LIMIT_LINES);

            let all_lines: Vec<&str> = text.split('\n').collect();
            let total = all_lines.len();
            let end = (offset + limit).min(total);
            let slice = &all_lines[offset.min(total)..end];

            let mut out = String::with_capacity(text.len() + 80);
            for (i, line) in slice.iter().enumerate() {
                out.push_str(&format!("{:>6}\t{line}\n", offset + i + 1));
            }

            let returned_lines = slice.len();
            let mut footer_parts: Vec<String> = Vec::new();
            if end < total {
                footer_parts.push(format!(
                    "[showing lines {}-{} of {} total]",
                    offset + 1,
                    end,
                    total
                ));
            }
            if truncated_bytes {
                footer_parts.push(format!(
                    "[file is {} bytes, only first {} bytes read]",
                    meta.len(),
                    FILE_READ_MAX_BYTES
                ));
            }
            if !footer_parts.is_empty() {
                out.push('\n');
                out.push_str(&footer_parts.join(" "));
                out.push('\n');
            }
            if returned_lines == 0 && !footer_parts.is_empty() {
                out.push_str("(no lines in this range)\n");
            }
            Ok(out)
        }
        Err(_) => {
            let preview = &content[..content.len().min(256)];
            let hex: Vec<String> = preview
                .chunks(16)
                .enumerate()
                .map(|(i, chunk)| {
                    let hex_str: String = chunk
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!("{:04x}: {hex_str}", i * 16)
                })
                .collect();
            Ok(format!(
                "(binary file, {} bytes)\n{}",
                meta.len(),
                hex.join("\n")
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// write_file (atomic)
// ---------------------------------------------------------------------------

fn tool_write_file(input: &Value) -> Result<String> {
    let path = input["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'path'"))?;
    let content = input["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'content'"))?;

    let path_obj = Path::new(path);
    if let Some(parent) = path_obj.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Cannot create directories for {path}: {e}"))?;
        }
    }

    // Atomic write: write a sibling tempfile then rename. On both Unix and
    // Windows (since Rust 1.5+, via `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`)
    // the rename will overwrite an existing file. If the rename still fails
    // (e.g. due to a different filesystem), we fall back to a non-atomic
    // remove-then-rename.
    let tmp = sibling_temp_path(path_obj);
    std::fs::write(&tmp, content)
        .map_err(|e| anyhow::anyhow!("Cannot write {}: {e}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, path_obj) {
        // Best-effort fallback (mostly for Windows quirks with cross-volume
        // moves or when the target is locked):
        let copy_result = std::fs::copy(&tmp, path_obj)
            .and_then(|_| std::fs::remove_file(&tmp))
            .map_err(|e2| anyhow::anyhow!("rename({}) failed: {e}; copy fallback also failed: {e2}", path));
        if let Err(err) = copy_result {
            let _ = std::fs::remove_file(&tmp);
            return Err(err);
        }
    }

    let lines = content.lines().count();
    let bytes = content.len();
    Ok(format!("Wrote {bytes} bytes ({lines} lines) to {path}"))
}

/// Build a sibling path used for atomic writes: same parent directory, name
/// is the original file name with `.claude-rs.tmp.<pid>` appended. Works on
/// every platform since we're string-concatenating onto the file name.
fn sibling_temp_path(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("tmpfile");
    let tmp_name = format!(".{file_name}.claude-rs.tmp.{pid}");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(tmp_name),
        _ => PathBuf::from(tmp_name),
    }
}

// ---------------------------------------------------------------------------
// str_replace (CRLF + quote normalization, replace_all)
// ---------------------------------------------------------------------------

fn tool_str_replace(input: &Value) -> Result<String> {
    let path = input["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'path'"))?;
    let old_str_raw = input["old_str"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'old_str'"))?;
    let new_str_raw = input["new_str"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'new_str'"))?;
    let replace_all = input["replace_all"].as_bool().unwrap_or(false);

    if old_str_raw == new_str_raw {
        anyhow::bail!("old_str and new_str are identical — nothing to do");
    }
    if old_str_raw.is_empty() {
        anyhow::bail!("old_str must not be empty");
    }

    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read {path}: {e}"))?;
    let had_crlf = raw.contains("\r\n");
    let content = raw.replace("\r\n", "\n");

    // Try exact match first. If no match, try a quote-normalized match
    // (curly → straight) and recover the actual matched substring so we can
    // splice in `new_str` while preserving original quoting.
    let (actual_old, count) = find_actual_match(&content, old_str_raw);

    if count == 0 {
        anyhow::bail!(
            "old_str not found in {path}. Make sure it matches exactly (whitespace and quotes count). \
             Tip: re-read the file first."
        );
    }
    if count > 1 && !replace_all {
        anyhow::bail!(
            "old_str found {count} times in {path}. \
             Provide more context to make the match unique, or pass replace_all=true."
        );
    }

    let new_content = if replace_all {
        content.replace(&actual_old, new_str_raw)
    } else {
        content.replacen(&actual_old, new_str_raw, 1)
    };

    let final_content = if had_crlf {
        new_content.replace('\n', "\r\n")
    } else {
        new_content
    };

    std::fs::write(path, &final_content)
        .map_err(|e| anyhow::anyhow!("Cannot write {path}: {e}"))?;

    let n = if replace_all { count } else { 1 };
    Ok(format!(
        "Replaced {n} occurrence{plural} in {path}",
        plural = if n == 1 { "" } else { "s" }
    ))
}

/// Try to locate `needle` in `haystack`, falling back to a curly-quote-
/// normalized lookup if exact match fails. Returns (the substring actually
/// present in `haystack`, occurrence count).
///
/// `normalize_quotes` is char-for-char length-preserving (one Unicode scalar
/// in, one out) but **not** byte-length preserving — curly quotes are 3 bytes
/// in UTF-8 while ASCII quotes are 1. So we map between the normalized and
/// original strings using char positions.
fn find_actual_match(haystack: &str, needle: &str) -> (String, usize) {
    let exact = haystack.matches(needle).count();
    if exact > 0 {
        return (needle.to_string(), exact);
    }

    // Even if the needle is already ASCII, the haystack may contain curly
    // quotes — so we still try the normalized lookup.
    let normalized_needle = normalize_quotes(needle);
    let normalized_hay = normalize_quotes(haystack);
    let count = normalized_hay.matches(&normalized_needle).count();
    if count == 0 {
        return (needle.to_string(), 0);
    }

    let needle_char_len = normalized_needle.chars().count();
    let byte_pos = normalized_hay.find(&normalized_needle).unwrap();
    let prefix_chars = normalized_hay[..byte_pos].chars().count();

    // Translate prefix char count back into a byte index in the original.
    let mut hay_chars = haystack.char_indices();
    let actual_start = hay_chars
        .by_ref()
        .nth(prefix_chars)
        .map(|(i, _)| i)
        .unwrap_or(haystack.len());
    let actual_end = haystack
        .char_indices()
        .nth(prefix_chars + needle_char_len)
        .map(|(i, _)| i)
        .unwrap_or(haystack.len());

    let actual = &haystack[actual_start..actual_end];
    (actual.to_string(), count)
}

fn normalize_quotes(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => '"',
            other => other,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// glob (rg --files when available)
// ---------------------------------------------------------------------------

fn tool_glob(input: &Value) -> Result<String> {
    let pattern = input["pattern"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'pattern'"))?;
    let base_dir = input["base_dir"].as_str().unwrap_or(".");

    let mut matches = if has_ripgrep() {
        glob_via_ripgrep(pattern, base_dir).unwrap_or_else(|_| glob_via_walk(pattern, base_dir))
    } else {
        glob_via_walk(pattern, base_dir)
    };

    matches.sort();
    matches.dedup();

    if matches.is_empty() {
        return Ok(format!("No files matching: {pattern}"));
    }

    let total = matches.len();
    if matches.len() > GLOB_RESULT_LIMIT {
        matches.truncate(GLOB_RESULT_LIMIT);
        Ok(format!(
            "{}\n\n({total} matches, showing first {GLOB_RESULT_LIMIT})",
            matches.join("\n")
        ))
    } else {
        Ok(format!("{}\n\n({total} matches)", matches.join("\n")))
    }
}

fn glob_via_ripgrep(pattern: &str, base_dir: &str) -> Result<Vec<String>> {
    let out = std::process::Command::new("rg")
        .args(["--files", "--hidden", "--no-ignore", "--glob"])
        .arg(pattern)
        .args(
            ALWAYS_IGNORE_DIRS
                .iter()
                .flat_map(|d| ["--glob".to_string(), format!("!**/{d}/**")]),
        )
        .arg(base_dir)
        .output()?;
    if !out.status.success() {
        anyhow::bail!("rg failed");
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect())
}

fn glob_via_walk(pattern: &str, base_dir: &str) -> Vec<String> {
    // The `glob` crate uses `/` as the separator on every platform, so we
    // build the search pattern as a forward-slash string instead of relying
    // on `Path::join` (which would introduce `\` on Windows).
    let pat_path = Path::new(pattern);
    let full = if pat_path.is_absolute() {
        pattern.to_string()
    } else {
        let base = base_dir.trim_end_matches(['/', '\\']);
        if base.is_empty() || base == "." {
            pattern.to_string()
        } else {
            format!("{base}/{pattern}")
        }
    };
    glob::glob(&full)
        .ok()
        .map(|iter| {
            iter.filter_map(|e| e.ok())
                .filter(|p| !path_in_ignored_dir(p))
                .map(|p| p.display().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn path_in_ignored_dir(p: &Path) -> bool {
    p.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map(|s| ALWAYS_IGNORE_DIRS.contains(&s))
            .unwrap_or(false)
    })
}

// ---------------------------------------------------------------------------
// grep (rg when available)
// ---------------------------------------------------------------------------

fn tool_grep(input: &Value) -> Result<String> {
    let pattern_str = input["pattern"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'pattern'"))?;
    let search_path = input["path"].as_str().unwrap_or(".");
    let file_glob = input["file_glob"].as_str();
    let case_sensitive = input["case_sensitive"].as_bool().unwrap_or(true);
    let max_results = input["max_results"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(GREP_RESULT_LIMIT);

    if has_ripgrep() {
        if let Ok(out) =
            grep_via_ripgrep(pattern_str, search_path, file_glob, case_sensitive, max_results)
        {
            return Ok(out);
        }
    }
    grep_via_walk(pattern_str, search_path, file_glob, case_sensitive, max_results)
}

fn grep_via_ripgrep(
    pattern: &str,
    search_path: &str,
    file_glob: Option<&str>,
    case_sensitive: bool,
    max_results: usize,
) -> Result<String> {
    let mut cmd = std::process::Command::new("rg");
    cmd.args(["--line-number", "--no-heading", "--color", "never", "--max-columns", "500"]);
    if !case_sensitive {
        cmd.arg("--ignore-case");
    }
    for d in ALWAYS_IGNORE_DIRS {
        cmd.args(["--glob", &format!("!**/{d}/**")]);
    }
    if let Some(g) = file_glob {
        cmd.args(["--glob", g]);
    }
    cmd.arg(pattern).arg(search_path);

    let out = cmd.output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);

    let lines: Vec<&str> = stdout.lines().collect();
    if lines.is_empty() {
        return Ok(format!("No matches for: {pattern}"));
    }
    let total = lines.len();
    let head: Vec<&str> = lines.iter().take(max_results).copied().collect();
    let body = head.join("\n");
    if total > max_results {
        Ok(format!("{body}\n\n({total} matches, showing first {max_results})"))
    } else {
        Ok(format!("{body}\n\n({total} matches)"))
    }
}

fn grep_via_walk(
    pattern_str: &str,
    search_path: &str,
    file_glob: Option<&str>,
    case_sensitive: bool,
    max_results: usize,
) -> Result<String> {
    let re = regex::RegexBuilder::new(pattern_str)
        .case_insensitive(!case_sensitive)
        .build()
        .map_err(|e| anyhow::anyhow!("Invalid regex: {e}"))?;

    let glob_matcher = file_glob.and_then(|g| glob::Pattern::new(g).ok());

    let base = Path::new(search_path);
    let files: Box<dyn Iterator<Item = PathBuf>> = if base.is_file() {
        Box::new(std::iter::once(base.to_path_buf()))
    } else {
        Box::new(
            walkdir::WalkDir::new(base)
                .follow_links(false)
                .into_iter()
                .filter_entry(|e| {
                    !e.file_name()
                        .to_str()
                        .map(|n| ALWAYS_IGNORE_DIRS.contains(&n))
                        .unwrap_or(false)
                })
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
                .map(|e| e.into_path()),
        )
    };

    let mut results: Vec<String> = Vec::new();
    'outer: for file_path in files {
        if let Some(matcher) = &glob_matcher {
            if let Some(name) = file_path.file_name().and_then(|s| s.to_str()) {
                if !matcher.matches(name) {
                    continue;
                }
            }
        }

        // Skip files that look binary (NUL byte in first 1024 bytes).
        let Ok(data) = std::fs::read(&file_path) else { continue };
        let head_len = data.len().min(1024);
        if data[..head_len].contains(&0) {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&data) else { continue };

        for (line_no, line) in text.lines().enumerate() {
            if re.is_match(line) {
                let display = if line.len() > 500 { &line[..500] } else { line };
                results.push(format!("{}:{}:{}", file_path.display(), line_no + 1, display));
                if results.len() >= max_results {
                    break 'outer;
                }
            }
        }
    }

    if results.is_empty() {
        return Ok(format!("No matches for: {pattern_str}"));
    }
    let total = results.len();
    let body = results.join("\n");
    if total >= max_results {
        Ok(format!("{body}\n\n(showing first {max_results} matches)"))
    } else {
        Ok(format!("{body}\n\n({total} matches)"))
    }
}

fn has_ripgrep() -> bool {
    static CACHED: Mutex<Option<bool>> = Mutex::new(None);
    let mut guard = CACHED.lock().unwrap();
    if let Some(v) = *guard {
        return v;
    }
    let ok = std::process::Command::new("rg")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    *guard = Some(ok);
    ok
}

// ---------------------------------------------------------------------------
// todo_write — durable per-session task list
// ---------------------------------------------------------------------------

static TODOS: Mutex<Vec<TodoItem>> = Mutex::new(Vec::new());

#[derive(Clone, Debug)]
struct TodoItem {
    content: String,
    status: TodoStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            _ => None,
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Self::Pending => "[ ]",
            Self::InProgress => "[~]",
            Self::Completed => "[x]",
        }
    }
}

fn tool_todo_write(input: &Value) -> Result<String> {
    let todos = input["todos"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing 'todos' array"))?;

    let mut parsed: Vec<TodoItem> = Vec::with_capacity(todos.len());
    for (i, t) in todos.iter().enumerate() {
        let content = t["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("todos[{i}].content must be a string"))?
            .to_string();
        let status_raw = t["status"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("todos[{i}].status must be a string"))?;
        let status = TodoStatus::parse(status_raw).ok_or_else(|| {
            anyhow::anyhow!(
                "todos[{i}].status = {status_raw:?}; must be pending|in_progress|completed"
            )
        })?;
        parsed.push(TodoItem { content, status });
    }

    let in_progress = parsed.iter().filter(|t| t.status == TodoStatus::InProgress).count();
    if in_progress > 1 {
        anyhow::bail!("at most one todo can be in_progress at a time (found {in_progress})");
    }

    *TODOS.lock().unwrap() = parsed.clone();

    let mut out = String::from("TODO list updated:\n");
    for t in &parsed {
        out.push_str(&format!("  {} {}\n", t.status.icon(), t.content));
    }
    let pending = parsed.iter().filter(|t| t.status == TodoStatus::Pending).count();
    let done = parsed.iter().filter(|t| t.status == TodoStatus::Completed).count();
    out.push_str(&format!(
        "\n({} total: {done} done, {in_progress} in-progress, {pending} pending)",
        parsed.len()
    ));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_normalization_finds_curly() {
        let hay = "let x = \u{201C}hello\u{201D};";
        let needle = "\"hello\"";
        let (actual, count) = find_actual_match(hay, needle);
        assert_eq!(count, 1);
        // Recovered the original curly substring
        assert_eq!(actual, "\u{201C}hello\u{201D}");
    }

    #[test]
    fn exact_match_preferred() {
        let (actual, count) = find_actual_match("foo bar foo", "foo");
        assert_eq!(actual, "foo");
        assert_eq!(count, 2);
    }

    #[test]
    fn no_match_returns_zero() {
        let (_, count) = find_actual_match("abc", "xyz");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn bash_captures_stdout_and_exit() {
        let (out, err) = execute("bash", &serde_json::json!({"command": "echo hello"})).await;
        assert!(!err);
        assert!(out.contains("hello"));
    }

    #[tokio::test]
    async fn bash_timeout_works() {
        // `sleep` exists on Unix; on Windows we use `ping` for the same effect.
        #[cfg(unix)]
        let cmd = "sleep 5";
        #[cfg(windows)]
        let cmd = "ping -n 6 127.0.0.1 > NUL";

        let (out, err) = execute(
            "bash",
            &serde_json::json!({"command": cmd, "timeout_secs": 1}),
        )
        .await;
        assert!(err);
        assert!(out.contains("timed out"));
    }

    #[test]
    fn sibling_temp_path_keeps_parent() {
        use std::path::PathBuf;
        let p = PathBuf::from("foo/bar/baz.rs");
        let tmp = sibling_temp_path(&p);
        assert_eq!(tmp.parent(), Some(std::path::Path::new("foo/bar")));
        let name = tmp.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with(".baz.rs.claude-rs.tmp."));
    }

    #[test]
    fn sibling_temp_path_no_parent() {
        let p = std::path::PathBuf::from("baz.rs");
        let tmp = sibling_temp_path(&p);
        assert_eq!(tmp.parent(), Some(std::path::Path::new("")));
        let name = tmp.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with(".baz.rs.claude-rs.tmp."));
    }

    #[test]
    fn build_shell_command_picks_native_shell() {
        // We can't run the command here without a real shell, but we can at
        // least verify that constructing the command doesn't panic and that
        // the program path is non-empty.
        let cmd = build_shell_command("echo ok");
        let std_cmd = cmd.as_std();
        assert!(!std_cmd.get_program().is_empty());
    }
}

/// Public accessor for the REPL `/todo` command.
pub fn current_todos_pretty() -> String {
    let todos = TODOS.lock().unwrap();
    if todos.is_empty() {
        return "(no todos)".into();
    }
    let mut s = String::new();
    for t in todos.iter() {
        s.push_str(&format!("  {} {}\n", t.status.icon(), t.content));
    }
    s
}
