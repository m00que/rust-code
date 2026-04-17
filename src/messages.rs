//! Message-list maintenance: normalization, token accounting, auto-compact.
//!
//! Mirrors (a small subset of) Claude Code's `normalizeMessagesForAPI`,
//! `tokenCountWithEstimation`, and `autoCompact` semantics.

use crate::api::{ApiMessage, ContentBlock, Usage};

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Sanitize a message list before sending to the API.
///
/// Rules (in order):
///   1. Drop assistant messages whose content is entirely empty/whitespace.
///   2. Merge runs of consecutive same-role messages into one (Bedrock-compat
///      and avoids API 400s when tools and meta-text mix).
///   3. Strip trailing assistant messages that contain only a tool_use without
///      a follow-up tool_result (these would make the API reject the request).
pub fn normalize_for_api(messages: &[ApiMessage]) -> Vec<ApiMessage> {
    let mut out: Vec<ApiMessage> = Vec::with_capacity(messages.len());

    for msg in messages {
        if msg.role == "assistant" && is_empty_assistant(msg) {
            continue;
        }

        match out.last_mut() {
            Some(prev) if prev.role == msg.role => {
                prev.content.extend(msg.content.iter().cloned());
            }
            _ => out.push(msg.clone()),
        }
    }

    // Final pass: every tool_use must have a matching tool_result in the next
    // user message. If not, drop that assistant turn and everything after.
    let mut truncate_at: Option<usize> = None;
    for (i, msg) in out.iter().enumerate() {
        if msg.role != "assistant" {
            continue;
        }
        let tool_use_ids: Vec<&str> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        if tool_use_ids.is_empty() {
            continue;
        }
        let next = out.get(i + 1);
        let satisfied = next.map(|m| {
            m.role == "user"
                && tool_use_ids.iter().all(|tid| {
                    m.content.iter().any(|b| matches!(b,
                        ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == tid))
                })
        }).unwrap_or(false);
        if !satisfied {
            truncate_at = Some(i);
            break;
        }
    }
    if let Some(idx) = truncate_at {
        out.truncate(idx);
    }

    out
}

fn is_empty_assistant(msg: &ApiMessage) -> bool {
    msg.content.iter().all(|b| match b {
        ContentBlock::Text { text } => text.trim().is_empty(),
        ContentBlock::Thinking { .. } => true, // alone, thinking is "empty" too
        _ => false,
    })
}

// ---------------------------------------------------------------------------
// Token accounting
// ---------------------------------------------------------------------------

/// Sum of input + cache + output tokens for a single API turn.
#[allow(dead_code)]
pub fn usage_total(u: &Usage) -> u64 {
    u.input_tokens
        + u.cache_creation_input_tokens
        + u.cache_read_input_tokens
        + u.output_tokens
}

/// Very rough token estimate: ~4 characters / token.
///
/// Used to estimate the size of *new* messages (since the last API turn) that
/// don't yet have authoritative usage numbers from the server.
pub fn estimate_tokens(messages: &[ApiMessage]) -> u64 {
    let mut chars = 0_u64;
    for m in messages {
        for b in &m.content {
            match b {
                ContentBlock::Text { text } => chars += text.len() as u64,
                ContentBlock::Thinking { thinking, .. } => chars += thinking.len() as u64,
                ContentBlock::ToolUse { name, input, .. } => {
                    chars += name.len() as u64;
                    if let Ok(s) = serde_json::to_string(input) {
                        chars += s.len() as u64;
                    }
                }
                ContentBlock::ToolResult { content, .. } => chars += content.len() as u64,
            }
        }
    }
    (chars / 4).max(1)
}

/// Compute the token budget where auto-compact should fire.
///
/// Mirrors Claude Code's `getAutoCompactThreshold`: leave a buffer below the
/// model's context window so we still have room for the next response.
pub fn auto_compact_threshold(context_window: u64) -> u64 {
    const BUFFER: u64 = 13_000;
    context_window.saturating_sub(BUFFER)
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

/// Result of [`compact_in_place`].
pub struct CompactSummary {
    pub kept: usize,
    pub dropped: usize,
}

/// Lightweight, *local* compaction: keep a system-style summary user message
/// plus the last `keep_recent` turns. This is the offline fallback used when
/// we don't want to spend a full API call summarizing.
///
/// (A full Claude-Code-style "summary via the model" lives in the REPL.)
pub fn compact_in_place(messages: &mut Vec<ApiMessage>, keep_recent: usize) -> CompactSummary {
    if messages.len() <= keep_recent + 1 {
        return CompactSummary { kept: messages.len(), dropped: 0 };
    }

    let cutoff = messages.len() - keep_recent;
    let removed = &messages[..cutoff];

    let mut summary = String::from(
        "[earlier conversation auto-compacted to save context]\n\nKey events:\n",
    );
    for (i, m) in removed.iter().enumerate() {
        let role = if m.role == "user" { "User" } else { "Assistant" };
        let mut snippet = String::new();
        for b in &m.content {
            match b {
                ContentBlock::Text { text } => {
                    snippet.push_str(text);
                }
                ContentBlock::ToolUse { name, .. } => {
                    snippet.push_str(&format!(" [used tool: {name}]"));
                }
                ContentBlock::ToolResult { content, is_error, .. } => {
                    let tag = if matches!(is_error, Some(true)) { "tool error" } else { "tool result" };
                    snippet.push_str(&format!(" [{tag}: {} chars]", content.len()));
                }
                ContentBlock::Thinking { .. } => {}
            }
        }
        let snippet = snippet.trim();
        if snippet.is_empty() {
            continue;
        }
        let preview: String = snippet.chars().take(200).collect();
        summary.push_str(&format!("- {} #{i}: {preview}\n", role));
    }

    let dropped = removed.len();
    let recent: Vec<ApiMessage> = messages.split_off(cutoff);
    messages.clear();
    messages.push(ApiMessage::user_text(summary));
    messages.extend(recent);
    CompactSummary { kept: messages.len(), dropped }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ContentBlock;

    fn user(text: &str) -> ApiMessage { ApiMessage::user_text(text) }
    fn assistant(text: &str) -> ApiMessage {
        ApiMessage::assistant(vec![ContentBlock::text(text)])
    }

    #[test]
    fn merges_consecutive_user_messages() {
        let msgs = vec![user("a"), user("b"), assistant("ok")];
        let n = normalize_for_api(&msgs);
        assert_eq!(n.len(), 2);
        assert_eq!(n[0].role, "user");
        assert_eq!(n[0].content.len(), 2);
    }

    #[test]
    fn drops_empty_assistant_turns() {
        let msgs = vec![user("hi"), assistant("   "), user("again")];
        let n = normalize_for_api(&msgs);
        // empty assistant dropped, then two users merge
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].role, "user");
    }

    #[test]
    fn truncates_orphan_tool_use() {
        let msgs = vec![
            user("do it"),
            ApiMessage::assistant(vec![ContentBlock::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            }]),
            // missing tool_result for t1
        ];
        let n = normalize_for_api(&msgs);
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].role, "user");
    }

    #[test]
    fn keeps_paired_tool_use_and_result() {
        let msgs = vec![
            user("do it"),
            ApiMessage::assistant(vec![ContentBlock::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
            }]),
            ApiMessage::user(vec![ContentBlock::tool_result("t1", "out", false)]),
        ];
        let n = normalize_for_api(&msgs);
        assert_eq!(n.len(), 3);
    }

    #[test]
    fn compact_keeps_recent_and_inserts_summary() {
        let mut msgs: Vec<ApiMessage> = (0..10).map(|i| user(&format!("msg {i}"))).collect();
        let s = compact_in_place(&mut msgs, 3);
        assert_eq!(s.kept, 4); // 1 summary + 3 kept
        assert_eq!(s.dropped, 7);
        assert!(msgs[0].content.iter().any(|b| matches!(b,
            ContentBlock::Text { text } if text.contains("auto-compacted"))));
    }
}
