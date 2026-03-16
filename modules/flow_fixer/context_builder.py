"""
Build flow-fixer context: split previous lines into parts (distinct segments) and optionally
summarize when context is long so we stay near token limits.
"""
from typing import List, Tuple

# Rough chars per token for limit check
CHARS_PER_TOKEN = 4
# When previous lines exceed this, we summarize older portion
SUMMARIZE_THRESHOLD_LINES = 20
# After summarization: keep this many most recent lines verbatim
RECENT_LINES_KEEP = 15
# Max summary lines to produce from older context
SUMMARY_MAX_LINES = 6
# Max total previous lines we send (summary + recent)
MAX_PREV_LINES = 25

PART_SIZE = 5  # Group every N lines into a "part" for clarity


def build_prev_lines_from_entries(previous_entries: List[dict]) -> List[str]:
    """Flatten previous_entries into a list of translation lines (order preserved)."""
    prev_lines: List[str] = []
    for ent in previous_entries:
        trans = ent.get("translations") or []
        for t in trans:
            if isinstance(t, str) and t.strip():
                prev_lines.append(t.strip())
    return prev_lines


def format_previous_as_parts(prev_lines: List[str]) -> str:
    """
    Format previous lines in parts so conversations that are apart are distinct.
    Parts are every PART_SIZE lines; labels: Part 1 (older), Part 2, ... Part N (most recent).
    """
    if not prev_lines:
        return "(none)\n"
    parts_text: List[str] = []
    global_idx = 1
    part_num = 1
    total_parts = (len(prev_lines) + PART_SIZE - 1) // PART_SIZE
    for i in range(0, len(prev_lines), PART_SIZE):
        chunk = prev_lines[i : i + PART_SIZE]
        label = "Part 1 (older):" if part_num == 1 else (
            f"Part {part_num} (most recent):" if part_num == total_parts else f"Part {part_num}:"
        )
        parts_text.append(label)
        for line in chunk:
            parts_text.append(f"{global_idx}. {line}")
            global_idx += 1
        part_num += 1
    return "\n".join(parts_text) + "\n"


def build_flow_fixer_user_content(
    prev_lines: List[str],
    new_translations: List[str],
    *,
    use_parts: bool = True,
    token_limit_chars: int = 3200,
) -> Tuple[List[str], str]:
    """
    Build the user message for the flow fixer.
    If prev_lines is long and we're near token limit, caller should first call
    summarize_older_context() and pass (summary + recent) as prev_lines.

    Returns (prev_lines_used, user_content_string).
    """
    if use_parts:
        prev_block = format_previous_as_parts(prev_lines)
    else:
        prev_block = "\n".join(f"{i}. {line}" for i, line in enumerate(prev_lines, 1)) + "\n" if prev_lines else "(none)\n"

    user_content = (
        "You improve subtitle flow so the sequence reads naturally. You may revise BOTH the previous lines and the new lines.\n\n"
    )
    if use_parts:
        user_content += (
            "Previous subtitle lines are grouped in parts (different moments may be in different parts). "
            "Only suggest revisions where flow is broken within or between parts; leave lines unchanged if they read fine.\n\n"
        )
    user_content += "Previous subtitle lines (revise these if it improves flow; return same count):\n"
    user_content += prev_block
    user_content += "\nNew line(s) to add (revise for flow; return same count):\n"
    for i, line in enumerate(new_translations, 1):
        user_content += f"{i}. {line}\n"
    n_prev_lines = len(prev_lines)
    user_content += (
        "\nReturn a JSON object with:\n"
        f"- \"revised_previous\": array of exactly {n_prev_lines} strings, one per previous line above in order (use the exact original line for any you do not change).\n"
        "- \"revised_new\": array of strings, one per new line above (same order and count).\n"
        "Revise wording only when it improves flow; keep meaning the same. "
        "Output only valid JSON, no markdown or </think> tags. "
        "Example: {\"revised_previous\": [\"Line one.\", \"Line two.\"], \"revised_new\": [\"New line.\"]}"
    )
    return prev_lines, user_content


def should_summarize(
    prev_lines: List[str],
    new_translations: List[str],
    token_limit_chars: int = 3200,
) -> bool:
    """True if we should summarize older context to stay near token limit."""
    if len(prev_lines) <= SUMMARIZE_THRESHOLD_LINES:
        return False
    approx = 800 + sum(len(ln) for ln in prev_lines) + sum(len(ln) for ln in new_translations)
    return approx > token_limit_chars


def get_summarize_prompt(old_lines: List[str]) -> str:
    """Prompt for the model to summarize older subtitle lines into fewer lines."""
    text = "\n".join(f"{i}. {line}" for i, line in enumerate(old_lines, 1))
    return (
        f"Summarize the following subtitle lines into 2–{SUMMARY_MAX_LINES} short lines that preserve the main meaning and context. "
        "Output only the summary lines, one per line. No numbering, no JSON, no explanation.\n\n"
        + text
    )


def parse_summary_response(content: str) -> List[str]:
    """Parse model response into a list of summary lines (non-empty, stripped)."""
    if not content:
        return []
    lines = [ln.strip() for ln in (content or "").strip().splitlines() if ln.strip()]
    return lines[:SUMMARY_MAX_LINES]
