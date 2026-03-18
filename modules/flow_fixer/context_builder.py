"""
Build flow-fixer context: split previous lines into parts (distinct segments) and optionally
summarize when context is long so we stay near token limits.
"""
from typing import Any, Dict, List, Tuple

# System message to discourage </think> / reasoning and force JSON-only output
FLOW_FIXER_SYSTEM_MESSAGE = (
    "You must respond with ONLY a single JSON object. No </think>, no reasoning, no explanation. "
    "Start your response with { and end with }. "
    "The JSON must have keys revised_previous (array of strings) and revised_new (array of strings). "
    "revised_previous must have exactly one string per previous line given in the user message (same order). "
    "revised_new must have exactly one string per new line given (same count as new lines—no extra items). "
    "Do NOT invent new facts or add new story content. Only do small edits: punctuation, minor wording, pronouns. "
    "Do NOT move meaning between lines. Each line must keep its original meaning; do not rewrite one line to contain another line's content. "
    "Do NOT add speaker labels or prefixes like 'I:' or 'Narrator:' unless they appear in the input line. "
    "When the user says the last previous line and the new line form one sentence, you MUST change at least one of them (add comma or subject). Copying the input unchanged is wrong. "
    "Any other output will cause failure."
)

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


def build_flow_fixer_response_schema(n_prev: int, n_new: int) -> Dict[str, Any]:
    """
    Build JSON Schema for flow fixer response (for structured output / LM Studio).
    Enforces exactly n_prev strings in revised_previous and n_new in revised_new.
    """
    return {
        "type": "object",
        "properties": {
            "revised_previous": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n_prev,
                "maxItems": n_prev,
                "description": "One string per previous line, same order.",
            },
            "revised_new": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n_new,
                "maxItems": n_new,
                "description": "One string per new line, same order.",
            },
        },
        "required": ["revised_previous", "revised_new"],
        "additionalProperties": False,
    }


def get_flow_fixer_response_format(n_prev: int, n_new: int) -> Dict[str, Any]:
    """
    Return response_format dict for OpenAI-compatible APIs that support
    structured output (e.g. LM Studio). Use in chat.completions.create().
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "flow_fixer_response",
            "strict": True,
            "schema": build_flow_fixer_response_schema(n_prev, n_new),
        },
    }


def build_prev_lines_from_entries(previous_entries: List[dict]) -> List[str]:
    """Flatten previous_entries into a list of translation lines (order preserved).
    Every entry's translations are included so the model receives all previous lines for context."""
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


def _continuation_hint(prev_lines: List[str], new_translations: List[str]) -> str:
    """If the last previous line and first new line look like one sentence split in two, return a short instruction so the model must add comma/subject."""
    if not prev_lines or not new_translations:
        return ""
    last_prev = (prev_lines[-1] or "").strip()
    first_new = (new_translations[0] or "").strip()
    if not last_prev or not first_new:
        return ""
    if last_prev.endswith(",") or last_prev.endswith(";"):
        return ""
    # New line looks like continuation: starts with past participle (Taken, Called, Left...) or "By..." and no leading subject
    first_lower = first_new.lower()
    continuation_starts = (
        first_new.startswith("Taken ") or first_new.startswith("Taken away")
        or first_new.startswith("Called ") or first_new.startswith("Left ")
        or first_new.startswith("Bringing ") or first_new.startswith("Having ")
        or (first_new.startswith("By ") and len(first_new) > 10)
        or first_lower.startswith("and ") or first_lower.startswith("but ")
    )
    if not continuation_starts:
        return ""
    return (
        "MANDATORY for this request: The last previous line and the new line are one sentence. "
        "In revised_previous, change the last item to end with a comma (e.g. 'Around thirty years old,'). "
        "In revised_new, you may add the subject to the first item (e.g. 'I was taken away from Earth by...'). "
        "Do not return those two lines unchanged."
    )


def build_flow_fixer_user_content(
    prev_lines: List[str],
    new_translations: List[str],
    *,
    use_parts: bool = True,
    token_limit_chars: int = 3200,
) -> Tuple[List[str], str]:
    """
    Build the user message for the flow fixer. The message includes every previous
    line (prev_lines) and every new line (new_translations) so the model has all
    lines and entries needed for each request. If prev_lines is long and we're
    near token limit, caller should first summarize and pass (summary + recent).

    Returns (prev_lines_used, user_content_string).
    """
    # Use flat numbered list when context is small so the model returns one string per line
    # (with parts, models often return one per "part" e.g. 5 items instead of 10)
    use_parts_effective = use_parts and len(prev_lines) > 10
    if use_parts_effective:
        prev_block = format_previous_as_parts(prev_lines)
    else:
        prev_block = "\n".join(f"{i}. {line}" for i, line in enumerate(prev_lines, 1)) + "\n" if prev_lines else "(none)\n"

    user_content = (
        "You improve subtitle flow. Each line is one on-screen subtitle—keep one string per line; do not merge lines.\n\n"
        "HARD RULES (timing/layout): Do not split, merge, reorder, or 'move content between lines'. Each revised line must keep the same meaning as that specific line.\n"
        "HARD RULES (no hallucination): Do not add new plot/story details that are not already in the provided text. If unsure, keep the line close to the original and only fix punctuation/pronouns.\n\n"
        "HARD RULES (format): Do not add speaker tags like 'I:' or 'Narrator:'. Do not add brackets like [SFX] unless present.\n\n"
        "CRITICAL: Do not return the same strings as the input. When the last previous line is the start of a sentence and the new line is the rest (e.g. 'Around thirty years old' then 'Taken away from Earth by...'), you MUST add a comma to that last previous line in revised_previous (e.g. 'Around thirty years old,') and/or add the subject in revised_new (e.g. 'I was taken away from Earth by...'). Copying the input unchanged in that case is invalid.\n\n"
        "Improving flow means:\n"
        "- Add missing pronouns when the speaker is clear but the line is fragmentary (e.g. 'I' alone → 'I am back' or keep with clear context; "
        "'Could it be that...' → 'Could it be that I...' if the speaker is first person). "
        "When two adjacent lines are clearly one thought split across two beats (e.g. 'I' then 'Chen Beixuan'), you may revise the second to 'I'm Chen Beixuan' so it reads properly as its own subtitle and in sequence.\n"
        "- Preserve speaker perspective: when the previous lines are first person (I, I'm back, my), the speaker is the protagonist—use first person for them. "
        "Fix pronoun errors: replace his/her with my when the speaker is referring to themselves (e.g. 'Because of his amazing talent' → 'Because of my amazing talent'); "
        "'Thus began the journey' → 'Thus I began my journey' in first-person context; use I not he for the speaker.\n"
        "- Add or fix punctuation when needed: use ? for questions and ! for exclamations or strong reactions.\n"
        "- **Continuation between lines:** When the last previous line is the start of a sentence and the new line is the continuation (e.g. 'Around thirty years old' then 'Taken away from Earth by the Cangqin Immortal...'), you MUST revise so they read as one sentence: add a **comma** at the end of the previous line ('Around thirty years old,') and/or add the subject to the new line ('I was taken away from Earth...') if the speaker is first person. Do not leave such pairs unconnected.\n"
        "- Smooth fragments and reduce repetition so the sequence flows when watched; do not merge distinct panel/beat lines into one.\n"
        "- Preserve natural relationship and casual-address phrasing (e.g. \"you're like a son to me\", \"bro\", \"buddy\"); if you see literal mistranslations like \"half your brother\" for a guardian figure, fix to natural English (e.g. \"you're like a son to me\"). Do not change correct colloquial terms to literal \"brother\" or \"sister\" when they mean friends.\n"
        "- **You must suggest real revisions when flow is broken.** Do not return the exact same strings when a small change would fix the flow (missing comma, missing 'I' or subject). At least revise the line(s) where punctuation or a subject would connect the thought. Returning identical text when flow is broken is wrong.\n"
        "- When it helps clarity or tone, you may use *italic* for emphasis or off-screen dialogue and **bold** for strong emphasis (these will be rendered on-screen). Use sparingly.\n\n"
    )
    n_prev_lines = len(prev_lines)
    n_new_lines = len(new_translations)
    if use_parts_effective:
        user_content += (
            "Previous subtitle lines are grouped in parts (different moments). "
            "Each numbered line (1., 2., 3., ...) is one subtitle; return one revised string per line—do not merge. "
            "When the last previous line and the new line are one sentence split in two, you must add a comma to the last previous and/or add the subject to the new; do not return them unchanged.\n\n"
        )
    else:
        user_content += (
            "Each numbered line below is one subtitle. Return exactly one revised string per line in the same order. "
            "When the last previous line and the new line are one sentence split in two, you must add a comma and/or subject; do not return them unchanged. "
            "If earlier lines are first person (I, I'm, my), fix 'his/her' → 'my', 'He/She' → 'I' for the speaker.\n\n"
        )
    # When last previous line has no trailing comma and new line looks like a continuation, inject explicit instruction
    continuation_hint = _continuation_hint(prev_lines, new_translations)
    if continuation_hint:
        user_content += continuation_hint + "\n\n"
    user_content += f"Previous subtitle lines ({n_prev_lines} lines; you must return exactly {n_prev_lines} strings in revised_previous):\n"
    user_content += prev_block
    user_content += f"\nNew line(s) to add ({n_new_lines} line(s); revise for flow; you must return exactly {n_new_lines} string(s) in revised_new—no more, no less):\n"
    for i, line in enumerate(new_translations, 1):
        user_content += f"{i}. {line}\n"
    user_content += (
        "\nReturn a JSON object with:\n"
        f"- \"revised_previous\": array of exactly {n_prev_lines} strings. One string for each previous line in order (line 1 → index 0, line 2 → index 1, ...). Do not merge lines. Count: you must return exactly {n_prev_lines} strings—no fewer, no more.\n"
        f"- \"revised_new\": array of exactly {n_new_lines} string(s). One string per new line above. Do not add extra items.\n"
        "Revise wording so the sequence flows when watched and when read in order (pronouns, his→my, fix fragments that are one thought; keep one subtitle per line; same meaning). "
        "Reply with ONLY the JSON object: no </think> tags, no reasoning, no markdown, no text before or after the JSON. "
        "Example: {\"revised_previous\": [\"Line one.\", \"Line two.\"], \"revised_new\": [\"New line.\"]}. "
        "Fragment example: if new line is \"Came back.\" and the speaker is first person, you may return \"I came back.\" in revised_new."
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


def build_flow_fixer_retry_prompt(prev_lines: List[str], new_translations: List[str]) -> str:
    """Minimal prompt for retry when model returned reasoning instead of JSON. No parts, strict JSON-only."""
    n_prev = len(prev_lines)
    n_new = len(new_translations)
    lines = []
    for i, line in enumerate(prev_lines, 1):
        lines.append(f"{i}. {line}")
    lines.append("")
    lines.append("New:")
    for i, line in enumerate(new_translations, 1):
        lines.append(f"{i}. {line}")
    lines.append("")
    lines.append(
        f"Reply with ONLY a JSON object. revised_previous: array of exactly {n_prev} strings (one per line above, same order). "
        f"revised_new: array of exactly {n_new} string(s)—no more, no less. Use exact line text if unchanged. "
        "Add missing pronouns (e.g. I/you) and punctuation (! or ?) when needed. When previous lines are first person, use my not his for the speaker. "
        "You may use *italic* and **bold** in revised strings when it helps (rendered on-screen)."
    )
    return "\n".join(lines)


def parse_summary_response(content: str) -> List[str]:
    """Parse model response into a list of summary lines (non-empty, stripped)."""
    if not content:
        return []
    lines = [ln.strip() for ln in (content or "").strip().splitlines() if ln.strip()]
    return lines[:SUMMARY_MAX_LINES]
