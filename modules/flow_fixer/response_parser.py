"""
Shared logic for parsing and validating flow fixer API responses.
Expects JSON: {"revised_previous": [...], "revised_new": [...]}
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import logger as LOGGER


_FLOW_FIXER_BANNED_SUBSTRINGS = (
    "unable to extract text from images",
    "unable to extract text from the image",
    "i can't extract text from images",
    "i cannot extract text from images",
    "i can't read text from images",
    "i cannot read text from images",
    "no relevant previous subtitles available",
    # Format leakage / schema keys showing up in revisions
    "revised_previous",
    "revised_new",
    "\"revised_previous\"",
    "\"revised_new\"",
)


def _tokenize_for_similarity(s: str) -> List[str]:
    s = (s or "").lower()
    # Keep words + numbers only; drop punctuation.
    return re.findall(r"[a-z0-9]+", s)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _looks_banned(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(x in t for x in _FLOW_FIXER_BANNED_SUBSTRINGS)


def sanitize_flow_fixer_revisions(
    prev_lines: List[str],
    new_lines: List[str],
    revised_previous: Optional[List[str]],
    revised_new: Optional[List[str]],
    *,
    allow_prev_edits_last_n: int = 2,
    min_prev_similarity: float = 0.55,
    min_new_similarity: float = 0.40,
    max_len_ratio: float = 2.5,
    max_abs_new_chars: int = 220,
    max_abs_prev_chars: int = 240,
    max_prev_len_ratio: float = 1.6,
) -> Tuple[Optional[List[str]], Optional[List[str]], Dict[str, int]]:
    """
    Safety gate for flow-fixer results.

    Goal: prevent the model from "moving content between lines" by rewriting older lines into
    unrelated text, or from hallucinating long new sentences that break subtitle timing.

    Returns (safe_revised_previous, safe_revised_new, stats).
    """
    stats = {
        "prev_reverted": 0,
        "new_reverted": 0,
        "prev_total": len(prev_lines or []),
        "new_total": len(new_lines or []),
    }

    safe_prev: Optional[List[str]] = None
    if isinstance(revised_previous, list) and prev_lines:
        safe_prev = []
        last_n = max(0, int(allow_prev_edits_last_n))
        allow_from = max(0, len(prev_lines) - last_n)
        for i, orig in enumerate(prev_lines):
            cand = str(revised_previous[i]).strip() if i < len(revised_previous) else ""
            if not cand:
                safe_prev.append(orig)
                continue
            if _looks_banned(cand):
                safe_prev.append(orig)
                stats["prev_reverted"] += 1
                continue
            # Only allow edits near the boundary; earlier lines must remain stable to avoid "line drift".
            if i < allow_from:
                if cand.strip() != (orig or "").strip():
                    stats["prev_reverted"] += 1
                safe_prev.append(orig)
                continue
            # Prevent giant expansions of previous lines (hallucinated continuations leaking backwards).
            o_len = len((orig or "").strip())
            c_len = len(cand)
            orig_tokens = _tokenize_for_similarity(orig)
            frag_like = (o_len <= 12) or (len(orig_tokens) <= 2)
            if c_len > max_abs_prev_chars:
                safe_prev.append(orig)
                stats["prev_reverted"] += 1
                continue
            if (not frag_like) and o_len > 0 and (c_len / max(1, o_len)) > max_prev_len_ratio:
                safe_prev.append(orig)
                stats["prev_reverted"] += 1
                continue
            # Similarity gate: punctuation-only changes should pass; major rewrites should not.
            sim = _jaccard(_tokenize_for_similarity(orig), _tokenize_for_similarity(cand))
            if sim < min_prev_similarity and (orig or "").strip() and cand.strip():
                safe_prev.append(orig)
                stats["prev_reverted"] += 1
                continue
            safe_prev.append(cand)

    safe_new: Optional[List[str]] = None
    if isinstance(revised_new, list) and new_lines:
        safe_new = []
        for i, orig in enumerate(new_lines):
            cand = str(revised_new[i]).strip() if i < len(revised_new) else ""
            if not cand:
                safe_new.append(orig)
                continue
            if _looks_banned(cand):
                safe_new.append(orig)
                stats["new_reverted"] += 1
                continue
            # Prevent giant expansions that will not fit the on-screen timing/layout.
            o_len = len((orig or "").strip())
            c_len = len(cand)
            if c_len > max_abs_new_chars:
                safe_new.append(orig)
                stats["new_reverted"] += 1
                continue
            # For very short fragments (e.g. "I", "Reached...", "Nowadays, me"), we must allow
            # adding a subject/pronoun to make it a valid standalone subtitle.
            orig_tokens = _tokenize_for_similarity(orig)
            frag_like = (o_len <= 12) or (len(orig_tokens) <= 2)
            if (not frag_like) and o_len > 0 and (c_len / max(1, o_len)) > max_len_ratio:
                safe_new.append(orig)
                stats["new_reverted"] += 1
                continue
            # Similarity gate: allow smoothing, but reject unrelated rewrites.
            sim = _jaccard(_tokenize_for_similarity(orig), _tokenize_for_similarity(cand))
            # For fragments, similarity can be low even for correct fixes ("Reached..." -> "I reached...").
            if (not frag_like) and sim < min_new_similarity and (orig or "").strip() and cand.strip():
                safe_new.append(orig)
                stats["new_reverted"] += 1
                continue
            safe_new.append(cand)

    return safe_prev, safe_new, stats


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> so we can parse JSON that follows reasoning. Uses last </think> if multiple."""
    if not content or "</think>" not in content:
        return content
    # Use last </think> so we get the final answer after any nested think
    parts = re.split(r"</think>", content, flags=re.IGNORECASE)
    return parts[-1].strip() if parts else content


def _find_balanced_braces(s: str, start: int) -> Optional[Tuple[int, int]]:
    """Return (start, end) of the next balanced { ... } starting at start, or None."""
    i = s.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j + 1)
    return None


def _extract_json_candidates(content: str) -> List[str]:
    """Return candidate strings to parse as JSON: after </think>, raw, code blocks, {...} with revised_* keys."""
    candidates = []
    content = (content or "").strip()
    if not content:
        return candidates
    content_no_think = _strip_think_tags(content)
    # Prefer content after think so we don't parse reasoning as JSON
    if content_no_think and content_no_think != content:
        candidates.append(content_no_think)
    candidates.append(content)
    # Markdown code block
    for src in (content_no_think, content):
        if not src:
            continue
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", src)
        if m:
            candidates.append(m.group(1).strip())
    # All balanced { ... } that contain flow-fixer keys (handles JSON inside </think> or after it)
    for src in (content, content_no_think):
        if not src or '"revised_previous"' not in src or '"revised_new"' not in src:
            continue
        pos = 0
        while True:
            span = _find_balanced_braces(src, pos)
            if not span:
                break
            start, end = span
            candidate = src[start:end]
            if '"revised_previous"' in candidate and '"revised_new"' in candidate:
                candidates.append(candidate)
            pos = end
    # When response is long (e.g. </think> then reasoning), JSON may be at the end; take last tail and try last {...}
    if len(content) > 800:
        tail = content[-2500:]
        idx = tail.rfind("{")
        if idx >= 0:
            span = _find_balanced_braces(tail, idx)
            if span:
                start, end = span
                candidates.append(tail[start:end])
    return candidates


def _try_fix_json(s: str) -> str:
    """Try to fix common JSON issues: trailing commas, missing commas between }/] and \", adjacent strings."""
    s = (s or "").strip()
    if not s:
        return s
    # Remove trailing comma before ] or }
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Missing comma between } or ] and next key (e.g. }"revised_new" -> },"revised_new")
    for _ in range(5):
        prev = s
        s = re.sub(r"([}\]])[\s\n\r]*(?=\"[a-zA-Z_])", r"\1,", s)
        if s == prev:
            break
    # Missing comma between adjacent quoted strings (e.g. "a" "b" or "a"\n"b" in arrays)
    s = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s+"', r'"\1", "', s)
    return s


def parse_flow_fixer_response(
    content: str,
    n_prev: int,
    n_new: int,
    log_prefix: str = "Flow fixer",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse API response into the expected flow fixer format.
    Returns (data, None) on success, or (None, error_message) on failure.
    data is validated to have revised_previous (list len n_prev) and revised_new (list len n_new).
    """
    if not content or not (content := content.strip()):
        return None, "empty response"
    candidates = _extract_json_candidates(content)
    last_error = None
    for raw in candidates:
        for attempt in (raw, _try_fix_json(raw)):
            if not attempt:
                continue
            try:
                data = json.loads(attempt)
            except json.JSONDecodeError as e:
                last_error = e
                continue
            if not isinstance(data, dict):
                last_error = ValueError("response is not a JSON object")
                continue
            revised_previous = data.get("revised_previous")
            revised_new = data.get("revised_new")
            if not isinstance(revised_new, list):
                err = f"invalid response: 'revised_new' must be an array (got {type(revised_new).__name__})"
                LOGGER.debug("%s: %s", log_prefix, err)
                return None, err
            # Empty revised_new: model returned no revisions (e.g. only <think>); treat as "keep originals"
            if len(revised_new) == 0:
                LOGGER.debug("%s: model returned empty revised_new; using original translations", log_prefix)
                data["revised_new"] = []
                data["revised_previous"] = None
                return data, None
            if len(revised_new) < n_new:
                err = f"invalid response: 'revised_new' must have length >= {n_new} (got {len(revised_new)})"
                LOGGER.debug("%s: %s. Snippet: %s", log_prefix, err, (content[:300] + "…").replace("\n", " "))
                return None, err
            if len(revised_new) > n_new:
                LOGGER.debug(
                    "%s: trimming 'revised_new' from %s to %s items (expected count)",
                    log_prefix,
                    len(revised_new),
                    n_new,
                )
                revised_new = revised_new[:n_new]
                data["revised_new"] = revised_new
            if not all(isinstance(x, str) for x in revised_new):
                err = "invalid response: 'revised_new' must contain only strings"
                LOGGER.debug("%s: %s", log_prefix, err)
                return None, err
            # revised_previous is optional: if present must be list of strings; length can differ (we use it only when len matches n_prev)
            if revised_previous is not None and not isinstance(revised_previous, list):
                err = f"invalid response: 'revised_previous' must be an array (got {type(revised_previous).__name__})"
                LOGGER.debug("%s: %s", log_prefix, err)
                return None, err
            if revised_previous is not None and not all(
                isinstance(x, str) for x in revised_previous
            ):
                err = "invalid response: 'revised_previous' must contain only strings"
                LOGGER.debug("%s: %s", log_prefix, err)
                return None, err
            if revised_previous is not None and len(revised_previous) != n_prev:
                LOGGER.debug(
                    "%s: revised_previous length %s (expected %s); fixer will pad/trim",
                    log_prefix,
                    len(revised_previous),
                    n_prev,
                )
                # Leave revised_previous as-is; fixer will pad with originals or trim to n_prev
            return data, None
    snippet = (content[:500] + "…") if len(content) > 500 else content
    err_msg = "invalid JSON in response"
    if last_error:
        err_msg += f": {last_error}"
    # Clearer message when model returned only reasoning (e.g. </think> with no JSON after)
    raw = content.strip().lower()
    if (raw.startswith("</think>") or "</think>" in raw[:80]) and '"revised_previous"' not in content:
        err_msg = "model returned only reasoning (no JSON); use originals or try a model that follows JSON-only"
    LOGGER.debug(
        "%s: %s. Response snippet: %s",
        log_prefix,
        err_msg,
        snippet.replace("\n", " ").strip(),
    )
    return None, err_msg
