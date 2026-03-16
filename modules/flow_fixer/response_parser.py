"""
Shared logic for parsing and validating flow fixer API responses.
Expects JSON: {"revised_previous": [...], "revised_new": [...]}
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import logger as LOGGER


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> so we can parse JSON that follows reasoning. Uses last </think> if multiple."""
    if not content or "</think>" not in content:
        return content
    # Use last </think> so we get the final answer after any nested think
    parts = re.split(r"</think>", content, flags=re.IGNORECASE)
    return parts[-1].strip() if parts else content


def _extract_json_candidates(content: str) -> List[str]:
    """Return candidate strings to parse as JSON: after </think>, raw, code blocks, first {...}."""
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
    # First balanced { ... } (greedy match from first { to last })
    for src in (content, content_no_think):
        start = src.find("{")
        if start >= 0:
            depth = 0
            end = -1
            for i in range(start, len(src)):
                if src[i] == "{":
                    depth += 1
                elif src[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > start:
                candidates.append(src[start : end + 1])
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
    err_msg = f"invalid JSON in response"
    if last_error:
        err_msg += f": {last_error}"
    LOGGER.debug(
        "%s: %s. Response snippet: %s",
        log_prefix,
        err_msg,
        snippet.replace("\n", " ").strip(),
    )
    return None, err_msg
