"""
Local server flow fixer: calls an OpenAI-compatible API (Ollama, LM Studio, etc.)
to improve subtitle flow. No cloud API; runs on localhost.
"""
from typing import Any, Dict, List, Tuple

from utils.logger import logger as LOGGER

from . import BaseFlowFixer, register_flow_fixer
from .context_builder import (
    RECENT_LINES_KEEP,
    build_flow_fixer_user_content,
    build_prev_lines_from_entries,
    get_summarize_prompt,
    parse_summary_response,
    should_summarize,
)
from .response_parser import parse_flow_fixer_response


@register_flow_fixer(name="local_server")
class LocalServerFlowFixer(BaseFlowFixer):
    """Flow fixer that calls a local OpenAI-compatible server (Ollama / LM Studio)."""

    def __init__(
        self,
        server_url: str = "http://localhost:1234/v1",
        model: str = "local",
        max_tokens: int = 256,
        timeout: float = 30.0,
        **kwargs: Any,
    ):
        self.server_url = (server_url or "").strip() or "http://localhost:1234/v1"
        self.model = (model or "").strip() or "local"
        self.max_tokens = max(64, int(max_tokens))
        self.timeout = max(5.0, min(120.0, float(timeout)))
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
            self._client = openai.OpenAI(
                base_url=self.server_url,
                api_key="ollama",  # Ollama/LM Studio often ignore key
                timeout=self.timeout,
            )
            return self._client
        except Exception as e:
            LOGGER.warning("Flow fixer: could not create OpenAI client for %s: %s", self.server_url, e)
            return None

    def improve_flow(
        self,
        previous_entries: List[Dict[str, Any]],
        new_translations: List[str],
        target_lang: str = "en",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not new_translations:
            return previous_entries, new_translations
        client = self._get_client()
        if client is None:
            return previous_entries, new_translations

        # Build context: all previous lines from entries (dialog already slices to flow_fixer_context)
        prev_lines = build_prev_lines_from_entries(previous_entries)

        # When near token limit, summarize older portion so we stay within context
        if should_summarize(prev_lines, new_translations):
            try:
                old_lines = prev_lines[:-RECENT_LINES_KEEP]
                summary_prompt = get_summarize_prompt(old_lines)
                sum_resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=min(256, self.max_tokens),
                    temperature=0.2,
                )
                if sum_resp and sum_resp.choices:
                    summary_content = (sum_resp.choices[0].message.content or "").strip()
                    summary_lines = parse_summary_response(summary_content)
                    if summary_lines:
                        prev_lines = summary_lines + prev_lines[-RECENT_LINES_KEEP:]
                        LOGGER.debug("Flow fixer (local): summarized older context to %d lines", len(prev_lines))
            except Exception as e:
                LOGGER.debug("Flow fixer (local): summarization failed, using full context: %s", e)

        _, user_content = build_flow_fixer_user_content(prev_lines, new_translations, use_parts=True)

        n_prev, n_new = len(prev_lines), len(new_translations)
        LOGGER.debug(
            "Flow fixer (local): improve_flow request n_prev=%s n_new=%s new_lines=%s",
            n_prev,
            n_new,
            new_translations[:3] if len(new_translations) > 3 else new_translations,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=self.max_tokens,
                temperature=0.2,
            )
        except Exception as e:
            LOGGER.debug("Flow fixer (local): request failed: %s", e)
            return previous_entries, new_translations

        if not response or not response.choices:
            LOGGER.debug("Flow fixer (local): empty response or no choices")
            return previous_entries, new_translations
        content = (response.choices[0].message.content or "").strip()
        if not content:
            LOGGER.debug("Flow fixer (local): empty message content")
            return previous_entries, new_translations

        data, parse_err = parse_flow_fixer_response(
            content, n_prev=n_prev, n_new=n_new, log_prefix="Flow fixer (local)"
        )
        if data is None:
            if parse_err and parse_err != "empty response":
                LOGGER.debug("Flow fixer (local): %s", parse_err)
            return previous_entries, new_translations

        revised_previous = data.get("revised_previous")
        revised_new = data.get("revised_new")
        if not revised_new:
            return previous_entries, new_translations
        revised_new = [str(s).strip() or orig for s, orig in zip(revised_new, new_translations)]
        # Pad or trim revised_previous to exactly n_prev so we can always map back to entries
        if isinstance(revised_previous, list) and prev_lines:
            if len(revised_previous) >= n_prev:
                adjusted_prev = [str(s).strip() or prev_lines[i] for i, s in enumerate(revised_previous[:n_prev])]
            else:
                adjusted_prev = [str(s).strip() or prev_lines[i] for i, s in enumerate(revised_previous)]
                adjusted_prev.extend(prev_lines[len(revised_previous) : n_prev])
            revised_previous = adjusted_prev
        n_prev_used = int(len(revised_previous)) if isinstance(revised_previous, list) else 0
        n_new_used = int(len(revised_new))
        LOGGER.debug("Flow fixer (local): applied revisions n_prev=%d n_new=%d", n_prev_used, n_new_used)

        # Map revised_previous back onto previous_entries (same order we built prev_lines)
        if isinstance(revised_previous, list) and len(revised_previous) == len(prev_lines):
            rev_idx = 0
            new_prev = []
            for ent in previous_entries:
                trans = ent.get("translations") or []
                n = len(trans)
                if rev_idx + n <= len(revised_previous):
                    new_prev.append({
                        **dict(ent),
                        "translations": [revised_previous[rev_idx + i].strip() for i in range(n)],
                    })
                    rev_idx += n
                else:
                    new_prev.append(dict(ent))
            if rev_idx == len(revised_previous):
                return new_prev, revised_new
        return previous_entries, revised_new
