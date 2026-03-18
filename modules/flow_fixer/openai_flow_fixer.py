"""
OpenAI / ChatGPT flow fixer: uses OpenAI API (e.g. gpt-4o-mini, gpt-3.5-turbo) to improve
subtitle flow. Good for using ChatGPT credits; cheap and fast with small models.
"""
from typing import Any, Dict, List, Tuple

from utils.logger import logger as LOGGER

from . import BaseFlowFixer, register_flow_fixer
from .context_builder import (
    FLOW_FIXER_SYSTEM_MESSAGE,
    RECENT_LINES_KEEP,
    build_flow_fixer_retry_prompt,
    build_flow_fixer_user_content,
    build_prev_lines_from_entries,
    get_summarize_prompt,
    parse_summary_response,
    should_summarize,
)
from .response_parser import parse_flow_fixer_response, sanitize_flow_fixer_revisions


@register_flow_fixer(name="openai")
class OpenAIFlowFixer(BaseFlowFixer):
    """Flow fixer that calls OpenAI API (ChatGPT credits). Use gpt-4o-mini or gpt-3.5-turbo for cheap flow passes."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        max_tokens: int = 256,
        timeout: float = 30.0,
        **kwargs: Any,
    ):
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip() or "gpt-4o-mini"
        self.max_tokens = max(64, int(max_tokens))
        self.timeout = max(5.0, min(120.0, float(timeout)))
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            LOGGER.warning("Flow fixer (OpenAI): no API key set; flow fixer disabled.")
            return None
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            return self._client
        except Exception as e:
            LOGGER.warning("Flow fixer (OpenAI): could not create client: %s", e)
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

        prev_lines = build_prev_lines_from_entries(previous_entries)

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
                        LOGGER.debug("Flow fixer (OpenAI): summarized older context to %d lines", len(prev_lines))
            except Exception as e:
                LOGGER.debug("Flow fixer (OpenAI): summarization failed, using full context: %s", e)

        _, user_content = build_flow_fixer_user_content(prev_lines, new_translations, use_parts=True)

        n_prev, n_new = len(prev_lines), len(new_translations)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": FLOW_FIXER_SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,
                stop=["</think>"],
            )
        except Exception as e:
            LOGGER.debug("Flow fixer (OpenAI): request failed: %s", e)
            return previous_entries, new_translations

        if not response or not response.choices:
            LOGGER.debug("Flow fixer (OpenAI): empty response or no choices")
            return previous_entries, new_translations
        content = (response.choices[0].message.content or "").strip()
        if not content:
            LOGGER.debug("Flow fixer (OpenAI): empty message content")
            return previous_entries, new_translations

        data, parse_err = parse_flow_fixer_response(
            content, n_prev=n_prev, n_new=n_new, log_prefix="Flow fixer (OpenAI)"
        )
        if data is None:
            if content.strip().lower().startswith("</think>") or (
                parse_err and "reasoning" in (parse_err or "").lower()
            ):
                retry_prompt = build_flow_fixer_retry_prompt(prev_lines, new_translations)
                try:
                    retry_resp = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": FLOW_FIXER_SYSTEM_MESSAGE},
                            {"role": "user", "content": retry_prompt},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=0,
                        stop=["</think>"],
                    )
                    if retry_resp and retry_resp.choices:
                        content = (retry_resp.choices[0].message.content or "").strip()
                        if content:
                            data, parse_err = parse_flow_fixer_response(
                                content, n_prev=n_prev, n_new=n_new, log_prefix="Flow fixer (OpenAI)"
                            )
                except Exception as e:
                    LOGGER.debug("Flow fixer (OpenAI): retry failed: %s", e)
            if data is None:
                if parse_err and parse_err != "empty response":
                    if "reasoning" in (parse_err or "").lower():
                        LOGGER.info("Flow fixer (OpenAI): %s", parse_err)
                    else:
                        LOGGER.debug("Flow fixer (OpenAI): %s", parse_err)
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
                missing = n_prev - len(adjusted_prev)
                # When one short, repeat last revision (model often drops last line); else pad with originals
                if missing == 1 and adjusted_prev:
                    adjusted_prev.append(adjusted_prev[-1])
                else:
                    adjusted_prev.extend(prev_lines[len(revised_previous) : n_prev])
            revised_previous = adjusted_prev

        # Safety gate: keep subtitles aligned with timing/layout; prevent line drift & hallucinations.
        safe_prev, safe_new, stats = sanitize_flow_fixer_revisions(
            prev_lines,
            new_translations,
            revised_previous if isinstance(revised_previous, list) else None,
            revised_new if isinstance(revised_new, list) else None,
            allow_prev_edits_last_n=2,
        )
        if stats.get("prev_reverted") or stats.get("new_reverted"):
            LOGGER.debug(
                "Flow fixer (OpenAI): safety gate reverted %d previous, %d new line(s)",
                int(stats.get("prev_reverted") or 0),
                int(stats.get("new_reverted") or 0),
            )
        if safe_new is not None:
            revised_new = safe_new
        if safe_prev is not None:
            revised_previous = safe_prev
        n_prev_used = int(len(revised_previous)) if isinstance(revised_previous, list) else 0
        n_new_used = int(len(revised_new))
        n_prev_changed = sum(
            1 for i in range(min(len(prev_lines), n_prev_used))
            if (revised_previous[i] or "").strip() != (prev_lines[i] or "").strip()
        ) if isinstance(revised_previous, list) else 0
        n_new_changed = sum(
            1 for i in range(min(len(new_translations), n_new_used))
            if (revised_new[i] or "").strip() != (new_translations[i] or "").strip()
        )
        if n_prev_changed or n_new_changed:
            LOGGER.debug("Flow fixer (OpenAI): revised %d previous, %d new line(s)", n_prev_changed, n_new_changed)

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
