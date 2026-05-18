"""Utility helpers for subtitle/ASR/OCR correction and timeline refinement via flow fixers.

These helpers are intentionally defensive: if a fixer is unavailable or errors,
original inputs are returned unchanged.
"""
from __future__ import annotations

from typing import Any, List

from utils.logger import logger as LOGGER


def _normalize_lines(lines: List[str]) -> List[str]:
    return [str(x or "").strip() for x in (lines or [])]


def _call_fixer_completion(flow_fixer: Any, *, task: str, payload: dict) -> Any:
    """Call optional generic completion API on fixer, if available."""
    if flow_fixer is None:
        return None
    fn = getattr(flow_fixer, "request_completion", None)
    if not callable(fn):
        return None
    try:
        return fn(task=task, payload=payload)
    except TypeError:
        try:
            return fn(task, payload)
        except Exception:
            return None
    except Exception:
        return None


def correct_ocr_via_fixer(flow_fixer: Any, texts: List[str], lang_hint: str | None = None, glossary: str = "") -> List[str]:
    src = _normalize_lines(texts)
    if not src:
        return src
    response = _call_fixer_completion(
        flow_fixer,
        task="correct_ocr",
        payload={"texts": src, "lang_hint": (lang_hint or "").strip(), "glossary": (glossary or "").strip()},
    )
    if isinstance(response, dict):
        out = response.get("texts") or response.get("corrected")
        if isinstance(out, list) and len(out) == len(src):
            return _normalize_lines(out)
    if isinstance(response, list) and len(response) == len(src):
        return _normalize_lines(response)
    return src


def correct_asr_via_fixer(flow_fixer: Any, texts: List[str], glossary: str = "") -> List[str]:
    src = _normalize_lines(texts)
    if not src:
        return src
    response = _call_fixer_completion(flow_fixer, task="correct_asr", payload={"texts": src, "glossary": (glossary or "").strip()})
    if isinstance(response, dict):
        out = response.get("texts") or response.get("corrected")
        if isinstance(out, list) and len(out) == len(src):
            return _normalize_lines(out)
    if isinstance(response, list) and len(response) == len(src):
        return _normalize_lines(response)
    return src


def reflect_translations_via_fixer(flow_fixer: Any, sources: List[str], translations: List[str], to_lang: str = "en", glossary: str = "") -> List[str]:
    src = _normalize_lines(sources)
    trn = _normalize_lines(translations)
    if not trn:
        return trn
    response = _call_fixer_completion(
        flow_fixer,
        task="reflect_translation",
        payload={"sources": src, "translations": trn, "to_lang": (to_lang or "en").strip(), "glossary": (glossary or "").strip()},
    )
    if isinstance(response, dict):
        out = response.get("translations") or response.get("revised")
        if isinstance(out, list) and len(out) == len(trn):
            return _normalize_lines(out)
    if isinstance(response, list) and len(response) == len(trn):
        return _normalize_lines(response)
    return trn


def improve_subtitle_timeline_via_fixer(
    flow_fixer: Any,
    timeline_texts: List[str],
    target_lang: str = "en",
    chunk_size: int = 80,
    context_lines: int = 20,
) -> List[str]:
    """Run flow fixer across timeline in chunks with rolling context.

    Returns a full list with same length as input; on any error keeps originals.
    """
    texts = _normalize_lines(timeline_texts)
    if not texts:
        return texts
    improve_flow = getattr(flow_fixer, "improve_flow", None)
    if not callable(improve_flow):
        return texts

    chunk_size = max(1, int(chunk_size or 80))
    context_lines = max(0, int(context_lines or 20))

    revised = list(texts)
    for i in range(0, len(texts), chunk_size):
        j = min(len(texts), i + chunk_size)
        chunk = list(revised[i:j])
        if not chunk:
            continue
        ctx_start = max(0, i - context_lines)
        prev_ctx = revised[ctx_start:i]
        prev_entries = [{"translations": [line]} for line in prev_ctx if line is not None]
        try:
            _new_prev, new_chunk = improve_flow(prev_entries, chunk, target_lang=target_lang)
            if isinstance(new_chunk, list) and len(new_chunk) == len(chunk):
                revised[i:j] = _normalize_lines(new_chunk)
        except Exception as e:
            LOGGER.debug("Flow fixer timeline review failed at chunk %d-%d: %s", i, j, e)
            continue
    return revised
