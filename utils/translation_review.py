"""Lightweight translation QA helpers: glossary extraction and guardrails."""
from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")


def extract_glossary_candidates(src_texts: List[str], translated_texts: List[str] | None = None, min_freq: int = 2) -> List[Dict[str, str]]:
    """Heuristic term miner (LLM-free fallback) for project glossary bootstrapping."""
    c = Counter()
    # translated_texts reserved for future bilingual scoring; kept for API stability.
    for t in src_texts or []:
        c.update(_TOKEN_RE.findall(t or ""))
    out = []
    for tok, freq in c.most_common():
        if freq < min_freq:
            break
        out.append({"source": tok, "target": ""})
    return out


def check_translation_guardrails(
    source: str,
    translated: str,
    glossary: List[Dict[str, str]] | None = None,
    max_len_ratio: float = 1.8,
) -> List[str]:
    issues: List[str] = []
    s = (source or "").strip()
    t = (translated or "").strip()
    if s and t and s == t:
        issues.append("Untranslated source carry-over detected.")
    if s and t and len(t) > max(1, int(len(s) * max_len_ratio)):
        issues.append(f"Translation may be overlong for bubble layout (len ratio > {max_len_ratio:.1f}).")
    if glossary:
        gl_map = {str(g.get('source', '')).strip(): str(g.get('target', '')).strip() for g in glossary if isinstance(g, dict)}
        for gs, gt in gl_map.items():
            if not gs or not gt:
                continue
            if gs in s and gt not in t:
                issues.append(f"Glossary mismatch: expected '{gt}' for source term '{gs}'.")
    return issues
