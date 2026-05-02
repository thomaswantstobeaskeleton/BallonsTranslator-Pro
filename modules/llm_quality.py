from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


@dataclass
class GlossaryEnforcementResult:
    text: str
    replacements: List[Tuple[str, str]]


def enforce_glossary(text: str, glossary: Dict[str, str]) -> GlossaryEnforcementResult:
    out = text
    replaced: List[Tuple[str, str]] = []
    for src, tgt in (glossary or {}).items():
        if not src or not tgt:
            continue
        if src in out:
            out = out.replace(src, tgt)
            replaced.append((src, tgt))
    return GlossaryEnforcementResult(text=out, replacements=replaced)


def chunk_text_by_budget(text: str, token_budget: int = 400) -> List[str]:
    if token_budget <= 0:
        token_budget = 400
    words = text.split()
    if len(words) <= token_budget:
        return [text]
    chunks: List[str] = []
    for i in range(0, len(words), token_budget):
        chunks.append(" ".join(words[i:i + token_budget]))
    return chunks


def back_translation_drift_score(source: str, translated_back: str) -> float:
    if not source and not translated_back:
        return 0.0
    return 1.0 - SequenceMatcher(None, source or "", translated_back or "").ratio()
