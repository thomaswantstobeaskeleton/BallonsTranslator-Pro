from __future__ import annotations

from typing import Any, Dict, List

from utils.translation_review import check_translation_guardrails

PROMPT_PROFILES: Dict[str, Dict[str, Any]] = {
    "dialogue": {"max_len_ratio": 1.7, "allow_same_text": False},
    "narration": {"max_len_ratio": 2.2, "allow_same_text": False},
    "sfx": {"max_len_ratio": 1.3, "allow_same_text": True},
    "signboard": {"max_len_ratio": 1.9, "allow_same_text": True},
    "system": {"max_len_ratio": 2.0, "allow_same_text": False},
}


def resolve_prompt_profile(name: str) -> Dict[str, Any]:
    key = (name or "dialogue").strip().lower()
    return PROMPT_PROFILES.get(key, PROMPT_PROFILES["dialogue"])


def build_translation_qa_report(
    blocks: List[Any],
    glossary: List[Dict[str, str]] | None,
    *,
    profile: str = "dialogue",
    retry_issue_threshold: int = 2,
    repetition_threshold: float = 0.45,
    untranslated_ratio_threshold: float = 0.85,
) -> Dict[str, Any]:
    cfg = resolve_prompt_profile(profile)
    rows: List[Dict[str, Any]] = []
    retry_candidates: List[int] = []
    for idx, b in enumerate(blocks or []):
        src = (getattr(b, "text", "") or getattr(b, "get_text", lambda: "")() or "").strip()
        tgt = (getattr(b, "translation", "") or "").strip()
        issues = check_translation_guardrails(src, tgt, glossary=glossary or [], max_len_ratio=float(cfg["max_len_ratio"]))
        src_tokens = [x for x in src.lower().split() if x]
        tgt_tokens = [x for x in tgt.lower().split() if x]
        rep_ratio = 0.0
        if tgt_tokens and not cfg.get("allow_same_text", False):
            rep_ratio = 1.0 - (len(set(tgt_tokens)) / float(len(tgt_tokens)))
            if rep_ratio >= float(repetition_threshold):
                issues.append(f"High repetition ratio ({rep_ratio:.2f})")
        carry_ratio = 0.0
        if src_tokens and tgt_tokens:
            src_set = set(src_tokens)
            overlap = sum(1 for tok in tgt_tokens if tok in src_set)
            carry_ratio = overlap / float(len(tgt_tokens))
            if carry_ratio >= float(untranslated_ratio_threshold) and not cfg.get("allow_same_text", False):
                issues.append(f"High source carry-over ratio ({carry_ratio:.2f})")
        if cfg.get("allow_same_text", False):
            issues = [x for x in issues if "Untranslated source carry-over" not in x]
        if len(issues) >= int(retry_issue_threshold):
            retry_candidates.append(idx)
        rows.append({"index": idx, "source": src, "translation": tgt, "issues": issues, "issue_count": len(issues), "repetition_ratio": round(rep_ratio, 4), "source_carry_ratio": round(carry_ratio, 4)})
    return {
        "profile": profile,
        "rows": rows,
        "total_blocks": len(rows),
        "issue_blocks": sum(1 for r in rows if r["issue_count"] > 0),
        "retry_candidates": retry_candidates,
        "thresholds": {
            "retry_issue_threshold": int(retry_issue_threshold),
            "repetition_threshold": float(repetition_threshold),
            "untranslated_ratio_threshold": float(untranslated_ratio_threshold),
        },
    }
