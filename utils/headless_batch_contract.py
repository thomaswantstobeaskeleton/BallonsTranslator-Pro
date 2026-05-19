from __future__ import annotations

from typing import Set

ALLOWED_STAGES = {"detect", "ocr", "translate", "inpaint"}


def parse_stage_set(raw: str) -> Set[str]:
    text = str(raw or "").strip().lower()
    if not text:
        return set(ALLOWED_STAGES)
    selected = {part.strip() for part in text.split(",") if part.strip()}
    invalid = sorted(selected - ALLOWED_STAGES)
    if invalid:
        raise ValueError(f"invalid stage(s): {', '.join(invalid)}; allowed: detect,ocr,translate,inpaint")
    return selected
