import re
from typing import Dict, List, Tuple


def default_profiles() -> Dict[str, List[Tuple[str, str, str]]]:
    return {
        "cleanup_spaces": [
            (r"[ \t]{2,}", " ", ""),
            (r"\s+\n", "\n", ""),
        ],
        "sfx_compact": [
            (r"([!?]){2,}", r"\1", ""),
            (r"\.{4,}", "...", ""),
        ],
    }


def apply_profile(text: str, rules: List[Tuple[str, str, str]]) -> str:
    out = text or ""
    for pattern, repl, flags in rules or []:
        f = 0
        if "i" in (flags or ""):
            f |= re.IGNORECASE
        out = re.sub(pattern, repl, out, flags=f)
    return out
