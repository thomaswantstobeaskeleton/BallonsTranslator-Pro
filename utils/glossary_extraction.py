from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}")


def extract_glossary_candidates(texts: List[str], *, min_freq: int = 2) -> List[Dict[str, object]]:
    c = Counter()
    for t in texts or []:
        for tok in _TOKEN.findall(str(t or "")):
            c[tok] += 1
    out: List[Dict[str, object]] = []
    for term, freq in c.items():
        if freq < max(1, int(min_freq)):
            continue
        category = "Recurring"
        if term[:1].isupper() and term[1:].islower():
            category = "Person/Name"
        elif any("\u4e00" <= ch <= "\u9fff" for ch in term):
            category = "CJK Term"
        out.append({"source": term, "target": "", "category": category, "frequency": int(freq)})
    out.sort(key=lambda r: (-int(r["frequency"]), str(r["source"])))
    return out
