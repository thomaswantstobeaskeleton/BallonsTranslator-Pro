import re


_PUNCT_ASCII_TO_FULL = str.maketrans({
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？",
    ":": "：",
    ";": "；",
})

_PUNCT_FULL_TO_ASCII = str.maketrans({v: k for k, v in _PUNCT_ASCII_TO_FULL.items()})


def normalize_text(text: str, profile: str = "balanced") -> str:
    if not text:
        return text
    profile = (profile or "balanced").strip().lower()
    out = text
    out = re.sub(r"\s+\n", "\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    if profile in ("balanced", "cjk"):
        out = out.translate(_PUNCT_ASCII_TO_FULL)
    if profile in ("balanced", "latin"):
        out = out.translate(_PUNCT_FULL_TO_ASCII)
    out = out.replace("…", "...")
    return out.strip()
