from __future__ import annotations

import os.path as osp
import re
from typing import Dict

_SAFE_FILENAME_RE = re.compile(r"[^\w.()\[\]{}#+=,@ -]+", re.UNICODE)
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename_component(value: str, fallback: str = "page") -> str:
    """Return a cross-platform filename component without path separators."""
    text = str(value or "").replace("/", "_").replace("\\", "_").strip()
    text = _SAFE_FILENAME_RE.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        text = fallback
    if text.upper() in _RESERVED_WINDOWS_NAMES:
        text = f"_{text}"
    return text[:180]


def render_export_filename(template: str, page_name: str, index: int, ext: str, source_kind: str = "rendered") -> str:
    """Render a safe export filename from a user/API template.

    Supported tokens intentionally mirror common manga-batch needs without
    exposing arbitrary format evaluation: ``{index}``, ``{index:03d}``,
    ``{page}``, ``{stem}``, ``{source}``, and ``{ext}``.
    """
    ext = ext if str(ext or "").startswith(".") else f".{ext or 'png'}"
    stem = osp.splitext(osp.basename(str(page_name or f"page_{index}")))[0]
    mapping: Dict[str, str] = {
        "index": str(int(index)),
        "page": sanitize_filename_component(osp.basename(str(page_name or stem)), f"page_{index}"),
        "stem": sanitize_filename_component(stem, f"page_{index}"),
        "source": sanitize_filename_component(source_kind or "rendered", "rendered"),
        "ext": ext.lstrip('.'),
    }
    tmpl = str(template or "{index:03d}").strip() or "{index:03d}"

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith("index:"):
            fmt = token.split(":", 1)[1]
            if re.fullmatch(r"0?\d+d", fmt or ""):
                return ("{0:" + fmt + "}").format(int(index))
            return str(int(index))
        return mapping.get(token, "")

    base = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*(?::[^{}]+)?)\}", repl, tmpl)
    base = sanitize_filename_component(base, f"{int(index):03d}")
    if base.lower().endswith(ext.lower()):
        return base
    return base + ext
