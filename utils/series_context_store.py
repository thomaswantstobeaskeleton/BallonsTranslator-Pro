"""
Series-level translation context storage for cross-chapter consistency.

Store glossary and recent source+translation context in a folder so all chapters
of a series (e.g. Rebirth of the Urban Immortal Cultivator) share the same terms
and style. The translator loads from this folder and appends after each page.

Folder layout:
  {series_context_dir}/
    glossary.txt          # One line per entry: source -> target
    recent_context.json   # Last N pages {sources, translations} across chapters
"""

import os
import os.path as osp
import json
from typing import List, Dict, Optional, Tuple

from .shared import PROGRAM_PATH

DEFAULT_SERIES_CONTEXT_DIR = osp.join(PROGRAM_PATH, "data", "translation_context")

# Default series ID when no path is set (so context is never blank)
DEFAULT_SERIES_ID = "default"

GLOSSARY_FILENAME = "glossary.txt"
RECENT_CONTEXT_FILENAME = "recent_context.json"


def get_series_context_dir(series_id_or_path: str) -> str:
    r"""
    Resolve to an absolute directory path for the series context.
    - If series_id_or_path contains a path separator (/ or \), treat as path
      (relative to PROGRAM_PATH if not absolute).
    - Otherwise treat as series ID and return {DEFAULT_SERIES_CONTEXT_DIR}/{series_id}.
    """
    if not (series_id_or_path or "").strip():
        return ""
    s = series_id_or_path.strip()
    if "/" in s or "\\" in s:
        if osp.isabs(s):
            return s
        return osp.join(PROGRAM_PATH, s)
    return osp.join(DEFAULT_SERIES_CONTEXT_DIR, s)


def ensure_series_dir(path: str) -> bool:
    """Create directory if it doesn't exist. Returns True if path is non-empty and ready."""
    if not path:
        return False
    os.makedirs(path, exist_ok=True)
    return True


def load_series_glossary(series_context_path: str) -> List[Tuple[str, str]]:
    """
    Load glossary from {series_context_path}/glossary.txt.
    Format: one entry per line, e.g. "source -> target" or "source = target".
    Target may use " | " for alternates (e.g. "Chu Province | Chuzhou"); the translator
    accepts any variant and uses the first for replacements. Returns list of (source, target) pairs.
    """
    if not series_context_path:
        return []
    path = osp.join(series_context_path, GLOSSARY_FILENAME)
    if not osp.isfile(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("->", "→", "=", ":"):
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        if k and v:
                            out.append((k, v))
                        break
    return out


def load_recent_context(
    series_context_path: str,
    max_pages: int = 10,
) -> List[Dict]:
    """
    Load recent page context from {series_context_path}/recent_context.json.
    Returns list of {"sources": [...], "translations": [...]}, at most max_pages.
    """
    if not series_context_path:
        return []
    path = osp.join(series_context_path, RECENT_CONTEXT_FILENAME)
    if not osp.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return data[-max_pages:] if len(data) > max_pages else data


def save_recent_context(
    series_context_path: str,
    pages_list: List[Dict],
    max_pages: int = 15,
) -> None:
    """
    Save recent page context to {series_context_path}/recent_context.json.
    pages_list is appended to existing content, then trimmed to last max_pages.
    Each item is {"sources": [...], "translations": [...]}.
    """
    if not series_context_path or not ensure_series_dir(series_context_path):
        return
    path = osp.join(series_context_path, RECENT_CONTEXT_FILENAME)
    existing = load_recent_context(series_context_path, max_pages=max_pages * 2)
    combined = existing + pages_list
    trimmed = combined[-max_pages:] if len(combined) > max_pages else combined
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def append_page_to_series_context(
    series_context_path: str,
    sources: List[str],
    translations: List[str],
    max_stored_pages: int = 15,
) -> None:
    """
    Append one page (sources + translations) to the series recent context file.
    Loads existing recent_context.json, appends this page, keeps last max_stored_pages, saves.
    """
    if not series_context_path or not ensure_series_dir(series_context_path):
        return
    page_entry = {
        "sources": [s if isinstance(s, str) else str(s) for s in sources],
        "translations": [t if isinstance(t, str) else str(t) for t in translations],
    }
    path = osp.join(series_context_path, RECENT_CONTEXT_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append(page_entry)
    trimmed = existing[-max_stored_pages:] if len(existing) > max_stored_pages else existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def merge_glossary_no_dupes(
    *sources: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Merge glossary lists; first occurrence of each source wins. Order preserved."""
    seen = set()
    out = []
    for lst in sources:
        for s, t in lst:
            if s not in seen and s and t:
                seen.add(s)
                out.append((s, t))
    return out


def append_to_series_glossary(
    series_context_path: str,
    entries: List[Tuple[str, str]],
) -> None:
    """
    Append (source, target) pairs to {series_context_path}/glossary.txt.
    One line per entry: source -> target. Skips duplicates (by source) already in file.
    """
    if not series_context_path or not entries:
        return
    if not ensure_series_dir(series_context_path):
        return
    path = osp.join(series_context_path, GLOSSARY_FILENAME)
    existing_sources = {s for s, _ in load_series_glossary(series_context_path)}
    with open(path, "a", encoding="utf-8") as f:
        for s, t in entries:
            if not s or not t or s in existing_sources:
                continue
            existing_sources.add(s)
            f.write(f"{s} -> {t}\n")
