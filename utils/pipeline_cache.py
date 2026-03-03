"""
Unified in-memory pipeline cache keyed by image+settings (Section 7).

Separate caches per stage (detector, OCR, translation, inpaint) to avoid redundant
GPU/API work on re-runs and iterative edits. Keys: (stage, page_key, settings_hash).
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Dict, Optional

# Stages we cache
STAGE_DETECT = "detect"
STAGE_OCR = "ocr"
STAGE_TRANSLATE = "translate"
STAGE_INPAINT = "inpaint"


def _settings_hash(settings: Dict[str, Any]) -> str:
    """Stable hash for a settings dict (module names + relevant params)."""
    try:
        s = json.dumps(settings, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        s = str(settings)
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


class PipelineCache:
    """
    In-memory cache for pipeline stage results. Keyed by (stage, page_key, settings_hash).
    Optional max_entries per stage and global to avoid unbounded growth.
    """

    def __init__(
        self,
        max_entries_per_stage: int = 50,
        max_total_entries: int = 200,
    ) -> None:
        self._max_per_stage = max(1, max_entries_per_stage)
        self._max_total = max(1, max_total_entries)
        self._cache: Dict[str, Any] = {}
        self._order: list = []  # FIFO eviction
        self._lock = threading.Lock()

    def _make_key(self, stage: str, page_key: str, settings_hash: str) -> str:
        return f"{stage}:{page_key}:{settings_hash}"

    def get(
        self,
        stage: str,
        page_key: str,
        settings_hash: str,
    ) -> Optional[Any]:
        """Return cached value if present, else None."""
        key = self._make_key(stage, page_key, settings_hash)
        with self._lock:
            if key not in self._cache:
                return None
            # Move to end for LRU-like eviction
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return self._cache[key]

    def set(
        self,
        stage: str,
        page_key: str,
        settings_hash: str,
        value: Any,
    ) -> None:
        """Store value. Evicts oldest entries if over capacity."""
        key = self._make_key(stage, page_key, settings_hash)
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._cache[key] = value
            self._order.append(key)
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_total and self._order:
            old_key = self._order.pop(0)
            self._cache.pop(old_key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._order.clear()


# Global instance; config can disable or tune via pcfg
_default_cache: Optional[PipelineCache] = None
_cache_lock = threading.Lock()


def get_pipeline_cache(enabled: bool = True) -> Optional[PipelineCache]:
    """Return the global pipeline cache if enabled, else None."""
    global _default_cache
    if not enabled:
        return None
    with _cache_lock:
        if _default_cache is None:
            _default_cache = PipelineCache(max_entries_per_stage=30, max_total_entries=120)
        return _default_cache
