"""
In-session block-level OCR and translation cache (comic-translate style).

Keys: image hash + module/settings; values: block_id -> text or {source_text, translation}.
Config: pcfg.module.ocr_cache_enabled, pcfg.module.translation_cache_enabled.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from utils.config import pcfg
from utils.textblock import TextBlock


def _generate_image_hash(img) -> str:
    """Stable hash for an image (numpy array). Samples every 10th pixel to keep it fast."""
    try:
        import numpy as np
        if img is None or not hasattr(img, 'shape'):
            return ""
        h, w = img.shape[:2]
        step = max(1, (h * w) // 2000)
        flat = img.reshape(-1)
        indices = slice(0, len(flat), step)
        sample = flat[indices].tobytes()
        return hashlib.md5(sample).hexdigest()[:16]
    except Exception:
        return ""


def _get_block_id(blk: TextBlock) -> str:
    """Stable string ID for a text block from bbox and angle."""
    xyxy = getattr(blk, 'xyxy', None)
    if not xyxy or len(xyxy) != 4:
        return ""
    x1, y1, x2, y2 = [int(x) for x in xyxy]
    angle = int(getattr(blk, 'angle', 0) or 0)
    return f"{x1}_{y1}_{x2}_{y2}_{angle}"


def _find_matching_block_id(
    cache_dict: Dict[str, Any],
    blk: TextBlock,
    tolerance_px: int = 5,
    tolerance_angle: int = 1,
) -> Tuple[Optional[str], Any]:
    """Find a cached block that matches blk within tolerance. Returns (block_id, value) or (None, None)."""
    xyxy = getattr(blk, 'xyxy', None)
    if not xyxy or len(xyxy) != 4:
        return None, None
    angle = int(getattr(blk, 'angle', 0) or 0)
    x1, y1, x2, y2 = [int(x) for x in xyxy]
    for bid, val in cache_dict.items():
        parts = bid.split("_")
        if len(parts) != 5:
            continue
        try:
            cx1, cy1, cx2, cy2, cangle = [int(p) for p in parts]
        except ValueError:
            continue
        if abs(cx1 - x1) <= tolerance_px and abs(cy1 - y1) <= tolerance_px and \
           abs(cx2 - x2) <= tolerance_px and abs(cy2 - y2) <= tolerance_px and \
           abs(cangle - angle) <= tolerance_angle:
            return bid, val
    return None, None


def _settings_hash(settings: Dict[str, Any]) -> str:
    try:
        s = json.dumps(settings, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        s = str(settings)
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


class PipelineCacheManager:
    """
    Block-level OCR and translation cache. Keyed by (image_hash, ocr_key, source_lang, device?)
    for OCR and (image_hash, translator_key, source_lang, target_lang, context_hash) for translation.
    """

    def __init__(self, max_ocr_entries: int = 100, max_trans_entries: int = 100) -> None:
        self._ocr_cache: Dict[str, Dict[str, List[str]]] = {}  # key -> {block_id: text (list of lines)}
        self._trans_cache: Dict[str, Dict[str, Dict[str, str]]] = {}  # key -> {block_id: {source_text, translation}}
        self._ocr_order: List[str] = []
        self._trans_order: List[str] = []
        self._max_ocr = max(1, max_ocr_entries)
        self._max_trans = max(1, max_trans_entries)
        self._lock = threading.Lock()

    def _ocr_key(self, image_hash: str, ocr_name: str, source_lang: str, device: str = "") -> str:
        return f"ocr:{image_hash}:{ocr_name}:{source_lang}:{device}"

    def _trans_key(
        self,
        image_hash: str,
        translator_name: str,
        source_lang: str,
        target_lang: str,
        context_hash: str = "",
    ) -> str:
        return f"trans:{image_hash}:{translator_name}:{source_lang}:{target_lang}:{context_hash}"

    def get_ocr_cache_key(self, image_hash: str, ocr_name: str, source_lang: str, device: str = "") -> str:
        return self._ocr_key(image_hash, ocr_name, source_lang, device)

    def get_translation_cache_key(
        self,
        image_hash: str,
        translator_name: str,
        source_lang: str,
        target_lang: str,
        context_hash: str = "",
    ) -> str:
        return self._trans_key(image_hash, translator_name, source_lang, target_lang, context_hash)

    def can_serve_all_blocks_from_ocr_cache(
        self,
        cache_key: str,
        blk_list: List[TextBlock],
        tolerance_px: int = 5,
        tolerance_angle: int = 1,
    ) -> bool:
        if not getattr(pcfg.module, 'ocr_cache_enabled', True):
            return False
        with self._lock:
            cached = self._ocr_cache.get(cache_key)
            if not cached:
                return False
            for blk in blk_list:
                _, val = _find_matching_block_id(cached, blk, tolerance_px, tolerance_angle)
                if val is None:
                    return False
            return True

    def apply_cached_ocr_to_blocks(
        self,
        cache_key: str,
        blk_list: List[TextBlock],
        tolerance_px: int = 5,
        tolerance_angle: int = 1,
    ) -> None:
        with self._lock:
            cached = self._ocr_cache.get(cache_key)
            if not cached:
                return
            cached = dict(cached)
        for blk in blk_list:
            bid, text_lines = _find_matching_block_id(
                cached, blk, tolerance_px, tolerance_angle
            )
            if text_lines is not None and isinstance(text_lines, list):
                blk.text = list(text_lines)

    def cache_ocr_results(
        self,
        cache_key: str,
        blk_list: List[TextBlock],
    ) -> None:
        if not getattr(pcfg.module, 'ocr_cache_enabled', True):
            return
        with self._lock:
            entry = {}
            for blk in blk_list:
                bid = _get_block_id(blk)
                if not bid:
                    continue
                text = getattr(blk, 'text', None)
                if text is not None:
                    entry[bid] = list(text) if isinstance(text, (list, tuple)) else [str(text)]
            if entry:
                self._ocr_cache[cache_key] = entry
                if cache_key in self._ocr_order:
                    self._ocr_order.remove(cache_key)
                self._ocr_order.append(cache_key)
                while len(self._ocr_cache) > self._max_ocr and self._ocr_order:
                    old = self._ocr_order.pop(0)
                    self._ocr_cache.pop(old, None)

    def can_serve_all_blocks_from_translation_cache(
        self,
        cache_key: str,
        blk_list: List[TextBlock],
        tolerance_px: int = 5,
        tolerance_angle: int = 1,
    ) -> bool:
        if not getattr(pcfg.module, 'translation_cache_enabled', False):
            return False
        with self._lock:
            cached = self._trans_cache.get(cache_key)
            if not cached:
                return False
            for blk in blk_list:
                _, val = _find_matching_block_id(cached, blk, tolerance_px, tolerance_angle)
                if val is None:
                    return False
                source = (blk.get_text() or "").strip() if hasattr(blk, 'get_text') else ""
                if (val.get("source_text") or "").strip() != source:
                    return False
            return True

    def apply_cached_translation_to_blocks(
        self,
        cache_key: str,
        blk_list: List[TextBlock],
        tolerance_px: int = 5,
        tolerance_angle: int = 1,
    ) -> None:
        with self._lock:
            cached = self._trans_cache.get(cache_key)
            if not cached:
                return
            cached = dict(cached)
        for blk in blk_list:
            bid, val = _find_matching_block_id(cached, blk, tolerance_px, tolerance_angle)
            if val and isinstance(val, dict) and "translation" in val:
                blk.translation = val.get("translation", "") or ""

    def cache_translation_results(
        self,
        cache_key: str,
        blk_list: List[TextBlock],
    ) -> None:
        if not getattr(pcfg.module, 'translation_cache_enabled', False):
            return
        with self._lock:
            entry = {}
            for blk in blk_list:
                bid = _get_block_id(blk)
                if not bid:
                    continue
                source = (blk.get_text() or "").strip() if hasattr(blk, 'get_text') else ""
                trans = getattr(blk, 'translation', None) or ""
                entry[bid] = {"source_text": source, "translation": trans}
            if entry:
                self._trans_cache[cache_key] = entry
                if cache_key in self._trans_order:
                    self._trans_order.remove(cache_key)
                self._trans_order.append(cache_key)
                while len(self._trans_cache) > self._max_trans and self._trans_order:
                    old = self._trans_order.pop(0)
                    self._trans_cache.pop(old, None)

    def clear(self) -> None:
        with self._lock:
            self._ocr_cache.clear()
            self._trans_cache.clear()
            self._ocr_order.clear()
            self._trans_order.clear()


_default_cache_manager: Optional[PipelineCacheManager] = None
_cache_manager_lock = threading.Lock()


def get_pipeline_cache_manager() -> PipelineCacheManager:
    global _default_cache_manager
    with _cache_manager_lock:
        if _default_cache_manager is None:
            _default_cache_manager = PipelineCacheManager(max_ocr_entries=80, max_trans_entries=80)
        return _default_cache_manager


def clear_block_level_caches() -> None:
    """Clear OCR and translation block-level caches (e.g. from Tools → Clear caches)."""
    global _default_cache_manager
    with _cache_manager_lock:
        if _default_cache_manager is not None:
            _default_cache_manager.clear()
