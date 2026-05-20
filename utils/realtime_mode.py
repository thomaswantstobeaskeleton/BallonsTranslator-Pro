from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import time
import hashlib
import numpy as np


@dataclass
class RealtimePrivacyConfig:
    persist_captures: bool = False
    persist_ocr_text: bool = False
    persist_translation_text: bool = False
    log_live_text: bool = False


@dataclass
class RealtimeRegion:
    region_id: str
    rect: Tuple[int, int, int, int]
    profile: str = "chrome_manhua_reader"
    paused: bool = False


@dataclass
class RealtimeState:
    status: str = "idle"
    last_hash: str = ""
    last_ocr_text: str = ""
    last_translation: str = ""
    updated_at: float = 0.0
    warnings: List[str] = field(default_factory=list)


class ScreenshotBackendBase:
    backend_name = "base"
    supports_window_exclusion = False
    supports_follow_window = False
    supports_high_dpi = True

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        raise NotImplementedError


class NumpyFrameBackend(ScreenshotBackendBase):
    backend_name = "numpy_test"

    def __init__(self, frames: List[np.ndarray]):
        self._frames = list(frames)
        self._idx = 0

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        if not self._frames:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        frame = self._frames[min(self._idx, len(self._frames)-1)]
        self._idx += 1
        return frame


def frame_hash(im: np.ndarray) -> str:
    return hashlib.sha1(im.tobytes()).hexdigest()


class RealtimeWatcher:
    def __init__(self, backend: ScreenshotBackendBase, ocr_fn: Callable[[np.ndarray], str], tr_fn: Callable[[str], str],
                 *, diff_threshold: float = 0.0, min_ocr_interval_sec: float = 0.0):
        self.backend = backend
        self.ocr_fn = ocr_fn
        self.tr_fn = tr_fn
        self.diff_threshold = float(diff_threshold)
        self.min_ocr_interval_sec = float(min_ocr_interval_sec)
        self.state_by_region: Dict[str, RealtimeState] = {}
        self._last_ocr_ts: Dict[str, float] = {}

    def tick(self, region: RealtimeRegion) -> RealtimeState:
        st = self.state_by_region.setdefault(region.region_id, RealtimeState())
        if region.paused:
            st.status = "paused"
            return st
        img = self.backend.capture_region(region.rect)
        h = frame_hash(img)
        if st.last_hash == h:
            st.status = "skipped_unchanged"
            return st
        now = time.time()
        if (now - self._last_ocr_ts.get(region.region_id, 0.0)) < self.min_ocr_interval_sec:
            st.status = "debounced"
            return st
        st.last_hash = h
        st.status = "ocr_running"
        ocr = (self.ocr_fn(img) or "").strip()
        self._last_ocr_ts[region.region_id] = now
        if ocr == st.last_ocr_text:
            st.status = "skipped_unchanged_text"
            st.updated_at = now
            return st
        st.last_ocr_text = ocr
        st.status = "translating"
        tr = (self.tr_fn(ocr) or "").strip()
        st.last_translation = tr
        st.status = "translated"
        st.updated_at = now
        return st


class RealtimeService:
    """In-process realtime control surface for local API/UI orchestration."""

    def __init__(self):
        self.enabled = False
        self.regions: Dict[str, RealtimeRegion] = {}
        self.states: Dict[str, RealtimeState] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {
            "chrome_manhua_reader": {"id": "chrome_manhua_reader", "label": "Chrome Manhua Reader"},
            "manga_reader": {"id": "manga_reader", "label": "Manga Reader"},
            "generic_screen_ocr": {"id": "generic_screen_ocr", "label": "Generic Screen OCR"},
        }
        self.watcher = RealtimeWatcher(NumpyFrameBackend([]), lambda _im: "", lambda _tx: "")

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "region_count": len(self.regions),
            "regions": sorted(self.regions.keys()),
        }
