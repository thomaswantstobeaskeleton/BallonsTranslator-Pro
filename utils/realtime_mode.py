from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import time
import hashlib
import numpy as np
try:
    import mss  # type: ignore
except Exception:
    mss = None
try:
    import platform
except Exception:
    platform = None


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
    window_id: str = ""
    follow_window: bool = False
    crop: Optional[Tuple[int, int, int, int]] = None


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

    def capture_window(self, window_id: str, crop: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        if crop is None:
            crop = (0, 0, 1, 1)
        return self.capture_region(crop)

    def list_windows(self) -> List[Dict[str, Any]]:
        return []

    def resolve_follow_window_rect(self, window_id: str, crop: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        for w in self.list_windows():
            if str(w.get("id", "")) == str(window_id):
                rect = tuple(w.get("rect", (0, 0, 1, 1))[:4])
                if crop:
                    x, y, ww, hh = [int(v) for v in rect]
                    cx, cy, cw, ch = [int(v) for v in crop]
                    return (x + cx, y + cy, cw, ch)
                return rect
        return None


class NumpyFrameBackend(ScreenshotBackendBase):
    backend_name = "numpy_test"
    supports_follow_window = True

    def __init__(self, frames: List[np.ndarray]):
        self._frames = list(frames)
        self._idx = 0
        self._windows: List[Dict[str, Any]] = []

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        if not self._frames:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        frame = self._frames[min(self._idx, len(self._frames)-1)]
        self._idx += 1
        return frame

    def list_windows(self) -> List[Dict[str, Any]]:
        return list(self._windows)

    def set_windows(self, wins: List[Dict[str, Any]]):
        self._windows = list(wins or [])


class QtFallbackBackend(ScreenshotBackendBase):
    backend_name = "qt_fallback"

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        # Safe fallback: return an empty frame if Qt screen capture path is unavailable.
        x, y, w, h = [int(v) for v in region]
        w = max(1, w)
        h = max(1, h)
        return np.zeros((h, w, 3), dtype=np.uint8)


class MSSBackend(ScreenshotBackendBase):
    backend_name = "mss"
    supports_follow_window = False
    supports_window_exclusion = False
    supports_high_dpi = True

    def __init__(self):
        if mss is None:
            raise RuntimeError("mss not available")
        self._mss = mss.mss()

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = [int(v) for v in region]
        w = max(1, w)
        h = max(1, h)
        shot = self._mss.grab({"left": x, "top": y, "width": w, "height": h})
        arr = np.array(shot)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        return np.zeros((h, w, 3), dtype=np.uint8)


class WindowsNativeBackend(ScreenshotBackendBase):
    backend_name = "windows_native_stub"
    supports_follow_window = True
    supports_window_exclusion = False
    supports_high_dpi = True

    def __init__(self):
        self._mss_fallback = MSSBackend() if mss is not None else QtFallbackBackend()

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        return self._mss_fallback.capture_region(region)


class OverlayExclusionBackend(ScreenshotBackendBase):
    backend_name = "overlay_exclusion_wrapper"

    def __init__(self, inner: ScreenshotBackendBase, hide_overlay_cb: Optional[Callable[[], None]] = None, show_overlay_cb: Optional[Callable[[], None]] = None):
        self.inner = inner
        self.hide_overlay_cb = hide_overlay_cb
        self.show_overlay_cb = show_overlay_cb
        self.supports_window_exclusion = bool(getattr(inner, "supports_window_exclusion", False))
        self.supports_follow_window = bool(getattr(inner, "supports_follow_window", False))
        self.supports_high_dpi = bool(getattr(inner, "supports_high_dpi", True))

    def capture_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        if self.supports_window_exclusion:
            return self.inner.capture_region(region)
        if self.hide_overlay_cb is not None:
            self.hide_overlay_cb()
        try:
            return self.inner.capture_region(region)
        finally:
            if self.show_overlay_cb is not None:
                self.show_overlay_cb()

    def capture_window(self, window_id: str, crop: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        if self.supports_window_exclusion:
            return self.inner.capture_window(window_id, crop)
        if self.hide_overlay_cb is not None:
            self.hide_overlay_cb()
        try:
            return self.inner.capture_window(window_id, crop)
        finally:
            if self.show_overlay_cb is not None:
                self.show_overlay_cb()

    def list_windows(self) -> List[Dict[str, Any]]:
        return self.inner.list_windows()


def create_screenshot_backend(preferred: str = "auto") -> ScreenshotBackendBase:
    choice = str(preferred or "auto").strip().lower()
    if choice in {"numpy_test"}:
        return NumpyFrameBackend([])
    if choice in {"windows_native"}:
        if platform is not None and platform.system().lower().startswith("win"):
            try:
                return WindowsNativeBackend()
            except Exception:
                return QtFallbackBackend()
        return QtFallbackBackend()
    if choice in {"mss"}:
        if mss is not None:
            return MSSBackend()
        return QtFallbackBackend()
    if choice in {"auto"} and mss is not None:
        try:
            if platform is not None and platform.system().lower().startswith("win"):
                try:
                    return WindowsNativeBackend()
                except Exception:
                    pass
            return MSSBackend()
        except Exception:
            return QtFallbackBackend()
    # In this baseline slice we keep compatibility-first fallback behavior.
    # Future slices can inject MSS/Windows-native backends here.
    return QtFallbackBackend()


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
        capture_rect = region.rect
        if region.follow_window and region.window_id and self.backend.supports_follow_window:
            resolved = self.backend.resolve_follow_window_rect(region.window_id, region.crop)
            if resolved is not None:
                capture_rect = resolved
                st.warnings = [w for w in st.warnings if w != "follow_window_unavailable"]
            else:
                if "follow_window_unavailable" not in st.warnings:
                    st.warnings.append("follow_window_unavailable")
        img = self.backend.capture_region(capture_rect)
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
        self._default_region_id = "default"

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "region_count": len(self.regions),
            "regions": sorted(self.regions.keys()),
            "states": {k: vars(v) for k, v in self.states.items()},
        }

    def ensure_region(self, region_id: str = "default", rect: Tuple[int, int, int, int] = (0, 0, 640, 360), profile: str = "generic_screen_ocr") -> RealtimeRegion:
        rid = str(region_id or "default").strip() or "default"
        if rid not in self.regions:
            self.regions[rid] = RealtimeRegion(region_id=rid, rect=tuple(int(v) for v in rect[:4]), profile=str(profile or "generic_screen_ocr"))
        return self.regions[rid]

    def apply_defaults(self, *, rect: Tuple[int, int, int, int], profile: str = "generic_screen_ocr", follow_window: bool = False):
        region = self.ensure_region(region_id=self._default_region_id, rect=rect, profile=profile)
        region.rect = tuple(int(v) for v in rect[:4])
        region.profile = str(profile or "generic_screen_ocr")
        region.follow_window = bool(follow_window)
        return region

    def tick_region(self, region_id: str = "default") -> Dict[str, Any]:
        region = self.ensure_region(region_id=region_id)
        state = self.watcher.tick(region)
        self.states[region.region_id] = state
        return {"region_id": region.region_id, "status": state.status, "ocr_text": state.last_ocr_text, "translation": state.last_translation}
