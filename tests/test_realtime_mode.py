import numpy as np
from utils.realtime_mode import RealtimeWatcher, RealtimeRegion, NumpyFrameBackend, RealtimePrivacyConfig, create_screenshot_backend, OverlayExclusionBackend


def test_realtime_watcher_skips_unchanged_frames_and_text():
    f0 = np.zeros((10, 10, 3), dtype=np.uint8)
    f1 = np.ones((10, 10, 3), dtype=np.uint8) * 255
    backend = NumpyFrameBackend([f0, f0, f1, f1])
    ocr_calls = []
    tr_calls = []
    def ocr(img):
        ocr_calls.append(1)
        return "hello" if img.mean() > 1 else ""
    def tr(text):
        tr_calls.append(1)
        return "你好" if text else ""

    w = RealtimeWatcher(backend, ocr, tr)
    r = RealtimeRegion("r1", (0, 0, 10, 10))
    s1 = w.tick(r)
    s2 = w.tick(r)
    s3 = w.tick(r)
    s4 = w.tick(r)

    assert s2.status == "skipped_unchanged"
    assert s4.status in {"skipped_unchanged", "skipped_unchanged_text"}
    assert len(ocr_calls) <= 2
    assert len(tr_calls) <= 1


def test_realtime_privacy_defaults_do_not_persist_or_log_text():
    p = RealtimePrivacyConfig()
    assert p.persist_captures is False
    assert p.persist_ocr_text is False
    assert p.persist_translation_text is False
    assert p.log_live_text is False


def test_screenshot_backend_factory_returns_compat_fallback():
    backend = create_screenshot_backend("auto")
    frame = backend.capture_region((0, 0, 12, 7))
    assert backend.backend_name in {"qt_fallback", "numpy_test"}
    assert frame.shape[0] == 7
    assert frame.shape[1] == 12


def test_realtime_follow_window_rect_resolution():
    f = np.ones((5, 5, 3), dtype=np.uint8)
    backend = NumpyFrameBackend([f])
    backend.set_windows([{"id": "chrome1", "rect": (100, 200, 400, 300)}])
    w = RealtimeWatcher(backend, lambda _im: "x", lambda _t: "y")
    r = RealtimeRegion("r2", (0, 0, 1, 1), follow_window=True, window_id="chrome1", crop=(10, 20, 50, 40))
    st = w.tick(r)
    assert st.status == "translated"
    assert "follow_window_unavailable" not in st.warnings


def test_overlay_exclusion_backend_calls_hide_show_when_no_native_exclusion():
    f = np.ones((4, 4, 3), dtype=np.uint8)
    base = NumpyFrameBackend([f])
    calls = []
    b = OverlayExclusionBackend(base, lambda: calls.append("hide"), lambda: calls.append("show"))
    _ = b.capture_region((0, 0, 4, 4))
    assert calls == ["hide", "show"]


def test_screenshot_backend_factory_supports_mss_selection_or_fallback():
    backend = create_screenshot_backend("mss")
    assert backend.backend_name in {"mss", "qt_fallback"}


def test_screenshot_backend_factory_supports_windows_native_selection_or_fallback():
    backend = create_screenshot_backend("windows_native")
    assert backend.backend_name in {"windows_native", "qt_fallback"}
