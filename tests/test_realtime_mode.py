import numpy as np
from utils.realtime_mode import RealtimeWatcher, RealtimeRegion, NumpyFrameBackend, RealtimePrivacyConfig


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
