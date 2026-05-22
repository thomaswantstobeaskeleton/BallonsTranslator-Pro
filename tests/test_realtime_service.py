import pytest
pytest.importorskip("cv2", exc_type=ImportError, reason="OpenCV runtime unavailable in test env")

from utils.realtime_mode import RealtimeService, NumpyFrameBackend, RealtimeWatcher
import numpy as np


def test_realtime_service_manual_tick_updates_state():
    svc = RealtimeService()
    frames = [np.ones((2, 2, 3), dtype=np.uint8) * 255]
    svc.watcher = RealtimeWatcher(NumpyFrameBackend(frames), lambda _im: 'src', lambda t: f'tr:{t}')
    out = svc.tick_region('r1')
    assert out['region_id'] == 'r1'
    assert out['status'] in {'translated', 'skipped_unchanged_text'}
    assert 'r1' in svc.states


def test_realtime_status_contains_states_map():
    svc = RealtimeService()
    st = svc.status()
    assert 'states' in st
    assert isinstance(st['states'], dict)


def test_realtime_apply_defaults_sets_default_region():
    svc = RealtimeService()
    region = svc.apply_defaults(rect=(11, 22, 333, 444), profile='manga_reader', follow_window=True)
    assert region.region_id == 'default'
    assert tuple(region.rect) == (11, 22, 333, 444)
    assert region.profile == 'manga_reader'
    assert region.follow_window is True
