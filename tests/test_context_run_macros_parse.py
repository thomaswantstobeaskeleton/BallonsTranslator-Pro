import sys
import types

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.__getattr__ = lambda name: 1
sys.modules["cv2"] = cv2_stub

from utils.config import parse_context_run_macros


def test_parse_context_run_macros_valid_and_invalid():
    out = parse_context_run_macros('[{"id":"m1","label":"Macro A","mode":2,"detect_first":false}]')
    assert out[0]['id'] == 'm1'
    assert out[0]['mode'] == 2

    try:
        parse_context_run_macros('{"bad":1}')
        assert False
    except ValueError:
        pass
