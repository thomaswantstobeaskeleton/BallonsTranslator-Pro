import sys
import types

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.__getattr__ = lambda name: 1
sys.modules["cv2"] = cv2_stub

from utils.config import ProgramConfig


def test_context_run_macros_default_present():
    cfg = ProgramConfig()
    assert isinstance(cfg.context_run_macros, list)
    assert cfg.context_run_macros
    ids = {m.get('id') for m in cfg.context_run_macros}
    assert 'detect_ocr_translate' in ids
    assert 'ocr_translate_inpaint' in ids
