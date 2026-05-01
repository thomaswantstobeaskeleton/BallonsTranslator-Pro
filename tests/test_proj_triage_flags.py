import sys
import types

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.__getattr__ = lambda name: 1
sys.modules["cv2"] = cv2_stub

from utils.proj_imgtrans import ProjImgTrans


def test_triage_flags_add_and_mark_reviewed():
    p = ProjImgTrans()
    p.pages = {'p1': []}
    p._image_info = {'p1': {'finish_code': 0, 'ignored': False}}

    p.add_triage_flags('p1', [1, 2])
    assert p._image_info['p1']['triage_flags']['1'] == 'open'
    assert p._image_info['p1']['triage_flags']['2'] == 'open'

    p.mark_triage_reviewed('p1', [2])
    assert p._image_info['p1']['triage_flags']['2'] == 'reviewed'
