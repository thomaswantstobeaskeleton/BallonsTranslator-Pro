import sys
import types

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.__file__ = getattr(cv2_stub, "__file__", "cv2_stub.py")
cv2_stub.IMREAD_COLOR = getattr(cv2_stub, "IMREAD_COLOR", 1)

def _cv2_missing_attr(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return 1

cv2_stub.__getattr__ = _cv2_missing_attr
sys.modules["cv2"] = cv2_stub

from utils.proj_imgtrans import ProjImgTrans


def test_page_completion_state_defaults_validates_and_serializes():
    p = ProjImgTrans()
    p.directory = "/tmp/nonexistent"
    p.pages = {"p1.png": []}
    p._image_info = {"p1.png": {"finish_code": 0, "ignored": False}}

    assert p.get_page_completion_state("p1.png") == "todo"
    p.set_page_completion_state("p1.png", "reviewed")
    assert p.get_page_completion_state("p1.png") == "reviewed"
    p.set_page_completion_state("p1.png", "bad-state")
    assert p.get_page_completion_state("p1.png") == "todo"
    p.set_page_completion_state("p1.png", "exported")

    d = p.to_dict()
    assert d["image_info"]["p1.png"]["completion_state"] == "exported"

    p2 = ProjImgTrans()
    p2.load_from_dict({"pages": {"p1.png": []}, "image_info": d["image_info"]})
    assert p2.get_page_completion_state("p1.png") == "exported"
