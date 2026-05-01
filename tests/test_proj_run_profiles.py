import sys
import types

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = getattr(cv2_stub, "IMREAD_COLOR", 1)
cv2_stub.IMREAD_GRAYSCALE = getattr(cv2_stub, "IMREAD_GRAYSCALE", 0)
cv2_stub.COLOR_GRAY2RGB = getattr(cv2_stub, "COLOR_GRAY2RGB", 0)
cv2_stub.COLOR_RGB2BGR = getattr(cv2_stub, "COLOR_RGB2BGR", 0)
cv2_stub.COLOR_RGBA2BGRA = getattr(cv2_stub, "COLOR_RGBA2BGRA", 0)
cv2_stub.IMWRITE_JPEG_QUALITY = getattr(cv2_stub, "IMWRITE_JPEG_QUALITY", 1)
cv2_stub.IMWRITE_WEBP_QUALITY = getattr(cv2_stub, "IMWRITE_WEBP_QUALITY", 64)
cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", lambda img, code: img)
cv2_stub.imencode = getattr(cv2_stub, "imencode", lambda ext, img, enc: (True, types.SimpleNamespace(tofile=lambda p: None)))
sys.modules["cv2"] = cv2_stub


def _missing_attr(name):
    return 1
cv2_stub.__getattr__ = _missing_attr

from utils.proj_imgtrans import ProjImgTrans


def test_proj_run_profiles_round_trip_in_dict_only():
    p = ProjImgTrans()
    p.directory = '/tmp/nonexistent'
    p.current_img = None
    p.run_profiles = {'fast': {'ocr': 'manga_ocr', 'translator': 'google'}}
    d = p.to_dict()
    assert d['run_profiles']['fast']['ocr'] == 'manga_ocr'

    p2 = ProjImgTrans()
    p2.load_from_dict({'pages': {}, 'image_info': {}, 'run_profiles': d['run_profiles']})
    assert 'fast' in p2.run_profiles
