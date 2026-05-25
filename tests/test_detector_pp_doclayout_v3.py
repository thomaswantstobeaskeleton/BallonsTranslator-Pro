"""Tests for PP-DocLayoutV3 detector module structure and registration."""
import sys
import os.path as osp
import types
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

# Stub transformers so we never pull heavy deps
trans = types.ModuleType("transformers")
trans.AutoModelForVisualDocumentUnderstanding = type(
    "AutoModelForVisualDocumentUnderstanding",
    (),
    {
        "from_pretrained": classmethod(
            lambda cls, *a, **k: type(
                "M", (), {"config": type("C", (), {"id2label": {0: "text"}})()}
            )()
        )
    },
)
trans.AutoProcessor = type(
    "AutoProcessor",
    (),
    {
        "from_pretrained": classmethod(
            lambda cls, *a, **k: type("P", (), {"__call__": lambda self, **kw: {"pixel_values": np.zeros((1, 3, 224, 224))}})()
        )
    },
)
sys.modules["transformers"] = trans

# Also stub PIL
pil_mod = types.ModuleType("PIL")
pil_mod.Image = type("Image", (), {"fromarray": classmethod(lambda cls, arr: object())})
sys.modules["PIL"] = pil_mod
pil_image = types.ModuleType("PIL.Image")
pil_image.Image = pil_mod.Image
sys.modules["PIL.Image"] = pil_image

from modules.textdetector.detector_pp_doclayout_v3 import (
    PPDocLayoutV3Detector,
    _PP_DOCLAYOUT_AVAILABLE,
    _is_text_label,
)
from modules.textdetector.base import TEXTDETECTORS


def test_module_available():
    assert _PP_DOCLAYOUT_AVAILABLE is True


def test_detector_registered():
    assert "pp_doclayout_v3" in TEXTDETECTORS.module_dict
    cls = TEXTDETECTORS.module_dict["pp_doclayout_v3"]
    assert cls is PPDocLayoutV3Detector


def test_params_structure():
    params = PPDocLayoutV3Detector.params
    assert "confidence threshold" in params
    assert "device" in params
    assert params["confidence threshold"]["type"] == "line_editor"


def test_detector_can_instantiate():
    det = PPDocLayoutV3Detector()
    assert det.name == "pp_doclayout_v3"


def test_is_text_label():
    assert _is_text_label("text") is True
    assert _is_text_label("paragraph") is True
    assert _is_text_label("title") is True
    assert _is_text_label("figure") is False
    assert _is_text_label("image") is False
    assert _is_text_label("") is False
