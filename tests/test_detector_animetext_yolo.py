"""Tests for AnimeText_yolo detector module structure and registration."""
import sys
import os.path as osp
import types
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

# Stub ultralytics and huggingface_hub so we never hit heavy deps or network
ult = types.ModuleType("ultralytics")
ult.YOLO = type("YOLO", (), {"predict": lambda *a, **k: [type("R", (), {"boxes": None, "obb": None})()]})()
sys.modules["ultralytics"] = ult

hf = types.ModuleType("huggingface_hub")
hf.hf_hub_download = lambda **k: "/fake/model.pt"
hf.list_repo_files = lambda *a, **k: ["animetext_yolo_n.pt", "README.md"]
sys.modules["huggingface_hub"] = hf

from modules.textdetector.detector_animetext_yolo import (
    AnimeTextYoloDetector,
    _ANIME_TEXT_YOLO_AVAILABLE,
)
from modules.textdetector.base import TEXTDETECTORS


def test_module_available():
    assert _ANIME_TEXT_YOLO_AVAILABLE is True  # because we stubbed it


def test_detector_registered():
    assert "animetext_yolo" in TEXTDETECTORS.module_dict
    cls = TEXTDETECTORS.module_dict["animetext_yolo"]
    assert cls is AnimeTextYoloDetector


def test_params_structure():
    params = AnimeTextYoloDetector.params
    assert "confidence threshold" in params
    assert "IoU threshold" in params
    assert "device" in params
    assert "model path" in params
    assert params["confidence threshold"]["type"] == "line_editor"


def test_detector_can_instantiate():
    det = AnimeTextYoloDetector()
    assert det.name == "animetext_yolo"


def test_detector_class_importable():
    assert AnimeTextYoloDetector is not None
