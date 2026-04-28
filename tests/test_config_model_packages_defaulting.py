import json
import sys
import types

# utils.config -> utils.io_utils imports cv2; provide/patch a minimal stub for test envs without libGL.
cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = getattr(cv2_stub, "IMREAD_COLOR", 1)
cv2_stub.IMREAD_GRAYSCALE = getattr(cv2_stub, "IMREAD_GRAYSCALE", 0)
cv2_stub.COLOR_GRAY2RGB = getattr(cv2_stub, "COLOR_GRAY2RGB", 0)
cv2_stub.COLOR_RGB2BGR = getattr(cv2_stub, "COLOR_RGB2BGR", 0)
cv2_stub.COLOR_RGBA2BGRA = getattr(cv2_stub, "COLOR_RGBA2BGRA", 0)
cv2_stub.IMWRITE_JPEG_QUALITY = getattr(cv2_stub, "IMWRITE_JPEG_QUALITY", 1)
cv2_stub.IMWRITE_WEBP_QUALITY = getattr(cv2_stub, "IMWRITE_WEBP_QUALITY", 64)
cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", lambda img, code: img)
cv2_stub.imencode = getattr(
    cv2_stub,
    "imencode",
    lambda ext, img, enc: (True, types.SimpleNamespace(tofile=lambda p: None)),
)
cv2_stub.imshow = getattr(cv2_stub, "imshow", lambda *args, **kwargs: None)
cv2_stub.waitKey = getattr(cv2_stub, "waitKey", lambda *args, **kwargs: 0)
sys.modules["cv2"] = cv2_stub

from utils.config import ProgramConfig


def _write_config(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_model_packages_enabled_defaults_to_core_when_missing(tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_config(
        cfg_path,
        {
            "module": {
                "translator": "google",
                "translator_params": {},
            }
        },
    )

    loaded = ProgramConfig.load(str(cfg_path))
    assert loaded.model_packages_enabled == ["core"]


def test_model_packages_enabled_defaults_to_core_when_null(tmp_path):
    cfg_path = tmp_path / "config.json"
    _write_config(
        cfg_path,
        {
            "model_packages_enabled": None,
            "module": {
                "translator": "google",
                "translator_params": {},
            },
        },
    )

    loaded = ProgramConfig.load(str(cfg_path))
    assert loaded.model_packages_enabled == ["core"]


def test_model_packages_enabled_preserves_explicit_value(tmp_path):
    cfg_path = tmp_path / "config.json"
    expected = ["core", "advanced_ocr"]
    _write_config(
        cfg_path,
        {
            "model_packages_enabled": expected,
            "module": {
                "translator": "google",
                "translator_params": {},
            },
        },
    )

    loaded = ProgramConfig.load(str(cfg_path))
    assert loaded.model_packages_enabled == expected
