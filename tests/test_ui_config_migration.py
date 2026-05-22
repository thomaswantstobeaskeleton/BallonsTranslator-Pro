import json

import pytest

pytest.importorskip("cv2", exc_type=ImportError, reason="OpenCV runtime unavailable in test env")

from utils.config import ProgramConfig


def test_ui_config_migration_defaults(tmp_path):
    cfg = {
        "module": {
            "translator": "google",
            "translator_params": {},
        },
        "show_welcome_screen": False,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = ProgramConfig.load(str(p))

    assert loaded.ui_mode == "advanced"
    assert loaded.show_experimental_features is False
    assert loaded.show_legacy_menus is True
    assert loaded.startup_mode == "last_used"
    assert loaded.show_home_on_startup is False
    assert loaded.recent_workflows == []
    assert loaded.omni_show_unavailable is False
    assert loaded.omni_result_type_filter == "all"
    assert loaded.omni_result_type_filter == "all"


def test_ui_config_migration_invalid_values_reset(tmp_path):
    cfg = {
        "module": {
            "translator": "google",
            "translator_params": {},
        },
        "ui_mode": "nope",
        "startup_mode": "???",
        "recent_workflows": "bad",
        "omni_result_type_filter": "bad_type",
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = ProgramConfig.load(str(p))

    assert loaded.ui_mode == "advanced"
    assert loaded.startup_mode == "last_used"
    assert loaded.recent_workflows == []
    assert loaded.omni_show_unavailable is False
    assert loaded.omni_result_type_filter == "all"


def test_invalid_recent_workflows_are_cleaned(tmp_path):
    cfg = {
        "module": {"translator": "google", "translator_params": {}},
        "recent_workflows": ["editor", "bad", "models", "", "EDITOR", "diag"],
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = ProgramConfig.load(str(p))
    assert loaded.recent_workflows == ["editor", "models"]


def test_context_menu_profile_migration_defaults_and_invalid(tmp_path):
    cfg = {
        "module": {"translator": "google", "translator_params": {}},
        "context_menu_profile": "invalid_profile",
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = ProgramConfig.load(str(p))
    assert loaded.context_menu_profile == "custom"


def test_startup_mode_models_and_diagnostics_are_preserved(tmp_path):
    for mode in ("settings", "batch", "models", "diagnostics"):
        cfg = {"module": {"translator": "google", "translator_params": {}}, "startup_mode": mode}
        p = tmp_path / f"cfg_{mode}.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        loaded = ProgramConfig.load(str(p))
        assert loaded.startup_mode == mode


def test_realtime_defaults_and_invalid_values(tmp_path):
    cfg = {
        "module": {"translator": "google", "translator_params": {}},
        "realtime_profile_id": "bad",
        "realtime_capture_interval_ms": -1,
        "realtime_min_ocr_interval_ms": "x",
    }
    p = tmp_path / "cfg_rt.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = ProgramConfig.load(str(p))
    assert loaded.realtime_profile_id == "generic_screen_ocr"
    assert loaded.realtime_capture_interval_ms == 100
    assert loaded.realtime_min_ocr_interval_ms == 0


def test_realtime_profile_valid_value_preserved(tmp_path):
    cfg = {
        "module": {"translator": "google", "translator_params": {}},
        "realtime_profile_id": "chrome_manhua_reader",
    }
    p = tmp_path / "cfg_rt_valid.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = ProgramConfig.load(str(p))
    assert loaded.realtime_profile_id == "chrome_manhua_reader"


def test_realtime_interval_clamps_upper_bound(tmp_path):
    cfg = {
        "module": {"translator": "google", "translator_params": {}},
        "realtime_capture_interval_ms": 999999,
        "realtime_min_ocr_interval_ms": 999999,
    }
    p = tmp_path / "cfg_rt_clamp.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = ProgramConfig.load(str(p))
    assert loaded.realtime_capture_interval_ms == 5000
    assert loaded.realtime_min_ocr_interval_ms == 5000


def test_realtime_region_rect_normalization(tmp_path):
    cfg = {
        "module": {"translator": "google", "translator_params": {}},
        "realtime_region_rect": [-10, "bad", 0, 200000],
    }
    p = tmp_path / "cfg_rt_rect.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    loaded = ProgramConfig.load(str(p))
    assert loaded.realtime_region_rect == [0, 0, 100, 100]
