from types import SimpleNamespace

from utils.workflow_presets import apply_workflow_preset, list_workflow_presets, workflow_stage_vector


def test_workflow_presets_apply_stage_flags():
    cfg = SimpleNamespace(enable_detect=False, enable_ocr=False, enable_translate=False, enable_inpaint=False, run_preset_name="")
    result = apply_workflow_preset(cfg, "detect+ocr")
    assert result["preset_id"] == "detect_ocr"
    assert workflow_stage_vector(cfg) == {
        "enable_detect": True,
        "enable_ocr": True,
        "enable_translate": False,
        "enable_inpaint": False,
        "run_preset_name": "Detect+OCR",
    }


def test_workflow_presets_include_lettering_review_no_ai_stage():
    presets = list_workflow_presets()
    assert presets["lettering_review"]["enable_detect"] is False
    cfg = SimpleNamespace(enable_detect=True, enable_ocr=True, enable_translate=True, enable_inpaint=True, run_preset_name="Full")
    apply_workflow_preset(cfg, "qa")
    assert cfg.run_preset_name == "Lettering QA"
    assert not any([cfg.enable_detect, cfg.enable_ocr, cfg.enable_translate, cfg.enable_inpaint])
