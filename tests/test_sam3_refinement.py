import numpy as np

from utils.sam3_refinement import (
    SAM3RefinementOptions,
    SAM3_REFINEMENT_MODEL_CUSTOM,
    SAM3_REFINEMENT_MODEL_DEFAULT,
    curated_sam3_refinement_models,
    cleanup_only_mode_plan,
    combine_refined_mask,
    default_refinement_settings,
    get_persistent_refinement_settings,
    mask_area,
    mask_iou,
    refine_mask_with_sam3,
    resolve_refinement_model_id,
    sam3_safe_areas_from_mask,
    should_use_sam3_for_live_translation,
)


def test_refine_mask_with_sam3_unions_prompt_mask_and_reports_safe_area():
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    initial = np.zeros((80, 100), dtype=np.uint8)
    initial[30:50, 35:65] = 255
    candidate = np.zeros((80, 100), dtype=np.uint8)
    candidate[22:58, 24:76] = 255

    def runner(_image, prompt):
        assert prompt == "speech bubble"
        return [candidate]

    result = refine_mask_with_sam3(
        img,
        initial,
        SAM3RefinementOptions(enabled=True, min_iou_with_initial=0.01),
        mask_runner=runner,
    )

    assert result.used_sam3 is True
    assert result.candidate_count == 1
    assert result.selected_candidate_count == 1
    assert mask_area(result.mask) > mask_area(initial)
    assert result.safe_areas
    assert result.safe_areas[0]["kind"] == "sam3_bubble_safe_area"


def test_refine_mask_with_sam3_rejects_unrelated_candidate():
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    initial = np.zeros((80, 100), dtype=np.uint8)
    initial[30:50, 35:65] = 255
    unrelated = np.zeros((80, 100), dtype=np.uint8)
    unrelated[0:10, 0:10] = 255

    result = refine_mask_with_sam3(
        img,
        initial,
        SAM3RefinementOptions(enabled=True, min_iou_with_initial=0.10),
        mask_runner=lambda _img, _prompt: [unrelated],
    )

    assert result.used_sam3 is False
    assert result.selected_candidate_count == 0
    assert mask_iou(result.mask, initial) == 1.0


def test_combine_refined_mask_intersect_and_replace_modes():
    initial = np.zeros((20, 20), dtype=np.uint8)
    initial[5:15, 5:15] = 255
    refined = np.zeros((20, 20), dtype=np.uint8)
    refined[10:19, 10:19] = 255

    inter = combine_refined_mask(initial, [refined], SAM3RefinementOptions(merge_mode="intersect"))
    repl = combine_refined_mask(initial, [refined], SAM3RefinementOptions(merge_mode="replace_if_nonempty"))

    assert mask_area(inter) == 25
    assert mask_area(repl) == mask_area(refined)


def test_sam3_safe_area_uses_mask_for_round_bubble():
    yy, xx = np.ogrid[:100, :160]
    mask = ((((xx - 80) / 78) ** 2 + ((yy - 50) / 48) ** 2) <= 1.0).astype(np.uint8) * 255
    safe = sam3_safe_areas_from_mask(mask, SAM3RefinementOptions())

    assert len(safe) == 1
    rect = safe[0]["safe_rect"]
    assert rect[2] < 160
    assert rect[3] < 100
    assert safe[0]["used_mask"] is True


def test_live_translation_sam3_gate_only_allows_slow_quality_profiles():
    assert should_use_sam3_for_live_translation("high_quality_slower") is True
    assert should_use_sam3_for_live_translation({"use_sam3": True, "quality_mode": "high_quality"}) is True
    assert should_use_sam3_for_live_translation("fast") is False
    assert should_use_sam3_for_live_translation({"use_sam3": True, "quality_mode": "low_latency"}) is False


def test_cleanup_only_mode_plan_disables_ocr_translation_and_render():
    plan = cleanup_only_mode_plan()
    assert plan["mode"] == "cleanup_only"
    assert plan["detector"] == "sam3_refiner"
    assert plan["run_detection"] is True
    assert plan["run_inpaint"] is True
    assert plan["run_ocr"] is False
    assert plan["run_translation"] is False
    assert plan["run_render"] is False


def test_curated_model_choices_only_include_sam_refinement_options():
    models = curated_sam3_refinement_models()
    assert SAM3_REFINEMENT_MODEL_DEFAULT in models
    assert SAM3_REFINEMENT_MODEL_CUSTOM in models
    assert all("gpt" not in m.lower() for m in models)
    assert all("ocr" not in m.lower() for m in models)
    assert all("lama" not in m.lower() for m in models)


def test_resolve_refinement_model_id_handles_custom_and_invalid_values():
    assert resolve_refinement_model_id(SAM3_REFINEMENT_MODEL_DEFAULT) == SAM3_REFINEMENT_MODEL_DEFAULT
    assert resolve_refinement_model_id(SAM3_REFINEMENT_MODEL_CUSTOM, "org/custom-sam3") == "org/custom-sam3"
    assert resolve_refinement_model_id(SAM3_REFINEMENT_MODEL_CUSTOM, "") == SAM3_REFINEMENT_MODEL_DEFAULT
    assert resolve_refinement_model_id("not-a-refiner") == SAM3_REFINEMENT_MODEL_DEFAULT


def test_persistent_settings_round_trip_from_textdetector_params():
    class ModuleCfg:
        textdetector_params = {
            "sam3_refinement": {
                "enabled": {"value": True},
                "base_detector": {"value": "paddle_det"},
                "refinement_model": {"value": SAM3_REFINEMENT_MODEL_CUSTOM},
                "custom_model_id": {"value": "org/custom-sam3"},
                "prompt": {"value": "speech bubble"},
            }
        }

    settings = get_persistent_refinement_settings(ModuleCfg())
    assert settings["enabled"] is True
    assert settings["base_detector"] == "paddle_det"
    assert settings["refinement_model"] == SAM3_REFINEMENT_MODEL_CUSTOM
    assert settings["custom_model_id"] == "org/custom-sam3"
    assert set(default_refinement_settings()) <= set(settings)
