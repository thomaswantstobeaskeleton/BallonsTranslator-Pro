import numpy as np

from utils.auto_text_layout import (
    auto_layout_effective_preset,
    auto_layout_preset_settings,
    auto_layout_profile_defaults,
    apply_auto_layout_profile,
    auto_layout_profile_summary,
    auto_layout_setting_hints,
    auto_layout_advanced_summary,
    auto_rendered_fit_scale,
    candidate_layout_widths,
    estimate_target_line_count,
    normalize_auto_layout_preset,
)
from utils.text_masking import bubble_safe_text_rect


def test_candidate_layout_widths_include_balanced_shape_specific_widths():
    widths = candidate_layout_widths(
        max_width=240,
        min_width=60,
        words_width=520,
        delimiter_total_width=40,
        line_height=24,
        target_box_height=120,
        balloon_shape="round",
    )

    assert widths[0] == 240
    assert len(widths) >= 10
    assert len(widths) == len(set(widths))
    assert all(60 <= width <= 240 for width in widths)
    # Round balloons should include compact candidates well below the max width,
    # not just the old near-max geometric shrink sequence.
    assert any(145 <= width <= 175 for width in widths)


def test_rectangle_alias_uses_simple_box_width_candidates():
    square = candidate_layout_widths(
        max_width=240,
        min_width=60,
        words_width=520,
        delimiter_total_width=40,
        line_height=24,
        target_box_height=120,
        balloon_shape="square",
    )
    rectangle = candidate_layout_widths(
        max_width=240,
        min_width=60,
        words_width=520,
        delimiter_total_width=40,
        line_height=24,
        target_box_height=120,
        balloon_shape="rectangle",
    )

    assert rectangle == square


def test_target_line_count_responds_to_tall_narrow_bubbles():
    wide = estimate_target_line_count(480, 24, 240, 80, "elongated")
    narrow = estimate_target_line_count(480, 24, 120, 240, "narrow")

    assert narrow > wide


def test_bubble_safe_text_rect_avoids_unsafe_corners():
    yy, xx = np.ogrid[:100, :160]
    cx, cy = 80, 50
    # Ellipse-like bubble mask: full bounding box has unsafe transparent corners.
    mask = (((xx - cx) / 78) ** 2 + ((yy - cy) / 48) ** 2 <= 1.0).astype(np.uint8) * 255

    safe = bubble_safe_text_rect(mask, [0, 0, 160, 100], min_coverage=0.90)
    x, y, w, h = safe["rect"]

    assert safe["used_mask"] is True
    assert w < 160
    assert h < 100
    assert x > 0
    assert y > 0
    assert safe["coverage"] >= 0.90


def test_text_layout_module_imports_without_image_runtime_side_effects():
    import utils.text_layout as text_layout

    assert hasattr(text_layout, "layout_text")


def test_auto_layout_profile_defaults_cover_advanced_knobs():
    balanced = auto_layout_profile_defaults("balanced")
    fit = auto_layout_profile_defaults("fit")
    readable = auto_layout_profile_defaults("readable")

    expected = {
        "layout_constrain_to_bubble",
        "layout_center_in_bubble_after_autolayout",
        "layout_center_in_bubble_min_gap_px",
        "layout_check_overflow_after_layout",
        "layout_use_mask_safe_area",
        "layout_box_size_check_model_id",
        "layout_optimal_breaks",
        "layout_hyphenation",
        "optimize_line_breaks",
        "layout_short_line_penalty",
        "layout_height_overflow_penalty",
        "layout_font_size_min",
        "layout_font_size_max",
        "layout_font_fit_bubble",
        "layout_font_binary_search",
        "layout_auto_final_fit_pass",
        "layout_balloon_shape",
        "layout_balloon_shape_auto_method",
        "layout_balloon_shape_model_id",
        "layout_min_line_width_px",
        "layout_max_line_width_frac_no_bubble",
        "layout_stub_penalty_1word",
    }
    assert expected <= set(balanced)
    assert balanced["layout_balloon_shape"] == "square"
    assert fit["layout_balloon_shape"] == "square"
    assert readable["layout_balloon_shape"] == "square"
    assert balanced["layout_balloon_shape_auto_method"] == "contour_ratio"
    assert balanced["layout_balloon_shape_model_id"] == ""
    assert fit["layout_height_overflow_penalty"] > balanced["layout_height_overflow_penalty"]
    assert readable["layout_font_size_min"] > balanced["layout_font_size_min"]
    assert "rectangular text boxes" in auto_layout_profile_summary("balanced")


def test_apply_auto_layout_profile_updates_config_like_object():
    class Cfg:
        layout_auto_preset = "balanced"
        layout_font_size_min = 1.0
        layout_balloon_shape = "auto"
        layout_balloon_shape_model_id = "old/model"

    cfg = Cfg()
    applied = apply_auto_layout_profile(cfg, "fit inside")
    assert cfg.layout_auto_preset == "fit"
    assert cfg.layout_font_size_min == applied["layout_font_size_min"] == 6.0
    assert cfg.layout_balloon_shape == "square"
    assert cfg.layout_balloon_shape_model_id == ""
    assert cfg.layout_constrain_to_bubble is True


def test_auto_layout_setting_hints_explain_numeric_values():
    hints = auto_layout_setting_hints({
        "layout_font_size_min": 6,
        "layout_font_size_max": 64,
        "layout_short_line_penalty": 130,
        "layout_height_overflow_penalty": 950,
        "layout_stub_penalty_1word": 2700,
        "layout_min_line_width_px": 70,
        "layout_max_line_width_frac_no_bubble": 0.9,
        "layout_center_in_bubble_min_gap_px": 0,
        "layout_box_size_check_model_id": "builtin",
        "layout_balloon_shape": "auto",
        "layout_balloon_shape_auto_method": "model_contour",
        "layout_balloon_shape_model_id": "shape/model",
    })

    assert hints["font_range"].startswith("6–64 pt")
    assert hints["short_line_penalty"] == "strict"
    assert hints["height_overflow_penalty"] == "strict fit"
    assert hints["stub_penalty"] == "maximum"
    assert hints["center_gap"] == "never skip centering"
    assert hints["box_model"] == "built-in CLIP"
    assert hints["shape_detection"] == "model-assisted"
    assert "Font 6–64 pt" in auto_layout_advanced_summary({
        "layout_font_size_min": 6,
        "layout_font_size_max": 64,
    })


def test_auto_layout_setting_hints_identify_simple_rectangle_shape():
    hints = auto_layout_setting_hints({
        "layout_balloon_shape": "square",
        "layout_balloon_shape_auto_method": "model_contour",
        "layout_balloon_shape_model_id": "shape/model",
    })

    assert hints["shape_detection"] == "simple rectangle"


def test_auto_layout_presets_normalize_and_shift_behavior():
    assert normalize_auto_layout_preset("strict") == "fit"
    assert normalize_auto_layout_preset("larger text") == "readable"

    fit = auto_layout_preset_settings("fit")
    readable = auto_layout_preset_settings("readable")

    assert fit.line_width_delta < 0
    assert fit.inner_inset_delta < 0
    assert fit.font_scale_multiplier < 1.0
    assert readable.line_width_delta > 0
    assert readable.inner_inset_delta > 0
    assert readable.font_scale_multiplier > 1.0


def test_balanced_preset_adapts_per_bubble_density():
    assert auto_layout_effective_preset("balanced", "Tiny", 180, 90, "round", line_count=1) == "readable"
    assert auto_layout_effective_preset("balanced", "word " * 40, 80, 80, "narrow", line_count=7) == "fit"
    assert auto_layout_effective_preset("readable", "word " * 40, 80, 80, "narrow", line_count=7) == "readable"


def test_auto_rendered_fit_scale_shrinks_overflow_and_limits_expand():
    assert auto_rendered_fit_scale(120, 60, 100, 80, "balanced") < 1.0
    readable_expand = auto_rendered_fit_scale(40, 30, 120, 80, "readable")
    fit_expand = auto_rendered_fit_scale(40, 30, 120, 80, "fit")

    assert 1.0 < fit_expand <= 1.04
    assert readable_expand > fit_expand
    assert auto_rendered_fit_scale(40, 30, 120, 80, "readable", allow_expand=False) == 1.0
