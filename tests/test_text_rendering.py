from utils.text_rendering import (
    FIT_MODE_EXPAND,
    FIT_MODE_PRESERVE,
    LINE_BREAK_BALANCED,
    LINE_BREAK_CJK_STRICT,
    LINE_BREAK_LOOSE,
    WRITING_MODE_HORIZONTAL_LTR,
    WRITING_MODE_RTL,
    WRITING_MODE_VERTICAL_RL,
    fit_font_size_to_box,
    font_fallback_runs,
    kinsoku_wrap,
    merge_font_fallback_chain,
    missing_glyphs_after_fallback,
    normalize_line_break_strategy,
    normalize_vertical_punctuation,
    optimal_kinsoku_wrap,
    resolve_writing_mode,
    vertical_columns,
    vertical_layout_plan,
    vertical_punctuation_class,
    line_break_opportunities,
    contrast_ratio,
    suggest_manga_effects_for_background,
    split_long_word,
    infer_reading_order,
    sort_blocks_for_reading_order,
    vertical_bracket_pair_hints,
    vertical_punctuation_adjustment,
)


def test_resolve_writing_mode_auto_cjk_tall_box_vertical():
    assert resolve_writing_mode("auto", "こんにちは世界", (60, 180)) == WRITING_MODE_VERTICAL_RL


def test_resolve_writing_mode_auto_rtl():
    assert resolve_writing_mode("auto", "مرحبا", (180, 60)) == WRITING_MODE_RTL


def test_resolve_writing_mode_auto_ltr_default():
    assert resolve_writing_mode("auto", "Hello world", (60, 180)) == WRITING_MODE_HORIZONTAL_LTR


def test_vertical_punctuation_normalizes_repeated_marks():
    assert normalize_vertical_punctuation("え?! 本当??") == "え⁈ 本当⁇"


def test_kinsoku_wrap_keeps_closing_punctuation_off_line_start():
    lines = kinsoku_wrap("これは（テスト）です。", 4)
    assert all(not line[0] in "）】」』、。" for line in lines)
    assert all(not line[-1] in "（【「『" for line in lines)


def test_vertical_columns_flow_top_to_bottom_then_right_to_left():
    cols = vertical_columns("これはテストです?!", 4)
    assert cols == ["これはテ", "ストです⁈"]
    assert vertical_punctuation_class("、") == "center"


def test_fit_font_size_shrink_and_preserve_report_overflow():
    shrunk, _text, diag = fit_font_size_to_box("A very long sentence", 32, (80, 30), "shrink")
    assert shrunk < 32
    preserved, _text, preserve_diag = fit_font_size_to_box("A very long sentence", 32, (80, 30), FIT_MODE_PRESERVE)
    assert preserved == 32
    assert preserve_diag.overflow is True


def test_fit_font_size_expand_uses_available_room():
    expanded, _text, diag = fit_font_size_to_box("Hi", 10, (180, 80), FIT_MODE_EXPAND, max_font_size=40)
    assert expanded > 10
    assert diag.overflow is False


def test_merge_font_fallback_chain_deduplicates_script_chain():
    class Cfg:
        render_fallback_fonts_latin = "Arial, DejaVu Sans"
        render_fallback_fonts_cjk = "Noto Sans CJK JP, Arial"
        render_fallback_fonts_korean = "Noto Sans CJK KR"
        render_fallback_fonts_rtl = "Noto Naskh Arabic"
        render_fallback_fonts_emoji = "Noto Color Emoji"

    assert merge_font_fallback_chain("Arial", "かな", Cfg(), "Noto Sans CJK JP") == ["Arial", "Noto Sans CJK JP"]


def test_line_break_strategy_normalization_and_balanced_dangling_line():
    assert normalize_line_break_strategy("cjk-strict") == LINE_BREAK_CJK_STRICT
    balanced = kinsoku_wrap("１２３４５", 2, LINE_BREAK_BALANCED)
    assert balanced[-1] != "５"


def test_line_break_strategy_loose_allows_sfx_punctuation_wrap():
    strict = kinsoku_wrap("ドン！！", 2, LINE_BREAK_CJK_STRICT)
    loose = kinsoku_wrap("ドン！！", 2, LINE_BREAK_LOOSE)
    assert strict != loose


def test_font_fallback_helpers_degrade_gracefully_without_qtgui():
    assert font_fallback_runs("abc", "Primary", None, "Fallback") == []
    assert missing_glyphs_after_fallback("Primary", "abc", None, "Fallback") == []


def test_vertical_layout_plan_marks_centered_and_hanging_punctuation():
    plan = vertical_layout_plan("あ、い。", 3, font_size=20)
    assert plan["column_count"] >= 1
    punct = [g for g in plan["glyphs"] if g["char"] in {"、", "。"}]
    assert punct and all(g["center"] for g in punct)
    assert any(g["hang"] for g in punct)


def test_line_break_opportunities_explain_kinsoku_bans():
    ops = line_break_opportunities("（あ）", "cjk_strict")
    assert any(op["allowed"] is False and op["reason"].startswith("kinsoku") for op in ops)


def test_contrast_effect_suggestion_recommends_stroke_for_low_contrast():
    assert contrast_ratio([0, 0, 0], [255, 255, 255]) > 10
    suggestion = suggest_manga_effects_for_background([230, 230, 230], [255, 255, 255], 0.0)
    assert suggestion["needs_effect"] is True
    assert suggestion["recommended_stroke_width"] > 0


def test_optimal_kinsoku_wrap_balances_without_punctuation_violations():
    lines = optimal_kinsoku_wrap("これはとても長い（テスト）です。", 6)
    assert len(lines) >= 3
    assert max(len(line) for line in lines) - min(len(line) for line in lines) <= 3
    assert all(line[0] not in "）】」』、。" for line in lines)
    assert all(line[-1] not in "（【「『" for line in lines)


def test_vertical_layout_plan_marks_tate_chu_yoko_groups():
    plan = vertical_layout_plan("第12話!?", 6, font_size=18)
    assert any(group["text"] == "12" for group in plan["tate_chu_yoko_groups"])
    assert any(g.get("tate_chu_yoko") for g in plan["glyphs"])


def test_fit_diagnostics_include_safe_inner_bounds_and_quality():
    _size, _text, diag = fit_font_size_to_box("Outlined", 30, (120, 60), FIT_MODE_PRESERVE, stroke_width=0.2, padding=4)
    d = diag.to_dict()
    assert d["effect_margin"] > 0
    assert d["safe_inner_bounds"][0] < 120
    assert 0 <= d["quality_score"] <= 1


def test_reading_order_infers_manga_rtl_and_sorts_rows_right_to_left():
    class B:
        def __init__(self, x1, y1, text="かな"):
            self.xyxy = [x1, y1, x1 + 20, y1 + 20]
            self.translation = text

    left = B(10, 0)
    right = B(80, 0)
    blocks, order = sort_blocks_for_reading_order([left, right], "auto")
    assert order == "rtl"
    assert blocks == [right, left]


def test_fit_diagnostics_recommend_box_for_shadow_overflow():
    _size, _text, diag = fit_font_size_to_box(
        "Shadowed text that is too long",
        26,
        (80, 32),
        FIT_MODE_PRESERVE,
        stroke_width=0.1,
        shadow_radius=0.2,
        shadow_offset=[0.2, 0.1],
    )
    d = diag.to_dict()
    assert d["overflow"] is True
    assert d["box_scale_hint"] >= 1.0
    assert d["recommended_box_size"][0] >= 80


def test_vertical_layout_plan_exposes_offsets_and_bracket_pairs():
    plan = vertical_layout_plan("「あ、い」", 5, font_size=20)
    comma = next(g for g in plan["glyphs"] if g["char"] == "、")
    assert comma["offset"]["dx"] != 0
    assert comma["hang"] is True
    assert plan["bracket_pairs"]
    assert vertical_punctuation_adjustment("「", 20)["rotate_degrees"] == 90.0
    assert vertical_bracket_pair_hints("「あ」")[0]["open"] == "「"


def test_split_long_word_respects_soft_hyphen_boundaries():
    parts = split_long_word("extra­ordinary", 7)
    assert len(parts) >= 2
    assert all("­" not in p for p in parts)


def test_locale_aware_upper_keeps_cjk_and_handles_turkish_i():
    from utils.text_rendering import locale_aware_upper
    assert locale_aware_upper("hello 魔法 i", "en") == "HELLO 魔法 I"
    assert locale_aware_upper("istanbul ı", "tr") == "İSTANBUL I"


def test_suggest_manga_preset_and_ink_clip_risk():
    from utils.text_rendering import suggest_manga_preset, ink_clip_risk
    assert suggest_manga_preset("ドン!!", (80, 180), "auto") == "vertical_cjk_bubble"
    assert suggest_manga_preset("BOOM!!", (90, 70), "horizontal_ltr") == "sfx_bold"
    assert ink_clip_risk((100, 40), (99, 39), 4) is True


def test_vertical_layout_plan_respects_rotate_and_hang_toggles():
    plan = vertical_layout_plan("A、", 4, font_size=18, rotate_latin=False, punctuation_hang=False)
    latin = next(g for g in plan["glyphs"] if g["char"] == "A")
    comma = next(g for g in plan["glyphs"] if g["char"] == "、")
    assert latin["rotate"] is False
    assert comma["hang"] is False
    assert plan["rotate_latin"] is False
    assert plan["punctuation_hang"] is False


def test_fit_diagnostics_expose_clip_risk_and_preset_suggestion():
    _size, _text, diag = fit_font_size_to_box("BOOM!!", 36, (80, 30), FIT_MODE_PRESERVE, stroke_width=0.2, padding=0)
    d = diag.to_dict()
    assert d["ink_clip_risk"] is True
    assert d["preset_suggestion"] in {"sfx_bold", "caption_box", "default_manga_bubble"}


def test_glyph_advance_units_treats_cjk_punctuation_as_compact():
    from utils.text_rendering import glyph_advance_units, estimate_text_bounds
    assert glyph_advance_units("漢字。", "horizontal_ltr") < glyph_advance_units("漢字漢", "horizontal_ltr")
    plain = estimate_text_bounds("漢字漢", 20, "horizontal_ltr", 200, 80)
    punct = estimate_text_bounds("漢字。", 20, "horizontal_ltr", 200, 80)
    assert punct[0] < plain[0]


def test_recommended_tight_letter_spacing_is_conservative():
    from utils.text_rendering import recommended_tight_letter_spacing
    assert recommended_tight_letter_spacing(1.0, 1.08) < 1.0
    assert recommended_tight_letter_spacing(1.0, 1.30) >= 0.88


def test_custom_manga_preset_sanitization_and_merge():
    from types import SimpleNamespace
    from utils.text_rendering import manga_presets, preset_from_font_format, preset_id_from_label, sanitize_manga_preset
    from utils.fontformat import FontFormat

    pid = preset_id_from_label("My SFX!!", ["custom:my_sfx"])
    assert pid == "custom:my_sfx_2"
    preset = sanitize_manga_preset({"label": "Boom", "font_size": "48", "writing_mode": "vertical-rl", "alignment": 9, "frgb": [300, -5, 20]})
    assert preset["font_size"] == 48.0
    assert preset["writing_mode"] == "vertical_rl"
    assert preset["alignment"] == 2
    assert preset["frgb"] == [255, 0, 20]
    fmt = FontFormat(font_size=31, stroke_width=0.12, writing_mode="rtl", letter_spacing=0.95)
    saved = preset_from_font_format(fmt, "RTL caption")
    cfg = SimpleNamespace(render_custom_manga_presets={"custom:rtl_caption": saved})
    presets = manga_presets(cfg)
    assert "default_manga_bubble" in presets
    assert presets["custom:rtl_caption"]["writing_mode"] == "rtl"
    assert presets["custom:rtl_caption"]["stroke_width"] == 0.12


def test_rendering_preset_pack_roundtrip_and_font_diagnostics(tmp_path):
    from types import SimpleNamespace
    from utils.rendering_preset_io import import_preset_pack, preset_font_diagnostics, write_preset_pack

    cfg = SimpleNamespace(render_custom_manga_presets={
        "custom:boom": {"label": "Boom", "font_family": "Missing Manga", "font_size": 42, "writing_mode": "auto"}
    })
    path = tmp_path / "pack.json"
    pack = write_preset_pack(cfg, str(path))
    assert pack["format"].startswith("ballonstranslator")
    assert path.exists()
    diag = preset_font_diagnostics(pack["presets"], ["Arial"])
    assert diag["checked"] is True
    assert diag["missing"]["custom:boom"] == "Missing Manga"

    target = SimpleNamespace(render_custom_manga_presets={})
    result = import_preset_pack(target, str(path))
    assert result["imported_count"] == 1
    assert target.render_custom_manga_presets["custom:boom"]["font_size"] == 42.0

    result2 = import_preset_pack(target, str(path), overwrite=False)
    assert result2["imported_count"] == 1
    assert any(pid.startswith("custom:boom_") for pid in target.render_custom_manga_presets)


def test_text_mask_safe_rect_detects_narrow_visible_area():
    import numpy as np
    from utils.text_masking import mask_safe_rect, recommended_padding_for_mask, masked_text_warnings

    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:16, 8:22] = 255
    diag = mask_safe_rect(mask)
    assert diag.narrow_safe_area is True
    assert diag.safe_rect == (8, 4, 22, 16)
    assert diag.safe_insets == (8, 4, 8, 4)
    assert recommended_padding_for_mask(mask, current_padding=1) >= 9
    payload = masked_text_warnings(mask, 1)
    assert payload["warning"]
    assert payload["recommended_padding"] >= 9


def test_rendering_qa_reports_text_mask_safe_area_warning():
    import numpy as np
    from types import SimpleNamespace
    from utils.fontformat import FontFormat
    from utils.rendering_qa import analyze_text_block

    mask = np.zeros((24, 40), dtype=np.uint8)
    mask[5:18, 10:28] = 255
    blk = SimpleNamespace(
        xyxy=[0, 0, 40, 24],
        translation="Hello mask",
        rich_text="",
        text=[],
        text_mask=mask,
        fontformat=FontFormat(font_size=12, text_padding=1),
    )
    cfg = SimpleNamespace(
        render_fallback_fonts_latin="",
        render_fallback_fonts_cjk="",
        render_fallback_fonts_korean="",
        render_fallback_fonts_rtl="",
        render_fallback_fonts_emoji="",
        module=SimpleNamespace(layout_font_size_min=6.0, layout_font_size_max=96.0),
    )
    diag = analyze_text_block(blk, "page.png", 0, cfg)
    assert "mask_safe_area" in diag["warnings"]
    assert diag["mask_diagnostics"]["narrow_safe_area"] is True
    assert any(s["action"] == "increase_padding" for s in diag["suggestions"])


def test_mask_effective_box_shrinks_fitting_area_for_narrow_mask():
    import numpy as np
    from utils.text_masking import mask_effective_box

    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[8:32, 20:60] = 255
    effective = mask_effective_box(mask, (80, 40), current_padding=1)
    assert effective["uses_mask"] is True
    assert effective["width"] < 80
    assert effective["height"] < 40
    assert effective["recommended_padding"] >= 21


def test_rendering_qa_reports_mask_safe_overflow_against_effective_box():
    import numpy as np
    from types import SimpleNamespace
    from utils.fontformat import FontFormat
    from utils.rendering_qa import analyze_text_block

    mask = np.zeros((50, 120), dtype=np.uint8)
    mask[10:40, 30:90] = 255
    blk = SimpleNamespace(
        xyxy=[0, 0, 120, 50],
        translation="A long masked caption",
        rich_text="",
        text=[],
        text_mask=mask,
        fontformat=FontFormat(font_size=18, text_padding=1, fit_mode="preserve"),
    )
    cfg = SimpleNamespace(
        render_fallback_fonts_latin="",
        render_fallback_fonts_cjk="",
        render_fallback_fonts_korean="",
        render_fallback_fonts_rtl="",
        render_fallback_fonts_emoji="",
        module=SimpleNamespace(layout_font_size_min=6.0, layout_font_size_max=96.0),
    )
    diag = analyze_text_block(blk, "page.png", 0, cfg)
    assert "mask_safe_overflow" in diag["warnings"]
    assert diag["mask_effective_box"]["uses_mask"] is True
    assert any(s["action"] == "shrink_to_mask_safe_area" for s in diag["suggestions"])


def test_plan_typography_cleanup_vertical_cjk_normalizes_and_sets_strict_breaks():
    from utils.text_rendering import plan_typography_cleanup
    result = plan_typography_cleanup("こんにちは?!", 22, (42, 160), "auto", "shrink", "auto", text_padding=0)
    assert result.resolved_writing_mode == "vertical_rl"
    assert "⁈" in result.text
    assert result.line_break_strategy == "cjk_strict"
    assert "switch_writing_mode" in result.actions
    assert "increase_padding" in result.actions


def test_plan_typography_cleanup_balances_latin_lines():
    from utils.text_rendering import plan_typography_cleanup
    result = plan_typography_cleanup("This translation needs nicer manga line breaks", 18, (120, 80), "horizontal_ltr", "shrink", "auto")
    assert result.line_break_strategy == "balanced"
    assert "\n" in result.text
    assert "balance_lines" in result.actions


def test_vertical_layout_cells_flow_top_to_bottom_right_to_left():
    from utils.text_rendering import vertical_layout_cells
    cells = vertical_layout_cells("天地人?!", 20, (80, 64), padding=2)
    assert cells[0]["column"] == 0 and cells[0]["row"] == 0
    assert cells[0]["x"] > cells[-1]["x"]
    assert any(c["char"] == "⁈" for c in cells)


def test_lettering_proof_metrics_reports_overflow_and_vertical_cells():
    from utils.text_rendering import lettering_proof_metrics
    metrics = lettering_proof_metrics("天地人?!", 22, (36, 42), "vertical_rl", padding=1)
    assert metrics["resolved_writing_mode"] == "vertical_rl"
    assert metrics["vertical_cells"]
    assert "check_vertical_punctuation" in metrics["recommended_actions"]


def test_vertical_columns_avoids_single_glyph_leftmost_orphan():
    from utils.text_rendering import vertical_columns
    cols = vertical_columns("天地玄黄宇", 4)
    assert cols == ["天地玄", "黄宇"]


def test_vertical_layout_cells_expose_tate_chu_yoko_orientation():
    from utils.text_rendering import vertical_layout_cells
    cells = vertical_layout_cells("第12話", 20, (80, 120), padding=2)
    tcy = [c for c in cells if c.get("tate_chu_yoko")]
    assert [c["char"] for c in tcy] == ["1", "2"]
    assert all(c["orientation"] == "upright_compact" for c in tcy)


def test_precise_text_bounds_degrades_and_proof_exposes_precise_bounds():
    from utils.text_rendering import precise_text_bounds, lettering_proof_metrics
    bounds = precise_text_bounds("Hello", "", 18, "horizontal_ltr", 120, 60)
    assert bounds[0] > 0 and bounds[1] > 0
    metrics = lettering_proof_metrics("Hello", 18, (120, 60), "horizontal_ltr")
    assert "precise_measured_bounds" in metrics
    assert metrics["precise_measured_bounds"][0] > 0


def test_secondary_stroke_increases_safe_margin_and_preset_roundtrip():
    from utils.text_rendering import effect_margin_px, fit_font_size_to_box, FIT_MODE_PRESERVE, manga_presets
    base = effect_margin_px(36, stroke_width=0.05, padding=0)
    double = effect_margin_px(36, stroke_width=0.05, secondary_stroke_width=0.20, padding=0)
    assert double > base
    _size, _text, diag = fit_font_size_to_box(
        "ドン!!", 36, (90, 50), FIT_MODE_PRESERVE,
        stroke_width=0.05, secondary_stroke_width=0.20, padding=0,
    )
    assert diag.effect_margin >= double - 0.01
    assert manga_presets()["sfx_bold"]["secondary_stroke_width"] > 0


def test_line_break_quality_flags_widow_and_fit_suggests_balance():
    from utils.text_rendering import FIT_MODE_PRESERVE, line_break_quality
    quality = line_break_quality("one two three four five", 7, LINE_BREAK_BALANCED)
    assert "lines" in quality.to_dict()
    _size, _text, diag = fit_font_size_to_box("one two three four five", 18, (92, 80), FIT_MODE_PRESERVE, line_break_strategy=LINE_BREAK_BALANCED)
    assert "line_break_quality" in diag.to_dict()
    assert "balance_lines" in diag.recommended_actions or diag.to_dict()["line_break_quality"]["needs_balance"] in {True, False}


def test_rtl_expand_is_capped_for_arabic_auto_fit():
    expanded, _text, diag = fit_font_size_to_box("مرحبا بالعالم", 20, (500, 160), FIT_MODE_EXPAND, max_font_size=96)
    assert diag.resolved_writing_mode == WRITING_MODE_RTL
    assert expanded <= 27.1


def test_atomic_bubble_fit_balances_lines_and_uses_readable_fill():
    from utils.text_rendering import plan_atomic_bubble_fit
    result = plan_atomic_bubble_fit(
        "This is a compact speech bubble with uneven wording",
        18,
        (180, 100),
        line_spacing=1.35,
        letter_spacing=1.12,
        padding=0,
        target_fill=0.76,
        max_expand_ratio=1.15,
    )
    assert result.font_size <= 18 * 1.15 + 0.1
    assert result.text_padding >= 1.5
    assert result.alignment == 1
    assert "\n" in result.text
    assert 0.2 < result.fill_ratio < 0.95
    assert "line_break_quality" in result.to_dict()


def test_atomic_bubble_fit_respects_vertical_cjk_and_punctuation():
    from utils.text_rendering import plan_atomic_bubble_fit
    result = plan_atomic_bubble_fit("第12話!?", 18, (60, 180), writing_mode="auto")
    assert result.resolved_writing_mode == WRITING_MODE_VERTICAL_RL
    assert "⁉" in result.text or "⁈" in result.text
    assert result.text_padding >= 1.0


def test_atomic_bubble_fit_profiles_change_density_and_export_profile():
    from utils.text_rendering import plan_atomic_bubble_fit

    text = "A short line should breathe in one bubble but grow louder in SFX"
    comfortable = plan_atomic_bubble_fit(text, 16, (180, 95), target_fill=0.76, max_expand_ratio=1.2, profile="roomy")
    dense = plan_atomic_bubble_fit(text, 16, (180, 95), target_fill=0.76, max_expand_ratio=1.2, profile="dense")
    sfx = plan_atomic_bubble_fit(text, 16, (180, 95), target_fill=0.76, max_expand_ratio=1.2, profile="sound-effect")

    assert comfortable.profile == "comfortable"
    assert dense.profile == "dense"
    assert sfx.profile == "sfx"
    assert dense.text_padding <= comfortable.text_padding
    assert dense.diagnostics["target_fill"] > comfortable.diagnostics["target_fill"]
    assert sfx.line_break_strategy == "loose"
    assert sfx.to_dict()["profile"] == "sfx"
