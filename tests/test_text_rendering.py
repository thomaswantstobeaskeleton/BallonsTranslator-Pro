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
