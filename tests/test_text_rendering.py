from utils.text_rendering import (
    FIT_MODE_EXPAND,
    FIT_MODE_PRESERVE,
    WRITING_MODE_HORIZONTAL_LTR,
    WRITING_MODE_RTL,
    WRITING_MODE_VERTICAL_RL,
    fit_font_size_to_box,
    kinsoku_wrap,
    merge_font_fallback_chain,
    normalize_vertical_punctuation,
    resolve_writing_mode,
    vertical_columns,
    vertical_punctuation_class,
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
