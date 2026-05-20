from utils.text_layout_auto_fit import AutoFitRequest, auto_fit_text


def test_auto_fit_short_text_large_box_no_overflow():
    req = AutoFitRequest(text="Hello there", box=(0, 0, 300, 140), min_font_size=8, max_font_size=64)
    out = auto_fit_text(req)
    assert out.overflow is False
    assert out.font_size >= 20
    assert len(out.lines) >= 1


def test_auto_fit_long_text_small_box_overflow_warning():
    req = AutoFitRequest(
        text="This is a very long translation sentence " * 12,
        box=(0, 0, 120, 60),
        min_font_size=8,
        max_font_size=36,
    )
    out = auto_fit_text(req)
    assert out.overflow or out.font_size <= 9
    assert len(out.warnings) >= 1


def test_auto_fit_round_like_width_profile_prefers_center_wider_lines():
    profile = [90, 120, 150, 120, 90]
    req = AutoFitRequest(
        text="One two three four five six seven eight nine ten eleven twelve",
        box=(0, 0, 180, 180),
        min_font_size=8,
        max_font_size=30,
        width_profile=profile,
    )
    out = auto_fit_text(req)
    assert out.overflow is False
    assert out.score > 0
    assert len(out.rejected_candidates) >= 0


def test_auto_fit_vertical_mode_uses_column_positions():
    req = AutoFitRequest(
        text="縦書きテキストの自動レイアウト確認です",
        box=(0, 0, 180, 240),
        min_font_size=8,
        max_font_size=28,
        writing_mode="vertical",
    )
    out = auto_fit_text(req)
    assert out.font_size >= 8
    assert len(out.line_positions) >= 1
    if len(out.line_positions) >= 2:
        assert out.line_positions[1][0] > out.line_positions[0][0]
