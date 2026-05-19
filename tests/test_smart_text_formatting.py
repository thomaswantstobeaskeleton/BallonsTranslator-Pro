from utils.text_rendering import smart_fit_text_to_box


def test_smart_fit_rebalances_long_text_for_narrow_box():
    text = "This is a very long sentence that should be wrapped more evenly across lines for comic bubbles"
    rst = smart_fit_text_to_box(
        text=text,
        font_size=24,
        box_size=(120, 70),
        writing_mode='horizontal_ltr',
        fit_mode='shrink',
        line_break_strategy='balanced',
    )
    assert isinstance(rst.text, str)
    assert '\n' in rst.text
    assert any(a in rst.actions for a in ['balance_lines', 'rebalance_for_overflow', 'shrink_to_fit'])


def test_smart_fit_keeps_vertical_mode_for_cjk_vertical_boxes():
    rst = smart_fit_text_to_box(
        text='第12話!? 魔王復活',
        font_size=24,
        box_size=(60, 180),
        writing_mode='auto',
        fit_mode='shrink',
        line_break_strategy='cjk_strict',
    )
    assert rst.resolved_writing_mode in {'vertical_rl', 'horizontal_ltr', 'rtl'}
    assert rst.diagnostics.get('fit') is not None
