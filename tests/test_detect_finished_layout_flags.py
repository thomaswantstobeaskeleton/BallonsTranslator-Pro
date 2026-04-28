from utils.detect_layout_flags import (
    is_detect_only_run,
    should_enable_auto_textlayout,
    should_run_post_detect_autofit,
)


def test_detect_only_autolayout_flag_enabled_with_program_font_size():
    assert should_enable_auto_textlayout(True, 0, True, False, False) is True
    assert is_detect_only_run(True, False, False) is True
    assert should_run_post_detect_autofit(True, 0, True, False, False) is True


def test_detect_only_post_autofit_disabled_for_manual_font_size_override():
    assert should_enable_auto_textlayout(True, 1, True, False, False) is False
    assert should_run_post_detect_autofit(True, 1, True, False, False) is False


def test_post_detect_autofit_only_for_detect_only_pipeline():
    assert should_run_post_detect_autofit(True, 0, True, True, False) is False
    assert should_run_post_detect_autofit(True, 0, False, True, True) is False
