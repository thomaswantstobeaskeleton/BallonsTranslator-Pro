from utils.shortcuts import classify_shortcut_conflicts, get_default_shortcuts, is_single_key_sequence


def test_default_shortcuts_have_no_hard_conflicts():
    conflicts = classify_shortcut_conflicts(get_default_shortcuts())
    assert conflicts["hard"] == {}


def test_single_key_detection_for_tool_shortcuts():
    assert is_single_key_sequence("B") is True
    assert is_single_key_sequence("Ctrl+B") is False
    assert is_single_key_sequence("PageDown") is False
