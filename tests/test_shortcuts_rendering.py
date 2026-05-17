from utils.shortcuts import classify_shortcut_conflicts, get_default_shortcuts, is_single_key_sequence


def test_default_shortcuts_have_no_hard_conflicts():
    conflicts = classify_shortcut_conflicts(get_default_shortcuts())
    assert conflicts["hard"] == {}


def test_single_key_detection_for_tool_shortcuts():
    assert is_single_key_sequence("B") is True
    assert is_single_key_sequence("Ctrl+B") is False
    assert is_single_key_sequence("PageDown") is False


def test_shortcut_conflicts_are_canonicalized_and_warn_for_single_key_tools():
    from utils.shortcuts import classify_shortcut_conflicts, normalize_shortcut_key, shortcut_safety_warnings
    assert normalize_shortcut_key("control++") == "Ctrl+Plus"
    conflicts = classify_shortcut_conflicts({"canvas.zoom_in": "Ctrl++", "custom.zoom": "control++"})
    assert "Ctrl+Plus" in conflicts["hard"]
    warnings = shortcut_safety_warnings({"draw.pen": "B", "format.bold": "Ctrl+B"})
    assert warnings and warnings[0]["action_id"] == "draw.pen"
