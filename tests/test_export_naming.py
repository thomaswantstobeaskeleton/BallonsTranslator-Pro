from utils.export_naming import render_export_filename, sanitize_filename_component


def test_export_filename_template_tokens_are_safe_and_extension_added():
    name = render_export_filename("{stem}_{index:03d}_{source}", "folder/Page:01?.png", 7, ".webp", "clean_fallback")
    assert name.endswith(".webp")
    assert "007" in name
    assert "clean_fallback" in name
    assert "/" not in name and "\\" not in name


def test_export_filename_sanitizes_windows_reserved_names():
    assert sanitize_filename_component("CON") == "_CON"
