from modules.translators.base import sanitize_translation_text


UNICODE_HYPHEN = chr(0x2010)


def test_sanitize_translation_text_removes_issue_137_artificial_hyphen_fragments():
    text = f"AH{UNICODE_HYPHEN}... NE{UNICODE_HYPHEN} VER MI{UNICODE_HYPHEN} ND{UNICODE_HYPHEN}..."
    assert sanitize_translation_text(text) == "AH... NEVER MIND..."


def test_sanitize_translation_text_handles_newline_word_fragments():
    assert sanitize_translation_text("NE-\nVER\nMI-\nND") == "NEVER\nMIND"


def test_sanitize_translation_text_preserves_real_hyphenated_words():
    text = "well-being and twenty-one should remain hyphenated"
    assert sanitize_translation_text(text) == text
