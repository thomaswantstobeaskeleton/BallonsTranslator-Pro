from utils.line_breaking import normalize_artificial_hyphenation, split_long_token_with_hyphenation


def measure(text: str) -> int:
    return len(text) * 10


def test_normalize_artificial_hyphenation_issue_137_google_translate_fragments():
    text = "AH\u2010... NE\u2010 VER MI\u2010 ND\u2010..."

    assert normalize_artificial_hyphenation(text) == "AH... NEVER MIND..."


def test_normalize_artificial_hyphenation_handles_newline_splits():
    text = "NE-\nVER\nMI-\nND"

    assert normalize_artificial_hyphenation(text) == "NEVER\nMIND"


def test_normalize_artificial_hyphenation_preserves_real_hyphenated_words():
    text = "well-being and twenty-one should remain hyphenated"

    assert normalize_artificial_hyphenation(text) == text


def test_short_translated_words_are_not_hyphenated_even_when_enabled():
    assert split_long_token_with_hyphenation("NEVER", measure, 10, hyphenate=True) == [("NEVER", 50)]
    assert split_long_token_with_hyphenation("MIND", measure, 10, hyphenate=True) == [("MIND", 40)]


def test_hyphenation_is_opt_in_by_default():
    token = "extraordinarilylongword"

    assert split_long_token_with_hyphenation(token, measure, 30) == [(token, measure(token))]
