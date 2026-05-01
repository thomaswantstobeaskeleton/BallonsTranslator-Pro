from utils.translation_review import extract_glossary_candidates, check_translation_guardrails


def test_extract_glossary_candidates_picks_repeated_terms():
    src = ["Alice uses Mana Burst", "Mana Burst hits Bob", "Alice recovers mana"]
    got = extract_glossary_candidates(src, [], min_freq=2)
    sources = {x['source'] for x in got}
    assert 'Alice' in sources
    assert 'Mana' in sources or 'Burst' in sources


def test_guardrails_flags_expected_issues():
    glossary = [{"source": "宗门", "target": "Sect"}]
    issues = check_translation_guardrails("宗门", "宗门", glossary=glossary, max_len_ratio=1.1)
    assert any("Untranslated source carry-over" in i for i in issues)
    assert any("Glossary mismatch" in i for i in issues)

    issues2 = check_translation_guardrails("short", "this is a very very long translation", glossary=[])
    assert any("overlong" in i for i in issues2)
