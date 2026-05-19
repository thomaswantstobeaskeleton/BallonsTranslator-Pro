from utils.glossary_extraction import extract_glossary_candidates


def test_extract_glossary_candidates_recurring_terms():
    rows = extract_glossary_candidates(['Alice used Fireball', 'Alice cast Fireball again', 'Bob watched'], min_freq=2)
    terms = {r['source']: r for r in rows}
    assert 'Alice' in terms
    assert 'Fireball' in terms
    assert terms['Alice']['frequency'] == 2
