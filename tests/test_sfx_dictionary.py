from utils.sfx_dictionary import default_sfx_dictionary, query_sfx_dictionary, merge_sfx_entries


def test_sfx_query_hits_defaults():
    rows = default_sfx_dictionary()
    hits = query_sfx_dictionary(rows, 'bang')
    assert hits
    assert hits[0]['target'] == 'BANG'


def test_sfx_merge_updates_and_adds():
    existing = [{'source': 'バン', 'target': 'BANG', 'style': 'impact'}]
    incoming = [{'source': 'バン', 'target': 'BOOM', 'style': 'impact'}, {'source': 'ギラ', 'target': 'GLINT', 'style': 'fx'}]
    out = merge_sfx_entries(existing, incoming)
    assert out['updated'] == 1
    assert out['added'] == 1
    assert any(r['target'] == 'BOOM' for r in out['entries'])
