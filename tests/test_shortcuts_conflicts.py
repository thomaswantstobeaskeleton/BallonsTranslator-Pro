from utils.shortcuts import find_shortcut_conflicts, classify_shortcut_conflicts, auto_resolve_shortcut_conflicts


def test_find_shortcut_conflicts_detects_duplicates_only():
    m = {
        'a': 'Ctrl+S',
        'b': 'Ctrl+S',
        'c': 'Ctrl+O',
        'd': '',
    }
    got = find_shortcut_conflicts(m)
    assert got == {'Ctrl+S': ['a', 'b']}


def test_classify_and_auto_resolve_conflicts():
    m = {
        'go.prev_page': 'A',
        'go.prev_page_alt': 'A',
        'file.open_folder': 'Ctrl+O',
        'file.save_proj': 'Ctrl+O',
    }
    c = classify_shortcut_conflicts(m)
    assert 'A' in c['alias']
    assert 'Ctrl+O' in c['hard']

    r = auto_resolve_shortcut_conflicts(m)
    assert r['go.prev_page'] == 'A'
    assert r['go.prev_page_alt'] == ''
    assert r['file.open_folder'] == 'Ctrl+O'
    assert r['file.save_proj'] == ''
