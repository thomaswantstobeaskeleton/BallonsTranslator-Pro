from utils.batch_find_replace import preview_batch_find_replace, apply_batch_find_replace


class B:
    def __init__(self, t):
        self.translation = t
        self.text = t


class P:
    def __init__(self):
        self.pages = {
            'p1.png': [B('Hello Cat'), B('Cat cat')],
            'p2.png': [B('No match')],
        }


def test_preview_regex_counts_hits():
    proj = P()
    rst = preview_batch_find_replace(proj, r'cat', 'dog', use_regex=True, case_sensitive=False)
    assert rst['count'] == 2


def test_apply_updates_translations_from_preview():
    proj = P()
    prev = preview_batch_find_replace(proj, r'cat', 'dog', use_regex=True, case_sensitive=False)
    changed, applied = apply_batch_find_replace(proj, prev)
    assert changed == 2
    assert len(applied) == 2
    assert proj.pages['p1.png'][0].translation == 'Hello dog'
