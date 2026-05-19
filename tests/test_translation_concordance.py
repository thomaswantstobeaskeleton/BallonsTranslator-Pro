from utils.translation_concordance import build_concordance_from_project, query_concordance


class B:
    def __init__(self, src, tgt, bid=''):
        self._src = src
        self.translation = tgt
        self.api_block_id = bid
    def get_text(self):
        return self._src


class P:
    def __init__(self):
        self.pages = {'001.png': [B('hello hero', '你好勇者', 'a1'), B('bye', '再见', 'a2')]}


def test_build_concordance_rows_include_page_and_block_id():
    rows = build_concordance_from_project(P())
    assert len(rows) == 2
    assert rows[0]['page'] == '001.png'
    assert rows[0]['block_id'] == 'a1'


def test_query_concordance_matches_source_and_target():
    rows = build_concordance_from_project(P())
    assert query_concordance(rows, 'hero')
    assert query_concordance(rows, '勇者')
