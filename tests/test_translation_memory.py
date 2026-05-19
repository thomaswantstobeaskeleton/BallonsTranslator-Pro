from utils.translation_memory import build_tm_from_project, query_tm, export_tm_payload, import_tm_payload


class B:
    def __init__(self, src, tr):
        self._src = src
        self.translation = tr
        self.api_block_id = ''
    def get_text(self):
        return self._src


class P:
    def __init__(self):
        self.pages = {'p1.png': [B('hello world', '你好世界'), B('good morning', '早上好')]}


def test_build_tm_from_project_and_query():
    store = build_tm_from_project(P())
    assert len(store) == 2
    hits = query_tm(store, 'hello worlds', min_score=0.6, limit=3)
    assert hits and hits[0]['target'] == '你好世界'


def test_tm_export_import_roundtrip():
    store = build_tm_from_project(P())
    payload = export_tm_payload(store)
    out = import_tm_payload(payload)
    assert len(out) == len(store)
    assert out[0]['source'] == store[0]['source']
