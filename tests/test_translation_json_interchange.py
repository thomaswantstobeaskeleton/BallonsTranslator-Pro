from utils.translation_json_interchange import export_translation_json, import_translation_json


class B:
    def __init__(self, src, tr=''):
        self._src = src
        self.translation = tr
        self.api_block_id = ''
    def get_text(self):
        return self._src


class P:
    def __init__(self):
        self.pages = {'p1.png': [B('A', 'AA'), B('B', 'BB')]}


def test_export_translation_json_has_schema_and_block_ids():
    payload = export_translation_json(P())
    assert payload['schema'] == 'ballonstranslator.translation_json.v1'
    bid = payload['pages'][0]['blocks'][0]['block_id']
    assert bid.startswith('tbx_')


def test_import_translation_json_prefers_block_id():
    proj = P()
    payload = export_translation_json(proj)
    # swap order but keep block ids
    blocks = payload['pages'][0]['blocks']
    blocks[0], blocks[1] = blocks[1], blocks[0]
    blocks[0]['translation'] = 'B_NEW'
    ok, rst = import_translation_json(proj, payload)
    assert ok is True
    assert proj.pages['p1.png'][1].translation == 'B_NEW'
    assert 'p1.png' in rst['matched_pages']
