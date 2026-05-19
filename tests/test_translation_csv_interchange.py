from utils.translation_csv_interchange import export_translation_csv_text, import_translation_csv_text


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


def test_export_csv_has_header_and_block_id():
    txt = export_translation_csv_text(P())
    assert 'page,index,block_id,source,translation' in txt
    assert 'tbx_' in txt


def test_import_csv_uses_block_id_matching():
    proj = P()
    txt = export_translation_csv_text(proj)
    lines = txt.strip().splitlines()
    header = lines[0]
    row1 = lines[1].split(',')
    row2 = lines[2].split(',')
    row2[4] = 'B_NEW'
    swapped = '\n'.join([header, ','.join(row2), ','.join(row1)]) + '\n'
    ok, rst = import_translation_csv_text(proj, swapped)
    assert ok is True
    assert proj.pages['p1.png'][1].translation == 'B_NEW'
    assert 'p1.png' in rst['matched_pages']
