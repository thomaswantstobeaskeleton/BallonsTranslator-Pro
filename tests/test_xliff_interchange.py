from utils.xliff_interchange import export_project_xliff, import_project_xliff


class B:
    def __init__(self, src, tr=''):
        self._src = src
        self.translation = tr
    def get_text(self):
        return self._src


class P:
    def __init__(self):
        self.pages = {'p1.png': [B('A', 'AA'), B('B', 'BB')]}


def test_export_xliff_contains_schema_and_units():
    xml = export_project_xliff(P())
    assert 'ballonstranslator.xliff.v1' in xml
    assert 'trans-unit' in xml
    assert 'p1.png::0' in xml


def test_import_xliff_updates_translations():
    proj = P()
    xml = export_project_xliff(proj).replace('<target>AA</target>', '<target>ZZ</target>')
    ok, rst = import_project_xliff(proj, xml)
    assert ok is True
    assert proj.pages['p1.png'][0].translation == 'ZZ'
    assert 'p1.png' in rst['matched_pages']
