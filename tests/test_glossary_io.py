from pathlib import Path
from utils.glossary_io import export_glossary_csv, import_glossary_csv, preview_glossary_merge


def test_glossary_csv_roundtrip(tmp_path: Path):
    entries = [{'source': '宗门', 'target': 'Sect'}, {'source': '灵石', 'target': 'Spirit Stone'}]
    out = tmp_path / 'glossary.csv'
    count = export_glossary_csv(entries, str(out))
    loaded = import_glossary_csv(str(out))
    assert count == 2
    assert loaded[0]['source'] == '宗门'
    assert loaded[1]['target'] == 'Spirit Stone'


def test_preview_glossary_merge_reports_add_and_skip():
    existing = [{'source': '宗门', 'target': 'Sect'}]
    incoming = [{'source': '宗门', 'target': 'SectX'}, {'source': '灵石', 'target': 'Spirit Stone'}]
    p = preview_glossary_merge(existing, incoming, mode='merge')
    assert p['added_count'] == 1
    assert p['skipped_count'] == 1
    assert p['result_count'] == 2
