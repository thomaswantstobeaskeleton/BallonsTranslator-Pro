from pathlib import Path

from utils.data_path_manager import resolve_data_path, free_space_gb, describe_data_path, migrate_data_path


def test_resolve_data_path_defaults_to_data():
    p = resolve_data_path("")
    assert p.endswith('data')


def test_free_space_gb_nonnegative(tmp_path):
    assert free_space_gb(str(tmp_path)) >= 0.0


def test_describe_data_path_has_space_fields(tmp_path):
    d = describe_data_path(str(tmp_path))
    assert d['exists'] is True
    assert d['free_gb'] >= 0.0


def test_migrate_data_path_dry_run_and_apply(tmp_path: Path):
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()
    (src / 'a.txt').write_text('x', encoding='utf-8')
    dry = migrate_data_path(str(src), str(dst), dry_run=True)
    assert dry['ok'] is True
    assert (src / 'a.txt').exists()
    run = migrate_data_path(str(src), str(dst), dry_run=False)
    assert run['ok'] is True
    assert (dst / 'a.txt').exists()
