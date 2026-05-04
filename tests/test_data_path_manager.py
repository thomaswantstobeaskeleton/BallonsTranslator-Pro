from utils.data_path_manager import resolve_data_path, free_space_gb


def test_resolve_data_path_defaults_to_data():
    p = resolve_data_path("")
    assert p.endswith('data')


def test_free_space_gb_nonnegative(tmp_path):
    assert free_space_gb(str(tmp_path)) >= 0.0
