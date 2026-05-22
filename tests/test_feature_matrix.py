from utils.feature_matrix import build_feature_matrix, feature_matrix_text


def test_feature_matrix_has_expected_rows():
    rows = build_feature_matrix()
    names = {r['feature'] for r in rows}
    assert 'Detection' in names
    assert 'Models: Segmentation' in names


def test_feature_matrix_text_format():
    txt = feature_matrix_text()
    assert 'Feature Matrix' in txt
    assert '- Detection:' in txt
