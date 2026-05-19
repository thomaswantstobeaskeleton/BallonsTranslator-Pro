from utils.renderer_diagnostics import collect_renderer_diagnostics


def test_renderer_diagnostics_shape():
    d = collect_renderer_diagnostics()
    assert d['default_renderer'] == 'qt'
    assert 'optional_modules' in d
    assert 'warnings' in d
