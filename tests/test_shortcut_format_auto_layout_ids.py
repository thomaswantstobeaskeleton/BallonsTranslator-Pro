from utils.shortcuts import get_default_shortcuts


def test_format_auto_layout_shortcut_ids_exist():
    d = get_default_shortcuts()
    for key in [
        'format.auto_fit',
        'format.auto_fit_binary',
        'format.re_auto_fit_selected',
        'format.re_auto_fit_page',
        'format.re_auto_fit_all',
    ]:
        assert key in d
