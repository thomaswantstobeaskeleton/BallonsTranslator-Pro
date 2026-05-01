from utils.shortcuts import get_default_shortcuts


def test_run_context_shortcut_ids_exist():
    d = get_default_shortcuts()
    for key in [
        'run.detect_page',
        'run.translate',
        'run.ocr',
        'run.ocr_translate',
        'run.ocr_translate_inpaint',
        'run.macro_detect_ocr_translate',
        'run.macro_ocr_translate_inpaint',
    ]:
        assert key in d
