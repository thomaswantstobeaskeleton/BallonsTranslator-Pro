import sys
import types

# Provide lightweight stubs for optional native deps during test collection.
cv2_stub = types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = 1
cv2_stub.IMREAD_GRAYSCALE = 0
cv2_stub.COLOR_GRAY2RGB = 0
cv2_stub.INTER_LINEAR = 1
cv2_stub.INTER_AREA = 3
cv2_stub.INTER_NEAREST = 0
cv2_stub.INTER_CUBIC = 2
cv2_stub.INTER_LANCZOS4 = 4
cv2_stub.BORDER_CONSTANT = 0
cv2_stub.BORDER_REFLECT = 2
cv2_stub.BORDER_REPLICATE = 1
cv2_stub.copyMakeBorder = lambda *args, **kwargs: None
cv2_stub.cvtColor = lambda img, code: img
sys.modules.setdefault("cv2", cv2_stub)

# BaseTranslator imports utils.textblock for type hints; provide a tiny stub so
# pytest can run in environments without OpenCV system libs.
textblock_stub = types.ModuleType("utils.textblock")
textblock_stub.TextBlock = type("TextBlock", (), {})
sys.modules.setdefault("utils.textblock", textblock_stub)

import pytest

from modules.translators.base import BaseTranslator
from modules.translators.trans_google import TransGoogle


class RecordingBatchTranslator(BaseTranslator):
    """Tiny translator stub to observe BaseTranslator.translate behavior."""

    concate_text = True

    def _setup_translator(self):
        for key in self.lang_map:
            self.lang_map[key] = key

    def _translate(self, src_list):
        self.calls.append(list(src_list))
        return list(src_list)


class MismatchTranslator(RecordingBatchTranslator):
    def _translate(self, src_list):
        # Return the wrong number of outputs to exercise mismatch handling.
        self.calls.append(list(src_list))
        return ["only-one"]


class PartialFailureTranslator(RecordingBatchTranslator):
    def _translate(self, src_list):
        self.calls.append(list(src_list))
        return ["", "ok"]


def _make_translator(translator_cls=RecordingBatchTranslator, *, translate_by_textblock=False):
    translator = translator_cls(lang_source="English", lang_target="日本語", raise_unsupported_lang=False)
    translator.translate_by_textblock = translate_by_textblock
    translator.calls = []
    return translator


def test_translate_list_concatenates_when_translate_by_textblock_is_false():
    translator = _make_translator(translate_by_textblock=False)

    src = ["line-1", "line-2"]
    out = translator.translate(src)

    assert out == src
    assert len(translator.calls) == 1
    assert translator.calls[0] == ["line-1\n##\nline-2"]


def test_translate_list_uses_per_item_path_when_translate_by_textblock_is_true():
    translator = _make_translator(translate_by_textblock=True)

    src = ["line-1", "line-2"]
    out = translator.translate(src)

    assert out == src
    assert len(translator.calls) == 1
    assert translator.calls[0] == src


def test_google_translator_uses_per_item_translation_by_default():
    # Google responses can rewrite delimiter markers, so default to per-item translation for stable block counts.
    assert TransGoogle.concate_text is False
    assert TransGoogle.translate_by_textblock is False


def test_translate_raises_on_output_input_count_mismatch():
    translator = _make_translator(MismatchTranslator, translate_by_textblock=True)

    with pytest.raises(AssertionError):
        translator.translate(["a", "b"])


def test_partial_failures_do_not_force_all_outputs_to_empty():
    translator = _make_translator(PartialFailureTranslator, translate_by_textblock=True)

    out = translator.translate(["a", "b"])

    assert out == ["", "ok"]
    assert out[1] == "ok"
