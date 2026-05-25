from __future__ import annotations

import pytest


# --- ApplyTranslationCandidateCommand tests ---

def test_apply_translation_candidate_command_redo_applies_text():
    """Regression: redo() must apply new_text on first call (QUndoStack calls redo on push)."""
    # Minimal mock objects to avoid heavy Qt dependencies in unit tests
    class MockBlk:
        def __init__(self):
            self.translation = "old"

    class MockBlkItem:
        def __init__(self):
            self.blk = MockBlk()
            self._plain = "old"

        def setPlainText(self, text):
            self._plain = text

    class MockTransEdit:
        def __init__(self):
            self._plain = "old"

        def setPlainText(self, text):
            self._plain = text

    from ui.textedit_commands import ApplyTranslationCandidateCommand
    blk_item = MockBlkItem()
    trans_edit = MockTransEdit()
    cmd = ApplyTranslationCandidateCommand(blk_item, trans_edit, "new_text")

    # On first redo (simulating QUndoStack.push), text should be applied
    cmd.redo()
    assert blk_item.blk.translation == "new_text"
    assert blk_item._plain == "new_text"
    assert trans_edit._plain == "new_text"

    # Undo restores old
    cmd.undo()
    assert blk_item.blk.translation == "old"
    assert blk_item._plain == "old"
    assert trans_edit._plain == "old"

    # Redo again applies new
    cmd.redo()
    assert blk_item.blk.translation == "new_text"
    assert blk_item._plain == "new_text"


def test_apply_translation_candidate_command_none_text():
    """Ensure empty/None text is handled gracefully."""
    class MockBlk:
        def __init__(self):
            self.translation = "old"

    class MockBlkItem:
        def __init__(self):
            self.blk = MockBlk()
            self._plain = "old"

        def setPlainText(self, text):
            self._plain = text

    from ui.textedit_commands import ApplyTranslationCandidateCommand
    blk_item = MockBlkItem()
    cmd = ApplyTranslationCandidateCommand(blk_item, None, None)
    cmd.redo()
    assert blk_item.blk.translation == ""
    assert blk_item._plain == ""


# --- CompareWorker tests ---

def test_compare_worker_module_registry_compare(monkeypatch):
    """Test that module registry compare returns candidates for detector/ocr/inpainter scopes."""
    import ui.translation_assist_worker as taw
    monkeypatch.setattr(taw, "GET_VALID_TEXTDETECTORS", lambda: ["ctd", "ysgyolo"])
    from ui.translation_assist_worker import CompareWorker

    worker = CompareWorker(
        scope="detector",
        providers=[],
        source_text="test",
        lang_source="Auto",
        lang_target="简体中文",
        current_module="ctd",
    )
    # run() emits a dict via finished signal; we can call _run_module_registry_compare directly
    cands, warns, tele = worker._run_module_registry_compare()
    assert len(cands) == 2
    # At least one candidate should mark is_current for the current module
    current_cands = [c for c in cands if c.get("telemetry", {}).get("is_current")]
    assert len(current_cands) == 1
    assert current_cands[0]["text"] == "ctd"


def test_compare_worker_translator_compare_with_mock(monkeypatch):
    """Test translator fan-out with a mock translator class."""
    from ui.translation_assist_worker import CompareWorker

    class FakeTranslator:
        def __init__(self, src, tgt, **kw):
            pass
        def translate(self, text):
            return f"translated:{text}"

    import modules.translators.base as trans_base
    import ui.translation_assist_worker as taw

    # Mutate the internal _module_dict directly since module_dict is a read-only property
    original_cls = trans_base.TRANSLATORS._module_dict.get("fake_translator")
    trans_base.TRANSLATORS._module_dict["fake_translator"] = FakeTranslator
    original_get = taw.GET_VALID_TRANSLATORS
    taw.GET_VALID_TRANSLATORS = lambda: ["fake_translator"]

    try:
        worker = CompareWorker(
            scope="translator",
            providers=["fake_translator"],
            source_text="hello",
            lang_source="English",
            lang_target="简体中文",
        )
        cands, warns, tele = worker._run_translator_compare()
        assert len(cands) == 1
        assert cands[0]["text"] == "translated:hello"
        assert cands[0]["provider"] == "fake_translator"
        assert "latency_ms" in cands[0].get("telemetry", {})
    finally:
        if original_cls is not None:
            trans_base.TRANSLATORS._module_dict["fake_translator"] = original_cls
        else:
            trans_base.TRANSLATORS._module_dict.pop("fake_translator", None)
        taw.GET_VALID_TRANSLATORS = original_get


# --- TranslationAssistDock refresh_provider_list test ---

def test_dock_refresh_provider_list_preserves_checked():
    """Test that refresh_provider_list preserves checked items when names overlap."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    from ui.translation_assist_dock import TranslationAssistDock

    app = QApplication.instance() or QApplication([])
    dock = TranslationAssistDock()
    dock.refresh_provider_list(["TM", "Glossary", "google", "deepl"])
    # Check two items
    from qtpy.QtCore import Qt
    dock.provider_list.item(0).setCheckState(Qt.Checked)
    dock.provider_list.item(2).setCheckState(Qt.Checked)

    # Refresh with overlapping names + new ones
    dock.refresh_provider_list(["TM", "Glossary", "SFX", "google", "openai"])
    texts = [dock.provider_list.item(i).text() for i in range(dock.provider_list.count())]
    assert texts == ["TM", "Glossary", "SFX", "google", "openai"]
    # TM and google should still be checked
    assert dock.provider_list.item(0).checkState() == Qt.Checked
    assert dock.provider_list.item(3).checkState() == Qt.Checked
    # Glossary should be unchecked (was unchecked before)
    assert dock.provider_list.item(1).checkState() == Qt.Unchecked


def test_build_candidates_from_sources_deduplicates():
    """Test that build_candidates_from_sources deduplicates by normalized text."""
    from utils.translation_assist import build_candidates_from_sources

    cands = build_candidates_from_sources(
        tm_hits=[{"target": "hello"}, {"target": "hello"}],
        glossary_hits=[{"target": "hello"}],
        sfx_hits=[{"common_en": "hi", "source": "hi"}],
        concordance_hits=[{"target": "world"}],
        max_candidates=10,
    )
    texts = [c["text"] for c in cands]
    # hello should appear once despite 3 sources
    assert texts.count("hello") == 1
    assert "hi" in texts
    assert "world" in texts


def test_dock_has_no_hardcoded_openai_model_combo():
    """Regression: the dock should not have a hardcoded OpenAI-only model combo."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    from ui.translation_assist_dock import TranslationAssistDock

    app = QApplication.instance() or QApplication([])
    dock = TranslationAssistDock()
    assert not hasattr(dock, "openai_model_combo")


def test_normalize_provider_warning_maps_codes():
    from utils.translation_assist import normalize_provider_warning

    assert normalize_provider_warning("google", warning_text="timeout")["code"] == "timeout"
    assert normalize_provider_warning("google", warning_text="api key invalid")["code"] == "auth"
    assert normalize_provider_warning("google", warning_text="rate limit hit")["code"] == "rate_limit"
    assert normalize_provider_warning("google", warning_text="generic")["code"] == "provider_warning"
    assert normalize_provider_warning("google") == {}
