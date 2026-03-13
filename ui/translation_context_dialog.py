"""
Dialog to edit project-level translation context: series path and glossary.
"""

from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QGroupBox,
    QDialogButtonBox,
)
from qtpy.QtCore import Qt

from utils.proj_imgtrans import ProjImgTrans


def _glossary_to_text(glossary: list) -> str:
    """Convert list of {source, target} to one line per entry: source -> target."""
    if not glossary:
        return ""
    lines = []
    for g in glossary:
        if isinstance(g, dict) and g.get("source") is not None and g.get("target") is not None:
            lines.append(f"{g['source']} -> {g['target']}")
    return "\n".join(lines)


def _text_to_glossary(text: str) -> list:
    """Parse one line per entry (source -> target) to list of {source, target}."""
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("->", "→", "=", ":"):
            if sep in line:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    s, t = parts[0].strip(), parts[1].strip()
                    if s and t:
                        out.append({"source": s, "target": t})
                    break
    return out


class TranslationContextDialog(QDialog):
    """Edit project series context path and project glossary. Call with ProjImgTrans or None (read-only/empty)."""

    def __init__(self, imgtrans_proj: ProjImgTrans, parent=None):
        super().__init__(parent)
        self.imgtrans_proj = imgtrans_proj
        self.setWindowTitle(self.tr("Translation context (project)"))
        layout = QVBoxLayout(self)

        group = QGroupBox(self.tr("Project translation context"))
        group_layout = QVBoxLayout(group)

        group_layout.addWidget(QLabel(self.tr("Series context path (folder or ID, e.g. urban_immortal_cultivator):")))
        self.series_path_edit = QLineEdit(self)
        self.series_path_edit.setPlaceholderText(
            self.tr("Leave empty to use default (data/translation_context/default)")
        )
        self.series_path_edit.setClearButtonEnabled(True)
        group_layout.addWidget(self.series_path_edit)

        group_layout.addWidget(QLabel(self.tr("Project glossary (one line per entry: source -> target):")))
        self.glossary_edit = QPlainTextEdit(self)
        self.glossary_edit.setPlaceholderText(
            self.tr("e.g.:\n丹田 -> dantian\n真气 -> true qi\n# lines starting with # are ignored")
        )
        self.glossary_edit.setMinimumHeight(120)
        group_layout.addWidget(self.glossary_edit)

        layout.addWidget(group)

        hint = QLabel(
            self.tr("Series path: glossary is loaded from data/translation_context/<path>/glossary.txt — you do not need to add those terms here or in Config → Translator. "
                    "Project glossary above is for extra terms that apply only to this project (merged with series + translator). "
                    "Optional: in Config → Translator set \"Series context prompt\" (e.g. \"This is a cultivation manhua. Keep place names and terms consistent.\") for style instructions.")
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if imgtrans_proj is not None:
            self.series_path_edit.setText(getattr(imgtrans_proj, "series_context_path", "") or "")
            self.glossary_edit.setPlainText(_glossary_to_text(getattr(imgtrans_proj, "translation_glossary", None) or []))
        else:
            self.series_path_edit.setEnabled(False)
            self.glossary_edit.setEnabled(False)

    def accept(self):
        if self.imgtrans_proj is None:
            super().accept()
            return
        self.imgtrans_proj.series_context_path = self.series_path_edit.text().strip()
        self.imgtrans_proj.translation_glossary = _text_to_glossary(self.glossary_edit.toPlainText())
        try:
            self.imgtrans_proj.save()
        except Exception:
            pass
        super().accept()
