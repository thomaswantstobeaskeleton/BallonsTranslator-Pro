from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QPlainTextEdit, QDialogButtonBox


def _map_to_text(gls: dict) -> str:
    lines = []
    for k, v in (gls or {}).items():
        k = str(k).strip()
        v = str(v).strip()
        if k and v:
            lines.append(f"{k} -> {v}")
    return "\n".join(lines)


def _text_to_map(text: str) -> dict:
    out = {}
    for ln in (text or '').splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if '->' in s:
            a, b = s.split('->', 1)
        elif ':' in s:
            a, b = s.split(':', 1)
        else:
            continue
        a = a.strip(); b = b.strip()
        if a and b:
            out[a] = b
    return out


class GlossaryMapDialog(QDialog):
    def __init__(self, glossary_map: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Glossary Map Editor'))
        self.resize(620, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(self.tr('One mapping per line: source -> target. Lines starting with # are ignored.')))
        self.edit = QPlainTextEdit(self)
        self.edit.setPlaceholderText('hero -> protagonist\nSFX:boom -> BOOM')
        self.edit.setPlainText(_map_to_text(glossary_map or {}))
        lay.addWidget(self.edit, 1)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_map(self) -> dict:
        return _text_to_map(self.edit.toPlainText())
