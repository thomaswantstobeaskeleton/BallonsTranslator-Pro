from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QPlainTextEdit, QDialogButtonBox


class RegexProfileDialog(QDialog):
    def __init__(self, profiles: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Regex Replace Profiles"))
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(self.tr("One rule per line: name|||pattern|||replacement|||flags")))
        self.editor = QPlainTextEdit(self)
        lay.addWidget(self.editor, 1)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        lines = []
        for name, rules in (profiles or {}).items():
            for pattern, repl, flags in rules:
                lines.append(f"{name}|||{pattern}|||{repl}|||{flags}")
        self.editor.setPlainText("\n".join(lines))

    def get_profiles(self) -> dict:
        out = {}
        for raw in self.editor.toPlainText().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|||")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            pattern = parts[1]
            repl = parts[2]
            flags = parts[3] if len(parts) > 3 else ""
            out.setdefault(name, []).append((pattern, repl, flags))
        return out
