"""Diagnostics page that runs actual system checks."""

from __future__ import annotations
import sys
import shutil

from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QPlainTextEdit, QGridLayout
from qtpy.QtCore import Qt

from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, StatusPill, AccentButton


class DiagnosticsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._check_cards = {}
        self._log = None
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Diagnostics", "System, rendering, OCR, model, and pipeline health checks."))

        self._grid = QGridLayout()
        self._grid.setSpacing(SPACING.lg)
        for i, name in enumerate(["GPU", "VRAM", "RAM", "Disk Space", "Python", "Dependencies"]):
            card = ShellCard(name)
            self._check_cards[name] = card
            self._grid.addWidget(card, i // 3, i % 3)
        root.addLayout(self._grid)

        log_card = ShellCard("Diagnostics Log")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        log_card.layout.addWidget(self._log)
        row = QHBoxLayout()
        run_btn = AccentButton("Run Full Diagnostics")
        run_btn.clicked.connect(self._run_diagnostics)
        row.addWidget(run_btn)
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self._export_report)
        row.addWidget(export_btn)
        row.addStretch()
        log_card.layout.addLayout(row)
        root.addWidget(log_card, 1)

        self._run_diagnostics()

    def _run_diagnostics(self):
        self._log.clear()
        self._log.appendPlainText("Running diagnostics...\n")

        results = []

        # GPU check
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                results.append(("GPU", f"{gpu}", 100, COLORS.success))
                results.append(("VRAM", f"{total:.1f} GB", int(min(total / 24, 1) * 100), COLORS.success))
                self._log.appendPlainText(f"GPU: {gpu}")
                self._log.appendPlainText(f"VRAM: {total:.1f} GB")
            else:
                results.append(("GPU", "CPU only", 50, COLORS.warning))
                results.append(("VRAM", "N/A", 0, COLORS.warning))
                self._log.appendPlainText("GPU: CPU only (no CUDA)")
        except Exception as e:
            results.append(("GPU", f"Error: {e}", 0, COLORS.danger))
            results.append(("VRAM", "Unknown", 0, COLORS.warning))
            self._log.appendPlainText(f"GPU check error: {e}")

        # RAM check
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_gb = mem.total / (1024**3)
            results.append(("RAM", f"{ram_gb:.1f} GB", int(mem.percent), COLORS.success if mem.percent < 80 else COLORS.warning))
            self._log.appendPlainText(f"RAM: {ram_gb:.1f} GB ({mem.percent}% used)")
        except Exception as e:
            results.append(("RAM", f"Error: {e}", 0, COLORS.warning))
            self._log.appendPlainText(f"RAM check error: {e}")

        # Disk check
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            pct = int(used / total * 100)
            results.append(("Disk Space", f"{free_gb:.1f} GB free", pct, COLORS.success if pct < 85 else COLORS.warning))
            self._log.appendPlainText(f"Disk: {free_gb:.1f} GB free")
        except Exception as e:
            results.append(("Disk Space", f"Error: {e}", 0, COLORS.warning))
            self._log.appendPlainText(f"Disk check error: {e}")

        # Python version
        results.append(("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", 100, COLORS.success))
        self._log.appendPlainText(f"Python: {sys.version}")

        # Dependencies check
        try:
            import torch
            import cv2
            import numpy
            results.append(("Dependencies", "Core OK", 100, COLORS.success))
            self._log.appendPlainText("Dependencies: torch, cv2, numpy OK")
        except ImportError as e:
            results.append(("Dependencies", f"Missing: {e}", 50, COLORS.warning))
            self._log.appendPlainText(f"Dependencies missing: {e}")

        self._update_cards(results)
        self._log.appendPlainText("\nDiagnostics complete.")

    def _update_cards(self, results):
        for name, status, pct, color in results:
            card = self._check_cards.get(name)
            if not card:
                continue
            # Clear old layout content
            while card.layout.count():
                item = card.layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            row = QHBoxLayout()
            row.addWidget(QLabel(status))
            row.addStretch()
            row.addWidget(StatusPill("OK" if pct >= 50 else "Check", color))
            card.layout.addLayout(row)
            pb = QProgressBar()
            pb.setValue(pct)
            card.layout.addWidget(pb)

    def _export_report(self):
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export Diagnostics", "diagnostics.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log.toPlainText())
