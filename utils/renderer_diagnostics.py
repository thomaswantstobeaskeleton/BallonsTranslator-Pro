from __future__ import annotations

from typing import Dict

from .renderer_backend_probe import probe_renderer_backends


def collect_renderer_diagnostics() -> Dict[str, object]:
    return probe_renderer_backends()
