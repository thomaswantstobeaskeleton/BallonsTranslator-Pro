from __future__ import annotations

import importlib.util
from typing import Dict, List


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def collect_renderer_diagnostics() -> Dict[str, object]:
    optional = {
        'uharfbuzz': _has_module('uharfbuzz'),
        'freetype': _has_module('freetype'),
        'python_bidi': _has_module('bidi'),
        'arabic_reshaper': _has_module('arabic_reshaper'),
    }
    advanced_ready = bool(optional['uharfbuzz'] and optional['freetype'])
    warnings: List[str] = []
    if not optional['uharfbuzz']:
        warnings.append('uharfbuzz not installed; advanced shaping backend unavailable.')
    if not optional['freetype']:
        warnings.append('freetype-py not installed; glyph metrics backend unavailable.')
    if not optional['python_bidi']:
        warnings.append('python-bidi not installed; RTL fallback diagnostics limited.')
    return {
        'default_renderer': 'qt',
        'advanced_backend_available': advanced_ready,
        'optional_modules': optional,
        'warnings': warnings,
    }
