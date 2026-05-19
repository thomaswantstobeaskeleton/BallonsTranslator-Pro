from __future__ import annotations

import importlib.util
import platform
from typing import Dict, List


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def probe_renderer_backends() -> Dict[str, object]:
    optional = {
        'uharfbuzz': _has_module('uharfbuzz'),
        'freetype': _has_module('freetype'),
        'python_bidi': _has_module('bidi'),
        'arabic_reshaper': _has_module('arabic_reshaper'),
        'PIL': _has_module('PIL'),
    }
    features = {
        'arabic_shaping': bool(optional['uharfbuzz'] and optional['python_bidi'] and optional['arabic_reshaper']),
        'bidi_support': bool(optional['python_bidi']),
        'glyph_bounds': bool(optional['freetype']),
        'vertical_cjk_layout_hints': True,  # qt/default path supports current layout hints
        'fallback_chain_diagnostics': True,
    }
    advanced_backend = bool(optional['uharfbuzz'] and optional['freetype'])
    warnings: List[str] = []
    install_hints: List[str] = []
    if not optional['uharfbuzz']:
        warnings.append('uharfbuzz not installed; advanced shaping backend unavailable.')
        install_hints.append('pip install uharfbuzz')
    if not optional['freetype']:
        warnings.append('freetype-py not installed; true glyph bounds backend unavailable.')
        install_hints.append('pip install freetype-py')
    if not optional['python_bidi']:
        warnings.append('python-bidi not installed; RTL/bidi diagnostics limited.')
        install_hints.append('pip install python-bidi')
    if not optional['arabic_reshaper']:
        warnings.append('arabic-reshaper not installed; Arabic fallback shaping is limited.')
        install_hints.append('pip install arabic-reshaper')

    return {
        'default_renderer': 'qt',
        'advanced_backend_available': advanced_backend,
        'optional_modules': optional,
        'features': features,
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'python': platform.python_version(),
        },
        'install_hints': sorted(set(install_hints)),
        'warnings': warnings,
    }
