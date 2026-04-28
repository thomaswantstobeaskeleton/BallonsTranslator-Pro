#!/usr/bin/env python3
"""Validate startup/model UI localization resources and zh_CN.ts coverage."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Set

ROOT = Path(__file__).resolve().parents[1]
EN_JSON = ROOT / "translate" / "startup_model_ui.en_US.json"
ZH_JSON = ROOT / "translate" / "startup_model_ui.zh_CN.json"
ZH_TS = ROOT / "translate" / "zh_CN.ts"

TARGET_FILES = [
    ROOT / "ui" / "model_package_selector_dialog.py",
    ROOT / "ui" / "model_manager_dialog.py",
    ROOT / "ui" / "mainwindow.py",
    ROOT / "ui" / "mainwindowbars.py",
]

MODEL_PACKAGE_FILE = ROOT / "utils" / "model_packages.py"

TR_LITERAL = re.compile(r"(?:self|parent)\.tr\(\s*(['\"])(.*?)\1\s*\)")
QCORE_TRANSLATE = re.compile(
    r"QCoreApplication\.translate\(\s*(['\"])(.*?)\1\s*,\s*(['\"])(.*?)\3\s*\)"
)

# Focus only on startup/model-management strings.
SUBSTRING_FILTERS = (
    "model",
    "download",
    "package",
    "compatibility",
    "hash",
    "incompatible",
    "Manage models",
    "Retry model downloads",
    "Core",
)


def _should_include(value: str) -> bool:
    low = value.lower()
    return any(token.lower() in low for token in SUBSTRING_FILTERS)


def _extract_code_strings(paths: Iterable[Path]) -> Set[str]:
    values: Set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for _, s in TR_LITERAL.findall(text):
            if _should_include(s):
                values.add(s)
        for _, _, _, s in QCORE_TRANSLATE.findall(text):
            if _should_include(s):
                values.add(s)
    return values


def _extract_package_labels() -> Set[str]:
    text = MODEL_PACKAGE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    values: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "QT_TRANSLATE_NOOP":
            continue
        if len(node.args) != 2:
            continue
        arg = node.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            values.add(arg.value)
    return values


def _extract_ts_sources() -> Set[str]:
    text = ZH_TS.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"<source>(.*?)</source>", text, flags=re.S))


def _load_json(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    en = _load_json(EN_JSON)
    zh = _load_json(ZH_JSON)
    en_keys, zh_keys = set(en), set(zh)

    err = False

    if en_keys != zh_keys:
        err = True
        print("[i18n] Key mismatch between EN/ZH resources.")
        print("  only in EN:", sorted(en_keys - zh_keys))
        print("  only in ZH:", sorted(zh_keys - en_keys))

    empty_zh = sorted(k for k, v in zh.items() if not str(v).strip())
    if empty_zh:
        err = True
        print("[i18n] Empty zh_CN values:", empty_zh)

    required_strings = _extract_code_strings(TARGET_FILES) | _extract_package_labels()
    en_values = set(en.values())
    missing_in_en = sorted(s for s in required_strings if s not in en_values)
    if missing_in_en:
        err = True
        print("[i18n] Startup/model UI strings missing in EN JSON:")
        for s in missing_in_en:
            print("  -", s)

    ts_sources = _extract_ts_sources()
    missing_in_ts = sorted(s for s in required_strings if s not in ts_sources)
    if missing_in_ts:
        err = True
        print("[i18n] Startup/model UI strings missing in translate/zh_CN.ts:")
        for s in missing_in_ts:
            print("  -", s)

    if err:
        print("[i18n] FAILED")
        return 1

    print("[i18n] OK: startup/model UI translation resources are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
