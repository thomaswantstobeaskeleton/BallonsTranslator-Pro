#!/usr/bin/env python3
"""Lint-style check: critical detector exception handlers should log before soft-failing."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "modules/textdetector/detector_hf_object_detection.py": {"_detect", "run_on_image", "_load_model"},
    "modules/textdetector/detector_paddle_det.py": {"_detect", "_load_model"},
    "modules/textdetector/detector_paddle_v5.py": {"_detect", "_load_model", "_split_block_by_image_gap"},
}


class Checker(ast.NodeVisitor):
    def __init__(self, target_functions: set[str]) -> None:
        self.target_functions = target_functions
        self.fn_stack: list[str] = []
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.fn_stack.append(node.name)
        self.generic_visit(node)
        self.fn_stack.pop()

    def _current_function(self) -> str | None:
        return self.fn_stack[-1] if self.fn_stack else None

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        fn = self._current_function()
        if fn in self.target_functions and _is_broad_exception(node.type):
            if not _contains_logger_call(node.body):
                self.violations.append(
                    f"{fn}: broad except at line {node.lineno} has no logger warning/error call"
                )
        self.generic_visit(node)


def _is_broad_exception(exc: ast.expr | None) -> bool:
    if exc is None:
        return True
    if isinstance(exc, ast.Name) and exc.id == "Exception":
        return True
    return False


def _contains_logger_call(nodes: list[ast.stmt]) -> bool:
    for n in ast.walk(ast.Module(body=nodes, type_ignores=[])):
        if not isinstance(n, ast.Call):
            continue
        func = n.func
        if isinstance(func, ast.Attribute) and func.attr in {"_warn_soft_fail", "_warn_rate_limited"}:
            return True
        if isinstance(func, ast.Name) and func.id in {"_warn_rate_limited"}:
            return True
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"warning", "error", "exception"}:
            continue
        base = func.value
        if isinstance(base, ast.Attribute) and base.attr == "logger":
            return True
        if isinstance(base, ast.Name) and "logger" in base.id.lower():
            return True
    return False


def main() -> int:
    all_violations: list[str] = []
    for rel_path, functions in TARGETS.items():
        src_path = ROOT / rel_path
        tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=rel_path)
        checker = Checker(functions)
        checker.visit(tree)
        for v in checker.violations:
            all_violations.append(f"{rel_path}: {v}")

    if all_violations:
        print("Detector exception logging check failed:")
        for v in all_violations:
            print(f"- {v}")
        return 1

    print("Detector exception logging check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
