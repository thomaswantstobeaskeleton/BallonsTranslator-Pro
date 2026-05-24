import importlib.util
import sys
import types
from pathlib import Path


class _FakePlatform(types.ModuleType):
    def system(self):
        return "Windows"

    def version(self):
        return "10.0.22631"


class _BlockWinsdkFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "winsdk" or fullname.startswith("winsdk."):
            raise ImportError("winsdk intentionally unavailable for test")
        return None


def test_ocr_windows_import_without_winsdk_does_not_raise_nameerror(monkeypatch):
    """Windows OCR is optional; missing winsdk must not crash module discovery.

    Regression coverage for issue #136: `LOGGER` used to be imported inside the
    same try block as winsdk, so missing winsdk caused the except block itself to
    raise `NameError: LOGGER is not defined`.
    """
    module_path = Path("modules/ocr/ocr_windows.py")
    spec = importlib.util.spec_from_file_location("_test_ocr_windows_missing_winsdk", module_path)
    module = importlib.util.module_from_spec(spec)
    finder = _BlockWinsdkFinder()
    monkeypatch.setitem(sys.modules, "platform", _FakePlatform("platform"))
    sys.meta_path.insert(0, finder)
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
