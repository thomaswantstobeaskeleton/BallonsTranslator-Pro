"""Lazy module exports for detector/OCR/inpaint/translator registries.

Avoid importing heavy optional backends (e.g. OpenCV) at package import time so
lightweight callers (such as translator-only tests) can import submodules
without pulling every dependency eagerly.
"""

from importlib import import_module


def _load_ocr_module():
    return import_module(".ocr", __name__)


def _load_textdetector_module():
    return import_module(".textdetector", __name__)


def _load_translator_module():
    return import_module(".translators", __name__)


def _load_inpaint_module():
    return import_module(".inpaint", __name__)


def _load_base_module():
    return import_module(".base", __name__)


def _valid_modules(registry, all_keys: list) -> list:
    """When dev_mode is False, return only module keys that are downloaded/ready; else return all keys."""
    try:
        from utils.config import pcfg
        if getattr(pcfg, "dev_mode", False):
            return all_keys
        from utils.model_manager import get_available_module_keys
        return get_available_module_keys(registry)
    except Exception:
        return all_keys


def GET_VALID_TEXTDETECTORS():
    textdet = _load_textdetector_module()
    return _valid_modules(textdet.TEXTDETECTORS, list(textdet.TEXTDETECTORS.module_dict.keys()))


def GET_VALID_TRANSLATORS():
    translators = _load_translator_module()
    return _valid_modules(translators.TRANSLATORS, list(translators.TRANSLATORS.module_dict.keys()))


def GET_VALID_INPAINTERS():
    inpaint = _load_inpaint_module()
    return list(inpaint.INPAINTERS.module_dict.keys())


def GET_VALID_OCR():
    ocr = _load_ocr_module()
    return _valid_modules(ocr.OCR, list(ocr.OCR.module_dict.keys()))


def _build_moduletype_to_registries():
    textdet = _load_textdetector_module()
    ocr = _load_ocr_module()
    inpaint = _load_inpaint_module()
    translators = _load_translator_module()
    return {
        "textdetector": textdet.TEXTDETECTORS,
        "ocr": ocr.OCR,
        "inpainter": inpaint.INPAINTERS,
        "translator": translators.TRANSLATORS,
    }


class _LazyModuleTypeRegistries(dict):
    def _ensure_loaded(self):
        if not self:
            self.update(_build_moduletype_to_registries())

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()


MODULETYPE_TO_REGISTRIES = _LazyModuleTypeRegistries()


def __getattr__(name):
    if name in {"OCR", "OCRBase"}:
        mod = _load_ocr_module()
        return getattr(mod, name)
    if name in {"TEXTDETECTORS", "TextDetectorBase"}:
        mod = _load_textdetector_module()
        return getattr(mod, name)
    if name in {"TRANSLATORS", "BaseTranslator"}:
        mod = _load_translator_module()
        return getattr(mod, name)
    if name in {"INPAINTERS", "InpainterBase"}:
        mod = _load_inpaint_module()
        return getattr(mod, name)
    if name in {
        "DEFAULT_DEVICE",
        "GPUINTENSIVE_SET",
        "LOGGER",
        "merge_config_module_params",
        "init_module_registries",
        "init_textdetector_registries",
        "init_inpainter_registries",
        "init_ocr_registries",
        "init_translator_registries",
    }:
        mod = _load_base_module()
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OCR",
    "OCRBase",
    "TEXTDETECTORS",
    "TextDetectorBase",
    "TRANSLATORS",
    "BaseTranslator",
    "INPAINTERS",
    "InpainterBase",
    "DEFAULT_DEVICE",
    "GPUINTENSIVE_SET",
    "LOGGER",
    "merge_config_module_params",
    "init_module_registries",
    "init_textdetector_registries",
    "init_inpainter_registries",
    "init_ocr_registries",
    "init_translator_registries",
    "GET_VALID_TEXTDETECTORS",
    "GET_VALID_TRANSLATORS",
    "GET_VALID_INPAINTERS",
    "GET_VALID_OCR",
    "MODULETYPE_TO_REGISTRIES",
]

# TODO: use manga-image-translator as backend...
