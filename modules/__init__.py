from .ocr import OCR, OCRBase
from .textdetector import TEXTDETECTORS, TextDetectorBase
from .translators import TRANSLATORS, BaseTranslator
from .inpaint import INPAINTERS, InpainterBase
from .base import DEFAULT_DEVICE, GPUINTENSIVE_SET, LOGGER, merge_config_module_params, \
    init_module_registries, init_textdetector_registries, init_inpainter_registries, init_ocr_registries, init_translator_registries

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
    return _valid_modules(TEXTDETECTORS, list(TEXTDETECTORS.module_dict.keys()))


def GET_VALID_TRANSLATORS():
    return _valid_modules(TRANSLATORS, list(TRANSLATORS.module_dict.keys()))


def GET_VALID_INPAINTERS():
    return list(INPAINTERS.module_dict.keys())


def GET_VALID_OCR():
    return _valid_modules(OCR, list(OCR.module_dict.keys()))


MODULETYPE_TO_REGISTRIES = {
    'textdetector': TEXTDETECTORS,
    'ocr': OCR,
    'inpainter': INPAINTERS,
    'translator': TRANSLATORS
}

# TODO: use manga-image-translator as backend...