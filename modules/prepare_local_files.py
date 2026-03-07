from typing import Union, List, Optional
import os.path as osp
import os

from . import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
from .base import BaseModule, LOGGER
import utils.shared as shared
from utils.download_util import download_and_check_files

# Optional ONNX inpainter models (modules only registered when onnxruntime is installed)
OPTIONAL_ONNX_MODELS = [
    {
        "url": "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
        "files": osp.join(shared.PROGRAM_PATH, "data/models/inpainting_lama_2025jan.onnx"),
    },
    {
        "url": "https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx",
        "files": osp.join(shared.PROGRAM_PATH, "data/models/lama_manga.onnx"),
    },
]


def download_and_check_module_files(module_class_list: Optional[List[BaseModule]] = None):
    if module_class_list is None:
        module_class_list = []
        for registered in [INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS]:
            for module_key in registered.module_dict.keys():
                module_class_list.append(registered.get(module_key))

    for module_class in module_class_list:
        if module_class.download_file_on_load or module_class.download_file_list is None:
            continue
        for download_kwargs in module_class.download_file_list:
            all_successful = download_and_check_files(**download_kwargs)
            if all_successful:
                continue
            LOGGER.error(f'Please save these files manually to specified path and restart the application, otherwise {module_class} will be unavailable.')


def download_optional_onnx_models():
    """Download optional ONNX inpainter assets (called when optional_onnx package is selected)."""
    for kw in OPTIONAL_ONNX_MODELS:
        download_and_check_files(**kw)


def prepare_pkuseg():
    try:
        import pkuseg
    except:
        import spacy_pkuseg as pkuseg

    flist = [
        {
            'url': 'https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip',
            'files': ['features.pkl', 'weights.npz'],
            'sha256_pre_calculated': ['17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d', '2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'],
            'archived_files': 'postag.zip',
            'save_dir': 'data/models/pkuseg/postag'
        },
        {
            'url': 'https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip',
            'files': ['features.msgpack', 'weights.npz'],
            'sha256_pre_calculated': ['fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18', '5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'],
            'archived_files': 'spacy_ontonotes.zip',
            'save_dir': 'data/models/pkuseg/spacy_ontonotes'
        },
    ]
    for files_download_kwargs in flist:
        download_and_check_files(**files_download_kwargs)

    PKUSEG_HOME = osp.join(shared.PROGRAM_PATH, 'data/models/pkuseg')
    pkuseg.config.pkuseg_home = PKUSEG_HOME

    # there must be data/models/pkuseg/postag.zip and data/models/pkuseg/spacy_ontonotes.zip
    # otherwise the dumb package download these models again becuz its dumb checking
    p = osp.join(PKUSEG_HOME, 'postag.zip')
    if not osp.exists(p):
        os.makedirs(p)

    p = osp.join(PKUSEG_HOME, 'spacy_ontonotes.zip')
    if not osp.exists(p):
        os.makedirs(p)


def prepare_local_files_forall():
    """
    Download model files for selected packages only.
    model_packages_enabled is None = legacy "download all". Otherwise only modules in selected packages.
    """
    import utils.config as program_config
    from utils.model_packages import (
        get_module_classes_for_packages,
        package_ids_include_pkuseg,
        package_ids_include_optional_onnx,
    )

    pcfg = program_config.pcfg
    package_ids = getattr(pcfg, "model_packages_enabled", None)

    module_list = get_module_classes_for_packages(package_ids)
    if module_list is None:
        # Legacy: download all modules
        download_and_check_module_files()
    else:
        # Selected packages only
        download_and_check_module_files(module_list)

    if package_ids_include_optional_onnx(package_ids):
        download_optional_onnx_models()

    if package_ids_include_pkuseg(package_ids):
        prepare_pkuseg()

    if shared.CACHE_UPDATED:
        shared.dump_cache()


