"""
Model manager: list modules with downloadable files, check status (downloaded / missing / hash mismatch),
and run downloads for selected modules. Used by the Manage models dialog.
"""
from typing import List, Dict, Any, Tuple, Optional
import os.path as osp

import utils.shared as shared
from utils.download_util import check_local_file

# Lazy import to avoid circular import and ensure registries are populated when UI runs
def _get_registries():
    from modules import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
    return [
        ('textdetector', TEXTDETECTORS, 'Text detection'),
        ('ocr', OCR, 'OCR'),
        ('inpainter', INPAINTERS, 'Inpainting'),
        ('translator', TRANSLATORS, 'Translation'),
    ]


def _resolve_save_paths_from_kwargs(download_kwargs: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """From one entry of download_file_list, return list of (absolute_path, sha256_or_none)."""
    files = download_kwargs.get('files')
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    save_files = download_kwargs.get('save_files')
    if save_files is None:
        save_files = files
    elif not isinstance(save_files, list):
        save_files = [save_files]
    sha256_list = download_kwargs.get('sha256_pre_calculated')
    if sha256_list is None:
        sha256_list = [None] * len(files)
    elif not isinstance(sha256_list, list):
        sha256_list = [sha256_list]
    while len(sha256_list) < len(files):
        sha256_list.append(None)
    save_dir = download_kwargs.get('save_dir')
    if save_dir is not None:
        save_files = [osp.join(save_dir, p) for p in save_files]
    root = getattr(shared, 'PROGRAM_PATH', osp.abspath(osp.dirname(osp.dirname(__file__))))
    result = []
    for i, savep in enumerate(save_files):
        if not osp.isabs(savep):
            savep = osp.join(root, savep)
        sha = sha256_list[i] if i < len(sha256_list) else None
        result.append((savep, sha))
    return result


def get_all_downloadable_modules() -> List[Dict[str, Any]]:
    """Return a list of module infos: category, key, display_name, module_class, can_download."""
    result = []
    for category_key, registry, category_label in _get_registries():
        for key in registry.module_dict.keys():
            try:
                module_class = registry.get(key)
            except Exception:
                continue
            dfl = getattr(module_class, 'download_file_list', None)
            on_load = getattr(module_class, 'download_file_on_load', False)
            can_download = bool(dfl is not None and not on_load)
            display_name = key.replace('_', ' ').title()
            result.append({
                'category': category_key,
                'category_label': category_label,
                'key': key,
                'display_name': display_name,
                'module_class': module_class,
                'can_download': can_download,
                'download_file_list': dfl,
                'download_file_on_load': on_load,
            })
    return result


def check_module_status(module_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check one module: files present and hash valid.
    Returns dict: status ('ok'|'missing'|'hash_mismatch'|'no_download_list'|'download_on_load'), details, file_results.
    """
    dfl = module_info.get('download_file_list')
    on_load = module_info.get('download_file_on_load', False)
    details = []
    file_results = []

    if on_load:
        return {'status': 'download_on_load', 'details': ['Downloads on first use (e.g. Hugging Face).'], 'file_results': []}
    if dfl is None:
        return {'status': 'no_download_list', 'details': ['No predefined file list; manual setup.'], 'file_results': []}

    all_ok = True
    any_bad_hash = False
    check_hash = getattr(shared, 'check_local_file_hash', True)

    for kwargs in dfl:
        for abs_path, sha in _resolve_save_paths_from_kwargs(kwargs):
            exists, valid_hash, _ = check_local_file(abs_path, sha, cache_hash=False)
            file_results.append({'path': abs_path, 'exists': exists, 'valid_hash': valid_hash, 'sha': sha})
            if not exists:
                all_ok = False
                details.append(f'Missing: {osp.basename(abs_path)}')
            elif sha and check_hash and not valid_hash:
                any_bad_hash = True
                all_ok = False
                details.append(f'Hash mismatch: {osp.basename(abs_path)}')
            else:
                details.append(f'OK: {osp.basename(abs_path)}')

    if all_ok:
        status = 'ok'
    elif any_bad_hash:
        status = 'hash_mismatch'
    else:
        status = 'missing'
    return {'status': status, 'details': details, 'file_results': file_results}


def check_module_import(module_info: Dict[str, Any]) -> Optional[str]:
    """Return None if module can be instantiated, else error message (e.g. missing dependency)."""
    module_class = module_info.get('module_class')
    category = module_info.get('category', '')
    try:
        if category == 'translator':
            # BaseTranslator.__init__ requires lang_source, lang_target; avoid raising on unsupported lang
            module_class('English', '简体中文', raise_unsupported_lang=False)
        else:
            module_class()
        return None
    except Exception as e:
        return str(e)


def check_all_models(include_import_check: bool = False) -> List[Dict[str, Any]]:
    """Run check on every module. include_import_check: try instantiating each module (slow, may load models)."""
    modules = get_all_downloadable_modules()
    results = []
    for mod in modules:
        check = check_module_status(mod)
        import_error = None
        if include_import_check:
            import_error = check_module_import(mod)
        results.append({
            'module_info': mod,
            'status': check['status'],
            'details': check['details'],
            'file_results': check.get('file_results', []),
            'import_error': import_error,
        })
    return results


def download_modules(module_class_list: List) -> Tuple[int, int, List[str]]:
    """
    Run download_and_check_module_files for the given module classes.
    Returns (success_count, fail_count, list of error messages).
    """
    from modules.prepare_local_files import download_and_check_module_files
    success = 0
    failed = 0
    errors = []
    for module_class in module_class_list:
        if getattr(module_class, 'download_file_on_load', False) or getattr(module_class, 'download_file_list', None) is None:
            continue
        try:
            download_and_check_module_files([module_class])
            success += 1
        except Exception as e:
            failed += 1
            name = getattr(module_class, '__name__', str(module_class))
            errors.append(f'{name}: {e}')
    return success, failed, errors
