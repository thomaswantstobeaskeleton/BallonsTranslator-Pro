"""
Centralized model manager for non-module assets (Section 8).

Manages HuggingFace downloads, cache dirs, gated tokens, and optional
faster downloads (Xet). Use this for SAM, upscalers, or any HF-based asset
so we have one place for cache dir, token, and cleanup.

Also provides get_all_downloadable_modules() and check_all_models() for the
Manage models dialog (Tools → Manage models).
"""
from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, List, Optional

from . import shared
from .logger import logger as LOGGER
from .download_util import check_local_file

# Default subdir under shared cache for HF assets
HF_CACHE_SUBDIR = "hub"

_model_manager: Optional["ModelManager"] = None


def _enable_hf_xet_when_token_present(token: Optional[str] = None) -> None:
    """Set HF_XET_HIGH_PERFORMANCE=1 when a HuggingFace token is present (faster first-time downloads)."""
    if token and (token := str(token).strip()):
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        LOGGER.debug("HF_XET_HIGH_PERFORMANCE=1 set (token present)")
    elif not os.environ.get("HF_XET_HIGH_PERFORMANCE"):
        if os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"):
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
            LOGGER.debug("HF_XET_HIGH_PERFORMANCE=1 set (env token present)")


class ModelManager:
    """
    One place to manage HF downloads, cache dirs, and gated tokens.
    Use for optional models (SAM, upscalers, YOLO weights from HF) so
    cache dir and token are consistent and cleanup is centralized.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._cache_dir = cache_dir or osp.join(shared.cache_dir, HF_CACHE_SUBDIR)
        self._hf_token: Optional[str] = None

    @property
    def cache_dir(self) -> str:
        """Default cache directory for HuggingFace assets."""
        return osp.normpath(osp.abspath(self._cache_dir))

    def set_hf_token(self, token: Optional[str]) -> None:
        """
        Set HuggingFace token for gated models and API. When token is present,
        also sets HF_XET_HIGH_PERFORMANCE=1 for faster first-time downloads (Xet).
        """
        raw = (token or "").strip() or None
        if raw:
            try:
                from .validation import validate_huggingface_token
                ok, msg = validate_huggingface_token(raw)
                if not ok and msg:
                    LOGGER.warning("HuggingFace token validation: %s", msg)
            except Exception:
                pass
        self._hf_token = raw
        if self._hf_token:
            os.environ["HF_TOKEN"] = self._hf_token
            os.environ["HF_HUB_TOKEN"] = self._hf_token
            _enable_hf_xet_when_token_present(self._hf_token)
        # Do not clear env if we're just clearing our stored ref; other code may have set it

    def get_hf_token(self) -> Optional[str]:
        """Return the token set via set_hf_token, or from env."""
        if self._hf_token:
            return self._hf_token
        return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or None

    def ensure_cache_dir(self) -> str:
        """Create cache dir if needed; return path."""
        d = self.cache_dir
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            LOGGER.warning("Could not create model cache dir %s: %s", d, e)
        return d

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Download a full repo from HuggingFace. Returns local path.
        Uses manager cache_dir and token if not passed.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required for HF model download. "
                "Install: pip install huggingface_hub"
            ) from e
        cache_dir = cache_dir or self.cache_dir
        token = token or self.get_hf_token()
        self.ensure_cache_dir()
        return snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            **kwargs,
        )

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Download a single file from a HuggingFace repo. Returns local path.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required for HF model download. "
                "Install: pip install huggingface_hub"
            ) from e
        cache_dir = cache_dir or self.cache_dir
        token = token or self.get_hf_token()
        self.ensure_cache_dir()
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            **kwargs,
        )


def get_model_manager(cache_dir: Optional[str] = None) -> ModelManager:
    """Return the global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(cache_dir=cache_dir)
    return _model_manager


def enable_hf_xet_if_token_in_env() -> None:
    """
    Call at startup to set HF_XET_HIGH_PERFORMANCE=1 when HF_TOKEN or HF_HUB_TOKEN
    is already in the environment (e.g. from huggingface-cli login). Faster first-time HF downloads.
    """
    _enable_hf_xet_when_token_present(None)


def get_all_downloadable_modules() -> List[Dict[str, Any]]:
    """
    Collect all registered modules (detectors, OCR, inpainters, translators) for the
    Manage models dialog. Includes every module so optional/pip-only modules (e.g.
    Nemotron) appear in the check table. can_download is True only when the module
    has a download_file_list and not download_file_on_load.
    Returns a list of dicts:
      - category_label
      - module_key
      - display_name
      - description
      - module_class
      - can_download
      - estimated_download_size_bytes (optional)
      - target_paths (relative/absolute paths where available)
    """
    from modules import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
    from .model_packages import get_module_manifest_metadata

    categories = [
        (TEXTDETECTORS, "Detect"),
        (OCR, "OCR"),
        (INPAINTERS, "Inpaint"),
        (TRANSLATORS, "Translator"),
    ]
    result = []
    for registry, category_label in categories:
        for module_key in registry.module_dict.keys():
            module_class = registry.get(module_key)
            if module_class is None:
                continue
            params = getattr(module_class, "params", None) or {}
            human_name = _extract_human_name(module_key, module_class, params)
            description = _extract_module_description(params)
            has_list = getattr(module_class, "download_file_list", None) is not None
            on_load = getattr(module_class, "download_file_on_load", False)
            display_name = f"{category_label} · {human_name} ({module_key})"
            can_download = has_list and not on_load
            manifest_meta = get_module_manifest_metadata(module_key)
            dl_meta = _extract_download_metadata(module_class)
            result.append({
                "category_label": category_label,
                "module_key": module_key,
                "display_name": display_name,
                "human_name": human_name,
                "description": description,
                "module_class": module_class,
                "can_download": can_download,
                "manifest_meta": manifest_meta,
                "estimated_download_size_bytes": dl_meta["estimated_download_size_bytes"],
                "target_paths": dl_meta["target_paths"],
            })
    return result


def _extract_human_name(module_key: str, module_class: Any, params: Dict[str, Any]) -> str:
    """Try to extract user-facing module name from params/class metadata."""
    for key in ("display_name", "name", "title", "label"):
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            v = val.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    class_name = getattr(module_class, "__name__", "") or ""
    return class_name or module_key


def _extract_module_description(params: Dict[str, Any]) -> str:
    desc = params.get("description")
    if isinstance(desc, str):
        return desc.strip()
    if isinstance(desc, dict):
        v = desc.get("value")
        if isinstance(v, str):
            return v.strip()
    return ""


def _extract_download_metadata(module_class: Any) -> Dict[str, Any]:
    """
    Extract optional metadata from download_file_list.
    Supports best-effort keys to keep backward compatibility:
      estimated_size_bytes / size_bytes / download_size_bytes / estimated_total_size_bytes.
    """
    dl_list = getattr(module_class, "download_file_list", None) or []
    target_paths: List[str] = []
    total_size = 0
    has_size = False
    for entry in dl_list:
        if not isinstance(entry, dict):
            continue
        target_paths.extend(_extract_target_paths_from_entry(entry))
        for key in ("estimated_size_bytes", "size_bytes", "download_size_bytes", "estimated_total_size_bytes"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                total_size += int(value)
                has_size = True
        file_sizes = entry.get("file_sizes")
        if isinstance(file_sizes, list):
            for size in file_sizes:
                if isinstance(size, (int, float)):
                    total_size += int(size)
                    has_size = True
    return {
        "estimated_download_size_bytes": total_size if has_size else None,
        "target_paths": target_paths,
    }


def _extract_target_paths_from_entry(entry: Dict[str, Any]) -> List[str]:
    files = entry.get("save_files")
    if files is None:
        files = entry.get("files")
    if files is None:
        return []
    if isinstance(files, str):
        file_list = [files]
    elif isinstance(files, list):
        file_list = [f for f in files if isinstance(f, str)]
    else:
        return []
    save_dir = entry.get("save_dir")
    if isinstance(save_dir, str) and save_dir.strip():
        return [osp.join(save_dir, f) for f in file_list]
    return file_list


def get_available_module_keys(registry) -> List[str]:
    """
    Return module keys that are ready to use: either no file list / download-on-load,
    or all required files are present with valid hash. Used to hide not-downloaded or
    incomplete modules from dropdowns when dev_mode is False. Also excludes modules
    that report is_environment_compatible() False (e.g. Python version requirements).
    """
    program_path = getattr(shared, "PROGRAM_PATH", "") or os.getcwd()
    result = []
    for module_key in registry.module_dict.keys():
        module_class = registry.get(module_key)
        if module_class is None:
            continue
        compat = getattr(module_class, "is_environment_compatible", None)
        if callable(compat) and not compat():
            continue
        mod_info = {
            "display_name": module_key,
            "module_class": module_class,
        }
        r = _check_one_module(mod_info, include_import_check=False)
        status = r["status"]
        # ok = downloaded and hash valid; no_download_list = no files needed (e.g. API); download_on_load = fetches on first use
        if status in ("ok", "no_download_list", "download_on_load"):
            result.append(module_key)
    return result


def _check_one_module(module_info: Dict[str, Any], include_import_check: bool) -> Dict[str, Any]:
    """Check one module's files; optionally try import. Returns dict with module_info, status, details, import_error?."""
    module_class = module_info["module_class"]
    details = []
    status = "ok"
    import_error = None

    if getattr(module_class, "download_file_on_load", False):
        return {
            "module_info": module_info,
            "status": "download_on_load",
            "details": [""],
        }
    download_file_list = getattr(module_class, "download_file_list", None)
    if not download_file_list:
        hint = getattr(module_class, "optional_install_hint", None)
        details_msg = [hint] if isinstance(hint, str) and hint.strip() else [""]
        return {
            "module_info": module_info,
            "status": "no_download_list",
            "details": details_msg,
        }

    program_path = getattr(shared, "PROGRAM_PATH", "") or os.getcwd()
    for d in download_file_list:
        files = d.get("files")
        if files is None:
            continue
        if isinstance(files, str):
            files = [files]
        save_dir = d.get("save_dir")
        save_files = d.get("save_files")
        if save_files is None:
            save_files = list(files)
        elif isinstance(save_files, str):
            save_files = [save_files]
        if save_dir:
            save_files = [osp.join(save_dir, p) for p in save_files]
        sha256_list = d.get("sha256_pre_calculated")
        if sha256_list is None:
            sha256_list = [None] * len(save_files)
        elif isinstance(sha256_list, str):
            sha256_list = [sha256_list]
        while len(sha256_list) < len(save_files):
            sha256_list.append(None)
        for savep, sha in zip(save_files, sha256_list):
            full = osp.join(program_path, savep) if not osp.isabs(savep) else savep
            exists, valid_hash, _ = check_local_file(full, sha, cache_hash=False)
            if not exists:
                details.append(f"Missing: {savep}")
                status = "missing"
            elif not valid_hash:
                details.append(f"Hash mismatch: {savep}")
                status = "hash_mismatch"
            else:
                details.append(savep)
    if not details:
        details = [""]

    if include_import_check and status == "ok":
        try:
            inst = module_class()
            inst.load_model()
        except Exception as e:
            import_error = str(e)

    return {
        "module_info": module_info,
        "status": status,
        "details": details,
        "import_error": import_error,
    }


def check_all_models(include_import_check: bool = False) -> List[Dict[str, Any]]:
    """
    Check all downloadable modules: file presence and optional hash.
    Optionally try loading each module (include_import_check) to detect missing deps.
    Returns a list of dicts: module_info, status ('ok'|'missing'|'hash_mismatch'|'no_download_list'|'download_on_load'), details, import_error?.
    """
    modules = get_all_downloadable_modules()
    results = []
    for mod in modules:
        results.append(_check_one_module(mod, include_import_check))
    return results
