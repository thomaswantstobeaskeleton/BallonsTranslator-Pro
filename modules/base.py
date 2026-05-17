import gc
import os
import time
from typing import Dict, List, Callable, Union
from copy import deepcopy
from collections import OrderedDict
import re
import importlib

from utils.logger import logger as LOGGER
from utils import shared
from utils.lock import aquire_model_loading_lock, release_model_loading_lock


GPUINTENSIVE_SET = {'cuda', 'mps', 'xpu', 'privateuseone'}

def register_hooks(hooks_registered: OrderedDict, callbacks: Union[List, Callable, Dict]):
    if callbacks is None:
        return
    if isinstance(callbacks, (Dict, OrderedDict)):
        for k, v in callbacks.items():
            hooks_registered[k] = v
    else:
        nhooks = len(hooks_registered)

        if isinstance(callbacks, Callable):
            callbacks = [callbacks]
        for callback in callbacks:
            hk = 'hook_' + str(nhooks).zfill(2)
            while True:
                if hk not in hooks_registered:
                    break
                hk = hk + '_' + str(time.time_ns())
            hooks_registered[hk] = callback
            nhooks += 1


def patch_module_params(cfg_param, module_params, module_name: str = ''):
    # cfg_param = config_params[module_key]
    cfg_key_set = set(cfg_param.keys())
    module_key_set = set(module_params.keys())
    for ck in cfg_key_set:
        if ck not in module_key_set:
            # Internal metadata keys are allowed and should not spam warnings.
            if isinstance(ck, str) and ck.startswith('__'):
                cfg_param.pop(ck)
                continue
            LOGGER.warning(f'Found invalid {module_name} config: {ck}')
            cfg_param.pop(ck)

    for mk in module_key_set:
        if mk not in cfg_key_set:
            if not mk.startswith('__') and mk != 'description':
                LOGGER.info(f'Found new {module_name} config: {mk}')
            cfg_param[mk] = module_params[mk]
        else:
            mparam = module_params[mk]
            cparam = cfg_param[mk]
            if isinstance(mparam, dict):
                tgt_type = mparam.get('data_type', type(mparam['value']))
                if isinstance(cparam, dict):
                    if 'value' in cparam:
                        v = cparam['value']
                    elif isinstance(mparam['value'], dict):
                        for k in mparam['value']:
                            if k in cparam:
                                mparam['value'][k] = cparam[k]
                        v = mparam['value']
                    else:
                        v = mparam['value']
                else:
                    v = cparam
                valid = True
                if tgt_type != type(v):
                    if isinstance(v, str):
                        v = v.strip()
                        if v == '':
                            v = mparam['value']
                            valid = True
                        elif tgt_type is float and ',' in v:
                            v = v.replace(',', '.')
                    if valid and type(v) is not tgt_type:
                        try:
                            v = tgt_type(v)
                        except (ValueError, TypeError):
                            valid = False
                            v = mparam['value']
                if valid:
                    mparam['value'] = v
                cfg_param[mk] = mparam
            else:
                if type(cparam) != type(mparam):
                    if not isinstance(mparam, dict) and isinstance(cparam, dict):
                        cparam = cparam['value']
                    try:
                        cfg_param[mk] = type(mparam)(cparam)
                    except ValueError:
                        LOGGER.warning(f'Invalid param value {cparam} for defined dtype: {type(mparam)}, it will be set to default value: {mparam}')
                        cfg_param[mk] = mparam
    
    cfg_key_list = list(cfg_param.keys())
    module_key_list = list(module_params.keys())
    if cfg_key_list != module_key_list:
        new_params = {key: cfg_param[key] for key in module_key_list}
        cfg_param.clear()
        cfg_param.update(new_params)
        module_key_set = set(module_params.keys())
    cfg_param['__param_patched'] = True
    return cfg_param


def merge_config_module_params(config_params: Dict, module_keys: List, get_module: Callable) -> Dict:
    for module_key in module_keys:
        module_params = get_module(module_key).params
        if module_key not in config_params or config_params[module_key] is None:
            config_params[module_key] = module_params
        else:
            patch_module_params(config_params[module_key], module_params, module_key)
    return config_params


def standardize_module_params(params):
    if params is None:
        return
    for k, v in params.items():
        if not isinstance(v, dict) and k not in {'description'}:  # remember to exclude special keys here
            v = {'value': v}
        if isinstance(v, dict) and 'data_type' not in v:
            v['data_type'] = type(v['value'])
        params[k] = v


class BaseModule:

    params: Dict = None
    logger = LOGGER

    _preprocess_hooks: OrderedDict = None
    _postprocess_hooks: OrderedDict = None

    download_file_list: List = None
    download_file_on_load = False

    _load_model_keys: set = None

    def __init__(self, **params) -> None:
        standardize_module_params(self.params)
        if self.params is not None and '__param_patched' not in params:
            params = patch_module_params(params, self.params, self)
        if params:
            if self.params is None:
                self.params = params
            else:
                self.params.update(params)

    @classmethod
    def register_postprocess_hooks(cls, callbacks: Union[List, Callable]):
        """
        these hooks would be shared among all objects inherited from the same super class
        """
        assert cls._postprocess_hooks is not None
        register_hooks(cls._postprocess_hooks, callbacks)

    @classmethod
    def register_preprocess_hooks(cls, callbacks: Union[List, Callable, Dict]):
        """
        these hooks would be shared among all objects inherited from the same super class
        """
        assert cls._preprocess_hooks is not None
        register_hooks(cls._preprocess_hooks, callbacks)

    def get_param_value(self, param_key: str):
        assert self.params is not None and param_key in self.params
        p = self.params[param_key]
        if isinstance(p, dict):
            return p['value']
        return p
    
    def set_param_value(self, param_key: str, param_value, convert_dtype=True):
        if self.params is None or param_key not in self.params:
            LOGGER.debug(
                "Module %s: ignoring set_param_value for unknown key %r (params keys: %s)",
                getattr(self, "name", self.__class__.__name__),
                param_key,
                list(self.params.keys()) if self.params else None,
            )
            return
        p = self.params[param_key]
        if isinstance(p, dict):
            if convert_dtype:
                val_type = p.get('data_type', type(p['value']))
                if isinstance(param_value, str):
                    param_value = param_value.strip()
                    if param_value == '':
                        param_value = p['value']
                    elif val_type is float and ',' in param_value:
                        param_value = param_value.replace(',', '.')
                if param_value != p['value'] or type(param_value) is not val_type:
                    try:
                        param_value = val_type(param_value)
                    except (ValueError, TypeError):
                        param_value = p['value']
            p['value'] = param_value
        else:
            if convert_dtype:
                if isinstance(param_value, str):
                    param_value = param_value.strip() or p
                if param_value != p:
                    try:
                        param_value = type(p)(param_value)
                    except (ValueError, TypeError):
                        param_value = p
            self.params[param_key] = param_value

    def updateParam(self, param_key: str, param_content):
        """Update a single param by key. Ignores unknown keys (Issue #19: avoid AssertionError from stale UI)."""
        if self.params is None or param_key not in self.params:
            return
        self.set_param_value(param_key, param_content)

    @property
    def low_vram_mode(self):
        if 'low vram mode' in self.params:
            return self.get_param_value('low vram mode')
        return False

    def is_cpu_intensive(self)->bool:
        if self.params is not None and 'device' in self.params:
            return self.params['device']['value'] == 'cpu'
        return False

    def is_gpu_intensive(self) -> bool:
        if self.params is not None and 'device' in self.params:
            return self.params['device']['value'] in GPUINTENSIVE_SET
        return False

    def is_computational_intensive(self) -> bool:
        if self.params is not None and 'device' in self.params:
            return True
        return False
    
    def unload_model(self, empty_cache=False):
        model_deleted = False
        if self._load_model_keys is not None:
            for k in self._load_model_keys:
                if hasattr(self, k):
                    model = getattr(self, k)
                    if model is not None:
                        if hasattr(model, 'unload_model'):
                            model.unload_model(empty_cache=False)
                        del model
                        setattr(self, k, None)
                        model_deleted = True
    
        if empty_cache and model_deleted:
            soft_empty_cache()

        return model_deleted

    def load_model(self):
        # Check and download required files before loading (inform UIs via dialog on failure)
        cls = self.__class__
        if getattr(cls, 'download_file_list', None) is not None and not getattr(cls, 'download_file_on_load', False):
            try:
                from modules.prepare_local_files import download_and_check_module_files
                download_and_check_module_files([cls])
            except Exception as e:
                self.logger.warning(f'Module file check/download failed: {e}')
        aquire_model_loading_lock()
        self._load_model()
        release_model_loading_lock()
        return

    def _load_model(self):
        return

    def all_model_loaded(self):
        if self._load_model_keys is None:
            return True
        for k in self._load_model_keys:
            if not hasattr(self, k) or getattr(self, k) is None:
                return False
        return True

    @classmethod
    def is_environment_compatible(cls) -> bool:
        """Override in subclasses to hide module from non-dev dropdown when env is incompatible (e.g. Python version). Default True."""
        return True

    def __del__(self):
        self.unload_model()

    @property
    def debug_mode(self):
        return shared.DEBUG
    
    def flush(self, param_key: str):
        return None

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch


def _safe_torch_version_attr(attr: str):
    try:
        return getattr(torch.version, attr, None)
    except Exception as e:
        return f'unavailable: {e}'


def _detect_runtime_devices() -> dict:
    """Collect torch/backend device diagnostics without advertising unusable devices."""
    diagnostics = {
        'torch': getattr(torch, '__version__', None),
        'torch_cuda': _safe_torch_version_attr('cuda'),
        'torch_xpu': _safe_torch_version_attr('xpu'),
        'cuda_available': False,
        'cuda_count': 0,
        'cuda_devices': [],
        'cuda_exception': None,
        'directml_available': False,
        'directml_count': 0,
        'directml_default_device': None,
        'directml_exception': None,
        'mps_available': False,
        'mps_built': None,
        'mps_exception': None,
        'xpu_available': False,
        'xpu_count': 0,
        'xpu_devices': [],
        'xpu_exception': None,
        'available_devices': ['cpu'],
        'default_device': 'cpu',
    }

    def append_device(device_name: str):
        if device_name and device_name not in diagnostics['available_devices']:
            diagnostics['available_devices'].append(device_name)

    try:
        if hasattr(torch, 'cuda'):
            diagnostics['cuda_available'] = bool(torch.cuda.is_available())
            diagnostics['cuda_count'] = int(torch.cuda.device_count())
            if diagnostics['cuda_available']:
                diagnostics['default_device'] = 'cuda'
                append_device('cuda')
                for idx in range(diagnostics['cuda_count']):
                    try:
                        diagnostics['cuda_devices'].append(torch.cuda.get_device_name(idx))
                    except Exception as name_exc:
                        diagnostics['cuda_devices'].append(f'cuda:{idx} name unavailable: {name_exc}')
                    append_device(f'cuda:{idx}')
            elif diagnostics['cuda_count'] > 0:
                for idx in range(diagnostics['cuda_count']):
                    try:
                        diagnostics['cuda_devices'].append(torch.cuda.get_device_name(idx))
                    except Exception as name_exc:
                        diagnostics['cuda_devices'].append(f'cuda:{idx} name unavailable: {name_exc}')
    except Exception as e:
        diagnostics['cuda_exception'] = str(e)

    try:
        if hasattr(torch, 'xpu'):
            diagnostics['xpu_available'] = bool(torch.xpu.is_available())
            if diagnostics['xpu_available']:
                diagnostics['xpu_count'] = int(torch.xpu.device_count()) if hasattr(torch.xpu, 'device_count') else 1
                diagnostics['default_device'] = 'xpu'
                append_device('xpu')
                for idx in range(diagnostics['xpu_count']):
                    try:
                        get_name = getattr(torch.xpu, 'get_device_name', None)
                        diagnostics['xpu_devices'].append(get_name(idx) if get_name else f'xpu:{idx}')
                    except Exception as name_exc:
                        diagnostics['xpu_devices'].append(f'xpu:{idx} name unavailable: {name_exc}')
    except Exception as e:
        diagnostics['xpu_exception'] = str(e)

    try:
        mps_backend = getattr(getattr(torch, 'backends', None), 'mps', None)
        if mps_backend is not None:
            is_built = getattr(mps_backend, 'is_built', None)
            diagnostics['mps_built'] = bool(is_built()) if callable(is_built) else None
            diagnostics['mps_available'] = bool(mps_backend.is_available())
            if diagnostics['mps_available']:
                diagnostics['default_device'] = 'mps'
                append_device('mps')
    except Exception as e:
        diagnostics['mps_exception'] = str(e)

    try:
        import torch_directml
        diagnostics['directml_count'] = int(torch_directml.device_count())
        diagnostics['directml_available'] = diagnostics['directml_count'] > 0
        if diagnostics['directml_available'] and hasattr(torch, 'privateuseone'):
            torch.dml = torch_directml
            diagnostics['directml_default_device'] = str(torch.dml.default_device())
            diagnostics['default_device'] = f'privateuseone:{torch.dml.default_device()}'
            for d in range(diagnostics['directml_count']):
                append_device(f'privateuseone:{d}')
    except Exception as e:
        diagnostics['directml_exception'] = str(e)

    return diagnostics


_DEVICE_DIAGNOSTICS = _detect_runtime_devices()
DEFAULT_DEVICE = _DEVICE_DIAGNOSTICS['default_device']
AVAILABLE_DEVICES = list(_DEVICE_DIAGNOSTICS['available_devices'])


def get_device_diagnostics() -> dict:
    """Return a copy of runtime backend diagnostics gathered at import time."""
    return deepcopy(_DEVICE_DIAGNOSTICS)


def get_available_devices() -> List[str]:
    """Return devices that passed runtime initialization checks."""
    return list(AVAILABLE_DEVICES)


def _cuda_unavailable_reason(diagnostics: dict) -> str:
    if diagnostics.get('cuda_exception'):
        return f"CUDA detection raised an exception: {diagnostics['cuda_exception']}"
    if diagnostics.get('torch_cuda') is None:
        return (
            'PyTorch appears to be CPU-only (torch.version.cuda is None). '
            'Install a CUDA-enabled PyTorch wheel for your NVIDIA driver. '
            'Suggested pip command: python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 '
            '(or choose the CUDA version matching your system at https://pytorch.org/get-started/locally/).'
        )
    if not diagnostics.get('cuda_available'):
        return (
            f"PyTorch was built with CUDA {diagnostics.get('torch_cuda')}, but CUDA is not usable by PyTorch. "
            'Check the NVIDIA driver, CUDA runtime visibility, and that no environment variable is hiding the GPU.'
        )
    return ''


def get_device_diagnostics_text() -> str:
    """Human-readable diagnostics for logs, tooltips, and settings UI."""
    d = _DEVICE_DIAGNOSTICS
    devices = ', '.join(d.get('available_devices', [])) or 'none'
    cuda_names = ', '.join(d.get('cuda_devices', [])) or 'none'
    xpu_names = ', '.join(d.get('xpu_devices', [])) or 'none'
    backend = DEFAULT_DEVICE.split(':', 1)[0] if DEFAULT_DEVICE != 'cpu' else 'cpu'
    lines = [
        f"Detected backend: {backend}",
        f"Available device selectors: {devices}",
        f"PyTorch: {d.get('torch')}",
        f"PyTorch CUDA build: {d.get('torch_cuda')}",
        f"CUDA available/count: {d.get('cuda_available')} / {d.get('cuda_count')}",
        f"CUDA GPU names: {cuda_names}",
        f"DirectML available/count: {d.get('directml_available')} / {d.get('directml_count')}",
        f"MPS available/built: {d.get('mps_available')} / {d.get('mps_built')}",
        f"XPU available/count: {d.get('xpu_available')} / {d.get('xpu_count')}",
        f"XPU device names: {xpu_names}",
    ]
    if d.get('cuda_exception'):
        lines.append(f"CUDA exception: {d['cuda_exception']}")
    if d.get('directml_exception'):
        lines.append(f"DirectML probe: {d['directml_exception']}")
    if d.get('mps_exception'):
        lines.append(f"MPS probe: {d['mps_exception']}")
    if d.get('xpu_exception'):
        lines.append(f"XPU probe: {d['xpu_exception']}")
    reason = _cuda_unavailable_reason(d)
    if DEFAULT_DEVICE == 'cpu' and reason:
        lines.append(f"CUDA unavailable: {reason}")
    return '\n'.join(lines)


def _get_device_selector_description(options: List[str]) -> str:
    d = _DEVICE_DIAGNOSTICS
    lines = [
        f"Detected backend: {DEFAULT_DEVICE.split(':', 1)[0] if DEFAULT_DEVICE != 'cpu' else 'cpu'}",
        f"Available devices: {', '.join(options) if options else 'cpu'}",
        f"PyTorch CUDA build: {d.get('torch_cuda')}",
        f"DirectML available/count: {d.get('directml_available')} / {d.get('directml_count')}",
    ]
    if d.get('cuda_devices'):
        lines.append(f"CUDA GPU names: {', '.join(d['cuda_devices'])}")
    reason = _cuda_unavailable_reason(d)
    if 'cuda' not in options and reason:
        lines.append(reason)
    return '\n'.join(lines)


LOGGER.info(
    'Runtime device diagnostics: torch=%s, torch_cuda=%s, cuda_available=%s, cuda_count=%s, devices=%s, cuda_exception=%s',
    _DEVICE_DIAGNOSTICS.get('torch'),
    _DEVICE_DIAGNOSTICS.get('torch_cuda'),
    _DEVICE_DIAGNOSTICS.get('cuda_available'),
    _DEVICE_DIAGNOSTICS.get('cuda_count'),
    _DEVICE_DIAGNOSTICS.get('available_devices'),
    _DEVICE_DIAGNOSTICS.get('cuda_exception'),
)
if DEFAULT_DEVICE == 'cpu':
    LOGGER.warning('Runtime device diagnostics: %s', _cuda_unavailable_reason(_DEVICE_DIAGNOSTICS))

try:
    BF16_SUPPORTED = DEFAULT_DEVICE == 'cuda' and torch.cuda.is_bf16_supported() or DEFAULT_DEVICE == 'xpu' and torch.xpu.is_bf16_supported()
except Exception:
    BF16_SUPPORTED = False

def is_nvidia():
    if DEFAULT_DEVICE == 'cuda':
        if torch.version.cuda:
            return True
    return False

def is_intel():
    if DEFAULT_DEVICE == 'xpu':
        if torch.version.xpu:
            return True
    return False

def soft_empty_cache():
    gc.collect()
    if DEFAULT_DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif DEFAULT_DEVICE == 'xpu':
       torch.xpu.empty_cache()
       # torch.xpu.ipc_collect()
    elif DEFAULT_DEVICE == 'mps':
        torch.mps.empty_cache()


def DEVICE_SELECTOR(not_supported: list[str] = None):
    not_supported = not_supported or []
    options = [opt for opt in AVAILABLE_DEVICES if all(device not in opt for device in not_supported)]
    value = DEFAULT_DEVICE if not any(DEFAULT_DEVICE in device for device in not_supported) else 'cpu'
    if value not in options:
        value = 'cpu'
    return deepcopy(
        {
            'type': 'selector',
            'options': options,
            'value': value,
            'description': _get_device_selector_description(options),
        }
    )

TORCH_DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

MODULE_SCRIPTS = {
    'translator': {'module_dir': 'modules/translators', 'module_pattern': r'trans_(.*?).py'},
    'textdetector': {'module_dir': 'modules/textdetector', 'module_pattern': r'detector_(.*?).py'},
    'inpainter': {'module_dir': 'modules/inpaint', 'module_pattern': r'inpaint_(.*?).py'},
    'ocr': {'module_dir': 'modules/ocr', 'module_pattern': r'ocr_(.*?).py'},
}
    
def init_module_registries(target_modules=None):
    def _load_module(module_dir: str, module_pattern: str):
        modules = os.listdir(module_dir)
        pattern = re.compile(module_pattern)
        module_path = module_dir.replace('/', '.')
        if not module_path.endswith('.'):
            module_path += '.'
        for module_name in modules:
            if pattern.match(module_name) is not None:
                try:
                    module = module_path + module_name.replace('.py', '')
                    importlib.import_module(module)
                except Exception as e:
                    LOGGER.warning(f'Failed to import {module}: {e}')

    if target_modules is None:
        target_modules = MODULE_SCRIPTS
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    for k in target_modules:
        _load_module(**MODULE_SCRIPTS[k])
    if target_modules is None or 'translator' in target_modules:
        _refresh_ensemble_translator_options()
        _refresh_chimera_ensemble_options()


def init_textdetector_registries():
    init_module_registries('textdetector')


def init_inpainter_registries():
    init_module_registries('inpainter')


def init_ocr_registries():
    init_module_registries('ocr')


def init_translator_registries():
    init_module_registries('translator')
    _refresh_ensemble_translator_options()
    _refresh_chimera_ensemble_options()


def _refresh_ensemble_translator_options():
    """Refresh Ensemble (3+1) candidate/judge selector options so the full translator list is available."""
    try:
        from modules.translators import TRANSLATORS
        ensemble = TRANSLATORS.module_dict.get('Ensemble (3+1)')
        if ensemble is not None and hasattr(ensemble, 'params'):
            from modules.translators.trans_ensemble import _valid_candidate_translators
            valid = _valid_candidate_translators()
            for key in ('candidate_1', 'candidate_2', 'candidate_3', 'judge_translator'):
                if key in ensemble.params and isinstance(ensemble.params[key], dict):
                    ensemble.params[key]['options'] = valid
    except Exception as e:
        LOGGER.warning('Could not refresh Ensemble translator options: %s', e)


def _refresh_chimera_ensemble_options():
    """Refresh Chimera (multi-source) candidate selector options so the full translator list is available."""
    try:
        from modules.translators.trans_chimera_ensemble import _refresh_chimera_ensemble_options as _do_refresh
        _do_refresh()
    except Exception as e:
        LOGGER.warning('Could not refresh Chimera (multi-source) candidate options: %s', e)

