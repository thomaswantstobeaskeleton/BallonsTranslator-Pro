"""
ONNX Runtime helpers: reduce C++ log spam when loading models (e.g. unused initializer warnings).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def quiet_ort_default_logger() -> None:
    """
    Lower the global ONNX Runtime logger so model load does not print hundreds of
    'Removing initializer ... not used by any node' lines. Safe no-op if API missing.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return
    # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    try:
        ort.set_default_logger_severity(3)
    except AttributeError:
        pass


def quiet_session_options(ort_module: Any) -> Any:
    """SessionOptions with session log level at ERROR (matches default logger tweak)."""
    so = ort_module.SessionOptions()
    so.log_severity_level = 3
    return so


def build_inpaint_onnx_session_options(ort_module: Any) -> Any:
    """
    SessionOptions for LaMa-style inpaint ONNX models: graph optimizations, memory patterns,
    optional intra-op thread cap (see pcfg.module inpaint_onnx_ort_* keys).
    """
    so = quiet_session_options(ort_module)
    try:
        from utils.config import pcfg

        cfg = pcfg.module
        glm = getattr(ort_module, "GraphOptimizationLevel", None)
        if glm is not None:
            level = str(getattr(cfg, "inpaint_onnx_ort_graph_optimization_level", "all") or "all").strip().lower()
            mapping = {
                "disable": getattr(glm, "ORT_DISABLE_ALL", None),
                "basic": getattr(glm, "ORT_ENABLE_BASIC", None),
                "extended": getattr(glm, "ORT_ENABLE_EXTENDED", None),
                "all": getattr(glm, "ORT_ENABLE_ALL", None),
            }
            lvl = mapping.get(level, mapping["all"])
            if lvl is not None:
                so.graph_optimization_level = lvl
        if bool(getattr(cfg, "inpaint_onnx_ort_enable_mem_pattern", True)):
            so.enable_mem_pattern = True
        if bool(getattr(cfg, "inpaint_onnx_ort_enable_cpu_mem_arena", True)):
            so.enable_cpu_mem_arena = True
        try:
            n_threads = int(getattr(cfg, "inpaint_onnx_ort_intra_op_num_threads", 0) or 0)
        except (TypeError, ValueError):
            n_threads = 0
        if n_threads > 0:
            so.intra_op_num_threads = n_threads
    except Exception:
        pass
    return so


def inference_session(
    ort_module: Any,
    model_path: str,
    *,
    providers: Optional[list] = None,
    use_inpaint_onnx_opts: bool = False,
):
    """
    Build InferenceSession with quiet logging. Call quiet_ort_default_logger() once per process
    before first session if you load multiple ONNX models.

    When use_inpaint_onnx_opts is True, apply build_inpaint_onnx_session_options (LaMa ONNX paths).
    """
    quiet_ort_default_logger()
    if use_inpaint_onnx_opts:
        so = build_inpaint_onnx_session_options(ort_module)
    else:
        so = quiet_session_options(ort_module)
    kwargs = {"sess_options": so}
    if providers is not None:
        kwargs["providers"] = providers
    return ort_module.InferenceSession(model_path, **kwargs)


def onnx_image_output_to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Convert an ONNX image tensor to H×W×3 uint8 for blending with OpenCV/RGB pipelines.

    Many exports use float32 in [0, 1], but others use [0, 255] or [-1, 1]. Using
    ``np.clip(x, 0, 1) * 255`` on a [0, 255] tensor clamps almost everything to 255
    (solid white in the inpainted region).
    """
    if arr is None:
        raise ValueError("onnx_image_output_to_uint8_hwc: arr is None")
    if arr.dtype == np.uint8:
        x = arr
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        return np.ascontiguousarray(x)
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if np.isnan(vmin) or np.isnan(vmax):
        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    # [-1, 1] (tanh-style)
    if vmin >= -1.05 and vmax <= 1.05 and vmin < -0.02:
        x = (x + 1.0) * 0.5 * 255.0
    # [0, 1] normalized
    elif vmax <= 1.01 and vmin >= -1e-3:
        x = np.clip(x, 0.0, 1.0) * 255.0
    # float32 in ~0–255 (common ONNX zoo / LaMa exports; avoids white-box bug from clip(...,0,1))
    elif vmax > 1.01:
        x = np.clip(x, 0.0, 255.0)
    else:
        x = np.clip(x, 0.0, 1.0) * 255.0
    return np.clip(np.round(x), 0, 255).astype(np.uint8)
