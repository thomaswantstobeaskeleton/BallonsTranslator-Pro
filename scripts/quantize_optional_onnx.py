"""Utilities to quantize optional ONNX inpainting models for smaller/faster CPU inference.

Supported modes:
- dynamic (recommended): weights-only quantization via QInt8.
- static: calibrated quantization using random calibration tensors as fallback.

This utility is intentionally optional and best-effort: it never blocks normal app startup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import tempfile
import numpy as np

OPTIONAL_MODELS = [
    Path("data/models/inpainting_lama_2025jan.onnx"),
    Path("data/models/lama_manga.onnx"),
]


def _default_output_path(model_path: Path, mode: str) -> Path:
    return model_path.with_suffix(f".{mode}.int8.onnx")


def _quantize_dynamic(model_path: Path, out_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )


def _quantize_static(model_path: Path, out_path: Path) -> None:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class _RandomReader(CalibrationDataReader):
        def __init__(self, input_name: str, count: int = 8):
            self._items = []
            for _ in range(count):
                sample = np.random.rand(1, 3, 512, 512).astype(np.float32)
                self._items.append({input_name: sample})
            self._iter = iter(self._items)

        def get_next(self):
            return next(self._iter, None)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = _RandomReader(input_name=input_name)

    with tempfile.TemporaryDirectory() as td:
        quantize_static(
            model_input=str(model_path),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            optimize_model=True,
        )


def quantize_optional_onnx_models(mode: str = "dynamic") -> Dict[str, List[str]]:
    mode = (mode or "dynamic").strip().lower()
    if mode not in {"dynamic", "static"}:
        raise ValueError(f"Unsupported mode: {mode}")

    quantizer = _quantize_dynamic if mode == "dynamic" else _quantize_static
    written, skipped = [], []

    for model_path in OPTIONAL_MODELS:
        if not model_path.exists():
            skipped.append(str(model_path))
            continue
        out_path = _default_output_path(model_path, mode)
        quantizer(model_path, out_path)
        written.append(str(out_path))

    return {"mode": mode, "written": written, "skipped": skipped}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize optional ONNX inpainting models.")
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()
    result = quantize_optional_onnx_models(mode=args.mode)
    print(result)
