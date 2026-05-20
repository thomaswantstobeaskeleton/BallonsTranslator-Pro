from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import os
import numpy as np
from PIL import Image, ImageOps


@dataclass
class TransformOptions:
    crop: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
    resize: Optional[Tuple[int, int]] = None  # w,h
    scale: Optional[float] = None
    border: int = 0
    border_color: Tuple[int, int, int] = (0, 0, 0)
    brightness: float = 0.0  # -255..255
    contrast: float = 1.0    # 0.1..4.0
    perspective_src: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    perspective_dst: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None


def _clamp_crop(shape, crop):
    h, w = shape[:2]
    x, y, cw, ch = [int(v) for v in crop]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    x2 = max(x + 1, min(w, x + max(1, cw)))
    y2 = max(y + 1, min(h, y + max(1, ch)))
    return x, y, x2 - x, y2 - y


def transform_image_array(img: np.ndarray, opts: TransformOptions) -> np.ndarray:
    out = img
    if opts.crop is not None:
        x, y, cw, ch = _clamp_crop(out.shape, opts.crop)
        out = out[y:y + ch, x:x + cw]
    if opts.perspective_src is not None and opts.perspective_dst is not None:
        src = np.array(opts.perspective_src, dtype=np.float64)
        dst = np.array(opts.perspective_dst, dtype=np.float64)
        a = []
        bvec = []
        for (x, y), (u, v) in zip(src, dst):
            a.append([x, y, 1, 0, 0, 0, -u*x, -u*y]); bvec.append(u)
            a.append([0, 0, 0, x, y, 1, -v*x, -v*y]); bvec.append(v)
        h = np.linalg.lstsq(np.array(a), np.array(bvec), rcond=None)[0]
        coeffs = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7])
        pil = Image.fromarray(out)
        pil = pil.transform((pil.width, pil.height), Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)
        out = np.array(pil)
    if opts.scale is not None and opts.scale > 0:
        nw = max(1, int(round(out.shape[1] * float(opts.scale))))
        nh = max(1, int(round(out.shape[0] * float(opts.scale))))
        pil = Image.fromarray(out)
        out = np.array(pil.resize((nw, nh), Image.Resampling.LANCZOS))
    if opts.resize is not None:
        rw, rh = [max(1, int(v)) for v in opts.resize]
        pil = Image.fromarray(out)
        out = np.array(pil.resize((rw, rh), Image.Resampling.LANCZOS))
    if int(opts.border) > 0:
        b = int(opts.border)
        color = tuple(int(c) for c in (opts.border_color or (0, 0, 0)))
        pil = Image.fromarray(out)
        out = np.array(ImageOps.expand(pil, border=b, fill=color))
    c = max(0.1, float(opts.contrast or 1.0))
    b = float(opts.brightness or 0.0)
    if abs(c - 1.0) > 1e-6 or abs(b) > 1e-6:
        arr = out.astype(np.float32) * c + b
        out = np.clip(arr, 0, 255).astype(np.uint8)
    return out


def transform_image_file(input_path: str, output_path: str, opts: TransformOptions) -> Dict[str, Any]:
    if not os.path.isfile(input_path):
        raise ValueError(f"input image not found: {input_path}")
    try:
        img = np.array(Image.open(input_path))
    except Exception as e:
        raise ValueError(f"failed to load image: {input_path} ({e})")
    transformed = transform_image_array(img, opts)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        Image.fromarray(transformed).save(output_path)
    except Exception as e:
        raise ValueError(f"failed to write transformed image: {output_path} ({e})")
    return {
        'input_path': input_path,
        'output_path': output_path,
        'input_shape': list(img.shape[:2][::-1]),
        'output_shape': list(transformed.shape[:2][::-1]),
    }
