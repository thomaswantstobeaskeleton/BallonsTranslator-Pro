from __future__ import annotations

import json
import os
import os.path as osp
import shutil
from typing import Dict, List, Tuple

from utils.io_utils import imread, imwrite


def _safe_copy(src: str, dst: str) -> bool:
    if not src or not osp.exists(src):
        return False
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_layered_psd_handoff(project, page_name: str, out_dir: str, final_image=None) -> Dict:
    """Export helper layers + editable text metadata for rebuilding a PSD in Photoshop/GIMP.

    This intentionally avoids writing a fake PSD when native editable PSD writing is unavailable.
    The manifest and JSX preserve text, geometry, style, and helper layer paths so downstream
    tools can create editable text layers instead of silently rasterizing text.
    """
    if not page_name:
        raise ValueError("page_name is required")
    os.makedirs(out_dir, exist_ok=True)
    base = osp.splitext(osp.basename(page_name))[0]
    layer_dir = osp.join(out_dir, f"{base}_layers")
    os.makedirs(layer_dir, exist_ok=True)
    warnings: List[str] = []
    layers: Dict[str, str] = {}

    orig = osp.join(project.directory, page_name)
    if _safe_copy(orig, osp.join(layer_dir, "01_original" + osp.splitext(page_name)[1])):
        layers["original"] = osp.relpath(osp.join(layer_dir, "01_original" + osp.splitext(page_name)[1]), out_dir)
    else:
        warnings.append("Original image was not found.")

    inpainted_path = project.get_inpainted_path(page_name)
    if _safe_copy(inpainted_path, osp.join(layer_dir, "02_inpainted.png")):
        layers["inpainted"] = osp.relpath(osp.join(layer_dir, "02_inpainted.png"), out_dir)
    else:
        warnings.append("Inpainted/clean layer was not available.")

    mask_path = project.get_mask_path(page_name)
    if _safe_copy(mask_path, osp.join(layer_dir, "03_mask.png")):
        layers["mask"] = osp.relpath(osp.join(layer_dir, "03_mask.png"), out_dir)
    else:
        warnings.append("Mask layer was not available.")

    if final_image is not None:
        final_path = osp.join(layer_dir, "05_final_composite.png")
        imwrite(final_path, final_image, ext=".png")
        layers["final_composite"] = osp.relpath(final_path, out_dir)
    else:
        result_path = project.get_result_path(page_name)
        if _safe_copy(result_path, osp.join(layer_dir, "05_final_composite.png")):
            layers["final_composite"] = osp.relpath(osp.join(layer_dir, "05_final_composite.png"), out_dir)
        else:
            warnings.append("Final rendered composite was not available; render the page first for a flattened reference layer.")

    from utils.rendering_qa import analyze_text_block

    text_layers = []
    for idx, blk in enumerate(project.pages.get(page_name, []) or []):
        ff = getattr(blk, "fontformat", None)
        xyxy = list(getattr(blk, "xyxy", [0, 0, 0, 0]) or [0, 0, 0, 0])
        text_value = getattr(blk, "translation", "") or "\n".join(getattr(blk, "text", []) or [])
        diagnostics = analyze_text_block(blk, page_name, idx)
        text_layers.append({
            "name": f"translated_text_{idx + 1:03d}",
            "text": text_value,
            "xyxy": xyxy,
            "font_family": getattr(ff, "font_family", "") if ff else "",
            "font_size_px": getattr(ff, "font_size", 24.0) if ff else 24.0,
            "fill_rgb": getattr(ff, "frgb", [0, 0, 0]) if ff else [0, 0, 0],
            "stroke_rgb": getattr(ff, "srgb", [0, 0, 0]) if ff else [0, 0, 0],
            "stroke_width": getattr(ff, "stroke_width", 0.0) if ff else 0.0,
            "writing_mode": getattr(ff, "writing_mode", "auto") if ff else "auto",
            "alignment": getattr(ff, "alignment", 0) if ff else 0,
            "line_spacing": getattr(ff, "line_spacing", 1.2) if ff else 1.2,
            "letter_spacing": getattr(ff, "letter_spacing", 1.0) if ff else 1.0,
            "opacity": getattr(ff, "opacity", 1.0) if ff else 1.0,
            "fit_mode": getattr(ff, "fit_mode", "shrink") if ff else "shrink",
            "line_break_strategy": getattr(ff, "line_break_strategy", "auto") if ff else "auto",
            "text_padding": getattr(ff, "text_padding", 0.0) if ff else 0.0,
            "fallback_font_chain": getattr(ff, "fallback_font_chain", "") if ff else "",
            "rendering_diagnostics": diagnostics,
        })
        for warning in diagnostics.get("warnings", []) if isinstance(diagnostics, dict) else []:
            warnings.append(f"Text layer {idx + 1}: {warning}")
    if not text_layers:
        warnings.append("No translated text layers were found for this page.")

    manifest = {
        "format": "ballonstranslator.layered_psd_handoff.v1",
        "page": page_name,
        "layers": layers,
        "text_layers": text_layers,
        "warnings": warnings,
        "notes": [
            "This package preserves editable text metadata and helper layers.",
            "Native PSD text-layer writing is not available in this Python build, so use the JSX script or manifest in Photoshop/GIMP.",
        ],
    }
    manifest_path = osp.join(out_dir, f"{base}_psd_handoff.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    jsx_path = osp.join(out_dir, f"{base}_rebuild_psd.jsx")
    with open(jsx_path, "w", encoding="utf-8") as f:
        f.write(_build_photoshop_jsx(manifest_path, manifest))
    manifest["manifest_path"] = manifest_path
    manifest["photoshop_jsx_path"] = jsx_path
    return manifest


def _build_photoshop_jsx(manifest_path: str, manifest: Dict) -> str:
    lines = [
        "// BallonsTranslator layered PSD handoff rebuild script",
        "// Open the original/inpainted layer manually if your Photoshop version blocks file IO.",
        f"// Manifest: {manifest_path}",
        "var doc = app.documents.length ? app.activeDocument : null;",
        "if (!doc) { alert('Open the original or inpainted page image first, then run this script.'); }",
    ]
    for layer in manifest.get("text_layers", []):
        text = json.dumps(layer.get("text", ""))
        name = json.dumps(layer.get("name", "translated_text"))
        x1, y1, x2, y2 = (layer.get("xyxy") or [0, 0, 0, 0])[:4]
        size_pt = max(1.0, float(layer.get("font_size_px", 24.0)) * 0.75)
        lines.extend([
            "if (doc) {",
            "  var lyr = doc.artLayers.add();",
            "  lyr.kind = LayerKind.TEXT;",
            f"  lyr.name = {name};",
            f"  lyr.textItem.contents = {text};",
            f"  lyr.textItem.position = [{float(x1):.2f}, {float(y1):.2f}];",
            f"  lyr.textItem.size = {size_pt:.2f};",
            f"  lyr.opacity = {max(0.0, min(100.0, float(layer.get('opacity', 1.0)) * 100.0)):.2f};",
            "  // Style notes: writing_mode=" + json.dumps(str(layer.get('writing_mode', 'auto'))) + ", fit_mode=" + json.dumps(str(layer.get('fit_mode', 'shrink'))) + ", fallback_chain=" + json.dumps(str(layer.get('fallback_font_chain', ''))) + ";",
            "}",
        ])
    lines.append("if (doc) { alert('Editable text layers created. Save as PSD from Photoshop.'); }")
    return "\n".join(lines) + "\n"
