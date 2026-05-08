from __future__ import annotations

import html
import json
import os
import os.path as osp
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional

from .layered_psd_export import build_layered_psd_handoff



def _safe_copy(src: str, dst: str) -> bool:
    if not src or not osp.exists(src):
        return False
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _rel(path: str, base: str) -> str:
    if not path:
        return ""
    try:
        return osp.relpath(path, base)
    except ValueError:
        return path


def _write_html_index(path: str, manifest: Dict, report: Dict, page_dir: str) -> None:
    blocks = []
    for page in report.get("pages", []) or []:
        for blk in page.get("blocks", []) or []:
            proof = blk.get("proof_metrics", {}) or {}
            blocks.append(
                "<tr>"
                f"<td>{html.escape(str(blk.get('index', '')))}</td>"
                f"<td>{html.escape(str(blk.get('resolved_writing_mode', blk.get('writing_mode', ''))))}</td>"
                f"<td>{html.escape(', '.join(blk.get('warnings', []) or []))}</td>"
                f"<td>{html.escape(', '.join(s.get('action', '') for s in (blk.get('suggestions', []) or []) if s.get('action')))}</td>"
                f"<td>{html.escape(str(proof.get('density', '')))}</td>"
                f"<td>{html.escape(str(proof.get('overflow_pixels', '')))}</td>"
                "</tr>"
            )
    def link(label: str, key: str) -> str:
        val = manifest.get(key) or ""
        if not val:
            return f"<li>{html.escape(label)}: unavailable</li>"
        return f'<li><a href="{html.escape(_rel(str(val), page_dir))}">{html.escape(label)}</a></li>'
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Lettering proof - {html.escape(str(manifest.get('page', '')))}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}code{{background:#f5f5f5;padding:1px 4px}}</style></head>
<body><h1>Lettering proof: {html.escape(str(manifest.get('page', '')))}</h1>
<p>Next actions: <code>{html.escape(', '.join(manifest.get('next_actions', []) or []) or 'none')}</code></p>
<h2>Files</h2><ul>{link('Typography QA JSON', 'typography_qa_json')}{link('Typography QA Markdown', 'typography_qa_markdown')}{link('Editable SVG handoff', 'svg_path')}{link('PSD helper manifest', 'psd_handoff_manifest')}{link('Final composite', 'final_composite')}</ul>
<h2>Warnings</h2><ul>{''.join('<li>'+html.escape(str(w))+'</li>' for w in (manifest.get('warnings', []) or ['none']))}</ul>
<h2>Text blocks</h2><table><thead><tr><th>#</th><th>Mode</th><th>Warnings</th><th>Suggestions</th><th>Density</th><th>Overflow px</th></tr></thead><tbody>{''.join(blocks) or '<tr><td colspan="6">No text blocks</td></tr>'}</tbody></table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def build_lettering_proof_pack(project, page_name: str, out_dir: str, final_image=None, final_image_path: Optional[str] = None, config_obj=None) -> Dict:
    """Build a compact per-page lettering QA/proof handoff.

    The proof pack is intentionally editor-agnostic: it includes QA JSON/Markdown,
    an editable SVG handoff, a PSD-helper manifest, copied helper layers where
    available, and a final composite reference when the caller supplies one.
    """
    if not page_name:
        raise ValueError("page_name is required")
    os.makedirs(out_dir, exist_ok=True)
    base = osp.splitext(osp.basename(page_name))[0]
    page_dir = osp.join(out_dir, base + "_lettering_proof")
    os.makedirs(page_dir, exist_ok=True)
    warnings = []

    final_ref = final_image_path or ""
    if final_image is not None:
        final_ref = osp.join(page_dir, "final_composite.png")
        from .io_utils import imwrite
        imwrite(final_ref, final_image, ext=".png")
    elif final_ref:
        copied = osp.join(page_dir, "final_composite" + (osp.splitext(final_ref)[1] or ".png"))
        if _safe_copy(final_ref, copied):
            final_ref = copied
    else:
        result_path = project.get_result_path(page_name)
        copied = osp.join(page_dir, "final_composite" + (osp.splitext(result_path)[1] or ".png"))
        if _safe_copy(result_path, copied):
            final_ref = copied
        else:
            warnings.append("Final rendered composite is missing; proof pack contains editable handoff and QA only.")

    from .rendering_qa import build_project_rendering_qa, rendering_qa_to_markdown
    from .svg_text_export import build_svg_text_handoff

    report = build_project_rendering_qa(project, pages=[page_name], include_ok=True, config_obj=config_obj)
    qa_json_path = osp.join(page_dir, "typography_qa.json")
    with open(qa_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    qa_md_path = osp.join(page_dir, "typography_qa.md")
    with open(qa_md_path, "w", encoding="utf-8") as f:
        f.write(rendering_qa_to_markdown(report))

    svg_manifest = build_svg_text_handoff(project, page_name, osp.join(page_dir, "svg"), final_image_path=final_ref or None)
    try:
        psd_manifest = build_layered_psd_handoff(project, page_name, osp.join(page_dir, "psd_handoff"), final_image=final_image)
    except Exception as exc:
        psd_manifest = {"manifest_path": "", "warnings": [f"PSD helper handoff unavailable: {exc}"]}
    warnings.extend(svg_manifest.get("warnings", []) or [])
    warnings.extend(psd_manifest.get("warnings", []) or [])

    summary = report.get("summary", {}) or {}
    manifest = {
        "format": "ballonstranslator.lettering_proof_pack.v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "page": page_name,
        "page_dir": page_dir,
        "final_composite": final_ref,
        "typography_qa_json": qa_json_path,
        "typography_qa_markdown": qa_md_path,
        "svg_manifest": svg_manifest.get("manifest_path"),
        "svg_path": svg_manifest.get("svg_path"),
        "psd_handoff_manifest": psd_manifest.get("manifest_path"),
        "summary": summary,
        "warnings": warnings,
        "next_actions": sorted({s.get("action", "") for p in report.get("pages", []) for b in p.get("blocks", []) for s in b.get("suggestions", []) if s.get("action")}),
    }
    html_index_path = osp.join(page_dir, "lettering_proof_index.html")
    manifest["html_index"] = html_index_path
    _write_html_index(html_index_path, manifest, report, page_dir)
    manifest_path = osp.join(page_dir, "lettering_proof_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest
