from __future__ import annotations

from typing import Dict, Any, List, Tuple
import xml.etree.ElementTree as ET


SCHEMA = "ballonstranslator.xliff.v1"


def export_project_xliff(project) -> str:
    root = ET.Element("xliff", {"version": "1.2", "bt_schema": SCHEMA})
    for page_name, blks in (getattr(project, "pages", {}) or {}).items():
        file_el = ET.SubElement(root, "file", {"original": str(page_name), "source-language": "auto", "target-language": "auto", "datatype": "plaintext"})
        body = ET.SubElement(file_el, "body")
        for idx, blk in enumerate(blks or []):
            trans_id = f"{page_name}::{idx}"
            unit = ET.SubElement(body, "trans-unit", {"id": trans_id, "resname": str(idx)})
            ET.SubElement(unit, "source").text = (getattr(blk, "get_text", lambda: "")() or "")
            ET.SubElement(unit, "target").text = (getattr(blk, "translation", "") or "")
    return ET.tostring(root, encoding="unicode")


def import_project_xliff(project, xml_text: str) -> Tuple[bool, Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    pages = getattr(project, "pages", {}) or {}
    matched_pages: List[str] = []
    missing_pages: List[str] = []
    unmatched_pages: List[str] = []

    for file_el in root.findall("file"):
        page = str(file_el.attrib.get("original", "") or "")
        if page not in pages:
            missing_pages.append(page)
            continue
        matched_pages.append(page)
        units = file_el.findall("./body/trans-unit")
        blk_list = pages.get(page, []) or []
        if len(units) != len(blk_list):
            unmatched_pages.append(page)
        for i, unit in enumerate(units[: len(blk_list)]):
            tgt = unit.findtext("target", default="") or ""
            blk_list[i].translation = str(tgt)

    all_matched = not missing_pages and not unmatched_pages
    return all_matched, {
        "matched_pages": set(matched_pages),
        "missing_pages": missing_pages,
        "unmatched_pages": unmatched_pages,
        "unexpected_pages": [],
    }
