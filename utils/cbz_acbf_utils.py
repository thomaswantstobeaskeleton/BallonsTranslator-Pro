"""
CBZ / ACBF support (#1141).
- CBZ: comic book zip (images + optional metadata). Extract to folder and open as project.
- ACBF: Advanced Comic Book Format (XML with page order, optional text layers). When opening
  a .cbz that contains an .acbf file, page order from ACBF can be used to sort images.
"""
from __future__ import annotations

import os
import zipfile
import tempfile
import re
import os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

from .logger import logger as LOGGER

CBZ_EXT = (".cbz", ".zip")
ACBF_EXT = (".acbf", ".xml")
# ACBF 1.x uses BookInfo or similar root; page list in body/pages
ACBF_PAGE_TAG = re.compile(r"^(?:{.*})?(?:page|Page)$", re.I)


def extract_cbz_to_folder(cbz_path: str, out_dir: Optional[str] = None) -> str:
    """
    Extract a CBZ (zip) to out_dir. If out_dir is None, use a temp directory.
    Returns the path to the extracted folder.
    """
    cbz_path = osp.abspath(cbz_path)
    if not osp.isfile(cbz_path):
        raise FileNotFoundError(cbz_path)
    if out_dir is None:
        base = Path(tempfile.gettempdir()) / "ballonstranslator_cbz"
        base.mkdir(parents=True, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix=Path(cbz_path).stem + "_", dir=str(base))
    else:
        out_dir = osp.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(cbz_path, "r") as zf:
        zf.extractall(out_dir)
    LOGGER.info("Extracted CBZ to %s", out_dir)
    return out_dir


def get_page_order_from_acbf(acbf_xml_path: str) -> Optional[List[str]]:
    """
    Parse an ACBF XML file and return image filenames in reading order, if present.
    Returns None if parsing fails or no page order is found.
    """
    if not osp.isfile(acbf_xml_path):
        return None
    try:
        tree = ET.parse(acbf_xml_path)
        root = tree.getroot()
        # ACBF: body/pages with page elements; image ref in page or linked data
        ns = {}
        if root.tag.startswith("{"):
            ns["acbf"] = root.tag[1 : root.tag.index("}")]
        pages_el = root.find(".//acbf:body/acbf:pages", ns) or root.find(".//body/pages", ns)
        if pages_el is None:
            pages_el = root.find(".//pages") or root.find(".//Pages")
        if pages_el is None:
            return None
        ordered: List[str] = []
        for page in pages_el:
            tag = page.tag.split("}")[-1] if "}" in page.tag else page.tag
            if not ACBF_PAGE_TAG.match(tag):
                continue
            # Image ref: often in image/@src or file-name or similar
            img_ref = (
                page.get("image")
                or page.get("src")
                or (page.find("image") is not None and page.find("image").get("src"))
            )
            if img_ref:
                name = osp.basename(img_ref)
                if name:
                    ordered.append(name)
            for child in page:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag in ("image", "Image"):
                    src = child.get("src") or child.get("href") or child.text
                    if src:
                        ordered.append(osp.basename(src.strip()))
                    break
        return ordered if ordered else None
    except Exception as e:
        LOGGER.debug("ACBF parse failed for %s: %s", acbf_xml_path, e)
        return None


def find_acbf_in_folder(folder: str) -> Optional[str]:
    """Return path to first .acbf or BookInfo.xml in folder (or subdirs), or None."""
    folder = osp.abspath(folder)
    for name in os.listdir(folder):
        if name.lower().endswith(".acbf") or name.lower() in ("bookinfo.xml", "comicinfo.xml"):
            return osp.join(folder, name)
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".acbf") or f.lower() in ("bookinfo.xml", "comicinfo.xml"):
                return osp.join(root, f)
    return None


def open_cbz_as_project_path(cbz_path: str, out_dir: Optional[str] = None) -> str:
    """
    Extract CBZ to folder and return the folder path suitable for opening as a project.
    If the extracted folder contains an ACBF file, page order is not applied here (caller
    may sort project pages by get_page_order_from_acbf if desired).
    """
    return extract_cbz_to_folder(cbz_path, out_dir)
