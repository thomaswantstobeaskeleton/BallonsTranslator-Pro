import io
import os
import os.path as osp
import zipfile
from typing import List
from urllib.parse import quote
from urllib.request import Request, urlopen


def install_google_font_family(family: str, fonts_dir: str, timeout: int = 30) -> List[str]:
    """Download a Google Fonts family ZIP and extract installable font files.

    Returns extracted font file paths.
    """
    fam = (family or "").strip()
    if not fam:
        raise ValueError("Font family is empty.")
    os.makedirs(fonts_dir, exist_ok=True)

    url = f"https://fonts.google.com/download?family={quote(fam)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    extracted: List[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            low = name.lower()
            if not (low.endswith(".ttf") or low.endswith(".otf")):
                continue
            out_name = osp.basename(name)
            out_path = osp.join(fonts_dir, out_name)
            with zf.open(name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted.append(out_path)

    if not extracted:
        raise RuntimeError("No TTF/OTF files found in downloaded Google Fonts package.")
    return extracted
