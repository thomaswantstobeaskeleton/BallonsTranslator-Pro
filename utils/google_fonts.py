import io
import os
import os.path as osp
import zipfile
from typing import List
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class GoogleFontInstallError(RuntimeError):
    """Raised when a Google Fonts family cannot be downloaded or extracted."""


def install_google_font_family(family: str, fonts_dir: str, timeout: int = 30) -> List[str]:
    """Download a Google Fonts family ZIP and extract installable font files.

    Returns extracted font file paths.
    """
    fam = (family or "").strip()
    if not fam:
        raise ValueError("Font family is empty.")
    try:
        os.makedirs(fonts_dir, exist_ok=True)
    except OSError as exc:
        raise GoogleFontInstallError(f"Cannot create font install folder '{fonts_dir}': {exc}") from exc

    url = f"https://fonts.google.com/download?family={quote(fam)}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            hint = "Check that the family name exactly matches Google Fonts."
        else:
            hint = "Google Fonts returned an HTTP error."
        raise GoogleFontInstallError(f"Could not download '{fam}' ({exc.code}). {hint}") from exc
    except URLError as exc:
        raise GoogleFontInstallError(f"Network error while downloading '{fam}': {exc.reason}") from exc
    except TimeoutError as exc:
        raise GoogleFontInstallError(f"Timed out downloading '{fam}' after {timeout} seconds.") from exc
    except Exception as exc:
        raise GoogleFontInstallError(f"Could not download '{fam}': {exc}") from exc

    extracted: List[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                low = name.lower()
                if not (low.endswith(".ttf") or low.endswith(".otf")):
                    continue
                out_name = osp.basename(name)
                if not out_name:
                    continue
                out_path = osp.join(fonts_dir, out_name)
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(out_path)
    except zipfile.BadZipFile as exc:
        raise GoogleFontInstallError(
            f"Google Fonts did not return a valid font ZIP for '{fam}'. Check the family name or try again later."
        ) from exc
    except OSError as exc:
        raise GoogleFontInstallError(f"Could not write font files to '{fonts_dir}': {exc}") from exc

    if not extracted:
        raise GoogleFontInstallError(
            f"No TTF/OTF files found in the Google Fonts package for '{fam}'. The family may be unavailable for download."
        )
    return extracted
