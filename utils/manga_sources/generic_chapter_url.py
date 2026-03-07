"""
Generic chapter-URL downloader (manhua-translator-style).
Fetches a chapter page HTML, extracts image URLs, downloads with retry/backoff,
and writes manifest.json compatible with manhua-translator.
Optional: use Playwright for JS-heavy/Cloudflare sites (pip install playwright; playwright install chromium).
"""
from __future__ import annotations

import ast
import json
import os
import os.path as osp
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from utils.logger import logger as LOGGER

USER_AGENT = "BallonsTranslator/1.0 (https://github.com/dmMaze/BallonsTranslator)"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
CLOUDFLARE_MARKERS = (
    "cf-browser-verification", "challenge-platform", "cloudflare ray id",
    "attention required", "just a moment",
)


def ensure_playwright_chromium_installed() -> tuple:
    """
    Ensure Playwright package is importable and Chromium browser is installed.
    Runs 'playwright install chromium' if needed. Returns (success, message).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Playwright not installed. Run: pip install playwright"
    try:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, "Chromium installed successfully."
        err = (result.stderr or result.stdout or "").strip() or "Exit code %s" % result.returncode
        return False, err
    except subprocess.TimeoutExpired:
        return False, "Installation timed out."
    except Exception as e:
        return False, str(e)


def _looks_like_challenge(html: str) -> bool:
    if not html:
        return False
    c = html.lower()
    return any(m in c for m in CLOUDFLARE_MARKERS)


def _extract_script_image_urls(html: str, base_url: str) -> List[str]:
    """Extract image URLs from JS arrays in reader pages (manhua-translator generic_playwright patterns)."""
    patterns = (
        r"chapter_preloaded_images\s*=\s*(\[[^\]]+\])",
        r"chapter_images\s*=\s*(\[[^\]]+\])",
        r"images\s*:\s*(\[[^\]]+\])",
    )
    out: List[str] = []
    for pattern in patterns:
        m = re.search(pattern, html, flags=re.S)
        if not m:
            continue
        raw = m.group(1).strip().rstrip(";")
        data = None
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(raw)
                break
            except Exception:
                pass
        if data is None:
            try:
                data = json.loads(raw.replace("'", '"'))
            except Exception:
                continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item and not item.startswith(("data:", "about:")):
                    out.append(urljoin(base_url, item))
    return out


def _throttle(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)


def _absolute_url(base: str, href: str) -> str:
    if not href or not href.strip():
        return ""
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    return urljoin(base, href)


def _is_image_url(url: str) -> bool:
    if not url or "?" in url:
        u = url.split("?")[0]
    else:
        u = url
    ext = Path(u).suffix.lower()
    return ext in IMAGE_EXTENSIONS or any(
        x in (url or "").lower() for x in ("/image", "img", ".jpg", ".png", ".webp")
    )


def _extract_image_urls_from_soup(soup: BeautifulSoup, page_url: str) -> List[str]:
    """Collect image URLs from common manga reader patterns."""
    seen = set()
    out: List[str] = []

    def add(u: str) -> None:
        u = _absolute_url(page_url, u)
        if not u or not _is_image_url(u) or u in seen:
            return
        seen.add(u)
        out.append(u)

    # img src
    for tag in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-srcset"):
            val = tag.get(attr)
            if not val:
                continue
            # data-srcset can be "url 1x, url2 2x"
            for part in val.split(","):
                part = part.strip().split()[0] if part.strip() else part.strip()
                if part:
                    add(part)

    # Links to images (e.g. <a href=".../page/1.jpg">)
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if _is_image_url(href):
            add(href)

    # Common reader containers: [data-src], [data-url], .chapter-img, etc.
    for tag in soup.find_all(attrs={"data-src": True}):
        add(tag["data-src"])
    for tag in soup.find_all(attrs={"data-url": True}):
        add(tag["data-url"])
    for tag in soup.find_all(class_=re.compile(r"chapter|page|reader|img", re.I)):
        src = tag.get("src") or tag.get("data-src")
        if src:
            add(src)

    return out


def fetch_html_playwright(
    url: str,
    wait_selector: Optional[str] = None,
    timeout_ms: int = 20000,
    headless: bool = True,
) -> Optional[str]:
    """
    Fetch a URL with Playwright and return the page HTML. Use for search/catalog pages
    when the site requires JavaScript or blocks plain HTTP. Returns None if Playwright
    not installed or fetch fails. When headless=True (default), the browser runs in the
    background with no visible window.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        LOGGER.warning(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        )
        return None
    if not url or not url.strip():
        return None
    url = url.strip()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=10000)
                except Exception:
                    pass
            page.wait_for_timeout(1500)
            html = page.content()
            context.close()
            browser.close()
        return html
    except Exception as e:
        LOGGER.warning("Playwright fetch %s failed: %s", url[:50], e)
        return None


def get_chapter_images_playwright(
    chapter_url: str,
    timeout_ms: int = 30000,
    scroll_steps: int = 6,
    scroll_wait_ms: int = 500,
    headless: bool = True,
) -> List[str]:
    """
    Fetch chapter page with Playwright (Chromium). When headless=True (default), runs in
    the background with no visible window. Scrolls to trigger lazy load, extracts image URLs.
    For JS-heavy/Cloudflare sites. Returns [] if Playwright not installed or fetch fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        LOGGER.warning(
            "Playwright not installed. For JS-heavy sites run: pip install playwright && playwright install chromium"
        )
        return []
    if not chapter_url or not chapter_url.strip():
        return []
    chapter_url = chapter_url.strip()
    out: List[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled"])
            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(chapter_url, wait_until="domcontentloaded", timeout=timeout_ms)
            if _looks_like_challenge(page.content()):
                LOGGER.warning("Page may be a Cloudflare challenge. Try using browser cookies or another source.")
            page.wait_for_selector("img", timeout=8000)
            for _ in range(scroll_steps):
                page.evaluate("window.scrollBy(0, 900)")
                page.wait_for_timeout(scroll_wait_ms)
            html = page.content()
            context.close()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
        out = _extract_image_urls_from_soup(soup, chapter_url)
        seen = set(out)
        for u in _extract_script_image_urls(html, chapter_url):
            if u and u not in seen and _is_image_url(u):
                seen.add(u)
                out.append(u)
    except Exception as e:
        LOGGER.warning("Playwright chapter fetch failed: %s", e)
    return out


class GenericChapterUrlClient:
    """
    Fetch a chapter page by URL, extract image URLs from HTML, download with retry/backoff,
    and write manifest.json (manhua-translator compatible).
    """

    def __init__(
        self,
        timeout: int = 25,
        request_delay: float = 0.5,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        self.request_delay = max(0.0, float(request_delay))
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.backoff_factor = float(backoff_factor)

    def _throttle(self) -> None:
        _throttle(self.request_delay)

    def _get(self, url: str, referer: Optional[str] = None) -> requests.Response:
        self._throttle()
        headers = {}
        if referer:
            headers["Referer"] = referer
        return self.session.get(url, timeout=self.timeout, headers=headers or None)

    def get_chapter_images(self, chapter_url: str, use_playwright: bool = False, headless: bool = True) -> List[str]:
        """
        Fetch chapter page HTML and return list of image URLs in reading order.
        If use_playwright=True, use browser (headless=headless for JS-heavy sites). Returns [] if fetch or parse fails.
        """
        if not chapter_url or not chapter_url.strip():
            return []
        chapter_url = chapter_url.strip()
        if use_playwright:
            urls = get_chapter_images_playwright(chapter_url, headless=headless)
            if urls:
                return urls
            LOGGER.warning("Playwright returned no images; falling back to HTTP.")
        for attempt in range(self.max_retries + 1):
            try:
                r = self._get(chapter_url)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                urls = _extract_image_urls_from_soup(soup, chapter_url)
                seen = set(urls)
                for u in _extract_script_image_urls(r.text, chapter_url):
                    if u and u not in seen and _is_image_url(u):
                        seen.add(u)
                        urls.append(u)
                if urls:
                    return urls
                # Maybe no images in first request (lazy load); try once more
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                return []
            except requests.RequestException as e:
                LOGGER.warning("Generic chapter fetch attempt %s failed: %s", attempt + 1, e)
                if attempt >= self.max_retries:
                    return []
                time.sleep(self.backoff_base * (self.backoff_factor ** attempt))
            except Exception as e:
                LOGGER.warning("Generic chapter parse failed: %s", e)
                return []
        return []

    def download_chapter(
        self,
        chapter_url: str,
        save_dir: str,
        manga_id: str = "generic",
        chapter_id: str = "chapter",
        on_progress: Optional[Any] = None,
        use_playwright: bool = False,
        headless: bool = True,
    ) -> Optional[str]:
        """
        Get image URLs from chapter_url (HTTP or Playwright if use_playwright), download to save_dir as 001.ext, ...,
        and write manifest.json (manhua-translator compatible). When using Playwright, headless=True runs browser in background.
        Returns save_dir on success.
        """
        urls = self.get_chapter_images(chapter_url, use_playwright=use_playwright, headless=headless)
        if not urls:
            LOGGER.warning("No images found at %s", chapter_url)
            return None
        os.makedirs(save_dir, exist_ok=True)
        pages: List[dict] = []
        total = len(urls)
        for i, url in enumerate(urls):
            page_index = i + 1
            ext = Path(urlparse(url).path).suffix or ".png"
            if not ext.lower().lstrip(".") in {"jpg", "jpeg", "png", "webp", "gif", "bmp"}:
                ext = ".png"
            page_name = f"{page_index:03d}{ext}"
            path = osp.join(save_dir, page_name)
            ok = False
            err: Optional[str] = None
            for attempt in range(self.max_retries + 1):
                try:
                    r = self._get(url, referer=chapter_url)
                    if r.status_code in (403, 429):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    with open(path, "wb") as f:
                        f.write(r.content)
                    ok = True
                    break
                except Exception as e:
                    err = str(e)
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_base * (self.backoff_factor ** attempt))
            pages.append({
                "index": page_index,
                "url": url,
                "path": path,
                "ok": ok,
                "error": err,
            })
            if on_progress:
                on_progress(page_index, total, page_name)
            if not ok:
                LOGGER.warning("Failed to download page %s: %s", page_index, err)
        manifest_path = osp.join(save_dir, "manifest.json")
        payload = {
            "manga_id": manga_id,
            "chapter_id": chapter_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        success_count = sum(1 for p in pages if p.get("ok"))
        if success_count == 0:
            return None
        return save_dir
