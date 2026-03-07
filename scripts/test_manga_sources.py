#!/usr/bin/env python3
"""
Quick tests for manga source clients (Mangakakalot, MangaForFree, ToonGod, NaruRaw, ManhwaRaw, 1kkk).
Run from repo root: python scripts/test_manga_sources.py
Uses HTTP only (no Playwright) for speed; enable use_playwright=True if a site blocks.
"""
from __future__ import annotations

import sys
import os

# Run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.manga_sources import (
    MangakakalotClient,
    MangaForFreeClient,
    ToonGodClient,
    NaruRawClient,
    ManhwaRawClient,
    OneKkkClient,
    GenericChapterUrlClient,
)


def _safe(s: str, max_len: int = 60) -> str:
    """Replace non-ASCII for Windows console."""
    if not s:
        return ""
    out = str(s).encode("ascii", "replace").decode("ascii")
    return (out[:max_len] + "..") if len(out) > max_len else out


def test_search(client, name: str, keyword: str = "one piece", use_playwright: bool = False):
    print(f"\n--- {name} search (keyword={_safe(keyword) or repr(keyword)}, playwright={use_playwright}) ---")
    try:
        results = client.search(keyword, limit=5, use_playwright=use_playwright, headless=True)
        print(f"  Results: {len(results)}")
        for r in results[:3]:
            print(f"    - {_safe(r.get('title', '?'))}  id={_safe(r.get('id', '?'))}")
        if not results:
            print("  (no results; site may require Playwright or block requests)")
        return bool(results)
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_feed(client, name: str, manga_id: str, use_playwright: bool = False):
    print(f"\n--- {name} feed (manga_id={_safe(manga_id) or repr(manga_id)}, playwright={use_playwright}) ---")
    try:
        chapters = client.get_feed(manga_id, limit=5, use_playwright=use_playwright, headless=True)
        print(f"  Chapters: {len(chapters)}")
        for ch in chapters[:3]:
            url_preview = (ch.get("id", "") or "")[:60]
            print(f"    - {_safe(ch.get('display', '?'))}  url={_safe(url_preview)}...")
        if not chapters:
            print("  (no chapters)")
        return bool(chapters)
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_download_generic(name: str, chapter_url: str, use_playwright: bool = False) -> bool:
    """Test that we can get chapter images (download path works). Uses GenericChapterUrlClient."""
    print(f"\n--- {name} download test (get chapter images, playwright={use_playwright}) ---")
    if not chapter_url or not chapter_url.startswith("http"):
        print("  Skip: no chapter URL")
        return True
    try:
        client = GenericChapterUrlClient(timeout=20, request_delay=0.3)
        urls = client.get_chapter_images(chapter_url, use_playwright=use_playwright, headless=True)
        n = len(urls) if urls else 0
        print(f"  Image URLs found: {n}")
        if n >= 1:
            print("  OK (download would succeed)")
            return True
        print("  No images; site may require Playwright or block requests.")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("Manga source client tests (HTTP only by default)")
    results = {}

    # Mangakakalot
    client_kakalot = MangakakalotClient(base_url="https://www.mangakakalot.gg", timeout=15, request_delay=0.5)
    results["Mangakakalot search"] = test_search(client_kakalot, "Mangakakalot")
    if results["Mangakakalot search"]:
        search_res = client_kakalot.search("one piece", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["Mangakakalot feed"] = test_feed(client_kakalot, "Mangakakalot", search_res[0]["id"])
            if results["Mangakakalot feed"]:
                chs = client_kakalot.get_feed(search_res[0]["id"], limit=1, use_playwright=False, headless=True)
                if chs and chs[0].get("id"):
                    results["Mangakakalot download"] = test_download_generic("Mangakakalot", chs[0]["id"])
                else:
                    results["Mangakakalot download"] = True
            else:
                results["Mangakakalot download"] = True
        else:
            results["Mangakakalot feed"] = False
            results["Mangakakalot download"] = True
    else:
        results["Mangakakalot feed"] = test_feed(client_kakalot, "Mangakakalot", "one-piece", use_playwright=False)
        results["Mangakakalot download"] = True

    # MangaForFree
    client_mff = MangaForFreeClient(base_url="https://mangaforfree.com", timeout=15, request_delay=0.5)
    results["MangaForFree search"] = test_search(client_mff, "MangaForFree")
    if results["MangaForFree search"]:
        search_res = client_mff.search("one piece", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["MangaForFree feed"] = test_feed(client_mff, "MangaForFree", search_res[0]["id"])
            if results["MangaForFree feed"]:
                chs = client_mff.get_feed(search_res[0]["id"], limit=1, use_playwright=False, headless=True)
                if chs and chs[0].get("id"):
                    results["MangaForFree download"] = test_download_generic("MangaForFree", chs[0]["id"])
                else:
                    results["MangaForFree download"] = True
            else:
                results["MangaForFree download"] = True
        else:
            results["MangaForFree feed"] = False
            results["MangaForFree download"] = True
    else:
        results["MangaForFree feed"] = False
        results["MangaForFree download"] = True

    # ToonGod
    client_tg = ToonGodClient(base_url="https://toongod.org", timeout=15, request_delay=0.5)
    results["ToonGod search"] = test_search(client_tg, "ToonGod")
    if results["ToonGod search"]:
        search_res = client_tg.search("solo", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["ToonGod feed"] = test_feed(client_tg, "ToonGod", search_res[0]["id"])
            if results["ToonGod feed"]:
                chs = client_tg.get_feed(search_res[0]["id"], limit=1, use_playwright=False, headless=True)
                if chs and chs[0].get("id"):
                    results["ToonGod download"] = test_download_generic("ToonGod", chs[0]["id"])
                else:
                    results["ToonGod download"] = True
            else:
                results["ToonGod download"] = True
        else:
            results["ToonGod feed"] = False
            results["ToonGod download"] = True
    else:
        results["ToonGod feed"] = False
        results["ToonGod download"] = True

    # NaruRaw (Japanese raw)
    client_naru = NaruRawClient(base_url="https://naruraw.net", timeout=15, request_delay=0.5)
    results["NaruRaw search"] = test_search(client_naru, "NaruRaw", "magic", use_playwright=False)
    if results["NaruRaw search"]:
        search_res = client_naru.search("magic", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["NaruRaw feed"] = test_feed(client_naru, "NaruRaw", search_res[0]["id"], use_playwright=False)
            if results["NaruRaw feed"]:
                chs = client_naru.get_feed(search_res[0]["id"], limit=1, use_playwright=False, headless=True)
                if chs and chs[0].get("id"):
                    results["NaruRaw download"] = test_download_generic("NaruRaw", chs[0]["id"])
                else:
                    results["NaruRaw download"] = True
            else:
                results["NaruRaw download"] = True
        else:
            results["NaruRaw feed"] = False
            results["NaruRaw download"] = True
    else:
        results["NaruRaw feed"] = False
        results["NaruRaw download"] = True

    # ManhwaRaw (Korean raw)
    client_mr = ManhwaRawClient(base_url="https://manhwaraw.club", timeout=15, request_delay=0.5)
    results["ManhwaRaw search"] = test_search(client_mr, "ManhwaRaw", "manager", use_playwright=False)
    if results["ManhwaRaw search"]:
        search_res = client_mr.search("manager", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["ManhwaRaw feed"] = test_feed(client_mr, "ManhwaRaw", search_res[0]["id"], use_playwright=False)
            if results["ManhwaRaw feed"]:
                chs = client_mr.get_feed(search_res[0]["id"], limit=1, use_playwright=False, headless=True)
                if chs and chs[0].get("id"):
                    results["ManhwaRaw download"] = test_download_generic("ManhwaRaw", chs[0]["id"])
                else:
                    results["ManhwaRaw download"] = True
            else:
                results["ManhwaRaw download"] = True
        else:
            results["ManhwaRaw feed"] = False
            results["ManhwaRaw download"] = True
    else:
        results["ManhwaRaw feed"] = test_feed(client_mr, "ManhwaRaw", "manager-kim", use_playwright=False)
        if results["ManhwaRaw feed"]:
            chs = client_mr.get_feed("manager-kim", limit=1, use_playwright=False, headless=True)
            results["ManhwaRaw download"] = test_download_generic("ManhwaRaw", chs[0]["id"]) if (chs and chs[0].get("id")) else True
        else:
            results["ManhwaRaw download"] = True

    # 1kkk (Chinese manhua) — use ASCII keyword to avoid Windows console encoding issues
    client_1kkk = OneKkkClient(base_url="https://www.1kkk.com", timeout=15, request_delay=0.5)
    results["1kkk search"] = test_search(client_1kkk, "1kkk", "manhua", use_playwright=False)
    if results["1kkk search"]:
        search_res = client_1kkk.search("manhua", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["1kkk feed"] = test_feed(client_1kkk, "1kkk", search_res[0]["id"], use_playwright=False)
            if results["1kkk feed"]:
                chs = client_1kkk.get_feed(search_res[0]["id"], limit=1, use_playwright=False, headless=True)
                if chs and chs[0].get("id"):
                    results["1kkk download"] = test_download_generic("1kkk", chs[0]["id"])
                else:
                    results["1kkk download"] = True
            else:
                results["1kkk download"] = True
        else:
            results["1kkk feed"] = False
            results["1kkk download"] = True
    else:
        results["1kkk feed"] = test_feed(client_1kkk, "1kkk", "manhua33991", use_playwright=False)
        if results["1kkk feed"]:
            chs = client_1kkk.get_feed("manhua33991", limit=1, use_playwright=False, headless=True)
            results["1kkk download"] = test_download_generic("1kkk", chs[0]["id"]) if (chs and chs[0].get("id")) else True
        else:
            results["1kkk download"] = True

    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    failed = sum(1 for v in results.values() if not v)
    if failed > 0:
        print(f"\n{failed} test(s) failed. Some sites may require 'Use browser (Playwright)' in the app.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
