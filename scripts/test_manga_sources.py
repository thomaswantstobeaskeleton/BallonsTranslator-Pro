#!/usr/bin/env python3
"""
Quick tests for manga source clients (MangaNato, MangaFire, MangaForFree, ToonGod, NaruRaw, ManhwaRaw, 1kkk).
Run from repo root: python scripts/test_manga_sources.py
Uses HTTP only (no Playwright) for speed; enable use_playwright=True if a site blocks.
"""
from __future__ import annotations

import sys
import os

# Run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.manga_sources import (
    MangaNatoClient,
    MangaFireClient,
    MangaForFreeClient,
    ToonGodClient,
    NaruRawClient,
    ManhwaRawClient,
    OneKkkClient,
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


def main():
    print("Manga source client tests (HTTP only by default)")
    results = {}

    # MangaNato
    client_nato = MangaNatoClient(base_url="https://manganato.com", timeout=15, request_delay=0.5)
    results["MangaNato search"] = test_search(client_nato, "MangaNato")
    if results["MangaNato search"]:
        search_res = client_nato.search("one piece", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["MangaNato feed"] = test_feed(client_nato, "MangaNato", search_res[0]["id"])
        else:
            results["MangaNato feed"] = False
    else:
        results["MangaNato feed"] = test_feed(client_nato, "MangaNato", "one-piece", use_playwright=False)

    # MangaFire
    client_fire = MangaFireClient(base_url="https://mangafire.to", timeout=15, request_delay=0.5)
    results["MangaFire search"] = test_search(client_fire, "MangaFire")
    if results["MangaFire search"]:
        search_res = client_fire.search("one piece", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["MangaFire feed"] = test_feed(client_fire, "MangaFire", search_res[0]["id"])
        else:
            results["MangaFire feed"] = False
    else:
        results["MangaFire feed"] = test_feed(client_fire, "MangaFire", "one-piecee.dkw", use_playwright=False)

    # MangaForFree
    client_mff = MangaForFreeClient(base_url="https://mangaforfree.com", timeout=15, request_delay=0.5)
    results["MangaForFree search"] = test_search(client_mff, "MangaForFree")
    if results["MangaForFree search"]:
        search_res = client_mff.search("one piece", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["MangaForFree feed"] = test_feed(client_mff, "MangaForFree", search_res[0]["id"])
        else:
            results["MangaForFree feed"] = False
    else:
        results["MangaForFree feed"] = False

    # ToonGod
    client_tg = ToonGodClient(base_url="https://toongod.org", timeout=15, request_delay=0.5)
    results["ToonGod search"] = test_search(client_tg, "ToonGod")
    if results["ToonGod search"]:
        search_res = client_tg.search("solo", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["ToonGod feed"] = test_feed(client_tg, "ToonGod", search_res[0]["id"])
        else:
            results["ToonGod feed"] = False
    else:
        results["ToonGod feed"] = False

    # NaruRaw (Japanese raw)
    client_naru = NaruRawClient(base_url="https://naruraw.net", timeout=15, request_delay=0.5)
    results["NaruRaw search"] = test_search(client_naru, "NaruRaw", "magic", use_playwright=False)
    if results["NaruRaw search"]:
        search_res = client_naru.search("magic", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["NaruRaw feed"] = test_feed(client_naru, "NaruRaw", search_res[0]["id"], use_playwright=False)
        else:
            results["NaruRaw feed"] = False
    else:
        results["NaruRaw feed"] = False

    # ManhwaRaw (Korean raw)
    client_mr = ManhwaRawClient(base_url="https://manhwaraw.club", timeout=15, request_delay=0.5)
    results["ManhwaRaw search"] = test_search(client_mr, "ManhwaRaw", "manager", use_playwright=False)
    if results["ManhwaRaw search"]:
        search_res = client_mr.search("manager", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["ManhwaRaw feed"] = test_feed(client_mr, "ManhwaRaw", search_res[0]["id"], use_playwright=False)
        else:
            results["ManhwaRaw feed"] = False
    else:
        results["ManhwaRaw feed"] = test_feed(client_mr, "ManhwaRaw", "manager-kim", use_playwright=False)

    # 1kkk (Chinese manhua) — use ASCII keyword to avoid Windows console encoding issues
    client_1kkk = OneKkkClient(base_url="https://www.1kkk.com", timeout=15, request_delay=0.5)
    results["1kkk search"] = test_search(client_1kkk, "1kkk", "manhua", use_playwright=False)
    if results["1kkk search"]:
        search_res = client_1kkk.search("manhua", limit=1, use_playwright=False, headless=True)
        if search_res:
            results["1kkk feed"] = test_feed(client_1kkk, "1kkk", search_res[0]["id"], use_playwright=False)
        else:
            results["1kkk feed"] = False
    else:
        results["1kkk feed"] = test_feed(client_1kkk, "1kkk", "manhua33991", use_playwright=False)

    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    failed = sum(1 for v in results.values() if not v)
    if failed > 0:
        print(f"\n{failed} test(s) failed. Some sites may require 'Use browser (Playwright)' in the app.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
