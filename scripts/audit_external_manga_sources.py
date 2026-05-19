#!/usr/bin/env python3
"""Audit local external manga downloader repositories for provider candidates.

This tool reads metadata from user-supplied/local folders only. It does not
scrape manga pages, download copyrighted content, or require network access.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDERS = [
    ROOT / "external/keiyoushi/extensions-source",
    ROOT / "external/hakuneko",
    ROOT / "external/FMD2",
    ROOT / "external/mangal-scrapers",
    ROOT / "external/manga-py",
    ROOT / "external/gallery-dl",
    ROOT / "external/mangadex-downloader",
]
OUTPUT = ROOT / "docs/EXTERNAL_SOURCE_AUDIT.md"
LICENSE_FILES = ("LICENSE", "LICENSE.md", "COPYING", "COPYING.txt")
SOURCE_PATTERNS = ("*.kt", "*.js", "*.ts", "*.lua", "*.py", "*.pas", "*.pp")
DOMAIN_RE = re.compile(r"https?://([A-Za-z0-9.-]+)")


@dataclass
class Candidate:
    ecosystem: str
    name: str
    path: str
    domains: list[str]
    portability: str
    active_hint: str
    requires_browser_or_cookies: str
    overlaps_existing: str


def ecosystem_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".kt":
        return "Kotlin"
    if suffix in {".js", ".ts"}:
        return "JS"
    if suffix == ".lua":
        return "Lua"
    if suffix == ".py":
        return "Python"
    if suffix in {".pas", ".pp"}:
        return "Pascal"
    return "Unknown"


def likely_name(path: Path, text: str) -> str:
    for pattern in [r'class\s+([A-Za-z0-9_]+)', r'name\s*[:=]\s*["\']([^"\']+)', r'TITLE\s*=\s*["\']([^"\']+)']:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return path.stem


def read_license(folder: Path) -> str:
    for name in LICENSE_FILES:
        p = folder / name
        if p.exists():
            first = p.read_text(errors="ignore")[:2000].lower()
            if "gpl" in first:
                return "GPL-family or mentions GPL; avoid copying unless obligations are handled"
            if "apache" in first:
                return "Apache-style; still verify per-file headers"
            if "mit license" in first or "permission is hereby granted" in first:
                return "MIT-style; verify per-file headers"
            return f"License file present: {name}; manual review needed"
    return "No top-level license detected; manual review required"


def classify_portability(ecosystem: str, text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("cloudflare", "captcha", "login", "cookie")):
        return "Medium/low: may need user browser or cookies"
    if ecosystem in {"Python", "JS", "Lua"}:
        return "Medium/high: selectors or JSON parsing likely portable"
    if ecosystem == "Kotlin":
        return "Medium: Tachiyomi/Mihon concepts portable, code is not direct Python"
    if ecosystem == "Pascal":
        return "Medium/low: useful inventory, parser rewrite needed"
    return "Unknown"


def browser_cookie_hint(text: str) -> str:
    lower = text.lower()
    hits = [word for word in ("playwright", "puppeteer", "webview", "cloudflare", "captcha", "cookie", "login") if word in lower]
    return ", ".join(hits) if hits else "No obvious browser/cookie requirement"


def overlap(domains: Iterable[str]) -> str:
    existing = ("mangadex", "comick", "mangakakalot", "manganato", "manganelo", "mangasee", "readmanga", "1kkk")
    matched = sorted({d for d in domains if any(x in d.lower() for x in existing)})
    return ", ".join(matched) if matched else "No obvious overlap"


def audit_folder(folder: Path, max_files: int = 2500) -> tuple[str, list[Candidate]]:
    candidates: list[Candidate] = []
    files: list[Path] = []
    for pattern in SOURCE_PATTERNS:
        files.extend(folder.rglob(pattern))
    for path in files[:max_files]:
        if any(part in {".git", "node_modules", "build", "dist", "vendor"} for part in path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")[:20000]
        except Exception:
            continue
        domains = sorted(set(DOMAIN_RE.findall(text)))[:8]
        if not domains and not re.search(r"manga|comic|source|connector|extractor", text, re.I):
            continue
        eco = ecosystem_for(path)
        candidates.append(Candidate(
            ecosystem=eco,
            name=likely_name(path, text),
            path=str(path.relative_to(ROOT)),
            domains=domains,
            portability=classify_portability(eco, text),
            active_hint="Recentness requires git history review" if (folder / ".git").exists() else "No git metadata in local copy",
            requires_browser_or_cookies=browser_cookie_hint(text),
            overlaps_existing=overlap(domains),
        ))
    return read_license(folder), candidates


def write_report(results: list[tuple[Path, str, list[Candidate]]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# External Manga Source Audit", "", "Generated from local folders only; no network access or content scraping is performed.", ""]
    if not results:
        lines += ["No external folders were found or supplied.", "", "Place repositories under `external/` or pass paths on the command line, then rerun:", "", "```bash", "python scripts/audit_external_manga_sources.py external/hakuneko", "```", ""]
    for folder, license_summary, candidates in results:
        lines += [f"## {folder}", "", f"License: {license_summary}", "", "| Ecosystem | Name | Path | Domains | Portability | Browser/cookies | Existing overlap | Active hint |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in candidates[:200]:
            lines.append(f"| {c.ecosystem} | {c.name} | `{c.path}` | {', '.join(c.domains) or '-'} | {c.portability} | {c.requires_browser_or_cookies} | {c.overlaps_existing} | {c.active_hint} |")
        if not candidates:
            lines.append("| - | - | - | - | No candidates detected | - | - | - |")
        lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional local external repository folders")
    args = parser.parse_args()
    folders = args.paths or DEFAULT_FOLDERS
    existing = [p.resolve() for p in folders if p.exists() and p.is_dir()]
    if not existing:
        write_report([])
        print("No external source folders found. Expected optional folders:")
        for p in DEFAULT_FOLDERS:
            print(f"  - {p.relative_to(ROOT)}")
        print(f"Wrote instructions to {OUTPUT.relative_to(ROOT)}")
        return 0
    results = []
    for folder in existing:
        license_summary, candidates = audit_folder(folder)
        results.append((folder, license_summary, candidates))
    write_report(results)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} for {len(results)} folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
