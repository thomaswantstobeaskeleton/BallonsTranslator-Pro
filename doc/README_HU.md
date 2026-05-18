# BallonsTranslator Pro

> Ez a fájl a `README.md` és `README_zh_CN.md` tartalmához lett igazítva (frissítve: 2026-05-18).

## Áttekintés

A BallonsTranslator Pro képregények/képek fordítására szolgál OCR → fordítás → szövegkorrekció → renderelés folyamattal.

## Fő funkciók

- Kötegelt feldolgozás manga/manhua/manhwa és általános képekhez.
- Helyi és felhős OCR motorok (PaddleOCR, MangaOCR, EasyOCR, LLM/VLM OCR stb.).
- Automatikus fordítás LLM és MT szolgáltatókkal.
- Buborék-, maszk-, stílus- és betűszerkesztés.
- Renderelés és export (SVG/felirat támogatással, ahol elérhető).
- Teljesítményhangolás (GPU/VRAM/opcionális modellek).

## Gyors telepítés

1. Klónozd a repót.
2. Telepítsd a függőségeket: `pip install -r requirements.txt`.
3. Indítás: `python launch.py`.

Windows szkriptek: `launch_win.bat`, `build_windows_installer.bat`.

## Dokumentáció

Lásd a `docs/` mappát fordítási, GPU, inpainting/renderelési, videó/felirat témákhoz.

## Támogatott felületi nyelvek

English (`en_US`), 简体中文 (`zh_CN`), Español (`es_MX`), Français (`fr_FR`), Português (Brasil) (`pt_BR`), Русский (`ru_RU`), 한국어 (`ko_KR`), Magyar (`hu_HU`).

## Közreműködés

- Lásd: `CONTRIBUTING.md`.
- Tartsd szinkronban a fordításokat a `README.md` és `README_zh_CN.md` frissítéseivel.
