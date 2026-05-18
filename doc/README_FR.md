# BallonsTranslator Pro

> Cette version est alignée sur `README.md` et `README_zh_CN.md` (mise à jour le 18 mai 2026).

## Aperçu

BallonsTranslator Pro est un outil de traduction d’images/bandes dessinées avec pipeline OCR → traduction → correction du texte → rendu final.

## Fonctions principales

- Traitement par lots pour manga/manhua/manhwa et images générales.
- Moteurs OCR locaux/cloud (PaddleOCR, MangaOCR, EasyOCR, OCR LLM/VLM, etc.).
- Traduction automatique avec fournisseurs LLM et MT.
- Outils d’édition: bulles, masques, styles de texte, polices.
- Rendu et export (dont SVG/sous-titres pour les workflows compatibles).
- Contrôle des performances (GPU/VRAM/modèles optionnels).

## Installation rapide

1. Cloner le dépôt.
2. Installer les dépendances: `pip install -r requirements.txt`.
3. Lancer: `python launch.py`.

Scripts Windows disponibles: `launch_win.bat`, `build_windows_installer.bat`.

## Documentation

Voir `docs/` pour:

- Traductions: `docs/TRANSLATIONS.md`, `docs/TRANSLATIONS_zh_CN.md`
- Performance/GPU: `docs/GPU_ACCELERATION.md`
- Inpainting/rendu: `docs/INPAINTING_QUALITY_AND_SPEED.md`, `docs/RENDERING_TEXT_FORMATTING.md`
- Vidéo/sous-titres: `docs/VIDEO_SUBTITLE_FLOW_EXAMPLE.md` et documents liés

## Langues UI prises en charge

English (`en_US`), 简体中文 (`zh_CN`), Español (`es_MX`), Français (`fr_FR`), Português (Brasil) (`pt_BR`), Русский (`ru_RU`), 한국어 (`ko_KR`), Magyar (`hu_HU`).

## Contribution

- Voir `CONTRIBUTING.md`.
- Garder les traductions synchronisées avec `README.md` et `README_zh_CN.md`.
