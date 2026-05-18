# BallonsTranslator Pro

> Esta versão está alinhada com `README.md` e `README_zh_CN.md` (atualizada em 18 de maio de 2026).

## Visão geral

BallonsTranslator Pro é uma ferramenta para traduzir quadrinhos/imagens com pipeline OCR → tradução → revisão de texto → renderização.

## Principais recursos

- Fluxo em lote para manga/manhua/manhwa e imagens gerais.
- OCR local e em nuvem (PaddleOCR, MangaOCR, EasyOCR, OCR com LLM/VLM etc.).
- Tradução automática com provedores LLM e MT.
- Edição de balões, máscaras, estilos e tipografia.
- Renderização/exportação (incluindo SVG e legendas em fluxos compatíveis).
- Controles de desempenho (GPU/VRAM/modelos opcionais).

## Instalação rápida

1. Clone o repositório.
2. Instale dependências: `pip install -r requirements.txt`.
3. Execute: `python launch.py`.

No Windows: `launch_win.bat`, `build_windows_installer.bat`.

## Documentação

Consulte `docs/` para traduções, GPU, inpainting/renderização e fluxo de vídeo/legendas.

## Idiomas de interface suportados

English (`en_US`), 简体中文 (`zh_CN`), Español (`es_MX`), Français (`fr_FR`), Português (Brasil) (`pt_BR`), Русский (`ru_RU`), 한국어 (`ko_KR`), Magyar (`hu_HU`).

## Contribuição

- Veja `CONTRIBUTING.md`.
- Mantenha traduções sincronizadas com `README.md` e `README_zh_CN.md`.
