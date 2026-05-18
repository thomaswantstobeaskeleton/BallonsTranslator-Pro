# BallonsTranslator Pro

> Esta versión está alineada con `README.md` y `README_zh_CN.md` (actualizado el 18 de mayo de 2026).

## Resumen

BallonsTranslator Pro es una herramienta para traducir cómics/imágenes con flujo OCR → traducción → corrección de texto → renderizado. Incluye múltiples motores OCR, traductores y opciones de inpainting/renderizado.

## Características principales

- Flujo completo por lotes para manga/manhua/manhwa e imágenes generales.
- Motores OCR locales y en la nube (PaddleOCR, MangaOCR, EasyOCR, LLM/VLM OCR, etc.).
- Traducción automática con proveedores LLM y MT.
- Herramientas de edición: detección de globos, limpieza de máscara, estilos y tipografía.
- Renderizado y exportación (incluyendo SVG/subtítulos en los flujos compatibles).
- Controles de rendimiento (GPU/VRAM/modelos opcionales).

## Instalación rápida

1. Clona el repositorio.
2. Instala dependencias con `pip install -r requirements.txt`.
3. Ejecuta `python launch.py`.

Para Windows también hay scripts: `launch_win.bat`, `build_windows_installer.bat`.

## Documentación

Consulta `docs/` para:

- Guía de traducciones: `docs/TRANSLATIONS.md`, `docs/TRANSLATIONS_zh_CN.md`
- Rendimiento/GPU: `docs/GPU_ACCELERATION.md`
- Inpainting/renderizado: `docs/INPAINTING_QUALITY_AND_SPEED.md`, `docs/RENDERING_TEXT_FORMATTING.md`
- Vídeo/subtítulos: `docs/VIDEO_SUBTITLE_FLOW_EXAMPLE.md` y archivos relacionados

## Idiomas de UI soportados

- English (`en_US`)
- 简体中文 (`zh_CN`)
- Español (`es_MX`)
- Français (`fr_FR`)
- Português (Brasil) (`pt_BR`)
- Русский (`ru_RU`)
- 한국어 (`ko_KR`)
- Magyar (`hu_HU`)

## Contribución

- Ver `CONTRIBUTING.md`.
- Mantén sincronizadas las traducciones cuando se actualicen `README.md` y `README_zh_CN.md`.
