# BallonsTranslator Pro

> Эта версия синхронизирована с `README.md` и `README_zh_CN.md` (обновлено: 18 мая 2026 г.).

## Обзор

BallonsTranslator Pro — инструмент перевода комиксов/изображений с конвейером OCR → перевод → правка текста → рендер.

## Основные возможности

- Пакетная обработка manga/manhua/manhwa и обычных изображений.
- Локальные и облачные OCR (PaddleOCR, MangaOCR, EasyOCR, LLM/VLM OCR и др.).
- Автоперевод через LLM/MT провайдеров.
- Редактирование пузырей, масок, стилей и шрифтов.
- Рендер и экспорт (включая SVG/субтитры в поддерживаемых сценариях).
- Настройки производительности (GPU/VRAM/доп. модели).

## Быстрый запуск

1. Клонируйте репозиторий.
2. Установите зависимости: `pip install -r requirements.txt`.
3. Запустите: `python launch.py`.

Для Windows есть `launch_win.bat` и `build_windows_installer.bat`.

## Документация

См. каталог `docs/`: переводы, GPU, inpainting/рендеринг, видео и субтитры.

## Поддерживаемые языки интерфейса

English (`en_US`), 简体中文 (`zh_CN`), Español (`es_MX`), Français (`fr_FR`), Português (Brasil) (`pt_BR`), Русский (`ru_RU`), 한국어 (`ko_KR`), Magyar (`hu_HU`).

## Вклад

- См. `CONTRIBUTING.md`.
- Поддерживайте синхронизацию переводов с `README.md` и `README_zh_CN.md`.
