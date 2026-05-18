# BallonsTranslator Pro

> 이 문서는 `README.md` 및 `README_zh_CN.md`와 동기화되었습니다(2026-05-18 기준).

## 개요

BallonsTranslator Pro는 OCR → 번역 → 텍스트 보정 → 렌더링 파이프라인으로 만화/이미지를 번역하는 도구입니다.

## 주요 기능

- 만화/이미지 일괄 처리 워크플로우.
- 로컬/클라우드 OCR(PaddleOCR, MangaOCR, EasyOCR, LLM/VLM OCR 등).
- LLM/MT 기반 자동 번역.
- 말풍선/마스크/텍스트 스타일/폰트 편집.
- 렌더링 및 내보내기(SVG/자막 워크플로우 포함).
- 성능 제어(GPU/VRAM/선택 모델).

## 빠른 설치

1. 저장소 클론
2. `pip install -r requirements.txt`
3. `python launch.py`

Windows 스크립트: `launch_win.bat`, `build_windows_installer.bat`.

## 문서

`docs/` 폴더에서 번역, GPU, 인페인팅/렌더링, 비디오/자막 문서를 확인하세요.

## 지원 UI 언어

English (`en_US`), 简体中文 (`zh_CN`), Español (`es_MX`), Français (`fr_FR`), Português (Brasil) (`pt_BR`), Русский (`ru_RU`), 한국어 (`ko_KR`), Magyar (`hu_HU`).

## 기여

- `CONTRIBUTING.md` 참고
- `README.md`, `README_zh_CN.md` 변경 시 번역 문서 동기화
