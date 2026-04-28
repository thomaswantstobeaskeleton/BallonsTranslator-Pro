# BallonsTranslator-Pro

Community fork of [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) focused on comic translation workflows with broader detector/OCR/inpainting choices and practical batch tools.

- 中文文档（与本页同结构）: [README_zh_CN.md](README_zh_CN.md)
- Full change history: [docs/CHANGELOG.md](docs/CHANGELOG.md)

**What’s new (short):** v1.7.0 refreshed auto-layout behavior and first-run model downloads (non-blocking startup). Details: [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Overview

BallonsTranslator-Pro keeps the upstream flow (detect → OCR → translate → inpaint) while adding larger optional module coverage, model management helpers, and manga/comic source utilities.

See also:
- Troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Model references: [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- Quality tiers: [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

## Install

1. Clone repository:
   ```bash
   git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro.git
   cd BallonsTranslator-Pro
   ```
2. Start directly (auto-installs base requirements on first run):
   ```bash
   python launch.py
   ```
3. Optional setup scripts:
   - Windows: `setup.bat`
   - Linux/macOS: `./setup.sh`

## First Run

### Quick start (5 steps) / 快速开始（5 步）

1. **Run app / 启动应用**: `python launch.py`
2. **Pick model package / 选择模型包**: choose Core only first for fastest startup.
3. **Open pages / 打开页面**: File → Open Folder (or Open Images).
4. **Select modules / 选择模块**: Config panel → Detector/OCR/Inpaint/Translator.
5. **Run pipeline / 运行流程**: click **Run** in the bottom bar.

### First-run model picker (explanation) / 首次运行模型选择器说明

- Trigger condition / 触发条件: shown when `config/config.json` does not exist (fresh install).
- Behavior / 行为:
  - Main window opens first.
  - Selected model packages download in background.
  - If download fails, use **Tools → Models → Retry model downloads**.
- Paths / 路径:
  - Config template: `config/config.example.json`
  - Runtime config: `config/config.json`
  - Downloaded models: `data/models/`
  - HF cache (optional cleanup): `.btrans_cache/hub`
- Screenshot references / 截图路径:
  - UI config panel (example): `doc/src/configpanel.png`
  - Add your local first-run picker screenshot as: `docs/images/first_run_model_picker.png`

## Model Packages

Use **Tools → Manage models** to add/remove optional packages after first launch.

Recommended baseline:
- Detector: `ctd`
- OCR: `manga_ocr`
- Inpaint: `aot`
- Translator: `google`

For best-practice module combinations, check:
- [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md)
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

## Startup Scripts

- `python launch.py`: main entrypoint.
- `setup.bat` / `setup.sh`: one-time environment bootstrap.
- `Launch BallonsTranslator.bat` (Windows): run with repo venv Python.

Tip: keep `data/` and `config/` beside `launch.py` for portable moves.

## Core Workflows

1. **Project workflow**: Open folder/images → Detect/OCR/Translate/Inpaint → save project.
2. **Batch workflow**: Tools → Batch queue for multi-folder processing.
3. **Source workflow**: Tools → Manga/Comic source to search/download chapters.
4. **Export workflow**: Tools → Export all pages (including PDF export options).

Related docs:
- Translation context/glossary: [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- InDesign LPtxt export: [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md)

## Troubleshooting

Primary guide: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

Common areas:
- GPU/VRAM limits and fallback strategies
- Optional dependency conflicts
- HuggingFace/network download failures
- OCR/box alignment tuning

## Contribution

- Read: [CONTRIBUTING.md](CONTRIBUTING.md)
- Open issues with reproducible info (OS, GPU/CPU, module names, logs, sample page).
- Keep changes small and focused.
