# BallonsTranslator-Pro

BallonsTranslator-Pro is an advanced fork of [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator), keeping the same assisted manga/comic translation pipeline while adding production-focused module options and quality-of-life workflow improvements for power users.

<img src="https://raw.githubusercontent.com/dmMaze/BallonsTranslator/master/doc/src/ui0.jpg" align="center">

BallonsTranslator-Pro helps you translate manga/comic pages with an assisted workflow:

1. Detect speech balloons/text areas
2. Read text (OCR)
3. Translate text
4. Clean original text from artwork (inpaint)
5. Edit and export pages

This fork focuses on practical, real-world comic workflows and gives you multiple module options for detection, OCR, translation, and inpainting.

- 中文文档（与本页同结构）: [README_zh_CN.md](README_zh_CN.md)
- Full change history: [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## What this fork keeps from base (and what it extends)

From the base project, Pro preserves the core five-stage pipeline (detect → OCR → translate → inpaint → edit/export) and WYSIWYG editing workflow.

The Pro fork focuses on:
- broader translator/model choices (including chainable and LLM-based translators),
- practical configuration for real chapter workflows,
- documentation for tuning quality/speed tradeoffs.

## Who this is for

This README is for regular users who just want to run the app and translate pages.

If you want deep technical details, see:
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

---

## Quick Start (recommended)

## Option A: Run with Python (Windows / Linux / macOS)

1. Install Python 3.10+.
2. Clone or download this repository.
3. In the project folder, run:

```bash
python launch.py
```

On first launch, base requirements are installed automatically if needed.

## Option B: Windows helper scripts

- `setup.bat`: one-time setup
- `Launch BallonsTranslator.bat`: run with local repo virtual environment
- `launch_win.bat` / `launch_win_with_autoupdate.bat` / `launch_win_amd_nightly.bat`: for portable bundle layouts

If you are unsure, start with **`python launch.py`**.

---

## First run: what to expect

When no `config/config.json` exists (fresh install), the app may ask which model package(s) to download.

- You can start with **Core only** (fastest, simplest)
- Downloads run in the background
- If a download fails, use **Tools → Models → Retry model downloads**
- You can add/remove packages later via **Tools → Manage models**

Default baseline after successful setup:
- Detector: `ctd`
- OCR: `manga_ocr`
- Inpaint: `aot`
- Translator: `google`

---

## Basic workflow (5 steps)

1. **Open images**: File → Open Folder (or Open Images)
2. **Pick modules**: Config panel → Detector / OCR / Inpaint / Translator
3. **Run pipeline**: click **Run** in the bottom bar
4. **Review & edit text** as needed
5. **Export results**: Tools → Export all pages

That’s enough to complete most projects.

### New: 2-step translation validation (Feature #55)

You can now run a two-step translation flow with online MT + LLM review/correction:

1. Set **Translator** to `Chain`
2. In Chain settings:
   - `chain_translators`: `google,LLM_API_Translator` (or `DeepL,LLM_API_Translator`)
   - `chain_intermediate_langs`: `English`
   - leave `chain_llm_review_mode` enabled
3. Configure your `LLM_API_Translator` provider/model normally

When enabled, the final LLM step receives both original source text and the first-pass draft so it can validate and refine wording, tone, and consistency before final output.

---

## Recommended beginner setup

If you want a stable starting point:

- Detector: `ctd`
- OCR: `manga_ocr`
- Inpaint: `aot`
- Translator: `google`

You can switch modules later when you want better quality for difficult pages.

For manhua-focused tuning, see: [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md)

---

## Where files are stored

- Config template: `config/config.example.json`
- Active config: `config/config.json`
- Downloaded models: `data/models/`
- Optional HuggingFace cache: `.btrans_cache/hub`
- Fonts folder: `fonts/`

Project outputs are saved with the project/pages you open (not in one global output folder).

---

## Common issues (fast fixes)

### Python command not found
Install Python 3.10+ and verify:

```bash
python --version
```

### pip problems
Try:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Torch install fails on first run
Retry with:

```bash
python launch.py --reinstall-torch
```

For GPU/CUDA/ROCm details, use: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## Helpful docs (when you need more)

- Troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Quality tiers: [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)
- Models reference: [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- Translation context & glossary: [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- InDesign LPtxt export: [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md)

---

## Notes

- This project includes many optional modules and fonts. You do **not** need all of them to get started.
- Start simple (core modules), verify your workflow, then expand.
- If you report issues, include your startup command and full terminal log.
