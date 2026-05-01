# BallonsTranslator-Pro

BallonsTranslator-Pro is an advanced fork of [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator). It keeps the original assisted manga/comic translation pipeline, then extends it with production-focused module choices and quality-of-life workflow improvements.

<img src="https://raw.githubusercontent.com/dmMaze/BallonsTranslator/master/doc/src/ui0.jpg" align="center">

At a high level, BallonsTranslator-Pro helps you:

1. Detect speech balloons/text areas
2. Read text (OCR)
3. Translate text
4. Clean original text from artwork (inpaint)
5. Edit and export pages

This means you can keep creative control (editing text and layout yourself) while removing most repetitive technical work.

- 中文文档（与本页同结构）: [README_zh_CN.md](README_zh_CN.md)
- Full change history: [docs/CHANGELOG.md](docs/CHANGELOG.md)

---

## Why use **Pro** (and not just the base fork)

If you are translating full chapters or frequent batches, Pro is designed to save time and reduce rework.

### Key reasons

- **More practical module choices** for detector/OCR/translator/inpaint combinations.
- **Better real-project flexibility** when some pages are easy and others are difficult.
- **Chain + LLM review workflows** to improve translation quality and consistency.
- **Beginner-friendly defaults** that work out-of-the-box for most users.

In short: base behavior is preserved, but Pro gives you more control when quality, speed, and consistency matter.

---

## Who this is for

This README is written for **non-developers and regular users** first.

You do **not** need programming knowledge to start.

If you want deeper technical details, use:
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)

---

## Quick start (non-developer friendly)

Choose **one** install method below.

### Option A (Easiest for most people): Download ZIP from GitHub

Use this if “git clone” is unfamiliar.

1. Open this repository page in your browser.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Extract the ZIP to a normal folder (for example: `Documents/BallonsTranslator-Pro`).
5. Open that folder.
6. Run:

```bash
python launch.py
```

> On first launch, required base packages install automatically.

---

### Option B: Clone with GitHub Desktop (no terminal required)

Use this if you want easier future updates but still avoid command-line Git.

1. Install **GitHub Desktop**.
2. In GitHub Desktop, choose **File → Clone repository**.
3. Paste this repo URL and choose a local folder.
4. After clone finishes, open the local folder.
5. Run:

```bash
python launch.py
```

---

### Option C: Clone with command line Git (for users comfortable with terminal)

```bash
git clone <REPO_URL>
cd BallonsTranslator-Pro
python launch.py
```

If you are not sure which option to choose, pick **Option A (Download ZIP)**.

---

## Requirements (simple version)

- Python **3.10 or newer** installed
- Internet connection for first-time dependency/model downloads
- Enough free disk space for models (size depends on selected packages)

Check Python quickly:

```bash
python --version
```

---

## First launch: what happens and why

On a fresh install (no `config/config.json` yet), Pro may ask what model packages to download.

- Start with **Core only** if you want the fastest setup
- Downloads run in background
- If a download fails: **Tools → Models → Retry model downloads**
- You can add/remove packages later: **Tools → Manage models**

Default baseline after setup:
- Detector: `ctd`
- OCR: `manga_ocr`
- Inpaint: `aot`
- Translator: `google`

These defaults are intentionally chosen to help beginners get usable results quickly.

---

## Basic workflow (5 steps)

1. **Open pages**: File → Open Folder (or Open Images)
2. **Pick modules** in Config panel (Detector / OCR / Inpaint / Translator)
3. Click **Run** in the bottom bar
4. **Review/edit** text bubbles as needed
5. **Export**: Tools → Export all pages

That is enough for most chapter workflows.

---

## When Pro is especially useful

Use Pro when you need one or more of the following:

- Better handling of mixed page difficulty (clean pages + messy pages)
- More translation control than one-click machine translation
- Consistency in terminology/tone across many pages
- A pipeline you can tune gradually rather than replacing everything at once

### 2-step translation validation (Feature #55)

You can run online MT first, then LLM review/correction second:

1. Set **Translator** to `Chain`
2. In Chain settings:
   - `chain_translators`: `google,LLM_API_Translator` (or `DeepL,LLM_API_Translator`)
   - `chain_intermediate_langs`: `English`
   - leave `chain_llm_review_mode` enabled
3. Configure `LLM_API_Translator` provider/model normally

This lets the final LLM stage validate and refine wording with access to both source text and first draft.

---

## Recommended beginner setup

If you want a stable starting point:

- Detector: `ctd`
- OCR: `manga_ocr`
- Inpaint: `aot`
- Translator: `google`

Then only change one module at a time when improving quality.

For manhua-focused tuning: [docs/MANHUA_BEST_SETTINGS.md](docs/MANHUA_BEST_SETTINGS.md)

---

## Where your files are

- Config template: `config/config.example.json`
- Active config: `config/config.json`
- Downloaded models: `data/models/`
- Optional HuggingFace cache: `.btrans_cache/hub`
- Fonts folder: `fonts/`

Outputs are saved with the project/pages you open (not one global output folder).

---

## Common issues (quick fixes)

### “python” command not found
Install Python 3.10+ and retry:

```bash
python --version
```

### pip problems

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Torch install fails during first run

```bash
python launch.py --reinstall-torch
```

For GPU/CUDA/ROCm details: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## Extra docs (when needed)

- Troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- Quality tiers: [docs/QUALITY_RANKINGS.md](docs/QUALITY_RANKINGS.md)
- Models reference: [docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md)
- Translation context & glossary: [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- InDesign LPtxt export: [docs/INDESIGN_LPTXT_WORKFLOW.md](docs/INDESIGN_LPTXT_WORKFLOW.md)

---

## Practical advice

- You do **not** need every model/module on day one.
- Start with default modules, confirm your end-to-end workflow, then expand.
- If you report an issue, include your startup command and full terminal log.
