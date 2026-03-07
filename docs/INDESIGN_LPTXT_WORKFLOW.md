# InDesign LPtxt workflow (Mangahanhua scripts)

This doc describes [Mangahanhua-Scripts-for-Indesign](https://github.com/jqk4388/Mangahanhua-Scripts-for-Indesign) and how BallonsTranslator’s **LPtxt export** fits into a translation → InDesign lettering pipeline.

---

## What Mangahanhua-Scripts-for-Indesign is

**Repo:** [jqk4388/Mangahanhua-Scripts-for-Indesign](https://github.com/jqk4388/Mangahanhua-Scripts-for-Indesign)

A suite of **Adobe InDesign ExtendScript (JSX)** scripts for manga layout and lettering. They run **inside InDesign only** (not a standalone app). The fork adds scripts aimed at **漫画汉化** (manga sinicization / Chinese localization); the original [Manga-Scripts-for-Indesign](https://github.com/papatangosierra/Manga-Scripts-for-Indesign) targeted 日漫英译 (Japanese → English).

- **Install path (example):**  
  `C:\Users\<you>\AppData\Roaming\Adobe\InDesign\Version 20.0-J\zh_CN\Scripts\Scripts Panel`  
  Or place a shortcut to the repo in that Scripts Panel folder.
- **Dependency:** Scripts rely on `Library/KTUlib.jsx`; keep it next to the scripts (or correctly referenced).
- **Tested:** InDesign CC 2025 (and similar versions).

---

## What the scripts do (high level)

| Category | Examples |
|----------|----------|
| **Import** | **LabelPlus TXT (LPtxt)** → InDesign: places translation text into text frames at coordinates, with optional style matching. |
| **Export** | **ID2LPtxt**: export text + coordinates from InDesign back to LPtxt (e.g. for proofreading or re-import elsewhere). |
| **Line-breaking** | 结巴断句 (jieba), 大模型LLM断句 (LLM-based), for Chinese; EN break-line scripts. |
| **Layout / style** | Create layers/frames, import styles/composite fonts, apply strokes (白字黑边 etc.), resize, text effects, object styles. |
| **Links / pages** | Relink images, PSD/TIF swap, page numbers, binding direction. |
| **Export (print/web)** | 1400 dpi TIF (print), 268 PNG (web), RGB for color pages, etc. |

So the **main link** with BallonsTranslator is: **we export LPtxt → you open it in InDesign with their “LabelPlus TXT 导入” script** to place and style the translated text.

---

## How the LPtxt format works (and compatibility)

The InDesign script **3.LabelPlus-script-id-UI.jsx** (and similar) parses:

1. **Header (optional):** e.g. `1,0`, `-`, `框内`, `框外`, `-`, `备注备注备注`.
2. **Page marker:** `>>>>>>>>[image_name]<<<<<<<<` (one per page).
3. **Block:**  
   - Line 1: `----------------[block_index]----------------[x,y,group]`  
   - Line 2+: the translation text (one or more lines).  
   Coordinates `x,y` are **normalized 0–1** (fraction of page width/height). `group` is used for style (e.g. 1 = 框内, 2 = 框外).

BallonsTranslator’s **Export translation to LPtxt** (File / Tools → Export translation to LPtxt) produces exactly this format:

- Same header and page markers.
- Block lines: `----------------[n]----------------[x,y,1]` with `x = left/total_width`, `y = top/total_height`, then the translation line.
- Optional **font/size info** in the text as `{字体：...}{字号：...}` when “Include font and size info” is checked; the InDesign script can use this for styling.

So **our LPtxt is compatible** with Mangahanhua’s LPtxt import. No code change in the scripts is required to consume BT’s export.

---

## Recommended workflow: BT → LPtxt → InDesign

1. **Translate in BallonsTranslator**  
   Open project → Detect → OCR → Translate → (optionally Inpaint/Render).  
   Use **Format / font** as you like; if you want InDesign to match font/size, enable “Include font and size info in the LPtxt output” when exporting.

2. **Export LPtxt from BallonsTranslator**  
   - **File** or **Tools** → **Export translation to LPtxt...**  
   - Choose whether to include font/size info.  
   - Output is `<project_name>_translations.txt` in the project folder (or the path you choose).

3. **InDesign**  
   - Create or open your InDesign document (page size/layout as needed).  
   - Run the Mangahanhua script: **LabelPlus TXT 导入** (e.g. “3.LabelPlus-script-id-UI.jsx” or the entry point your bundle uses).  
   - In the script UI: select the `*_translations.txt` file, set single-line vs multi-line mode, page range, and any text replacement / style matching options.  
   - Script places text frames at the coordinates and can apply object/paragraph/character styles.

4. **Optional round-trip**  
   - Use **ID2LPtxt** (export from InDesign to LPtxt) to send text+coordinates back out for proofreading or for re-use in other tools.

---

## What we do *not* implement

- The **InDesign scripts themselves** cannot run inside BallonsTranslator; they require Adobe InDesign and ExtendScript. We only **export LPtxt** that their scripts **import**.
- We do not run jieba/LLM line-breaking inside InDesign; that stays in their script suite (and optionally in BT for pre-processing before export, if you want).
- We do not implement their export presets (1400 TIF, 268 PNG, etc.); those remain InDesign-side.

---

## Summary

| Item | Description |
|------|-------------|
| **Mangahanhua** | InDesign JSX suite for manga lettering: LPtxt import, ID2LPtxt export, jieba/LLM line-breaking, styles, links, export. |
| **Our role** | Produce **LPtxt** that matches their format so you can **Export translation to LPtxt** in BT and **import that file in InDesign** with their scripts. |
| **Compatibility** | BT’s LPtxt (page markers, `----------------[n]----------------[x,y,1]`, optional `{字体：...}{字号：...}`) is compatible with their parser. |
| **Links** | [Mangahanhua-Scripts-for-Indesign](https://github.com/jqk4388/Mangahanhua-Scripts-for-Indesign), [Manga-Scripts-for-Indesign (original)](https://github.com/papatangosierra/Manga-Scripts-for-Indesign). |

For related projects and what we adopted from other tools, see [RELATED_PROJECTS.md](RELATED_PROJECTS.md).
