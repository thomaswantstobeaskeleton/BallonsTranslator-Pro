# UI competitor visual research

This document records visual/product references for the BallonsTranslator-Pro UI/UX rework. It is intentionally implementation-oriented: each reference is converted into concrete UI decisions for Pro.

## Scope

References reviewed:

- ImageTrans / BasicCAT
- Dango Translator / 团子翻译器
- manga-image-translator
- Project Naptha-style browser image text interaction
- MangaDex/Tachiyomi-style library and reader conventions

## ImageTrans takeaways

ImageTrans is closest to the professional scanlation/CAT-tool workflow.

Observed layout:

- Top menu bar: File, Edit, Project, Tools, About.
- Top command strip: OCR button, source/target language selectors, engine selector, checkboxes for automatic OCR/translation cleanup behavior.
- Left workspace: image canvas with detected text boxes visible as colored rectangles.
- Right workspace: source OCR text, target translation text, save/spellcheck/cleanup controls.
- Lower right tabs: Text areas, Translation assist, Panel.
- Translation assist table compares multiple machine translation outputs with a provider/source note column.
- Term/glossary hints appear near the translation/editor area.

What to emulate:

1. Keep a professional editor mode with a big canvas and an inspector/assist area.
2. Make Translation Assist a first-class dock/tab, not a hidden menu command.
3. Show multiple translation candidates in a table with provider names.
4. Keep selected block source/target editing close to the canvas.
5. Expose reading order/text-area data in a tabular panel.
6. Keep manual adjustment available for every automatic step.

What not to emulate directly:

1. The old gray desktop style feels dated.
2. Too many controls in the top strip can overwhelm new users.
3. The top-level menu names are simple, but they do not scale to Pro's feature count.

## Dango Translator takeaways

Dango is strongest as a consumer-friendly live/image translator experience.

Observed layout patterns:

- First-run or mode screen uses large illustrated cards for two clear modes: realtime translation and manga translation.
- Manga mode uses a left image list, a central before/after workspace, and a right style/settings inspector.
- Top area shows import/one-click translate/export actions and progress/status details.
- Right inspector has tabs/categories for text box style, global style, and templates.
- Live translation uses a lightweight overlay above another app/game with translated text in place.
- Recent Dango notes emphasize capture-window exclusion so the overlay does not pollute OCR.

What to emulate:

1. Add a friendly Home/Launcher with workflow cards.
2. Split Pro into clear modes: Editor, Live Translation, Quick Image, Raw Downloader, Batch, Assist/QA, Models, Settings.
3. Use a left navigation rail for major modes instead of relying only on menus.
4. Use a right inspector for selected text/style/provider details.
5. Add top job/progress status that is always visible for long-running workflows.
6. Make live translation feel like a separate lightweight product surface.
7. Use softer visual hierarchy, clearer spacing, and more obvious primary actions.

What not to emulate directly:

1. Do not make the UI too cute or brand-specific.
2. Do not hide advanced scanlation features behind a consumer-only flow.
3. Do not make cloud/account sync required.

## manga-image-translator takeaways

manga-image-translator is strongest for automation and server/headless workflows, not a polished desktop editor.

What to emulate:

1. Clear stage pipeline mental model: detection, OCR, translation, inpaint, render/export.
2. A compact job/progress surface for long tasks.
3. Presets for common workflows: full translate, OCR only, cleanup only, translate only.
4. API/headless parity should be visible in Diagnostics/Automation, not mixed into normal editing menus.

## Project Naptha-style takeaways

Project Naptha's browser-extension model is relevant to Pro's live Chrome/manhua translation mode.

What to emulate:

1. Image text should feel selectable and interactive.
2. Browser/live mode should minimize friction: select region, translate, copy, pause.
3. OCR overlays should not feel like a full project unless promoted.

## MangaDex/Tachiyomi-style reader takeaways

These are not translation editors, but their library/reader conventions matter for raw downloader and live reading workflows.

What to emulate:

1. Library/source/downloader views should be card/list based and searchable.
2. Reader/live mode should prioritize low chrome and easy navigation.
3. Chapter/page lists should be persistent and resumable.

## Final UI direction for Pro

Pro should have two personalities:

1. Friendly workflow launcher for new users and quick tasks.
2. Professional scanlation workspace for serious editing.

The target layout:

- App shell
  - left mode rail
  - top command/status bar
  - center workspace
  - right inspector
  - bottom job/status drawer
- Home mode
  - workflow cards
  - recent projects
  - setup health
  - recommended next actions
- Editor mode
  - left page thumbnails
  - center canvas
  - right inspector tabs: Text, Style, Layout, Assist, OCR, QA, Metadata
  - bottom job/status area
- Live mode
  - region/window picker
  - profiles
  - overlay controls
  - OCR/translation history
- Downloader mode
  - source browser
  - search/results
  - chapter queue
  - output/import controls
- Assist/QA mode
  - selected text block assist panel
  - provider candidate comparison
  - TM/glossary/concordance/SFX tabs
- Models/Providers mode
  - setup wizard
  - health checks
  - model packages
  - provider keys/endpoints
- Settings mode
  - searchable grouped settings
  - simple/advanced/developer filters

## Visual design rules

1. Prefer cards for workflow choices and setup tasks.
2. Prefer inspectors for selected-object controls.
3. Prefer tabs for related detail panes, not top-level workflows.
4. Prefer a command palette for rare actions.
5. Keep menus, but make them secondary to workflow navigation.
6. Use primary action buttons per mode, not dozens of equally weighted toolbar buttons.
7. Use status chips for provider/model/job health.
8. Use warning banners instead of modal dialogs for non-critical issues.
9. Preserve manual controls and advanced features.
10. Keep legacy menus available during migration.

## Emulation summary

- Emulate ImageTrans for professional editor + Translation Assist placement.
- Emulate Dango for mode launcher, live translation flow, right-side style inspector, and progress/status clarity.
- Emulate manga-image-translator for pipeline/job mental model.
- Emulate Project Naptha for low-friction live/browser image text interaction.
- Emulate reader apps for source/chapter/library organization.
