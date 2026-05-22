# UI/UX rework implementation prompt

You are working in BallonsTranslator-Pro, a Python/PyQt fork of dmMaze/BallonsTranslator.

The goal is to complete a workflow-centered UI/UX rework while preserving every existing feature, project format, config field, model-manager flow, startup script, local API route, and advanced scanlation workflow.

Use the competitor visual research in `docs/UI_COMPETITOR_VISUAL_RESEARCH.md` as the product-design reference.

## Design targets

Emulate the good parts of these tools:

- ImageTrans: professional editor layout, selected text block inspector, Translation Assist pane, text-area tables, CAT-tool workflow.
- Dango Translator: friendly mode launcher, live translation flow, soft modern visual hierarchy, right-side style inspector, always-visible job/status progress.
- manga-image-translator: clear automation pipeline, headless/API mental model, stage presets.
- Project Naptha-style browser OCR: low-friction selected-region/live translation interaction.
- Manga reader apps: left library/page/chapter navigation and resumable source/downloader workflows.

Do not clone any competitor UI directly. Adapt the patterns to Pro's existing PyQt architecture.

## Non-negotiable rules

1. Do not remove features.
2. Do not break existing menus until replacements are proven.
3. Do not break existing shortcuts.
4. Do not break config loading/saving.
5. Do not break startup scripts.
6. Do not break first-run model picker or model manager.
7. Do not break local API route discovery.
8. Keep Windows compatibility.
9. Keep README.md and README_zh_CN.md parity for user-facing changes.
10. Make changes in small reviewable PR-sized phases.

## Target app shell

Create a consistent shell with:

- Left mode rail
- Top command/status bar
- Center workspace
- Right inspector
- Bottom job/status drawer
- Command palette
- Legacy menus available during migration

Top-level modes:

1. Home
2. Editor
3. Live Translation
4. Quick Image
5. Raw Downloader
6. Batch Queue
7. Assist / QA
8. Models / Providers
9. Settings
10. Diagnostics / Help

## Phase 1: Action registry and command palette

Status: initial implementation exists on this branch.

Continue improving:

- Register every menu action.
- Add stable action IDs.
- Add category, workflow mode, simple/advanced visibility, shortcut, tooltip, and danger level.
- Build command palette from the registry.
- Add command result badges: Command, Setting, Page, Text block, Recent, Help.
- Add disabled-action reasons.
- Add duplicate shortcut detection.
- Add JSON export for action inventory.

Acceptance:

- No action is lost.
- Omni-search/command palette uses the registry.
- Tests cover duplicate IDs, duplicate shortcuts, and command discovery.

## Phase 2: Menu cleanup

Reorganize menus into:

- File
- Edit
- View
- Pipeline
- Tools
- Translation
- Export
- Models
- Diagnostics
- Help

Rules:

- Tools must not be a dumping ground.
- Translation Assist, TM, glossary, concordance, SFX, prompt profiles, and translation QA belong under Translation.
- Proof packs, CBZ/ZIP/PDF, SVG/PSD handoff, XLIFF, Excel, Word, LabelPlus, OCR JSON belong under Export.
- Model/provider/cache/setup actions belong under Models.
- Logs, doctors, startup reports, action-registry validation, safe mode, and API route discovery belong under Diagnostics.

Acceptance:

- Existing handlers are reused.
- Legacy menu locations remain reachable through command palette and optionally legacy menus.

## Phase 3: Home / Launcher

Build a new Home mode inspired by Dango's workflow cards.

Cards:

- Open Project
- Open Folder / Images
- Open CBZ/ZIP/CBR
- Continue Recent Project
- Manga & Comic Editor
- Chrome Manhua Live Translation
- Image Quick Translation
- Raw Downloader
- Batch Queue
- Translation Assist / QA
- Models / Provider Setup
- Diagnostics

Each card should show:

- short description
- recommended user/task
- setup health/warnings
- last-used state where relevant

Config:

- startup_mode: home | editor | last_used | live | downloader
- show_home_on_startup
- recent_workflows

Acceptance:

- New users know where to start.
- Existing power users can bypass Home.

## Phase 4: Editor workspace

Rework Editor mode around:

- Left page thumbnails/project navigation
- Center canvas
- Right inspector
- Bottom job/status bar

Right inspector tabs:

- Text
- Style
- Layout
- Assist
- OCR
- QA
- Metadata

Emulate ImageTrans' professional workspace: source/target text and Translation Assist should be visible near the selected block, not hidden in menus.

Acceptance:

- Selecting a text block updates the inspector.
- Common text/style/layout actions require fewer clicks.
- Translation Assist is a dock/tab, not just a menu action.

## Phase 5: Settings rework

Reorganize settings into searchable groups:

- General
- Appearance
- Startup
- Editor
- Canvas
- Text & Typesetting
- Auto Layout
- OCR
- Translation
- Translation Assist
- Glossary / TM / SFX
- Inpaint / Cleanup
- Raw Downloader
- Live Translation
- Models
- Providers
- Performance
- Storage / Data Paths
- Shortcuts
- Integrations
- Advanced
- Developer

Requirements:

- Search at top.
- Breadcrumbs.
- Simple/advanced/developer filter.
- Reset-to-default per setting.
- Inline validation.
- Requires-restart badges.
- Requires-model/provider badges.
- Setup wizard links.

Acceptance:

- Existing config keys still load/save.
- Settings are easier to scan and search.

## Phase 6: Workflow-specific modes

### Live Translation

Dango-inspired, low-friction UI:

- region/window picker
- Chrome Manhua Reader preset
- profile selector
- OCR/translation provider status
- overlay controls
- history panel
- privacy status
- high-quality/slower options such as SAM-assisted region refinement

### Quick Image

- drag/drop images
- run default OCR/translate/inpaint/render
- before/after preview
- promote to full project

### Raw Downloader

- source browser
- search/results
- chapters queue
- output/import controls
- source health/status badges

### Batch Queue

- parent/child CBZ/folder processing
- resumable queue
- progress/cancel
- export manifests

### Assist / QA

ImageTrans-inspired professional area:

- candidate comparison
- TM fuzzy matches
- glossary hits/violations
- concordance
- SFX dictionary
- QA warnings

Acceptance:

- Each major workflow has a clear home instead of being buried in Tools.

## Phase 7: Visual design system

Create or continue:

- design tokens
- spacing scale
- typography scale
- icon sizes
- panel padding
- cards
- badges
- warning banners
- empty states
- job status pills
- provider status chips
- inspector sections

Files may include:

- `ui/design_tokens.py`
- `ui/ui_components/`
- `docs/UI_DESIGN_SYSTEM.md`

Acceptance:

- New UI surfaces look consistent.
- Styling is not duplicated randomly.

## Phase 8: Notifications and jobs

Create a unified job/status drawer:

- OCR
- translation
- inpaint
- render
- export
- raw download
- live translation
- model download
- batch queue
- Translation Assist provider calls

Requirements:

- progress
- cancel/pause where supported
- warnings
- copy error details
- avoid modal spam for non-critical errors

Acceptance:

- Users always know what is running and why.

## Phase 9: Migration safety

Add:

- legacy menu layout option
- UI reset/troubleshooting docs
- config migration tests
- action registry inventory tests
- startup mode tests
- manual QA checklist

Acceptance:

- Existing users keep their workflows.
- New users get a cleaner app.

## Minimum next implementation milestone

1. Keep the existing ActionRegistry work.
2. Add competitor visual research document.
3. Add this implementation prompt.
4. Add a lightweight design-token module.
5. Add a Home/Launcher skeleton or improve existing welcome screen with workflow cards.
6. Add tests for mode/action visibility if possible.
7. Update PR body with research-driven design direction.

## Final acceptance

- Pro has a workflow-centered app shell.
- Menus are logical and no longer overloaded.
- Home/Launcher makes the app approachable.
- Editor mode feels like a professional scanlation workspace.
- Live mode feels like a lightweight reader/overlay tool.
- Translation Assist is visible and useful.
- Settings are searchable and organized.
- Power users have command palette access.
- All existing functionality remains accessible.
