# Default UI rework commitment

The experimental shell is not intended to remain a hidden side feature. It is scaffolding for safely replacing the default BallonsTranslator-Pro UI.

## Product decision

The reworked UI should become the default UI.

Legacy UI should become:

- a temporary fallback during migration
- a troubleshooting/rollback option
- eventually a compatibility mode, not the primary product surface

## Design principle

Modern does not mean simple.

For BallonsTranslator-Pro, modern means:

- advanced
- dense where useful
- professional
- polished
- discoverable
- fast for power users
- approachable for new users without hiding expert tools

The goal is not to remove menus, panels, QA tools, CAT features, model controls, diagnostics, or export formats. The goal is to organize them into a high-quality interface with clear modes, strong visual hierarchy, and fewer confusing duplicate entry points.

## Default target shell

The default app should move toward:

- left mode rail
- top command/status bar
- center workflow workspace
- right inspector
- bottom job/status drawer
- command palette
- legacy menus available during migration

Default modes:

1. Home
2. Editor
3. Live Translation
4. Quick Image
5. Raw Downloader
6. Batch Queue
7. Translation Assist / QA
8. Models / Providers
9. Settings
10. Diagnostics / Help

## Migration rule

Do not delete the legacy editor until the modern shell has feature parity.

Instead:

1. Make Home/Launcher modern by default.
2. Add the mode rail as the default navigation surface.
3. Add the bottom job/status drawer as the default status surface.
4. Add the editor inspector as the default right-side shell, initially alongside legacy panels.
5. Embed the existing editor canvas into the Editor page.
6. Move existing right-side text/style/OCR panels into the EditorInspector tabs.
7. Feed real pipeline/export/download jobs into JobStatusDrawer.
8. Wire dashboard actions to existing handlers via `dashboard_action_dispatcher.py`.
9. Move raw downloader, live translation, batch queue, models, and diagnostics into their mode pages.
10. Keep old menus and keyboard shortcuts during migration.
11. Add `use_legacy_ui` as the rollback option.

## Current branch state

Already added:

- `ui/design_tokens.py`
- `ui/workflow_home.py`
- `ui/mode_rail.py`
- `ui/default_modern_shell.py`
- `ui/editor_inspector.py`
- `ui/job_status_drawer.py`
- `ui/mode_dashboard.py`
- `ui/experimental_app_shell.py`
- `ui/experimental_shell_preview_dialog.py`
- `ui/dashboard_action_router.py`
- `ui/dashboard_action_dispatcher.py`
- `ui/experimental_shell_settings.py`
- `ui/experimental_shell_menu.py`
- `ui/experimental_shell_bootstrap.py`

Already default-facing:

- `WelcomeWidget` embeds `WorkflowHomeWidget`.
- Existing welcome screen now uses workflow cards.
- `WelcomeWidget` now schedules `install_default_modern_navigation()` so `ModeRail` is inserted into the default `MainWindow` layout beside the legacy `LeftBar`.
- `ModeRail` routes default modes to existing handlers without replacing legacy controls yet.
- `ModeRail` selection is kept in sync when legacy handlers navigate to Home, Editor, Live, Downloader, Batch, Assist, Models, Settings, or Diagnostics.
- `WelcomeWidget.open_assist_requested` has a fallback connection to Translation Assist through `install_default_welcome_signal_fallbacks()`.
- `install_default_job_drawer()` inserts a collapsed `JobStatusDrawer` into the default `MainWindow` layout before the legacy `BottomBar` when available.
- The job drawer is a default status surface but does not replace legacy progress dialogs yet.
- Pipeline stage events are mirrored into the drawer as a `pipeline-current` job after legacy handlers run.
- Pipeline completion marks the drawer's current pipeline job as succeeded after legacy completion logic runs.
- `install_default_editor_inspector()` adds `EditorInspector` to the existing right-side editor stack.
- `EditorInspector` buttons reuse existing handlers for Translation Assist, OCR crop/triage, layout review, and typography QA.
- The inspector is additive and does not replace `DrawingPanel`, `TextPanel`, `SpellCheckPanel`, `PipelineInsightsWidget`, `MaskDiagnosticsWidget`, or `OcrCropInspectorWidget` yet.

Covered by tests:

- `tests/test_default_modern_shell.py` verifies that the rail is inserted before `LeftBar`, installation is idempotent, mode routing reuses existing handlers, legacy navigation keeps the rail synced, welcome Assist fallback connects once, the collapsed job drawer installs before `BottomBar`, jobs can be upserted into the drawer, pipeline stage events update the drawer, pipeline finish marks the drawer job succeeded, the editor inspector installs into the right stack, and inspector buttons call existing handlers.

Still to make default-facing:

- Feed export/download/model jobs into `JobStatusDrawer`.
- Make dashboard actions call existing handlers from the default shell.
- Make the experimental shell become the default app shell once legacy editor embedding is complete.
- Replace inspector placeholders with embedded legacy controls after parity is confirmed.

## Next implementation order

### Milestone A: default Home

Status: mostly complete.

- Modern workflow cards are in the default welcome screen.
- Continue polishing setup health, recent workflows, and model/provider warnings.

### Milestone B: default navigation

Status: initial default-facing implementation complete.

- `ModeRail` is installed beside the legacy `LeftBar` by `ui/default_modern_shell.py`.
- Legacy `LeftBar` stays visible and functional during migration.
- Mode routes currently map to existing handlers:
  - Home -> welcome screen
  - Editor -> existing editor
  - Live -> realtime translator
  - Quick Image -> open images
  - Downloader -> raw downloader
  - Batch -> batch queue
  - Assist -> Translation Assist dock
  - Models -> model manager
  - Settings -> config panel
  - Diagnostics -> environment doctor

Remaining navigation work:

- Add a `use_legacy_ui` / `use_legacy_left_bar` rollback setting before hiding or replacing the legacy left bar.
- Add richer visual state for project/job/provider health.

### Milestone C: default job/status drawer

Status: initial default-facing implementation complete.

- `JobStatusDrawer` is installed collapsed into the default main layout before `BottomBar`.
- It can accept `JobStatusSpec` entries through `upsert_default_job()`.
- Pipeline stage events update a current pipeline job with stage, page, progress, and cancel availability.
- Pipeline completion updates that job to `succeeded` at 100%.
- It intentionally does not replace existing progress dialogs yet.

Remaining drawer work:

- Feed batch/export/archive/download/model jobs into the drawer.
- Add clear completed, details, cancel, pause, and resume behavior per real job type.
- Preserve legacy progress dialogs until the drawer has parity.

### Milestone D: default editor inspector

Status: initial default-facing implementation complete.

- `EditorInspector` is installed into `rightComicTransStackPanel` as an additional panel.
- Its buttons call existing handlers instead of duplicating editor logic.
- Legacy panels remain available until parity is confirmed.

Remaining inspector work:

- Make it the primary right-side panel when opening projects.
- Move/embed existing text, style, layout, OCR, assist, QA, and metadata controls into inspector tabs.
- Update selected-block previews as canvas selection changes.
- Keep a fallback path to the current TextPanel/DrawingPanel stack while migrating.

### Milestone E: default advanced dashboards

- Use `ModeDashboard` pages for mode landing screens.
- Wire dashboard actions through `dashboard_action_dispatcher.py`.
- Replace placeholder pages with real workflow UI.

### Milestone F: legacy fallback

- Add `use_legacy_ui` config.
- Keep legacy layout reachable while modern shell becomes default.
- Add UI reset/troubleshooting docs.

## Acceptance criteria

The rework is not complete until:

- the modern shell is the default app UI
- legacy UI is a fallback, not the main path
- all current features remain accessible
- existing projects/configs continue to work
- model manager and first-run picker continue to work
- local API routes continue to work
- startup scripts continue to work
- power-user menu/shortcut workflows continue to work
- modern dashboards expose advanced tools instead of hiding them
- the app feels polished like high-quality modern software, with Dango-like friendliness and ImageTrans-like professional depth
