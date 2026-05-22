# UI shell components

This document tracks reusable UI pieces added during the BallonsTranslator-Pro UI/UX rework.

## WorkflowHomeWidget

File: `ui/workflow_home.py`

Purpose:

- Dango-style workflow-card launcher.
- Can live inside the existing `WelcomeWidget` now.
- Can become the Home mode of the final app shell later.

Workflow cards:

- Editor
- Live Translation
- Quick Image
- Raw Downloader
- Batch Queue
- Translation Assist / QA
- Models / Providers
- Diagnostics / Help

Current integration:

- Embedded in `ui/welcome_widget.py`.
- Emits `workflow_requested(str)`.
- `WelcomeWidget` maps workflow keys to existing signals where possible.

Next integration step:

- Connect `WelcomeWidget.open_assist_requested` in `MainWindow` once a stable Translation Assist panel entry point is chosen.
- Consider adding a first-run setup-health row above the workflow cards.

## ModeRail

File: `ui/mode_rail.py`

Purpose:

- Reusable left navigation rail for the final app shell.
- Independent from `MainWindow`, so it can be tested and iterated before replacing or sitting beside the legacy `LeftBar`.

Modes:

- Home
- Editor
- Live
- Quick Image
- Raws
- Batch
- Assist
- Models
- Settings
- Help / Diagnostics

Current integration:

- Component exists, but is not yet embedded into `MainWindow`.

Next integration step:

- Add `ModeRail` beside the existing `LeftBar` behind a config flag, for example `enable_experimental_mode_rail`.
- Route `mode_requested(str)` to existing handlers:
  - `home` -> `_show_welcome_screen()`
  - `editor` -> `setupImgTransUI()`
  - `live` -> `on_open_realtime_translator()`
  - `quick_image` -> `leftBar.onOpenImages()`
  - `downloader` -> `on_open_manga_source()`
  - `batch` -> `on_open_batch_queue()`
  - `assist` -> Translation Assist dock entry point
  - `models` -> `on_open_manage_models()`
  - `settings` -> `setupConfigUI()`
  - `diagnostics` -> `on_environment_doctor()`

## Design tokens

File: `ui/design_tokens.py`

Purpose:

- Shared spacing, radius, typography, color, badge, card, and inspector-section styling helpers.
- Prevents every new UI widget from inventing a different style.

Next integration step:

- Use tokens in the future right inspector and job drawer components.
- Add `docs/UI_DESIGN_SYSTEM.md` once more components use the tokens.
