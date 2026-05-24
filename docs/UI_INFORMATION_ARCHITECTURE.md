# UI Information Architecture (Target)

## Target top-level modes
1. Home / Launcher
2. Manga & Comic Editor
3. Live Screen / Chrome Manhua Translator
4. Image Quick Translator
5. Raw Downloader
6. Batch Queue
7. Translation Assist / QA
8. Models & Providers
9. Settings
10. Diagnostics / Help

## First milestone (implemented in this PR)
- Introduce a **central action metadata layer** (`ui/action_registry.py`).
- Keep current visible IA intact while enabling future migration.
- Feed omni-search from the same registry-backed action list.

## Transitional IA principle
- **Do not move user workflows yet**.
- Build an action graph first, then remap menus/modes in later phases.

## Implemented progress update
- Added **UI mode control** in Theme & UI Customizer (Simple/Advanced/Developer).
- Added runtime menu visibility filtering for simple mode (advanced-heavy commands hidden, legacy handlers preserved).

- Omni search now supports fuzzy matching and an explicit **Show unavailable** toggle to include disabled commands with availability reasons.

- Startup and UI complexity controls are now available in the Settings > General startup section (startup mode, fallback-to-home, legacy menus, omni unavailable commands).

- Omni search now includes typed result badges and sources beyond commands: **Command, Setting, Text block, Page, Recent, Help**.
- Omni search can directly open recent projects and trigger help entries (Documentation/About/Keyboard Shortcuts).

- Added non-breaking top-level **Translation** and **Diagnostics** menus to reduce Tools overload while preserving legacy Tools entries.

- Added non-breaking top-level **Export** and **Models** menus to continue splitting overloaded Tools responsibilities while preserving legacy access points.

- Omni-search now supports a persistent result-type filter (All/Command/Setting/Text block/Page/Recent/Help), configurable from both search UI and Settings.

- Welcome/Home now includes direct workflow launch buttons (Editor, Live, Downloader, Batch, Models, Diagnostics) that route to existing handlers.

- Legacy menu layout toggle is now functional at runtime: Tools can be hidden while Translation/Diagnostics/Export/Models remain available.

- Added dedicated top-level **Help** menu to avoid burying docs/about/shortcuts inside View and improve discoverability for new users.
