# UI Rework Audit (Phase 0)

## Scope
This audit inventories current action surfaces before any behavior-moving UI rework.

Primary audited modules:
- `ui/mainwindowbars.py`
- `ui/mainwindow.py`
- `ui/configpanel.py`
- `ui/textedit_area.py`
- `ui/scenetext_manager.py`
- `utils/shortcuts.py`
- `utils/config.py`

## Current action surfaces
- Left rail quick controls: open/project/save/run/config/search.
- Title bar menus: File, Edit, View, Go, Pipeline, Tools.
- Context menus: canvas/text actions (configured by context-menu options dialog).
- Omni search: currently scans menu actions + settings + canvas blocks.

## Findings
1. **Action creation is distributed** across `LeftBar`, `TitleBar`, and `MainWindow`.
2. **Tools menu mixes categories** (project QA, export, sources, queue, models, diagnostics).
3. **Omni search depends on recursive runtime menu walk**, not a reusable registry.
4. **Shortcut ownership is split** between `QAction` and `QShortcut`, with conflict-avoidance comments already present.

## Action inventory status
This PR introduces the first centralized registry and migrates discoverability metadata for existing:
- File actions (title bar File + left-bar file-related actions)
- Edit actions
- View actions
- Go actions
- Pipeline actions
- Tools actions

Detailed per-action mapping is tracked in `docs/UI_ACTION_REGISTRY.md` and will be expanded in follow-up PRs to include dialogs, context menus, and mode-level surfaces.

## Compatibility constraints captured
- Preserve action handlers/signals and existing shortcuts.
- Preserve existing menu layout/labels in this milestone.
- Preserve omni-search behavior while sourcing command entries from registry.
