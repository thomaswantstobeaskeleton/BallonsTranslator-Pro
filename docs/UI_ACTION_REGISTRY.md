# UI Action Registry

## Purpose
Provide one source of truth for menu/command discoverability metadata before visual navigation changes.

## Implementation
- New module: `ui/action_registry.py`
- Core model: `ActionRecord`
- Registry API:
  - `register_qaction(...)`
  - `register_menu_tree(...)`
  - `discoverable_actions()`
  - `duplicate_shortcuts()`

## Current migration coverage
Registry-backed discoverability is now wired for existing title-bar menu trees:
- File
- Edit
- View
- Go
- Pipeline
- Tools

## Notes
- This milestone intentionally does not rebind handlers or rewrite menu construction.
- Action IDs are deterministic slug IDs generated from top-level/menu-path/label.
- Next phase will add explicit stable IDs for backward command aliases.

## Enhancements in current pass
- Registry now tracks action enabled-state metadata.
- Discoverability API supports `show_unavailable` so command palette can optionally include disabled commands with reasons.

- Added registry export support: users can now export full action metadata to JSON from the UI for audit/migration verification.

- Added `summary_stats()` and in-app **Validate action registry now** command to quickly audit totals, visibility distribution, and duplicate shortcuts.

- Registry rows now include `top_level`, `category`, and `workflow_mode` metadata for stronger downstream remapping/reporting.
