# UI Rework Migration Plan

## Compatibility-first sequence
1. Build complete action inventory (menus + shortcuts + handlers + context menus).
2. Centralize actions in registry and keep legacy menu wiring intact.
3. Switch command palette + shortcuts diagnostics to registry source.
4. Add simple/advanced/developer visibility metadata and filtering.
5. Add launcher/startup flow settings + migration-safe defaults.
6. Reorganize menus and navigation only after parity checks pass.

## This iteration
- Extended registry-backed discoverability through titlebar menu trees.
- Added startup/UI mode config defaults and load-time migration guards.
- Added tests for config migration defaults and registry behaviors.

## Backward compatibility contract
- Existing project files remain unchanged.
- Existing shortcut strings are preserved.
- Legacy menu layout remains active while `show_legacy_menus=True`.
- New config keys have safe defaults when loading older config files.

## Newly implemented in this pass
- Theme customizer now persists `ui_mode` and `show_legacy_menus`.
- MainWindow applies UI mode changes immediately via TitleBar action visibility filtering.
- Simple mode now reduces menu noise without removing underlying functionality or shortcuts.

## Startup mode behavior now wired
- `startup_mode` is now consumed at startup (`home|editor|last_used|settings|live|downloader|batch|models|diagnostics`).
- `recent_workflows` is now updated at runtime when entering editor/settings/live/downloader/batch/home surfaces.
- `show_home_on_startup` now participates in last-used fallback routing.
- Config-load startup-mode validation now explicitly accepts all routed modes (`settings|batch|models|diagnostics`) instead of collapsing them back to `last_used`.
- Live Translator now persists and restores runtime profile/capture/debounce/follow-window preferences through config-backed defaults.
- Realtime API `translate_now` now performs an immediate watcher tick for a target region and returns OCR/translation/status payload (instead of a placeholder queued response).
- Realtime service now applies config-backed defaults (profile/follow-window/region rect) when initialized and when live mode is enabled, improving startup consistency between Settings and live runtime behavior.
- Manage Models preset selector now includes a user-facing preset summary (packages/size/dependency notes) so community/full-stack model bundles are understandable before download.
- Manage Models now persists the last selected model preset, so repeated sessions reopen with the same package-bundle context.

## Live settings propagation improvements
- ConfigPanel now emits `ui_mode_changed` and MainWindow applies mode visibility immediately.
- ConfigPanel now emits omni option changes so command search cache refreshes without restart.
- Startup-dependent settings UI now validates fallback option relevance by selected startup mode.

- Continued phased menu split: introduced top-level Translation/Diagnostics/Export/Models while retaining legacy Tools to preserve workflow compatibility.

- Added persistent omni result-type filtering to reduce result noise and support simple-vs-power-user navigation styles.

- Simple/Advanced visibility now prefers ActionRegistry metadata when available, reducing heuristic-only filtering.

- Added registry validation loop (in-app + export payload summary) to support parity checks during phased menu remapping.

- Added Home workflow launch shortcuts wired to existing handlers to improve first-run discoverability without breaking legacy flows.

- `show_legacy_menus` now actively toggles legacy Tools visibility at runtime (non-destructive; new top-level menus remain).

- Theme customizer now controls startup target directly, complementing ConfigPanel startup settings.

- Registry enrichment pass now tags actions with top-level/category/workflow metadata to support deterministic menu-to-mode migration.

## Completion status snapshot
- Foundation (registry/config/startup/menu split/launcher): largely implemented.
- Context-menu taxonomy rewrite: substantially implemented (canvas action categories now aligned to Text box/OCR/Translation/Style/Layout/QA/Export; lightweight debug grouping remains mode-aware in editor list context menus).
- Context-menu quick profiles now persist last-applied preset (`editing|pipeline|layout|custom`) for clearer recoverability across restarts.
- Full end-state rework (workspace redesign, full settings IA, all wizard flows): still in progress.
