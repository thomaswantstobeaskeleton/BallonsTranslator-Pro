# Default mode dashboard migration

This document records the default-facing dashboard slice for the UI/UX rework.

## What changed

The modern mode dashboards are no longer only experimental preview components. They are now installed into the default `MainWindow.centralStackWidget` through `ui/default_mode_dashboards.py`.

The installer appends dashboard landing pages after the existing default pages:

1. Welcome
2. Legacy editor
3. Legacy settings
4. Live Translation dashboard
5. Quick Image dashboard
6. Raw Downloader dashboard
7. Batch Queue dashboard
8. Translation Assist / QA dashboard
9. Models / Providers dashboard
10. Diagnostics / Help dashboard

This keeps the legacy editor/settings surfaces intact while giving the mode rail real landing pages.

## Routing behavior

`install_default_mode_dashboards()` also reroutes the default `ModeRail` after it has been installed.

Mode behavior:

- Home still opens the existing welcome screen.
- Editor still opens the existing editor when a project is loaded, or returns to welcome when no project is open.
- Settings still opens the existing settings panel.
- Live, Quick Image, Downloader, Batch, Assist, Models, and Diagnostics now show dashboard landing pages first.

Dashboard action buttons then call existing handlers through `dashboard_action_dispatcher.py`.

Supported dispatch targets:

- registered QAction IDs when present
- direct MainWindow handlers, for example `on_open_realtime_translator`
- dotted child-widget handlers, for example `leftBar.onOpenImages`

## Why this is safe

The dashboards are additive.

They do not remove:

- legacy `LeftBar`
- legacy menus
- welcome screen
- editor canvas
- settings panel
- text/style panels
- progress dialogs
- model manager flows
- local API routes

They only add dashboard pages and route mode-rail clicks to those pages.

If dashboard installation fails, the rail falls back to the existing handlers.

## Current tests

Covered by `tests/test_default_mode_dashboards.py`:

- dashboard pages are appended to the central stack
- installation is idempotent
- dashboards can be shown by mode key
- rail routing prefers dashboard pages when available
- legacy fallback still works when dashboards are not installed
- installed `ModeRail` routes to dashboard pages
- dashboard actions can dispatch to existing child-widget handlers and log status

Also covered by `tests/test_dashboard_action_dispatcher.py`:

- action registry dispatch
- direct handler fallback
- dotted child-handler fallback
- editor layout-review route
- known safe routes

## Remaining work

Dashboard pages still need real live data:

- project page count
- current project status
- pipeline stage state
- model/provider health
- warning counts
- downloader source health
- batch queue counts
- realtime latency/state
- Translation Assist candidate/glossary/TM status

Until those metrics are wired, dashboards act as modern navigation and action launchers rather than full operational dashboards.

## Migration rule

Do not replace the legacy editor with dashboards until feature parity is confirmed.

The next dashboard PR should focus on metrics and status wiring, not removing old surfaces.
