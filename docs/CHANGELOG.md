# Changelog

All notable updates for BallonsTranslator-Pro are tracked here.

## v1.7.0

### Highlights
- Auto-layout behavior was reworked and tuning options were expanded.
- First-run model package download became non-blocking (main window opens first).
- Retry path for failed first-run downloads added via model tools menu.
- UI/menu organization was cleaned up for clearer workflow grouping.
- Stability fixes landed across pipeline flow, module parameter handling, and project save paths.

### UX / Accessibility updates (post-v1.7.0 maintenance)
- Right-side text editor list now includes direct context-menu entry points for compare flows:
  - Compare translation providers
  - Compare OCR engines
  - Compare detectors
  - Compare inpainters
- Compare selection UI was upgraded from numeric input prompts to a dedicated table dialog with:
  - provider/source/latency/current markers,
  - searchable/filterable candidate rows,
  - in-dialog candidate preview,
  - one-click copy/apply actions.
- Translation Assist dock opening was hardened for frameless host windows by falling back to floating mode when dock APIs are unavailable.
- Advanced text formatting now includes lightweight 3D lettering controls (`Perspective X`, `Perspective Y`, `3D Depth`) for SFX/title styling.

### Notes
- This file intentionally holds volatile release notes so `README.md` and `README_zh_CN.md` stay concise.
- For issue-level details and implementation notes, review commit history and issue tracker.
