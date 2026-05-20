# Realtime Translation Mode Plan (Phase 0/1 Baseline)

## Baseline audit summary
- 🟨 Existing repo already has a realtime dialog entry and minimal watcher primitives.
- 🟨 Local automation API already exposes realtime namespace skeleton.
- 🟨 Privacy defaults exist in `RealtimePrivacyConfig` and default to non-persistent behavior.
- ⛔ Region/window picker UX, follow-window backend integration, and robust overlay exclusion are partial/incomplete.

## First milestone implementation target
- Keep manga editor unchanged except launcher/menu entry points.
- Preserve project-less live mode.
- Stabilize watcher change-detection behavior and route discovery exposure.
- Keep privacy defaults strict:
  - do not persist captures by default
  - do not persist OCR/translation text by default
  - do not log live text by default

## Migration risk
- Low: additive mode, no project schema change in this first slice.
- Primary risk is optional dependency handling around capture backends; keep fallback backend active.

## Tests in this milestone
- unchanged frames are skipped
- unchanged OCR text is skipped
- overlay/capture persistence defaults stay disabled

## Next slices
- capture backend implementations (Windows native + MSS + Qt fallback)
- multi-region/profile manager
- follow-window coordinate tracking and HiDPI diagnostics
- richer overlay modes and promote-to-project workflow

## 2026-05-20 incremental completion
- Added follow-window rect resolution path in watcher/backend abstraction (`window_id` + optional `crop`) with diagnostics warning (`follow_window_unavailable`) when window metadata cannot be resolved.
- Added optional MSS backend path in screenshot backend factory with automatic fallback to Qt compatibility backend when MSS is unavailable.
- Added Windows-native backend selection path (stub + fallback) in screenshot backend factory to preserve API shape for platform-specific backend rollout.
