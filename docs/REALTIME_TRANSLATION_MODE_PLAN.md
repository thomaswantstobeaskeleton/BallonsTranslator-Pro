# Realtime Translation Mode Plan (Phase 1 Skeleton)

Last updated: 2026-05-19

## Scope for first PR series

- Add launcher/menu entry for a project-less realtime dialog.
- Add screenshot backend abstraction (interface + safe fallback backend).
- Add rectangular region selection model for watcher pipeline.
- Add change-detection watch loop (skip unchanged frame / unchanged OCR text).
- Add OCR + translation provider selection controls in realtime UI.
- Add floating overlay display path.
- Add privacy-first defaults (no persistence/logging by default).
- Add minimal local API routes for realtime status/control (`/realtime_*`) without exposing screenshots by default.
- Add realtime SSE snapshot endpoint (`GET /realtime/events`) for external status dashboards.

## Existing audit baseline

- Local API discovery and job routes already exist and are being extended incrementally.
- Main manga/comic editor workflow remains primary and must stay unchanged.
- New realtime mode should be optional and tool-launched.

## Incremental design

### Components

1. `utils/realtime_mode.py`
   - `ScreenshotBackendBase`
   - `NumpyFrameBackend` (test fallback backend)
   - `RealtimeRegion`
   - `RealtimePrivacyConfig`
   - `RealtimeWatcher.tick()` status machine

2. `ui/realtime_translator_dialog.py`
   - OCR/trans provider comboboxes
   - privacy hint text
   - Start/Pause/Translate now controls
   - floating overlay label

3. Menu entry
   - Tools → Realtime Screen Translator

### Privacy defaults

- No screenshot persistence by default.
- No OCR text persistence by default.
- No translation persistence by default.
- No live-text logging by default.

## Risks / follow-up

- Real native window capture backends (MSS/Windows APIs) and overlay exclusion are follow-up slices.
- Follow-window and named-region persistence are follow-up slices.
- Realtime local API routes (`/realtime/*`) are follow-up slices.
- This phase intentionally avoids browser scripting/extensions and works from visible pixels only.

## Tests added for skeleton

- skip unchanged frames and unchanged text avoids duplicate OCR/translation calls
- privacy config defaults remain non-persistent/non-logging
