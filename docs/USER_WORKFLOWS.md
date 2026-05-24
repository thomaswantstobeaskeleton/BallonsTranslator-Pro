# User Workflows (Current Increment)

## Home / Launcher workflows
From the Welcome screen users can now start directly from workflow shortcuts:
- Manga Editor
- Live Translator
- Raw Downloader
- Batch Queue
- Models
- Diagnostics

These shortcuts route to existing handlers and preserve current behavior.

## Compatibility
- Legacy menu paths remain available.
- Existing project open flows and startup behavior are unchanged aside from the added launcher shortcuts.

## Startup preference
Theme & UI Customizer and Settings now include startup target selection (Home/Editor/Last-used/Settings/Live/Downloader/Batch/Models/Diagnostics).

## Help & diagnostics access
A dedicated top-level Help menu now provides Documentation, About, Update, Keyboard Shortcuts, Context Menu Options, and a Feature Matrix / Model Coverage summary.

## Current limitation note
Some project-dependent commands appear as unavailable until a project is opened; omni-search now explains this state explicitly.

## Live Translator setup status
- Live Translator now initializes with real screenshot backend selection (`auto` backend), persists capture/debounce/follow-window preferences, and restores them on next launch.
- Settings → General now exposes Live Translator defaults (profile/capture interval/min OCR interval/follow-window), and automation API supports `realtime_regions` update payloads.
- Live Translator also remembers the last region rectangle (x/y/width/height) to speed up repeated session startup.

## Community model bundle coverage
- Model packages now include an optional **Community showcase models** bundle with entries for YSG comic segmenters, MangaLens bubble segmentation, SAM2/3 segmentation variants, Flux inpaint variants, PaddleOCR-VL-1.5, and Real-CUGAN/AnimeSharp upscaling metadata.
- A dedicated detector key `mangalens_bubble_segmentation` is now registered, using YOLO-backed bubble-seg defaults and a default checkpoint path `data/models/mangalens_bubble_segmentation.pt`.
