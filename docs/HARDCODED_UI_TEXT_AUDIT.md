# Hardcoded UI Text Audit (non-`.ts`/`.qm` localizable)

This audit lists user-facing UI text that is currently hardcoded in Python and therefore does not flow through Qt `.ts/.qm` translation files.

## Scope and method

- Scanned `ui/*.py` for Qt widget/message APIs with direct string literals (`QLabel`, `QPushButton`, `QCheckBox`, `QGroupBox`, `setText`, `setToolTip`, `setPlaceholderText`, etc.).
- Filtered to human-visible English UI strings.
- Logging-only messages are **not** translation blockers for end-user UI and are excluded from the action list.

## High-priority hardcoded UI strings (should be wrapped with `self.tr(...)`)

### `ui/merge_dialog.py`
- Window/group/button labels and tooltips are heavily hardcoded in English:
  - `Region merge tool settings`
  - `Main settings`
  - `Text merge order (by label)`
  - `Label merge rules`
  - `Geometry merge parameters`
  - `Advanced options`
  - `Merge result type`
  - `Run on current page`, `Run on all pages`, `Cancel`
  - Multiple placeholder/tooltips such as `label1,label2,...`, `e.g. label1,label2`, overlap/gap descriptions.

### `ui/mainwindowbars.py`
- Branding label text: `BallonsTranslatorPro` (if intended as immutable brand text, keep; otherwise localize presentation copy nearby).

### `ui/configpanel.py`
- Enum-like combobox literals not wrapped: `never`, `all_pages`, `current_page` (if visible to users in UI, localize labels and map back internally).

### `ui/subtitle_file_translator_dialog.py`
- Format keys inserted via literals (`auto`, `srt`, `txt`) in some `addItem(...)` calls; visible labels should always use localized text.

### `ui/video_translator_dialog.py`
- Several placeholder texts are direct literals (e.g., `ja, en, zh, ...`, `mp4v`, API key placeholders). These should be reviewed case-by-case:
  - Technical tokens can remain raw examples.
  - Explanatory placeholder text should be wrapped with `self.tr(...)`.

## Why this matters

Any user-visible English text that bypasses `self.tr(...)` will remain English even when Chinese (or any other language) is selected, which creates a mixed-language UI and undermines translation completeness.

## Recommended follow-up patch plan

1. Refactor `ui/merge_dialog.py` first (largest untranslated surface).
2. Normalize combobox labels in `ui/configpanel.py` and `ui/subtitle_file_translator_dialog.py` with localized display text + stable internal values.
3. Review placeholders in `ui/video_translator_dialog.py` and localize explanatory ones.
4. Regenerate/update `.ts` files and compile `.qm` with `lrelease` in an environment where Qt Linguist tools are installed.
