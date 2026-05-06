# Koharu-inspired Rendering and Workflow Controls

This page summarizes the active BallonsTranslator-Pro controls and automation outputs added/extended for Koharu-style manga lettering passes.

## Text rendering / typography controls

- **Writing mode**: per-textbox `auto`, `horizontal_ltr`, `vertical_rl`, and `rtl` are stored on `FontFormat.writing_mode`. Auto mode resolves CJK text in tall boxes to vertical RL and Arabic/Hebrew text to RTL.
- **Fit mode**: `shrink`, `expand`, `preserve`, and `balance` are stored on `FontFormat.fit_mode` and used by layout review, QA, and project-wide fixes.
- **Line breaking**: `auto`, `cjk_strict`, `balanced`, and `loose` are stored on `FontFormat.line_break_strategy`. Diagnostics expose kinsoku break opportunities, and balanced fixes use dynamic-programming wrapping to reduce ragged lines without violating punctuation rules.
- **Vertical CJK diagnostics**: vertical layout plans list columns in right-to-left order and glyphs top-to-bottom with punctuation classes (`center`, `rotate`, `punct`, `normal`) plus hang/center flags for pause/stop punctuation.
- **Fallback fonts**: per-style fallback chains and config fallback chains are included in Typography QA and PSD handoff metadata.
- **Manga effects**: stroke, padding, shadow, opacity, spacing, and preset fields continue to persist per style. Typography QA now suggests/applies a conservative outline when light text would be hard to read on a light manga bubble/background.

## Workflow improvements

- **Typography QA / project fixes**: automation and UI flows can list renderer issues, then apply checked-row conservative fixes such as shrink-to-fit, vertical switch, punctuation normalization, padding increase, fallback chain, and contrast stroke.
- **Headless export**: the local automation API `export` route supports rendered batch export, current-page render, structured OCR JSON, and PSD handoff without file dialogs.
- **Export manifest**: batch export writes `export_manifest.json` with exported paths, missing pages, page completion states, export options, and warnings; successful batch exports are also marked `Exported` in project page state.
- **PSD handoff diagnostics**: PSD handoff JSON now includes fit mode, line-break strategy, fallback chain, padding, and per-layer renderer diagnostics so unsupported cases are explicit rather than silent.

## Known limitations

- Native editable PSD writing is still a handoff/JSX workflow, not a guaranteed direct PSD writer.
- Vertical OpenType alternates and Arabic shaping still depend on future optional shaping-engine work.
- Current contrast suggestions assume a light bubble/background when exact mask sampling is unavailable.
