# Koharu Issue Backlog

_Refreshed: 2026-05-17. Source: GitHub REST API because `gh` is unavailable; scanned 300 all-state Koharu issues plus topic searches for rendering/text/font/layout (143), export/layers (34), workflow/editor/batch/setup (96), API/automation (28), and feature/UX (77). This document is intentionally kept as a living implementation backlog._

## Implemented or advanced in this pass (2026-05-17)

- [#698](https://github.com/mayocream/koharu/issues/698) **Double outline** — implemented a persisted back/second outline for manga SFX and high-contrast lettering: `FontFormat.secondary_stroke_width`, `secondary_srgb`, renderer compositing, SVG editable-text handoff, text-panel controls, Rendering settings default, layout-review suggestion/action, and tests.
- [#594](https://github.com/mayocream/koharu/issues/594), [#446](https://github.com/mayocream/koharu/issues/446), [#447](https://github.com/mayocream/koharu/issues/447), [#545](https://github.com/mayocream/koharu/issues/545) — advanced clipping-safe typography by making effect-margin and fit diagnostics account for the larger of primary and back outlines.
- [#612](https://github.com/mayocream/koharu/issues/612), [#610](https://github.com/mayocream/koharu/issues/610), [#691](https://github.com/mayocream/koharu/issues/691) — extended the local automation/page-list workflow with `list_pages(include_rendering_qa=true)` so headless tools can list pages, textboxes, writing/fit modes, and current rendering issues in one call.
- [#593](https://github.com/mayocream/koharu/issues/593), [#595](https://github.com/mayocream/koharu/issues/595), [#640](https://github.com/mayocream/koharu/issues/640) — advanced style/preset UX by wiring SFX preset back outlines and project-default application into visible controls instead of hidden renderer-only state.


- Follow-up 2026-05-17: [#454](https://github.com/mayocream/koharu/issues/454), [#558](https://github.com/mayocream/koharu/issues/558), and [#698](https://github.com/mayocream/koharu/issues/698) advanced again by preserving secondary/back-outline metadata in layered PSD handoff manifests/JSX and proof metrics.
- Follow-up 2026-05-17: [#610](https://github.com/mayocream/koharu/issues/610), [#612](https://github.com/mayocream/koharu/issues/612), and [#691](https://github.com/mayocream/koharu/issues/691) advanced by allowing multi-page PSD handoff export through the local automation API.
- Follow-up 2026-05-17: shortcut/keybind search results and UX requests advanced with canonical conflict detection plus live single-key typing-safety warnings in the Keyboard Shortcuts dialog.
- Second follow-up 2026-05-17: [#117](https://github.com/mayocream/koharu/issues/117), [#594](https://github.com/mayocream/koharu/issues/594), and [#509](https://github.com/mayocream/koharu/issues/509) advanced with line-break quality diagnostics, proof metrics, UI status, layout-review `balance_lines`, and rendering-QA auto-fix support for widows/ragged/kinsoku-problem wraps.
- Second follow-up 2026-05-17: [#602](https://github.com/mayocream/koharu/issues/602) and [#583](https://github.com/mayocream/koharu/issues/583) advanced with an RTL expand-to-fill cap so Arabic/Hebrew auto-fit no longer inflates dramatically in large boxes.
- Second follow-up 2026-05-17: [#535](https://github.com/mayocream/koharu/issues/535), [#568](https://github.com/mayocream/koharu/issues/568), and [#610](https://github.com/mayocream/koharu/issues/610) advanced by adding persisted batch export filename templates in Settings, the export dialog, and the automation API.

## Issue backlog

| Issue | Title | Category | Labels | Maps to Pro | Implemented in Pro | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#698](https://github.com/mayocream/koharu/issues/698) | [Feature request]: Double outline | Text rendering / typography / fonts | none | Yes | Yes/advanced this pass | High | Back/second outline is persisted per textbox/style, exposed in the text panel and default Rendering settings, rendered in Qt, exported in SVG handoff, and proposed by layout review for SFX-style blocks. Native PSD editable double-stroke fidelity remains deferred to PSD writer work. |
| [#602](https://github.com/mayocream/koharu/issues/602) | Issues with Arabic Text Rendering and Pipeline Export | Vertical CJK / RTL / punctuation | bug, renderer | Yes | Partial | High | Existing RTL writing mode/export diagnostics remain; full HarfBuzz shaping still deferred because it needs optional dependency and UI compatibility validation. |
| [#545](https://github.com/mayocream/koharu/issues/545) | Intermittent text rendering overlap and restricted rendering area at 125% scaling | Text fitting / layout / overflow | bug, renderer, windows, dpi scaling | Yes | Partial/advanced this pass | High | Back-outline margin now participates in fit/QA calculations; DPI-specific Qt scene behavior remains next-batch validation. |
| [#454](https://github.com/mayocream/koharu/issues/454) | PSD text rendering issues if font is changed after export | PSD/export/layers | bug, renderer, psd, font, export | Yes | Partial | High | SVG/PSD handoff manifests preserve more style metadata; native PSD editable text with double-outline effects deferred. |
| [#535](https://github.com/mayocream/koharu/issues/535) | Custom filename on export & include unrendered pages in export | PSD/export/layers | feature request, export | Yes | Yes/advanced this pass | High | Batch export now includes unrendered-page fallbacks and persisted/API filename templates with safe token substitution. |
| [#583](https://github.com/mayocream/koharu/issues/583) | auto font size exaggerates the font size, in arabic | Vertical CJK / RTL / punctuation | bug, renderer, rtl, arabic, text layout | Yes | Partial/advanced this pass | High | RTL expand-to-fill is now capped to avoid exaggerated Arabic/Hebrew sizing; full shaping still deferred. |
| [#509](https://github.com/mayocream/koharu/issues/509) | Vertical text layout wraps to next column despite sufficient box size | Vertical CJK / RTL / punctuation | bug, renderer, vertical text | Yes | Yes/advanced previous pass | High | Vertical layout/kinsoku/tate-chu-yoko exists; continue with punctuation tuning. |
| [#597](https://github.com/mayocream/koharu/issues/597) | Add manual text direction toggle to render controls | Vertical CJK / RTL / punctuation | none | Yes | Yes | High | Writing-mode controls are present; this pass keeps layout review/export aware of richer style state. |
| [#117](https://github.com/mayocream/koharu/issues/117) | Intelligent Word Splitting for Enhanced Text Rendering | Text fitting / layout / overflow | renderer, feature request, text layout | Yes | Partial/advanced this pass | Medium | Line-break quality now detects widows, kinsoku violations, and ragged wraps and can trigger balance-line fixes; hyphenation dictionaries remain deferred. |
| [#640](https://github.com/mayocream/koharu/issues/640) | Render with fixed font options | Text rendering / typography / fonts | none | Yes | Yes/advanced this pass | High | Default project style application now includes secondary outline and shadow defaults. |
| [#594](https://github.com/mayocream/koharu/issues/594) | Advanced text formatting (gradients, line spacing, kerning) | Text rendering / typography / fonts | renderer, feature request, font | Yes | Yes/advanced this pass | High | Added double-outline effect path plus diagnostics; OpenType shaping remains deferred. |
| [#593](https://github.com/mayocream/koharu/issues/593) | Text presets | Text rendering / typography / fonts | ui, feature request, font | Yes | Yes/advanced this pass | High | `sfx_bold` preset now carries a back outline. |
| [#595](https://github.com/mayocream/koharu/issues/595) | Font UX improvements | Text rendering / typography / fonts | ui, feature request, font | Yes | Partial/advanced this pass | Medium | Visible back-outline controls reduce round-trips; font local names still deferred. |
| [#558](https://github.com/mayocream/koharu/issues/558) | PSD export rasterizes all text layers | PSD/export/layers | bug, psd, export, text layer | Yes | Partial | High | SVG editable text and PSD handoff are improved; true PSD editable text authoring remains deferred. |
| [#610](https://github.com/mayocream/koharu/issues/610) | Bulk process folders/CBZs as parent/child projects | Batch/project workflow | none | Yes | Partial/advanced this pass | Medium | Automation list-pages QA reduces batch triage clicks; true parent/child project queue deferred. |
| [#612](https://github.com/mayocream/koharu/issues/612) | API/headless workflow requests | Automation/API/headless/MCP | inferred from topic search | Yes | Yes/advanced this pass | Medium | `list_pages` can now include rendering QA for headless page/textbox inspection. |
| [#691](https://github.com/mayocream/koharu/issues/691) | Resume Interrupted Task + Quick Image Navigation | UI/UX/editor workflow | none | Yes | Partial/advanced this pass | Medium | QA-enriched page listing complements previous next-rendering-issue navigation; pause/resume queue remains deferred. |

## Deferred high-value items

- Native editable PSD text/effects (#454/#558/#587): deferred because the current PSD handoff is manifest/helper-layer based and a true PSD text writer needs cross-app validation.
- HarfBuzz/OpenType shaping for Arabic/Indic scripts (#602/#583): deferred to avoid adding a hard dependency without PyQt5/PyQt6 packaging validation.
- Full queue pause/cancel/model unload (#705/#691/#610): deferred because it crosses pipeline thread lifecycle and model-cache ownership.

## Next batch candidates

1. Native PSD editable translated text with helper layers and unsupported-effect warnings.
2. Optional HarfBuzz/uharfbuzz shaping path for RTL scripts with graceful fallback.
3. Queue pause/resume/cancel and model unload controls for VRAM recovery.
4. Font local-name/family-weight picker improvements.
5. DPI/125% scaling visual regression diagnostics for text bounds.
6. Parent/child CBZ project queue.
7. Provider/model offline cold-start diagnostics.
8. Vertical punctuation pair tuning for brackets and prolonged marks.
