# Koharu Gap Analysis and Implementation Audit

_Last audited: 2026-05-07 against BallonsTranslator-Pro current branch and public `mayocream/koharu` issues via GitHub REST API (652 all-state issues/PRs scanned, including #660 through #1)._

## Audit scope

This audit compares BallonsTranslator-Pro with Koharu's local-first manga translation workflow: detect/bubble/mask → OCR → inpaint → translate → render → export. The priority remains advanced manga lettering, vertical/RTL text, font fallback, constrained fitting, layered export handoff, local automation/API, runtime settings, batch workflow, and issue-requested UX improvements.

The repository already had substantial Koharu-inspired foundations before this pass: per-textbox writing modes, fit modes, line-break strategies, fallback font chains, manga presets, vertical CJK controls, layout review agent, local automation API, structured OCR export, page completion state, batch styling, and layered PSD handoff. Current work extends those systems instead of duplicating them.

## Newly implemented in this pass (2026-05-07)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Reading-order workflow | Added configurable default textbox reading order (`auto`, manga RTL columns, LTR rows, top-to-bottom/webtoon), shared reading-order sorting utilities, structured OCR source-index preservation, and a local API `list_pages` route that returns pages/textboxes in resolved reading order. | #660 reading order dropdown/LTR order, #612 granular workflow/API, #613 cross-page batch handoff | `utils/text_rendering.py`, `utils/config.py`, `ui/configpanel.py`, `utils/structured_ocr_export.py`, `ui/mainwindow.py`, `tests/test_text_rendering.py`, `tests/test_structured_ocr_export.py` |
| Effect-aware fit diagnostics | Fit-to-box now accounts for shadow radius/offset as well as stroke/padding, and diagnostics include recommended box size plus a scale hint for resize/review actions. | #630 centering/fitting inconsistencies, #637 mask/collision-aware sizing groundwork, #640 fixed font options | `utils/text_rendering.py`, `utils/rendering_qa.py`, `tests/test_text_rendering.py` |
| QA/API export metadata | Rendering QA now exposes recommended box size/scale hints, while structured OCR exports include writing mode, fit mode, line-break strategy, source order, and resolved reading order for headless agents. | #649/#648 project-wide text fixes, #660 reading order, #612 API workflow | `utils/rendering_qa.py`, `utils/structured_ocr_export.py`, `ui/mainwindow.py` |
| Archive export workflow | Batch export dialog now supports manga-reader CBZ archives in addition to ZIP, with clearer archive wording and backend status messages. | #626 streaming ZIP export, #610 bulk comic/archive workflow, #591 export/interoperability requests | `ui/export_dialog.py`, `ui/mainwindow.py` |

## Latest implementation pass (2026-05-07 continued: reusable presets and font workflow)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| User-saved manga presets | Added persisted custom manga lettering presets built from the current textbox/style, with sanitization for font, size, stroke, colors, spacing, writing mode, fit mode, line breaks, padding, opacity, fallback chain, and font weight. Presets can now be imported/exported as JSON packs with missing-font diagnostics. | #593 text presets, #594 advanced formatting, #595 font UX | `utils/text_rendering.py`, `utils/rendering_preset_io.py`, `utils/config.py`, `ui/text_panel.py`, `ui/fontformat_commands.py`, `utils/text_style_batch.py`, `tests/test_text_rendering.py`, `tests/test_text_style_batch.py` |
| Preset UI and batch/API parity | The text panel now has connected **Save/Import/Export preset** actions, built-in + custom presets appear in the preset picker, batch style override can apply custom presets, and automation can list/save/import/export/delete presets with `rendering_presets`. | #593 text presets, #612 automation/API workflow, #648 project-wide style changes | `ui/text_panel.py`, `ui/mainwindow.py`, `utils/text_style_batch.py`, `utils/rendering_preset_io.py` |
| Recent font workflow | Font choices are remembered as recent lettering fonts and surfaced alongside favorites, reducing repeated font-list scrolling during manga lettering. | #595 font UX improvements, #589 workflow/font requests | `utils/config.py`, `ui/text_panel.py` |
| Review/custom preset application | Layout review preset actions and direct formatting commands resolve the merged built-in/custom preset library, so saved presets are first-class renderer actions rather than text-panel-only UI state. | #594 advanced formatting, #593 presets, #649/#648 style repair | `ui/fontformat_commands.py`, `ui/scenetext_manager.py`, `utils/text_style_batch.py` |

## Latest implementation pass (2026-05-07 continued: font UX, fit quality, export workflow)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Visual-advance lettering fit | Replaced raw character-count width estimates with script/punctuation-aware visual advance units, making CJK punctuation, emoji/symbols, combining marks, and Latin text fit diagnostics closer to actual manga lettering; added conservative tracking recommendations when text is horizontally overfull. | #594 advanced formatting/kerning, #630 fit/centering inconsistencies, #640 fixed font options | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Layout review tracking fix | Layout review and Typography QA now expose/apply `tighten_letter_spacing` before more destructive shrink/resize fixes, preserving lettering weight while reducing wide-line overflow. | #594 advanced formatting, #649/#648 project-wide style repair | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `utils/rendering_qa.py` |
| Favorite lettering fonts | Added persistent favorite font chains in Rendering settings and a one-click Favorites picker/★ capture button in the text formatting panel so common manga fonts can be reused without scrolling the full font list. | #595 font UX improvements, #593 text presets | `utils/config.py`, `ui/configpanel.py`, `ui/text_panel.py` |
| Export handoff workflow | Added persistent “open output folder after batch export” control in Settings and the export dialog, reducing clicks after rendered/CBZ/ZIP/manifest/helper export. | #591/#587 export handoff/interoperability, #626 archive workflow | `utils/config.py`, `ui/configpanel.py`, `ui/export_dialog.py`, `ui/mainwindow.py` |
| Automation archive export | Extended the local automation export route with `kind=archive`, `kind=zip`, and `kind=cbz`, packing batch rendered output plus optional helper images/manifests for headless workflows. | #612 API workflow, #626 ZIP export, #610 bulk CBZ workflow | `ui/mainwindow.py` |

## Latest implementation pass (2026-05-07 continued again)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Text editor comfort / onboarding polish | Added persistent side text-editor viewport top padding with a live Config control, fixing the first/selected visible textbox feeling pressed into the top edge while preserving compact list density even after scrolling. | #519 workflow/editor polish, #612 fewer-friction setup/API workflow | `ui/textedit_area.py`, `ui/configpanel.py`, `utils/config.py` |
| Script-aware uppercase lettering | Replaced raw `.upper()` usage in translation post-processing and selected-block uppercase actions with dependency-free locale/script-aware casing that preserves CJK/RTL text and handles Turkish/Azeri dotted-i. | #572/#567 universal locale-aware uppercase conversion | `utils/text_rendering.py`, `ui/mainwindow.py`, `tests/test_text_rendering.py` |
| Ink-safe typography diagnostics | Renderer fit diagnostics now report `ink_clip_risk` plus `preset_suggestion`; Typography QA and layout review can propose/apply padding and preset actions when strokes/shadows may clip or geometry suggests SFX/caption/vertical presets. | #594 advanced formatting, #630 centering/fitting, #649/#648 project-wide style repair | `utils/text_rendering.py`, `utils/rendering_qa.py`, `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `tests/test_text_rendering.py` |
| Configurable vertical punctuation behavior | Vertical layout plans now carry and honor runtime `rotate_latin` and `punctuation_hang` toggles, and QA forwards Config settings into the plan metadata for API/export parity. | #624 advanced typesetting/kinsoku, #602/#583 RTL/CJK layout fidelity | `utils/text_rendering.py`, `utils/rendering_qa.py`, `tests/test_text_rendering.py` |
| Automation onboarding workflow | Added `POST /recent_projects` with path/existence/project-json metadata so local tools and first-run helpers can list reopenable projects before running pipeline actions. | #612 granular workflow/API, #610 project workflow, #519 fewer manual processing steps | `ui/mainwindow.py` |


## Follow-up implementation pass (2026-05-07 continued)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Batch style repair | Extended Batch Text Style Override and added shared `apply_text_style_batch` backend/API support, including auto-sized-only targeting, stroke/shadow controls, and fit min/max clamps for project-wide lettering cleanup. | #649 global Auto font override, #648 project-wide alignment/style changes, #640 fixed font options | `ui/mainwindow.py`, `utils/text_style_batch.py`, `tests/test_text_style_batch.py` |
| Vertical punctuation diagnostics | Added punctuation offset/rotation/scale hints and bracket-pair diagnostics to vertical layout plans, plus soft-hyphen-aware Latin token splitting. | #624 advanced typesetting/kinsoku, #602/#583 RTL/text layout quality | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Export helper images | Batch export can include clean inpainted pages and masks in `clean/` and `masks/` subfolders, and automation can request the same via `include_intermediate`. | #626 export workflow, #587/#558 PSD/layer handoff issues | `ui/export_dialog.py`, `ui/mainwindow.py`, `docs/RENDERING_TEXT_FORMATTING.md` |
| SVG editable text handoff | Added SVG export with editable translated text elements, vertical/RTL attributes, per-layer diagnostics, menu wiring, and API export kind. | #591 SVG/XCF export request, #587/#558 layer fidelity, #602 RTL export | `utils/svg_text_export.py`, `ui/mainwindowbars.py`, `ui/mainwindow.py`, `tests/test_svg_text_export.py` |
| Layout review action completion | Review now consumes recommended fit box diagnostics and can propose/apply targeted box resize, vertical punctuation normalization, RTL alignment, contrast stroke, and low quality-score flags. | #637 mask/collision-aware sizing groundwork, #630 centering/fitting, #624 vertical punctuation, #602/#583 RTL | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `utils/rendering_qa.py`, `tests/test_layout_review_agent.py` |

## Previous 2026-05-07 implementation passes retained

- Google Fonts installer dialog with error reporting and immediate Qt registration.
- Clearer Settings/right-side text-panel ownership for project defaults vs current-selection overrides.
- Live lettering diagnostics in the text panel, typography QA filtering/action summaries, tate-chu-yoko hints, effect-aware safe inner bounds, and side-panel text-box reorder discoverability.
- Per-style fit-size clamps and UI controls.

## Current status by workstream

| Workstream | Status | Implemented | Partial / deferred |
| --- | --- | --- | --- |
| Advanced manga text rendering / formatting | **Partially implemented and advanced this pass** | Writing modes, vertical CJK plans, kinsoku/balanced wrapping, tate-chu-yoko hints, configurable latin-rotation/punctuation-hang metadata, vertical punctuation offset/rotation/bracket hints, fallback chains, favorite lettering fonts, manga presets, visual-advance punctuation-aware fit estimates, conservative letter-spacing review fixes, script-aware uppercase conversion, fit clamps, ink-clip risk diagnostics, effect-aware safe bounds, shadow-aware fit diagnostics, recommended resize hints, and live QA. | Full HarfBuzz/ICU shaping, OpenType vertical alternates, native kerning tables, and true ink-bound measurement remain deferred pending optional dependency/runtime design. |
| Layout review agent | **Implemented/verified and advanced** | Selected/page review, heuristic fallback, settings, action application, recommended-box resize, vertical punctuation normalization, RTL alignment, contrast stroke, and richer QA metadata are present. | Visual before/after previews and true mask-aware squeezing still need rendered snapshots/mask geometry. |
| PSD/export | **Partially implemented and improved** | Layered PSD handoff manifests/Photoshop JSX, editable text metadata, export manifests, current-page render API, ZIP/CBZ batch archive workflow, API archive export, optional clean/mask helper image export, and persistent open-output-folder workflow. | Native PSD text layer writer and streaming archive writer remain deferred. |
| Settings/config | **Implemented and improved** | Dedicated rendering defaults include font, writing mode, fit mode, line break, reading order, effects, overflow warnings, diagnostics overlay, script fallback chains, recent/favorite fonts, and user-saved manga lettering presets. | Per-project profile import/export and model/provider wizard remain future work. |
| Keyboard/tool UX | **Partially implemented** | Existing shortcut conflict tests and text-field guard behavior remain in place; no shortcut ambiguity changes were introduced in this pass. | Broader one-owner QAction/QShortcut cleanup is deferred to a focused shortcut pass. |
| Automation/API/headless | **Implemented and improved** | Existing local API supports project open/run/edit/export/review/render/QA/fixes; `list_pages` exposes page states and ordered textboxes, `recent_projects` exposes reopenable project metadata for agents/onboarding flows, and archive/ZIP/CBZ export is now callable headlessly. | Event streaming/progress subscription and MCP server parity are deferred. |
| Workflow enhancements from issues | **Improved this pass** | Reading-order-aware exports/API reduce manual agent cleanup; ZIP/CBZ archive export reduces comic delivery steps; the side text editor now has configurable comfort padding; `recent_projects` helps agents/onboarding flows reopen projects safely. | Bulk parent/child CBZ project processing and provider/model setup wizard are next candidates. |

## Issue-inspired items implemented in this pass

- **#660 Reading Order Dropdown + LTR Reading Order**: implemented a BT-Pro equivalent through rendering settings, shared sorting utilities, structured OCR export order, and automation `list_pages` output.
- **#626 / #610 archive/bulk comic workflow**: implemented CBZ archive export alongside ZIP to support manga-reader delivery without manual renaming.
- **#630 / #640 fitting/fixed font controls**: extended fit diagnostics to include shadow-aware bounds and actionable resize recommendations.
- **#649 / #648 project-wide style cleanup**: extended batch style override and automation so users can adjust font/style/fitting fields across current page or project while optionally limiting changes to auto-sized boxes.
- **#587 / #558 helper export handoff**: batch export can now include clean/inpainted pages and masks beside rendered pages for downstream PSD/GIMP review.
- **#591 SVG/XCF interop request**: added an SVG text handoff path with editable translated text, vertical/RTL attributes, and companion diagnostics manifest.
- **#637/#630/#624 review fixes**: layout review can now turn fit diagnostics into targeted resize actions and apply vertical punctuation, RTL alignment, contrast-stroke, ink-safe padding, and preset-suggestion fixes.
- **#572/#567 locale uppercase**: uppercase post-processing and context actions are now script-aware instead of applying raw uppercase to every character.
- **#612/#519 workflow polish**: automation can list recent projects and the side text editor has configurable breathing room for faster editing.
- **#594/#595 typography workflow**: visual-advance fit diagnostics, letter-spacing review fixes, persistent favorite/recent lettering fonts, and custom saved presets improve final lettering and reduce font-selection clicks.
- **#593/#612 preset workflow**: custom manga presets are saved from the text panel, reused in batch style override/layout review, and exposed to automation clients for headless style consistency.
- **#626/#610/#591 export workflow**: API archive export and open-output-folder settings make rendered/CBZ/ZIP handoff faster and more discoverable.

## Deferred with reasons

| Deferred item | Reason |
| --- | --- |
| #637 mask-aware collision detection/squeezing | Needs access to bubble/mask geometry in the renderer/review loop and visual verification; this pass added resize-hint groundwork. |
| #587 native editable PSD fidelity | Current handoff preserves metadata and Photoshop JSX, but native PSD text-layer writing needs a dedicated library/design pass. |
| #602/#583 Arabic shaping/export fidelity | Correct Arabic joining/bidi export needs optional shaping dependencies and renderer integration. |
| #626 streaming ZIP writer | Current archive export is practical for normal projects; true streaming should be added with progress/cancel plumbing. |
| #625 model recommendation wizard | Useful workflow request, but lower priority than lettering/export/API work for this pass. |

## Next batch candidates

1. Shared per-project preset packs with thumbnail previews for lettering teams (#593/#595).
2. Weight/style-aware font browser with localized family names and deeper missing-font diagnostics (#595).
3. True ink-bound glyph measurement / OpenType shaping for clipping-free advanced formatting (#594/#572/#624).
4. Mask-aware collision detection/squeezing using bubble masks and text bounds (#637).
5. Before/after thumbnails for checked-row Typography QA fixes (#649/#648/#630).
6. Native editable PSD text writer or stronger Photoshop/GIMP handoff validation (#587/#558/#602).
7. Canvas-side text block drag-to-reorder with undo/read-order preview (#601/#660).
8. HarfBuzz/ICU shaping experiment for Arabic joining and vertical OpenType alternates (#602/#583/#213/#624).
9. Streaming ZIP/CBZ export with progress/cancel and parent/child batch processing (#626/#610).
10. Provider/model setup wizard with OCR model recommendation and failure retry diagnostics (#625/#652).
11. Runtime GPU memory profiles and safer device fallback guidance (#638/#600).
