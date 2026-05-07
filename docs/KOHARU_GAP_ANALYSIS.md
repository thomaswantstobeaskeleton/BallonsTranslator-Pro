# Koharu Gap Analysis and Implementation Audit

_Last audited: 2026-05-07 against BallonsTranslator-Pro current branch and public `mayocream/koharu` issues via GitHub REST API (651 all-state issues/PRs scanned, including #660 through #1)._

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
| Advanced manga text rendering / formatting | **Partially implemented and advanced this pass** | Writing modes, vertical CJK plans, kinsoku/balanced wrapping, tate-chu-yoko hints, vertical punctuation offset/rotation/bracket hints, fallback chains, manga presets, fit clamps, effect-aware safe bounds, shadow-aware fit diagnostics, recommended resize hints, and live QA. | Full HarfBuzz/ICU shaping, OpenType vertical alternates, and true ink-bound measurement remain deferred pending optional dependency/runtime design. |
| Layout review agent | **Implemented/verified and advanced** | Selected/page review, heuristic fallback, settings, action application, recommended-box resize, vertical punctuation normalization, RTL alignment, contrast stroke, and richer QA metadata are present. | Visual before/after previews and true mask-aware squeezing still need rendered snapshots/mask geometry. |
| PSD/export | **Partially implemented and improved** | Layered PSD handoff manifests/Photoshop JSX, editable text metadata, export manifests, current-page render API, ZIP/CBZ batch archive workflow, and optional clean/mask helper image export. | Native PSD text layer writer and streaming archive writer remain deferred. |
| Settings/config | **Implemented and improved** | Dedicated rendering defaults include font, writing mode, fit mode, line break, reading order, effects, overflow warnings, diagnostics overlay, and script fallback chains. | Per-project profile presets and model/provider wizard remain future work. |
| Keyboard/tool UX | **Partially implemented** | Existing shortcut conflict tests and text-field guard behavior remain in place; no shortcut ambiguity changes were introduced in this pass. | Broader one-owner QAction/QShortcut cleanup is deferred to a focused shortcut pass. |
| Automation/API/headless | **Implemented and improved** | Existing local API supports project open/run/edit/export/review/render/QA/fixes; new `list_pages` route exposes page states and ordered textboxes for agents. | Event streaming/progress subscription and MCP server parity are deferred. |
| Workflow enhancements from issues | **Improved this pass** | Reading-order-aware exports/API reduce manual agent cleanup; ZIP/CBZ archive export reduces comic delivery steps. | Bulk parent/child CBZ project processing and provider/model setup wizard are next candidates. |

## Issue-inspired items implemented in this pass

- **#660 Reading Order Dropdown + LTR Reading Order**: implemented a BT-Pro equivalent through rendering settings, shared sorting utilities, structured OCR export order, and automation `list_pages` output.
- **#626 / #610 archive/bulk comic workflow**: implemented CBZ archive export alongside ZIP to support manga-reader delivery without manual renaming.
- **#630 / #640 fitting/fixed font controls**: extended fit diagnostics to include shadow-aware bounds and actionable resize recommendations.
- **#649 / #648 project-wide style cleanup**: extended batch style override and automation so users can adjust font/style/fitting fields across current page or project while optionally limiting changes to auto-sized boxes.
- **#587 / #558 helper export handoff**: batch export can now include clean/inpainted pages and masks beside rendered pages for downstream PSD/GIMP review.
- **#591 SVG/XCF interop request**: added an SVG text handoff path with editable translated text, vertical/RTL attributes, and companion diagnostics manifest.
- **#637/#630/#624 review fixes**: layout review can now turn fit diagnostics into targeted resize actions and apply vertical punctuation, RTL alignment, and contrast-stroke fixes.

## Deferred with reasons

| Deferred item | Reason |
| --- | --- |
| #637 mask-aware collision detection/squeezing | Needs access to bubble/mask geometry in the renderer/review loop and visual verification; this pass added resize-hint groundwork. |
| #587 native editable PSD fidelity | Current handoff preserves metadata and Photoshop JSX, but native PSD text-layer writing needs a dedicated library/design pass. |
| #602/#583 Arabic shaping/export fidelity | Correct Arabic joining/bidi export needs optional shaping dependencies and renderer integration. |
| #626 streaming ZIP writer | Current archive export is practical for normal projects; true streaming should be added with progress/cancel plumbing. |
| #625 model recommendation wizard | Useful workflow request, but lower priority than lettering/export/API work for this pass. |

## Next batch candidates

1. Mask-aware collision detection/squeezing using bubble masks and text bounds (#637).
2. Before/after thumbnails for checked-row Typography QA fixes (#649/#648/#630).
3. Native editable PSD text writer or stronger Photoshop/GIMP handoff validation (#587/#558/#602).
4. Canvas-side text block drag-to-reorder with undo/read-order preview (#601/#660).
5. HarfBuzz/ICU shaping experiment for Arabic joining and vertical OpenType alternates (#602/#583/#213/#624).
6. Streaming ZIP/CBZ export with progress/cancel and parent/child batch processing (#626/#610).
7. Provider/model setup wizard with OCR model recommendation and failure retry diagnostics (#625/#652).
8. Runtime GPU memory profiles and safer device fallback guidance (#638/#600).
