# Koharu Gap Analysis and Implementation Audit

_Last audited: 2026-05-06 against BallonsTranslator-Pro current branch and public `mayocream/koharu` issues via GitHub REST API (649 all-state issues/PRs scanned)._

## Audit scope

This audit compares BallonsTranslator-Pro with Koharu's local-first manga translation workflow: detect/bubble/mask → OCR → inpaint → translate → render → export, with special attention to advanced manga lettering, vertical/RTL text, font fallback, constrained fitting, layered PSD handoff, automation/API, runtime settings, batch workflow, and issue-requested UX improvements.

Before this implementation pass, the repository already had substantial Koharu-inspired foundations: per-textbox writing modes, fit modes, line-break strategies, fallback font chains, manga presets, vertical CJK controls, layout review agent, local automation API, structured OCR export, page completion state, batch styling, and layered PSD handoff. This pass therefore extends existing systems rather than duplicating them.

## Current status by workstream

| Workstream | Status after this pass | Implemented / newly advanced | Partial / deferred |
| --- | --- | --- | --- |
| Advanced manga text rendering / formatting | **Partially implemented, advanced this pass** | Added renderer-independent line-break opportunity diagnostics, vertical-RL glyph layout plans, punctuation center/rotate/hang metadata, fit diagnostics with overflow axes/recommended actions, contrast-aware manga effect suggestions, QA/fix support for horizontal CJK in tall bubbles, vertical punctuation normalization, and low-contrast stroke suggestions. | Full HarfBuzz/ICU shaping, OpenType vertical alternates, DP hyphenation, mask-aware squeezing, and visual preview thumbnails remain deferred. |
| Layout review agent | **Implemented/verified foundation; advanced diagnostics** | Existing selected/page review and API fixes now consume richer rendering diagnostics through QA/API paths; project fixes can shrink, switch writing mode, normalize vertical punctuation, increase padding, apply fallback chains, and add contrast stroke. | Needs render-after-fix screenshot verification and a visual approval queue. |
| PSD/export improvements | **Partially implemented, advanced this pass** | Batch export writes `export_manifest.json` with paths, missing pages, completion states, options, and warnings. Automation `export` now supports dialog-free rendered batch/current page/structured OCR/PSD handoff. PSD handoff text layers now include fit mode, line-break strategy, padding, fallback chain, and per-layer rendering diagnostics/warnings. | Native PSD editable text writing and streaming ZIP/CBZ export remain deferred. |
| Settings/config polish | **Partially implemented** | Existing Rendering/Text Formatting settings persist writing mode, fit mode, fallback fonts, diagnostics toggles, default font, and effect defaults. New diagnostics are data-compatible with those settings. | Google/web font UX and font favorites/localized names remain deferred. |
| Keyboard/tool UX | **Implemented foundation** | Existing shortcuts have conflict detection and single-key typing guards; no new ambiguous QAction/QShortcut owners were introduced. | Text block drag-to-reorder and deeper shortcut ownership audit remain next candidates. |
| Automation/API/headless | **Implemented, advanced this pass** | Existing local API was extended semantically: `export` can run batch rendered export, current-page render, structured OCR, and PSD handoff without file dialogs; `list_rendering_issues` now returns full Typography QA blocks/rows with suggestions and fit/vertical diagnostics. | Event streaming/progress callbacks for long API operations remain deferred. |
| Workflow enhancements from Koharu issues | **Partially implemented, advanced this pass** | Export manifests reduce uncertainty after batch exports; headless export route removes dialog clicks; Typography QA/project fixes reduce manual per-textbox lettering cleanup; PSD handoff warnings expose unsupported cases. | Parent/child CBZ processing, reorder workflow, and setup wizard remain deferred. |

## Newly implemented in this pass (2026-05-06)

| Area | Feature | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Text rendering / line breaking | Added `line_break_opportunities` so kinsoku bans and CJK/word break reasons are inspectable by UI/API/review code. | #624 advanced typesetting / kinsoku | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Vertical CJK / punctuation | Added `vertical_layout_plan` with top-to-bottom/right-to-left columns and glyph metadata for center/rotate/hang punctuation. | #598/#597 manual text direction controls, #624 vertical typesetting | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Fitting / overflow diagnostics | Extended `TextRenderDiagnostics` with overflow axes and recommended actions; fitting now reports balance/vertical-punctuation actions. | #630 centering/fitting inconsistencies, #649/#640 fixed font options | `utils/text_rendering.py`, `utils/rendering_qa.py` |
| Manga text effects | Added contrast ratio and conservative stroke/shadow suggestions for low-contrast lettering; project rendering fixes can apply a minimum stroke color/width. | #649/#640/#630 final lettering quality | `utils/text_rendering.py`, `utils/rendering_qa.py`, `tests/test_rendering_qa.py` |
| Typography QA / project fixes | Rendering QA now flags horizontal CJK in tall bubbles, vertical punctuation needing normalization, low-contrast no-effect text, and includes vertical plans/line-break opportunities/fit diagnostics in reports. | #624/#598/#649/#648 | `utils/rendering_qa.py`, `tests/test_rendering_qa.py` |
| Automation API / headless workflow | `list_rendering_issues` returns full QA blocks and flattened rows; `export` supports rendered batch/current page/structured OCR/PSD handoff without dialogs. | #651 RPC workflows, #612/#613 batch automation, #626 export workflow | `ui/mainwindow.py` |
| Export workflow | Batch export writes a stable `export_manifest.json` recording exported paths, missing pages, completion states, options, and warnings. | #626 streaming/export status, #651 page state | `utils/export_manifest.py`, `ui/mainwindow.py` |
| PSD handoff | Editable text metadata now includes fit mode, line-break strategy, padding, fallback chain, rendering diagnostics, and per-layer warnings. | #587/#558 PSD fidelity, #602 export of complex text | `utils/layered_psd_export.py` |
| Living issue pipeline | Refreshed the curated Koharu backlog from multiple issue pages and categorized implemented/deferred items without keeping a large raw snapshot in-tree. | Required ongoing harvesting | `docs/KOHARU_ISSUE_BACKLOG.md` |


## Follow-up implementation pass after PR review (2026-05-06)

| Area | Feature | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Text rendering / line breaking | Added `optimal_kinsoku_wrap`, a dynamic-programming balanced wrapper that keeps CJK punctuation rules while reducing ragged lines and one-character danglers. | #624 advanced typesetting / kinsoku | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Typography QA UI/workflow | Typography QA preview now has a checked-row fix queue instead of all-or-nothing application, and backend fixes honor selected page/textbox/action rows. | #649/#648/#640 global/fixed formatting with user approval | `ui/typography_qa_dialog.py`, `utils/rendering_qa.py`, `ui/mainwindow.py`, `tests/test_rendering_qa.py` |
| Export workflow | Batch rendered exports now mark successfully exported pages as `Exported`, save completion state, refresh page-list styling, and include the count in returned API/UI status. | #651 page completion state, #626 export status | `ui/mainwindow.py`, `utils/export_manifest.py`, `tests/test_export_manifest.py` |
| Documentation hygiene | Removed the large raw issue snapshot from the repo and kept the living curated backlog as the durable issue-harvest artifact. | Required ongoing backlog maintenance | `docs/KOHARU_ISSUE_BACKLOG.md`, `docs/KOHARU_GAP_ANALYSIS.md` |

## Progress audit

| Capability | Implemented | Partially implemented | Missing | Newly implemented this pass | Deferred with reason |
| --- | --- | --- | --- | --- | --- |
| Per-textbox writing mode persistence/UI | yes | — | — | QA can now detect explicit horizontal CJK in tall bubbles and auto-fix to vertical. | Preview UI for suggested switch deferred. |
| Real vertical CJK layout | partial | yes | full OpenType vertical alternates | Vertical layout plan and punctuation metadata. | Qt renderer still lacks HarfBuzz/ICU glyph substitution. |
| Punctuation-aware vertical layout | partial | yes | complete glyph substitution | Center/rotate/hang diagnostics and normalization QA. | Exact glyph-specific offsets remain renderer-specific. |
| Script-aware wrapping/fitting | partial | yes | full hyphenation | Break-opportunity diagnostics, richer fit diagnostics, and dynamic-programming balanced kinsoku wrapping. | Hyphenation dictionaries/dependency selection deferred. |
| Font fallback | partial | yes | localized font UX | QA/export paths now include fallback chain and missing glyph context. | Font favorites/localized names deferred. |
| Manga effects | partial | yes | live preview | Contrast-aware stroke/shadow suggestions and conservative project fixes. | Background sampling from actual bubble mask deferred. |
| Precise bounds/placement | partial | yes | render-after-fix diff | Overflow axes/actions added. | Screenshot verification deferred. |
| Renderer diagnostics overlay/API | partial | yes | complete overlay | API/QA returns vertical plans, break opportunities, fit diagnostics. | More canvas overlay drawing deferred. |
| Rendering presets | yes | partial | richer preset previews | Existing presets are used by review/batch paths; diagnostics now explain effects. | Icon/thumbnail previews deferred. |
| Layout review selected/page/settings | yes | partial | provider screenshot recheck | Diagnostics fields are now richer for review/API. | Visual approval queue deferred. |
| PSD/export handoff | partial | yes | native PSD writer | Export manifests and richer text-layer diagnostics. | True PSD editable writer deferred. |
| Local automation/API | yes | partial | progress stream | Dialog-free exports and structured rendering issue rows. | Server-sent events/progress deferred. |
| Workflow polish | partial | yes | setup wizard, reorder | Export status manifests, exported-page auto-state, and checked-row Typography QA fixes reduce manual work. | Reorder/CBZ/setup flows deferred. |

## Issue-inspired items implemented this pass

- **#624 Advanced typesetting / kinsoku**: explainable break opportunities, dynamic-programming balanced wrapping, and fit diagnostics.
- **#598/#597 Manual text direction controls**: vertical-RL layout plan and QA/fix for tall CJK bubbles.
- **#649/#648/#640 Global/fixed formatting**: project-level conservative typography fixes now cover vertical switch, punctuation normalization, fallback chain, padding, shrink, and contrast stroke, with checked-row preview application.
- **#626 Export workflow/status**: batch export manifests, exported-page completion-state updates, and headless export route.
- **#587/#558/#602 PSD/export fidelity**: PSD handoff now records renderer diagnostics and style fields that affect editable text reconstruction.
- **#651/#612 automation/batch**: API export and issue listing are more useful for headless workflows.

## High-value deferred items and reasons

| Deferred item | Issues | Reason |
| --- | --- | --- |
| Mask-aware collision detection and squeezing | #637 | Requires bubble/mask geometry integration and likely renderer iteration; too large for this pass after QA/API/export work. |
| Native PSD editable text writer | #587/#558/#602 | Python PSD text-layer writing support is limited; current safe handoff avoids fake/silent rasterization. |
| HarfBuzz/ICU shaping and OpenType vertical alternates | #602/#213/#624 | Needs optional dependency design and Qt compatibility testing. |
| Text block drag-to-reorder | #601 | Needs scene/list model ordering, undo commands, and export/read-order validation. |
| Streaming ZIP/CBZ and parent-child batch projects | #626/#610 | Requires batch archive IO design; manifest/status was prioritized first. |
| Provider/model setup wizard | #625/#652 | Existing diagnostics are present; a guided wizard needs broader UI work. |

## Next batch candidates

1. Mask-aware text squeezing/collision detection against bubble masks (#637).
2. Previewed Typography QA fix queue with before/after thumbnails and selective apply (#649/#648/#630).
3. HarfBuzz/ICU shaping experiment for Arabic joining and vertical OpenType alternates (#602/#213/#624).
4. Native PSD text layer writer or stronger Photoshop/GIMP handoff validator (#587/#558).
5. Text block drag-to-reorder with undo/read-order/export validation (#601).
6. Streaming ZIP/CBZ export and parent/child project batch processing (#626/#610).
7. Setup/model recommendation wizard for OCR/inpainting/provider failures (#625/#652).
8. Runtime GPU memory profiles and fallback device policy (#638/#600).
