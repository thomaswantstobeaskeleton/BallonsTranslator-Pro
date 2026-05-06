# Koharu Gap Analysis and Implementation Audit

_Last audited: 2026-05-05 against BallonsTranslator Pro current branch and public `mayocream/koharu` issue/API surface._

## Audit scope

This audit compares BallonsTranslator Pro against Koharu's current public direction: a local-first staged manga translation stack with provider/runtime settings, MCP/API automation, vertical/PSD-focused rendering, and recent issue themes such as page completion state, skipping satisfied pipeline work, global/fixed text styling, dictionary/glossary workflow, and bulk project processing.

Repository search before implementation found that BallonsTranslator Pro already had substantial foundations: Pipeline Insights, local automation actions, mask diagnostics, OCR crop inspection, reading-order editing, glossary guardrails, run profiles, skip ignored pages, and heuristic layout review wiring. The work below therefore completes gaps and avoids duplicating existing features.

## Existing implementations found

| Area | Existing status | Evidence in repo |
| --- | --- | --- |
| Pipeline observability | Substantially implemented: job/progress UI, warnings, provider health, engine registry, events. | `ui/pipeline_insights_widget.py`, `ui/module_manager.py` |
| Layout review agent | Partial before this PR: heuristic/provider planner existed and menu/shortcut/API entry points existed, but page-level provider application could act on stale selection and there was no human-readable report/approval UI. | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `ui/mainwindow.py` |
| Local automation API | Implemented basic command surface including `layout_review`. | `ui/mainwindow.py`, `tests/test_local_automation_api.py` |
| OCR/text QA | Implemented triage, OCR crop inspector, translation QA, auto glossary extraction. | `ui/ocr_crop_inspector_widget.py`, `utils/translation_review.py` |
| Project ops/history | Implemented initial JSON project ops dialog/protocol; not a full MCP op log. | `ui/project_ops_dialog.py`, `tests/test_project_ops_protocol.py` |
| Page skipping | Partial: skip ignored pages and skip already translated text existed; skip already-satisfied stage work did not exist for normal full runs. | `ui/mainwindow.py`, `ui/configpanel.py` |
| Page completion state | Missing before this PR. | No `completion_state` implementation before this audit. |
| Fixed/global text style | Partial: rich text style presets existed; no simple batch fixed font size/alignment override like Koharu issue requests. | `ui/text_style_presets.py`, `ui/scenetext_manager.py` |
| Credential separation | Partial: keyring support exists for selected flows only; full translator/LLM key migration remains open. | `utils/credential_store.py` |
| Rendering parity | Partial: vertical CJK controls and many text effects exist; PSD text/effect fidelity mode remains incomplete. | `ui/scene_textlayout.py`, export scripts |

## Highest-impact missing or partial items selected for this PR

1. **Complete layout review end-to-end**: make review auditable, provider/API safe, undo-friendly, and selection-correct.
2. **Page completion state** (Koharu issue #651): add durable user-visible page state for needs-work/translated/reviewed/exported.
3. **Skip already-satisfied pipeline pages** (Koharu issue #650): avoid rerunning pages whose persisted finish state already satisfies enabled stages.
4. **Global/fixed text style override** (Koharu issues #649/#640/#648): add a connected UI and backend to apply fixed font size, font family, alignment, and auto-fit across current page or project.
5. **Structured OCR export** (Koharu issue #555): expose page/block geometry, OCR text, translations, font hints, and workflow state for LLM/tooling integrations.
6. **Visible UI polish**: improve Pipeline Insights empty state/action affordance and add icon assets for completion/export workflows.

## Implemented in this PR

### 1) Layout review agent completion

- Added a report dialog that summarizes issues, actions, and score before applying fixes.
- Added per-block/action checkboxes so reviewers can approve only selected proposed fixes instead of applying an all-or-nothing batch.
- Changed selected/page menu flows to run provider-backed review, show the report, and apply through `SceneTextManager.apply_review_result` only after user approval.
- Fixed provider/page review application so target blocks are temporarily selected before low-level auto-fit/center/resize commands run; this prevents page-level/API review from accidentally mutating only the previously selected block.
- Enhanced the automation API `layout_review` action to return structured summary data and applied action count.

### 2) Page completion state

- Added per-page `completion_state` persisted in project `image_info` with validated states: `todo`, `translated`, `reviewed`, `exported`.
- Added page list context menu actions to set completion state for selected pages.
- Added visible page-list row styling and tooltips for completion state while preserving existing ignored-page behavior.
- Added automation API support to list/set page completion state.
- Added an optional automatic transition that marks pages as `translated` when a completed run satisfies the enabled stage finish contract.

### 3) Skip already-satisfied pipeline pages

- Added a persisted setting `skip_satisfied_pipeline_steps`.
- Added General settings UI for the setting.
- Full runs now keep already-satisfied pages out of `pages_to_process` before resetting finish codes, with Pipeline Insights telemetry for skipped pages and a status message when everything is already satisfied.

### 4) Batch fixed text styling

- Added a Pipeline Insights action for batch text style overrides.
- Added connected backend behavior for current-page or whole-project scope.
- Supports font family, fixed font size in points, alignment, and enabling auto-fit after the override.
- Persists project changes and refreshes current page scene text.

### 5) Structured OCR export

- Added a stable `ballonstranslator.structured_ocr.v1` JSON payload builder for current project state.
- Export includes page dimensions, ignored/completion state, finish code, block order, geometry, OCR/source text, translation, labels, confidence, and font hints.
- Added a Tools → Export action and automation API command for structured OCR JSON export.

### 6) UI polish and assets

- Added a clearer Pipeline Insights empty state.
- Highlighted the Layout Review Agent action as the primary review entry point.
- Added `icons/page_completion.svg` and `icons/structured_ocr_export.svg` for the new completion/export workflows.

## Remaining limitations

- Layout review provider schema is intentionally conservative and still expects OpenAI-compatible chat-completions JSON responses; richer provider adapters and model-specific response parsing remain future work.
- Page completion can auto-mark `translated`, but `reviewed` and `exported` remain manual to avoid surprising users.
- Skip already-satisfied pages uses the existing `finish_code` contract. If users change provider/model settings and want a clean rerun, they should leave the new setting off or manually rerun selected pages.
- Batch text style override changes project/block formats directly, but does not yet expose named reusable project-level style policies.
- Full translator/LLM keyring migration, PSD text/effect fidelity mode, and bubble-aware differential mask expansion remain open.

## Updated phased plan

### Completed / substantially implemented

- Layout review agent end-to-end report/apply/API flow with selectable proposed actions.
- Page completion state with persistence, page-list UI, automation API, and optional translated auto-marking.
- Skip already-satisfied full-run pages.
- Batch fixed text style override for page/project scope.
- Structured OCR JSON export for LLM/tooling workflows.
- Pipeline Insights empty/action-state polish.
- Existing foundations retained: warnings/events, local automation API, mask diagnostics, OCR crop inspector, reading-order editor, glossary guardrails, run profiles, runtime HTTP controls, engine registry.

### Next recommended tranche

1. Document and harden the local automation contract for external agents/MCP clients.
2. Add guarded automatic `reviewed`/`exported` completion transitions where user intent is unambiguous.
3. Add reusable project style policies/templates rather than one-off batch overrides.
4. Extend layout review provider adapters beyond OpenAI-compatible JSON chat completions.
5. Continue keyring migration for all translator/LLM API keys.

---

## 2026-05-05 major rendering-focused implementation pass

This pass re-audited the repository and Koharu's public issue tracker before implementation. Searches covered renderer/text layout/font/stroke/shadow/vertical text, PSD/export, layout review, config, shortcuts, and automation code paths in `ui/`, `utils/`, `tests/`, and docs.

### Koharu issue research used

Relevant public Koharu issues reviewed via the GitHub API and documentation pages:

| Koharu issue/doc | Relevant idea | BallonsTranslator-Pro outcome |
| --- | --- | --- |
| [#597](https://github.com/mayocream/koharu/issues/597) manual text direction toggle | Add user-visible writing direction control instead of relying only on heuristics. | Implemented per-textbox writing mode in the text panel and context menu; persisted in `FontFormat`; review can switch modes. |
| [#602](https://github.com/mayocream/koharu/issues/602), [#583](https://github.com/mayocream/koharu/issues/583), [#431](https://github.com/mayocream/koharu/issues/431) Arabic/RTL render quality | RTL should be detected and not treated as LTR/CJK. | Auto writing-mode resolution detects Arabic/Hebrew; diagnostics/API report resolved RTL and review can flag mode mismatch. Full shaping remains dependent on Qt. |
| [#594](https://github.com/mayocream/koharu/issues/594) advanced text formatting | Spacing/effects/kerning/gradient-style formatting should be first-class. | Added fit policies, preset-driven stroke/spacing/padding/font-size, diagnostics-aware bounds, and config defaults; existing gradient/path/warp controls remain connected. |
| [#593](https://github.com/mayocream/koharu/issues/593), [#640](https://github.com/mayocream/koharu/issues/640), [#649](https://github.com/mayocream/koharu/issues/649), [#528](https://github.com/mayocream/koharu/issues/528) presets/fixed/global font options | Users need repeatable lettering styles across projects/pages. | Added manga lettering presets with font size/stroke/spacing/alignment/writing mode/padding and default render config fields. Existing batch style override remains available. |
| [#595](https://github.com/mayocream/koharu/issues/595), [#490](https://github.com/mayocream/koharu/issues/490) font UX/fallback/family weights | Font choice/fallback needs better diagnostics. | Added per-script fallback chains, missing-glyph diagnostics, fallback chain reporting, and config UI. Weight subfamily selection is deferred. |
| [#545](https://github.com/mayocream/koharu/issues/545), [#462](https://github.com/mayocream/koharu/issues/462), [#446](https://github.com/mayocream/koharu/issues/446), [#447](https://github.com/mayocream/koharu/issues/447) bounds/overlap/cropped outlines | Bounds must account for stroke, shadow, punctuation, and DPI. | Added renderer-independent bounds estimates, overlay diagnostics, stroke/shadow expansion in text item bounds, padding quick fixes, and layout-review overflow actions. |
| [#558](https://github.com/mayocream/koharu/issues/558), [#587](https://github.com/mayocream/koharu/issues/587), [#535](https://github.com/mayocream/koharu/issues/535), [#568](https://github.com/mayocream/koharu/issues/568) PSD/export reliability | Export should preserve editable text and surface unsupported cases. | Added a layered PSD handoff export with helper layers, editable text metadata, JSX rebuild script, status feedback, and warnings instead of silent rasterization/failure. Native PSD writing remains deferred. |
| [#519](https://github.com/mayocream/koharu/issues/519), [#410](https://github.com/mayocream/koharu/issues/410), [#365](https://github.com/mayocream/koharu/issues/365), [#646](https://github.com/mayocream/koharu/issues/646) keyboard/tool polish | Tool shortcuts and brush-size shortcuts should be discoverable and safe while editing. | Added text eraser and brush-size shortcuts, visible shortcut tooltips, single-key shortcut input guards, and a duplicate-hard-conflict test. |
| Koharu docs: text rendering/vertical CJK layout and PSD export docs | Columns flow top-to-bottom/right-to-left, punctuation is normalized/recentered, bounds are tight, PSD has helper layers/editable text. | Implemented Python/PyQt equivalents where practical: vertical column diagnostics/wrapping, punctuation classes, fit-to-box, fallback diagnostics, helper-layer PSD handoff. |

### Progress audit

| Category | Implemented | Partially implemented | Missing/deferred with reason | Newly implemented in this pass |
| --- | --- | --- | --- | --- |
| Writing mode controls | Per-textbox `writing_mode` values Auto/Horizontal LTR/Vertical RL/RTL are normalized, persisted in `FontFormat`, visible in the text panel, available from context menu, and used by layout/review/API diagnostics. | Qt's underlying RTL shaping is used; complex HarfBuzz-level shaping is not bundled. | Full OpenType feature control (`vert`/`vrt2`) is deferred because PyQt's text stack does not expose a stable cross-binding API. | `utils/text_rendering.py`, `ui/text_panel.py`, `ui/fontformat_commands.py`, `ui/canvas.py`, `ui/scenetext_manager.py`, `utils/fontformat.py`. |
| Vertical CJK | Existing real `VerticalTextDocumentLayout` already stacks glyphs into vertical columns; this pass added text normalization, vertical punctuation classes, diagnostic vertical columns, and fit/layout integration. | Font-specific vertical alternates still depend on Qt/font behavior. | Publishing-grade vertical shaping remains deferred for optional native shaping dependency evaluation. | `utils/text_rendering.py`, `ui/scene_textlayout.py`, `ui/textitem.py`, `tests/test_text_rendering.py`. |
| Script-aware wrapping/fitting | CJK kinsoku wrapping, line balancing, vertical columns, shrink/expand/preserve/balance fit modes, and fit diagnostics are implemented and tested. | The canvas layout still has older heuristic paths for bubble masks; the new helper is integrated before those paths rather than replacing them wholesale. | Knuth-Plass/ICU segmentation is deferred to avoid adding heavy dependencies. | `utils/text_rendering.py`, `ui/scenetext_manager.py`, `tests/test_text_rendering.py`. |
| Font fallback | Per-script fallback chains, chain merging/deduplication, missing-glyph diagnostics, config UI, layout review warnings, and automation issue listing are implemented. | Actual multi-font glyph-run substitution is still limited by Qt document behavior. | Full fallback shaping per glyph is deferred until a cross-platform shaping/font stack is chosen. | `utils/text_rendering.py`, `utils/config.py`, `ui/configpanel.py`, `ui/scenetext_manager.py`, `ui/textitem.py`. |
| Manga effects/presets | Presets now set font size, stroke, spacing, alignment, writing mode, and padding; defaults for stroke/shadow/padding are configurable; existing stroke/shadow/gradient/path/warp remain connected. | Live preview is limited to the selected canvas text item plus optional diagnostics overlay. | Separate mini preview widget deferred because canvas already provides live rendering and avoids fake/unconnected UI. | `utils/text_rendering.py`, `ui/text_panel.py`, `ui/fontformat_commands.py`, `utils/config.py`, `docs/RENDERING_TEXT_FORMATTING.md`. |
| Precise bounds/placement | Text item bounds now expand for stroke and shadow; diagnostics include measured bounds, overflow, missing glyphs, and fallback chain; quick action can recenter text in its box. | Bounds estimates are renderer-independent approximations; final Qt ink metrics can still differ by platform/font. | Pixel-perfect HarfBuzz ink-bounds deferred. | `ui/textitem.py`, `utils/text_rendering.py`, `ui/canvas.py`, `ui/scenetext_manager.py`. |
| Renderer diagnostics | Config toggle and canvas overlay show text box, measured bounds, overflow, writing mode, and missing glyphs; automation can list issues. | Overlay is intended for QA and not exported. | None for current QA scope. | `ui/textitem.py`, `ui/configpanel.py`, `ui/mainwindow.py`. |
| Layout review agent | Selected and whole-page review use persisted settings, provider fallback, user report, style-aware snapshots, and new actions: shrink, balance, writing-mode switch, recenter, padding, preset, missing-glyph flag. | Remote provider response parsing remains conservative and falls back to heuristics on provider errors. | Rich multimodal critique is deferred to provider availability. | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `ui/mainwindow.py`, `ui/layout_review_report_dialog.py`. |
| PSD/export | Current-page image export has status feedback; structured OCR exists; new layered PSD handoff exports original/inpainted/mask/final helper layers, editable text JSON, JSX rebuild script, and warnings. | It is a PSD handoff package rather than native `.psd` binary creation. | Native editable PSD writing deferred because available pure-Python libraries do not reliably write Photoshop text layers cross-platform. | `utils/layered_psd_export.py`, `ui/mainwindowbars.py`, `ui/mainwindow.py`, `docs/RENDERING_TEXT_FORMATTING.md`. |
| Settings/config | Dedicated Rendering/Text Formatting config section added with font, default writing/fit mode, stroke/shadow/padding defaults, overflow/diagnostic toggles, fallback chains, and a clearly labeled web-font future stub. | Google/web fonts are not exposed as an active renderer control. | Web font download/cache integration deferred to avoid dead UI and Program Files permission issues. | `utils/config.py`, `ui/configpanel.py`. |
| Keyboard/tool UX | Shortcut schema now includes text eraser and brush size; single-key shortcut firing is guarded while typing; hard duplicate conflict test added; tooltips show shortcuts. | Existing QAction menu shortcuts still exist for menu actions where appropriate. | A full shortcut ownership registry migration is deferred; current changes avoid adding ambiguity. | `utils/shortcuts.py`, `ui/mainwindow.py`, `ui/drawingpanel.py`, `tests/test_shortcuts_rendering.py`. |
| Automation/API | Local API now supports layout review, render current page, list rendering issues, page state, and structured OCR export. | Long-running render/pipeline progress still uses existing UI events rather than streaming HTTP events. | Headless server expansion is deferred to avoid duplicating existing local API infrastructure. | `ui/mainwindow.py`, `utils/local_automation_api.py`. |

### Deferred Koharu items and reasons

- **Native PSD editable text-layer writer**: deferred because this Python stack does not currently include a reliable cross-platform PSD text-layer writer. The handoff package preserves editability via manifest and Photoshop JSX instead of silently producing raster-only PSDs.
- **Full HarfBuzz/OpenType shaping pipeline**: deferred because adding and packaging a shaping stack requires dependency and platform validation. Current implementation uses PyQt text layout plus explicit vertical/CJK heuristics and diagnostics.
- **Google/web font renderer integration**: deferred as a labeled future stub. Font downloads require cache, licensing, offline behavior, and Windows Program Files write-safety work.
- **Full shortcut ownership rewrite**: deferred because existing QAction/QShortcut menu integration is broad. This pass adds guards and conflict tests without destabilizing menu accelerators.

## 2026-05-06 Progress audit addendum — typography line-breaking and automation preset pass

| Area | Implemented | Partially implemented | Missing | Newly implemented in this pass | Deferred with reason | File references |
| --- | --- | --- | --- | --- | --- | --- |
| Text rendering / line breaking | Writing mode, vertical CJK, fitting, fallback diagnostics, presets, padding/effects, and diagnostics existed before this addendum. | The renderer still uses Qt text layout plus Python heuristics rather than a full HarfBuzz/ICU shaping stack. | Native glyph-run fallback and full OpenType vertical alternates. | Added persistent per-style line-break strategy (`auto`, `cjk_strict`, `balanced`, `loose`), strategy-aware CJK kinsoku wrapping, balanced dangling-line cleanup, loose SFX wrapping, preset defaults, diagnostics serialization, and tests. | Full ICU segmentation is deferred to avoid adding heavy packaging/dependency risk in this pass. | `utils/text_rendering.py`, `utils/fontformat.py`, `tests/test_text_rendering.py` |
| Text/style panel | Writing mode, fit mode, presets, effects, and padding controls existed. | The panel is dense; further UX grouping is still desirable. | Dedicated live mini-preview remains absent because the canvas is the connected preview. | Added a connected Line-break Strategy combo that writes to the selected text style and persists with project data. | A separate preview widget is deferred to avoid dead UI and duplication of canvas rendering. | `ui/text_panel.py` |
| Settings/config | Rendering defaults/fallback font chains/diagnostic toggles existed. | Defaults currently apply through helpers/presets rather than every legacy text creation path. | Google/web font download/cache integration. | Added default line-break strategy to config persistence and the Rendering/Text Formatting config section. | Web fonts remain a labeled future stub because cache, licensing, and Windows permission behavior need separate validation. | `utils/config.py`, `ui/configpanel.py` |
| Layout review agent | Selected/page review, provider settings, heuristic fallback, reports, style-aware snapshots, and actions existed. | Provider responses remain conservative and normalized. | Multimodal second-pass re-render scoring. | Added line-break strategy to snapshots, heuristic issue detection for vertical CJK with weak wrapping, and a safe `set_line_break_strategy` review action. | Second-pass visual scoring is deferred until provider/runtime cost is validated. | `utils/layout_review_agent.py`, `ui/scenetext_manager.py` |
| Automation/API | Local API already supported layout review, rendering, page state, structured OCR, and rendering diagnostics. | No streaming progress channel for every long action yet. | Full headless server/MCP protocol parity. | Added `apply_rendering_preset` automation action to apply connected manga presets to current-page text boxes and save project data. | Full MCP parity is deferred to avoid duplicating existing local automation infrastructure. | `ui/mainwindow.py` |
| Koharu issue harvesting | Gap analysis had issue references for #651/#650/#649/#640/#648/#555. | Some dependency-only Koharu issues are tracked only as low-priority signals. | Continuous auto-sync tooling is not yet checked into the repo. | Refreshed `docs/KOHARU_ISSUE_BACKLOG.md` from 649 GitHub issues across 7 REST pages and recorded issue-inspired implementation notes/next candidates. | Automated scheduled refresh is deferred because this repository does not have a task runner for docs refreshes. | `docs/KOHARU_ISSUE_BACKLOG.md` |

### Issue-inspired items implemented in this addendum

- Koharu #624 / #649 / #640 / #648 / #630 inspired the per-style line-break strategy, strict CJK kinsoku, balanced final-line cleanup, style panel control, config default, diagnostics/API payloads, and layout-review fix action.
- Koharu API/RPC/editor workflow issues such as #651 inspired the `apply_rendering_preset` automation action so external tools can apply connected lettering presets without adding fake UI.

### Deferred Koharu items after this addendum

- Mask-aware collision squeezing (#637): useful but requires deeper mask/balloon geometry integration and should be handled in the next layout-review batch.
- Native editable PSD writing: still deferred because the current safe handoff avoids silently producing raster-only fake PSD text layers.
- Full HarfBuzz/ICU shaping: deferred until a cross-platform dependency strategy is selected.
- Shortcut ownership registry rewrite: still deferred; no new QAction/QShortcut ambiguity was introduced by this pass.

## 2026-05-06 Follow-up progress audit — font fallback, RTL, review fixes

| Area | Implemented | Partially implemented | Missing | Newly implemented in this pass | Deferred with reason | File references |
| --- | --- | --- | --- | --- | --- | --- |
| Font fallback / missing glyphs | Global per-script fallback chains and diagnostics existed. | Previous pass warned about missing glyphs but still relied mostly on platform font substitution. | Font favorites/localized family names and full glyph shaping remain absent. | Added explicit per-character fallback font runs, after-fallback missing-glyph detection, per-style fallback-chain UI, and a fallback status label in the text panel. | Font favorites and localized names are deferred to a larger font browser pass. | `utils/text_rendering.py`, `ui/textitem.py`, `ui/text_panel.py`, `ui/fontformat_commands.py` |
| RTL lettering | Writing-mode auto detects Arabic/Hebrew and layout review can flag mode mismatches. | Arabic joining still depends on Qt shaping and installed fonts. | HarfBuzz-level shaping and bidi run inspection. | RTL writing mode now sets the QTextDocument text direction to right-to-left and right-aligns left-default text, reducing reversed/incorrect Arabic rendering cases inspired by Koharu #602/#583/#213. | Full shaping is deferred until dependency strategy is validated. | `ui/textitem.py`, `utils/text_rendering.py` |
| Layout review agent | Selected/page review, provider settings, reporting, and style-aware actions are connected. | Provider responses are still normalized to conservative safe actions. | Second-pass visual verification after applying fixes. | Added `apply_font_fallback` review action, fed review snapshots with after-fallback missing-glyph diagnostics, and added regression coverage for fallback actions. | Screenshot-based re-review is deferred due runtime/provider cost. | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `tests/test_layout_review_agent.py` |
| Automation/API | Local automation supports render/list issues/layout review/presets. | No streaming progress for every long-running fix action. | Full MCP protocol parity. | Added `fix_rendering_issues` automation action that runs connected layout-review fixes and reports remaining renderer issues. | MCP parity is deferred to avoid duplicating existing local API infrastructure. | `ui/mainwindow.py` |
| Issue backlog quality | Previous backlog existed but included dependency-only noise. | Manual categorization is still used. | A checked-in refresh script/scheduled bot. | Rebuilt the backlog from 649 current issues while filtering dependency churn and prioritizing text/font/RTL/layout/export/API issues. | Automated refresh script deferred until docs tooling is chosen. | `docs/KOHARU_ISSUE_BACKLOG.md` |

### Issue-inspired items implemented in this follow-up

- Koharu #595/#77 inspired per-style fallback-chain controls and live fallback diagnostics.
- Koharu #602/#583/#213 inspired RTL document direction handling and fallback-aware Arabic diagnostics.
- Koharu #624/#120/#117 inspired fixing final fit bounds so selected line-break strategy affects overflow diagnostics consistently.
- Koharu API/RPC workflow themes (#651 and related issues) inspired `fix_rendering_issues` for headless layout-review repair.

### Deferred after this follow-up

- Font favorites, localized font names, and rich family/weight previews (#595/#77) are deferred to a dedicated font-browser pass.
- Full HarfBuzz/ICU shaping for Arabic and OpenType vertical alternates (#602/#213/#583) remains deferred for dependency/package validation.
- Native PSD editable text-layer writing (#587/#558) remains deferred; current handoff is intentionally explicit rather than fake PSD text.

## 2026-05-06 Extended progress audit — project typography QA and bulk preset controls

| Area | Implemented | Partially implemented | Missing | Newly implemented in this pass | Deferred with reason | File references |
| --- | --- | --- | --- | --- | --- | --- |
| Project typography QA | Per-textbox diagnostics, layout review, and current-page issue listing existed. | Previous automation/UI flows focused on selected/current page more than whole-project QA. | Visual before/after screenshots in the QA report. | Added pure project-level rendering QA that scans pages/text boxes for overflow, missing glyphs, weak vertical CJK line breaks, RTL alignment, and stroke padding; reports summaries and safe suggestions. | Screenshot diffs are deferred because project-wide rendering can be expensive and should be optional. | `utils/rendering_qa.py`, `tests/test_rendering_qa.py` |
| Bulk text formatting controls | Batch font family/size/alignment override existed. | Some preset/style operations still require the text panel for fine details. | Full project font favorite/weight browser. | Expanded the connected batch style override dialog to support manga presets, writing mode, fit mode, line-break strategy, fallback chain, and padding across current page or whole project. | Font favorite/weight browser remains deferred to a dedicated font UX pass. | `ui/mainwindow.py` |
| Pipeline Insights UX | Layout review and batch style buttons existed. | No dedicated project typography QA entry point. | Rich table preview of every warning before export/apply. | Added a Typography QA Report action that exports JSON, can include clean boxes, and optionally applies conservative fixes while saving/refreshing the project. | Table preview is deferred to avoid a large new widget in this pass. | `ui/pipeline_insights_widget.py`, `ui/mainwindow.py` |
| Automation/API | Local API could list current rendering issues and run current-page fixes. | Long-running progress events are still coarse. | Streaming progress for every project page. | Added project-level `export_rendering_qa` and `apply_project_rendering_fixes` automation actions. | Streaming events are deferred until API transport supports long-running event channels. | `ui/mainwindow.py`, `utils/rendering_qa.py` |
| Issue-inspired coverage | Backlog tracked #649/#648/#640/#637/#595/#77/#602/#583. | Native PSD and shaping gaps remain. | Native PSD text writer, HarfBuzz/ICU shaping, mask-aware squeezing. | This pass implements issue-inspired project-wide style/QA controls for #649/#648/#640, font QA controls for #595/#77, and layout issue reporting for #545/#630. | Deferred items remain next-batch candidates with dependency/runtime reasons. | `docs/KOHARU_ISSUE_BACKLOG.md` |

### Issue-inspired items implemented in this extended pass

- Koharu #649/#648/#640 inspired project-wide manga preset/style override expansion beyond font size/alignment.
- Koharu #595/#77 inspired project-level font fallback QA and fallback-chain batch application.
- Koharu #545/#630 inspired typography QA reporting of overflow/placement-risk signals before export.
- Koharu API/RPC workflow themes inspired project-level `export_rendering_qa` and `apply_project_rendering_fixes` endpoints.

### Deferred after this extended pass

- Mask-aware squeezing (#637) remains the next highest-impact layout implementation because it requires bubble/mask geometry scoring.
- Native PSD editable text writer (#587/#558) remains deferred; the handoff path is still the safe export story.
- Rich QA preview tables and screenshot diffs are deferred until the lightweight JSON/action pipeline proves stable.

## 2026-05-06 QA preview follow-up — reviewable typography reports

| Area | Implemented | Partially implemented | Missing | Newly implemented in this pass | Deferred with reason | File references |
| --- | --- | --- | --- | --- | --- | --- |
| Typography QA preview | JSON project QA and conservative fixes existed. | Previous UI exported/applied without a row-level preview. | Screenshot diff thumbnails are still absent. | Added a reusable Typography QA dialog with sortable warning rows, summaries, JSON/Markdown export, and optional conservative fixes. | Screenshot thumbnails are deferred until rendering cost and cache behavior are profiled. | `ui/typography_qa_dialog.py`, `ui/mainwindow.py` |
| QA export formats | JSON report export existed. | Markdown handoff was only documented as future-friendly. | CSV and PSD-linked QA metadata. | Added flattened QA rows plus Markdown conversion and automation support for `.md` output. | CSV/PSD metadata is deferred until native PSD handoff evolves. | `utils/rendering_qa.py`, `tests/test_rendering_qa.py` |
| Main window maintainability | Project QA was wired but inline. | Main window still owns many workflow dialogs. | Full workflow-dialog extraction. | Moved Typography QA UI into a focused helper dialog module to avoid further `mainwindow.py` bloat while keeping connected behavior. | Other legacy dialogs are deferred because they are outside this typography pass. | `ui/typography_qa_dialog.py`, `ui/mainwindow.py` |

### Issue-inspired items implemented in this QA preview follow-up

- Koharu #545/#630 inspired previewable typography warning rows before applying fixes.
- Koharu #649/#648/#640 inspired reviewable project-wide style/fix workflows rather than silent bulk changes.
- Koharu automation/API themes inspired Markdown QA export from `export_rendering_qa` for headless review handoff.
