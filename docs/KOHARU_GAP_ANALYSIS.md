# Koharu Gap Analysis and Implementation Audit

_Last audited: 2026-05-17 against BallonsTranslator-Pro current branch, public `mayocream/koharu` issues via GitHub REST API (300 all-state issues plus topic searches scanned), upstream `dmMaze/BallonsTranslator` issues via REST API (300 all-state issues plus topic searches scanned), and recent upstream `dev` commits fetched from Git._


## Newly implemented in this pass (2026-05-17 second follow-up: line-quality QA, RTL fit safety, export naming)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| Line-break quality / final lettering QA | Added renderer-neutral `line_break_quality` diagnostics for widows, kinsoku violations, and ragged lines; fit diagnostics/proof metrics now recommend `balance_lines`, the Text panel displays line-balance status, and project rendering fixes can apply balanced wraps. | Koharu #117 word splitting, #594 advanced formatting, #509 vertical wrap pain, #649/#528 global consistent lettering; upstream #1077 fit-to-box and #995 punctuation/width conversion reports. | `utils/text_rendering.py`, `utils/rendering_qa.py`, `ui/text_panel.py`, `tests/test_text_rendering.py`, `tests/test_rendering_qa.py` |
| RTL/Arabic auto-fit guard | Expand-to-fill sizing now caps RTL expansion to avoid Arabic/Hebrew text ballooning in large boxes while preserving shrink/preserve behavior. | Koharu #602 Arabic rendering/export, #583 Arabic auto font size exaggeration; upstream font/layout bug reports #862/#972. | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Effect-aware fit regression fix | Final fit bounds now include `secondary_stroke_width` in the non-preserve path, so back-outline margins are not lost after binary fitting. | Koharu #698 double outline, #446/#447 cropping, upstream #1077 fit-to-box. | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Layout review line-quality action | Scene snapshots pass line-break quality to the layout review planner, which can now propose `balance_lines` even when text technically fits but has a poor final lettering wrap. | Koharu #117/#594 layout quality; upstream #1077. | `ui/scenetext_manager.py`, `utils/layout_review_agent.py`, `tests/test_layout_review_agent.py` |
| Batch export filename workflow | Batch export dialog, settings, automation API, export manifest, and a new safe filename renderer now support templates such as `{stem}_{index:03d}` while sanitizing Windows/path-unsafe names. | Koharu #535 custom filename/include unrendered pages, #568 export skips pages, #610 batch projects; upstream #229 export suggestions. | `ui/export_dialog.py`, `ui/configpanel.py`, `ui/mainwindow.py`, `utils/config.py`, `utils/export_naming.py`, `tests/test_export_naming.py` |

### Progress audit update for this second follow-up

| Capability | Status | Newly implemented | Deferred with reason |
| --- | --- | --- | --- |
| Advanced text rendering / formatting | Implemented / advanced | Line-quality diagnostics now flow through fit, proof packs, QA, layout review, UI diagnostics, and auto-fix; RTL expand is safer; secondary outline fit margins are preserved. | Full HarfBuzz shaping/ligature metrics remain deferred to an optional dependency pass. |
| Layout review agent | Implemented / advanced | Review snapshots include line quality and planner proposes balance-line fixes for poor wraps, not only overflow. | Provider prompt schema can be expanded later to include the raw line-quality object. |
| Export/batch workflow | Implemented / advanced | Users and API callers can choose persisted filename templates with safe sanitization and manifest recording. | Per-page export queue progress events remain deferred. |
| Upstream BallonsTranslator sync | Active | Reviewed current upstream `dev` head (`6649de1` through `91f5d13`) again; this pass adapted issue-inspired export naming and typography safety instead of unsafe broad cherry-picks. | Requirements/provider commits still deferred pending Pro dependency/provider matrix. |


## Newly implemented in this pass (2026-05-17 follow-up: PSD style fidelity, shortcut safety, settings color polish)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| PSD/export style fidelity | Layered PSD handoff manifests and Photoshop JSX notes now preserve secondary/back outline width and color, and proof metrics include the secondary outline in effect margins without requiring OpenCV-backed global config import. | Koharu #454/#558 PSD text-layer issues, #698 double outline, #591 export interoperability; upstream #50/#364 Photoshop/PSD export. | `utils/layered_psd_export.py`, `tests/test_layered_psd_export.py` |
| Batch/headless PSD handoff workflow | The local automation `export` route can now export layered PSD handoff manifests for multiple requested pages in one call, using the live final composite for the current page and saved result fallbacks for other pages with explicit warnings. | Koharu #610 batch projects, #612 API/headless, #558/#587 export handoff; upstream #1094 automation and #1020 batch queue. | `ui/mainwindow.py` |
| Shortcut/keybind safety polish | Shortcut conflict detection now canonicalizes equivalent key strings (for example `Ctrl++` and `control++`) and the keyboard shortcuts dialog shows live warnings for single-key tool/navigation shortcuts that are intentionally suppressed while typing. | Koharu workflow/keybind search, #519 fewer processing friction; upstream #1180 spacebar panning, #568 keyboard block navigation, #1104 workflow polish. | `utils/shortcuts.py`, `ui/shortcuts_dialog.py`, `tests/test_shortcuts_rendering.py` |
| Rendering settings color completion | The Rendering/Text Formatting settings section now exposes a persisted default back/second-outline color picker, completing the previous width-only default and making project defaults fully discoverable. | Koharu #595 font UX, #593 text presets, #698 double outline. | `ui/configpanel.py`, `utils/config.py` |

### Progress audit update for this follow-up

| Capability | Status | Newly implemented | Deferred with reason |
| --- | --- | --- | --- |
| Advanced text rendering / formatting | Implemented / advanced | Back-outline style now reaches settings color defaults, PSD handoff metadata, proof metrics, JSX notes, and tests. | Native PSD editable layer effects are still handoff/JSX metadata rather than true PSD text effects. |
| PSD/export workflow | Partially implemented / advanced | Multi-page API PSD handoff export and richer text style metadata reduce manual export steps. | Native PSD writer remains deferred for cross-app validation. |
| Keyboard/tool UX | Implemented / advanced | Canonical conflict detection and live single-key safety warnings improve shortcut setup without adding duplicate QAction/QShortcut owners. | Spacebar panning itself remains deferred to a dedicated canvas input pass. |
| Automation/API/headless | Implemented / advanced | `export(kind=psd_handoff, pages=[...])` now handles multi-page handoff exports with warnings. | Streaming progress events remain deferred. |
| Upstream BallonsTranslator sync | Active | Reviewed recent `dev` commits again and documented safe/unsafe ports; no blind cherry-picks. | Dependency/provider/inpaint commits remain deferred pending Pro compatibility matrix. |

## Newly implemented in this pass (2026-05-17: double-outline lettering, QA-enriched automation, upstream stage-restore fix)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| Manga double-outline/back-stroke rendering | Added persisted `secondary_stroke_width` and `secondary_srgb` style fields, normalized legacy/project data on load, drew the back outline before the primary outline/fill in normal and text-on-path rendering, and included the larger outline in effect-cache rendering. | Koharu #698 double outline, #594 advanced formatting, #446/#447 border cropping; upstream #35 wishes, #1138 text-on-path, #1077 fit-to-box. | `utils/fontformat.py`, `ui/textitem.py`, `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Effect-aware fit/bounds diagnostics | Updated fit diagnostics, precise/proof metrics, renderer QA, and layout snapshots so the back outline contributes to safe inner bounds, overflow, recommended box sizing, and clipping risk. | Koharu #545 DPI/restricted rendering area, #594 formatting, #640 fixed font options; upstream #1077 auto-size. | `utils/text_rendering.py`, `utils/rendering_qa.py`, `ui/scenetext_manager.py`, `ui/text_panel.py` |
| Visible text formatting controls and defaults | Added text-panel controls for back-outline width/color, a persisted Rendering setting for default back-outline width, and project-default application that carries primary stroke, secondary stroke, shadow, padding, writing, fit, and line-break defaults. | Koharu #593 presets, #595 font UX, #640 fixed font options, #698 double outline. | `ui/text_panel.py`, `ui/configpanel.py`, `utils/config.py`, `ui/fontformat_commands.py` |
| Layout review agent action | Layout review now flags SFX-style text missing a back outline and can apply an `apply_double_outline` action through the existing scene-text manager, preserving user undo/update behavior. | Koharu #698, #594; upstream #1077. | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `tests/test_layout_review_agent.py` |
| Editable export/handoff metadata | SVG editable text handoff now emits a separate back-outline text layer before the main editable text layer and records secondary outline metrics in the manifest/proof metrics. | Koharu #454/#558 PSD text-layer issues, #591 export interoperability; upstream #50/#364/#749 export requests. | `utils/svg_text_export.py` |
| Automation/headless page QA workflow | Extended the local automation `list_pages` backend with `include_rendering_qa=true`, returning each textbox's rendering QA warnings/metrics alongside reading-order blocks so headless/batch tools can triage pages without separate calls. | Koharu #612 API/headless, #610 batch projects, #691 quick navigation; upstream #1094 automation, #1020/#841 batch queue/resume. | `ui/mainwindow.py` |
| Upstream workflow/stability adaptation | Reviewed upstream dev `6649de1` and adjacent workflow fixes, then fixed a Pro-specific selected-page detect-only restore tuple regression that could crash pipeline-finished stage restoration. | dmMaze #1178 / dev `6649de1`; upstream workflow stability. | `ui/mainwindow.py` |

### Progress audit update for this pass

| Capability | Status | Newly implemented | Deferred with reason |
| --- | --- | --- | --- |
| Advanced text rendering / formatting | Implemented / advanced | Double-outline/back-stroke rendering, text-on-path support, SVG handoff support, effect-aware fit/proof diagnostics, text-panel controls, and preset/default integration. | HarfBuzz/OpenType shaping and native editable PSD text effects still need optional dependency and cross-app validation. |
| Layout review agent | Implemented / advanced | SFX missing-back-outline issue and `apply_double_outline` action added; snapshots include secondary outline style. | LLM provider quality and full model-specific review remain provider-dependent. |
| PSD/export workflow | Partially implemented / advanced | SVG editable text handoff preserves back-outline as a separate layer; proof metrics include secondary outline. | Native PSD editable text writer remains deferred. |
| Settings/config polish | Implemented / advanced | Rendering defaults now include a persisted back-outline width; project-default application applies outline/shadow/padding formatting defaults. | Default back-outline color UI is stored with config but not yet exposed as a dedicated color picker in Settings; the text panel exposes per-style color. |
| Keyboard/tool UX | Maintained | No new shortcut owners were added; UI controls are direct widgets, avoiding QAction/QShortcut ambiguity. | Spacebar panning/custom shortcut conflict UI remains a future pass. |
| Automation/API/headless | Implemented / advanced | `list_pages(include_rendering_qa=true)` gives pages/textboxes/rendering issues in one call for batch/headless tooling. | Streaming progress/events remain deferred. |
| Upstream BallonsTranslator sync | Active | Reviewed recent dev commits and adapted a safe Pro stage-restore stability fix inspired by `6649de1`; dependency commits documented as deferred. | Requirements pins and large provider/inpaint commits remain deferred pending Pro compatibility tests. |

### Koharu issue-inspired items implemented/deferred

- Implemented/advanced: #698 double outline, #594 advanced text formatting, #593 presets, #595 font UX, #640 fixed font options, #612/#610/#691 API/batch/navigation QA workflow.
- Deferred: #454/#558 native editable PSD text/effects, #602/#583 full RTL shaping, #705 VRAM model cancellation, #693 large-history Save As crash until reproducible fixtures are available.

### Upstream BallonsTranslator issue-inspired items implemented/deferred

- Implemented/advanced: #1077 auto-size/effect-aware fitting, #1138 text-on-path effects, #1094 automation, #1020/#841 batch/headless triage, #1178 workflow stability area.
- Deferred: #1179/#1177/#1175 dependency pins, #1172 translator fix, #1180 spacebar panning, #50/#364 native PSD handoff.

### Upstream dev commits ported/deferred

- Ported/adapted: `6649de1` reviewed; Pro-specific detect-only `_run_stages_restore` stability fix applied.
- Reviewed/deferred: `c80eb81`, `04c3414`, `88d4969` dependency changes; `485bbe8` FLUX inpaint; `4c14019` replace/render responsiveness; `1958f66` Ollama provider; `64a5713` Google translator fix.

### Next batch candidates

1. Native PSD editable translated text/effects with helper layers and warnings.
2. Optional HarfBuzz shaping for RTL/Arabic with package-safe fallback.
3. Shortcut conflict UI and spacebar panning without QAction/QShortcut ambiguity.
4. Dependency matrix for PaddleOCRVLManga/transformers/gguf upstream fixes.
5. Replace-all/render-all responsiveness and save-state audit against `4c14019`.
6. Batch pause/resume/cancel and model-unload workflow.
7. Default back-outline color picker in Rendering settings.
8. More vertical punctuation/bracket optical placement fixtures.


---

## Previous implementation history

## Newly implemented in this pass (2026-05-14: review response, batch lettering dialog, live fixes, upstream shortcut dependency port)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| Real vertical CJK rendering improvement | Extended the Qt vertical text layout to keep short ASCII tate-chu-yoko groups upright/compact instead of rotating each digit/mark separately, while preserving the prior right-to-left column model and punctuation diagnostics. | Koharu #597 manual direction, #509 vertical layout bug, #583 Arabic/auto-size pain, #594 advanced text formatting; upstream #1128 vertical text and #1132 Latin text after original language. | `ui/scene_textlayout.py`, `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Precise lettering bounds diagnostics | Added `precise_text_bounds` with Qt font metrics fallback and surfaced `precise_measured_bounds` in proof metrics so QA/proof packs can compare stable estimates against real glyph metrics for clipping/overflow review. | Koharu #545 DPI/restricted rendering area, #547 render size display, #649/#640 fixed/global font workflows; upstream #1169/#1077 fit-to-box requests. | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| Live selected-textbox diagnostic fixes | Added a visible **Apply diagnostics fixes** button in the text formatting panel that applies typography polish + smart fit to the selected textbox: writing mode, vertical punctuation, line breaks, padding, fallback chain, and font size. | Koharu #649 global font override, #640 fixed font options, #594 line spacing/kerning, #595 font UX, #583 RTL sizing. | `ui/text_panel.py` |
| Batch lettering workflow dialog | Replaced the previous one-click message flow with a reviewable dialog for current, selected, or whole-project scope, showing ordered workflow steps and highest-priority textboxes before applying fixes/proof/render actions. Added page-list context action for selected pages. | Koharu #691 quick navigation/resume, #601 reorder/textbox workflow, #559 multi-select editing, #519 processing shortcuts/workflow, #592 workflow presets. | `ui/lettering_workflow_dialog.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py` |
| Multi-page workflow/API behavior | `lettering_workflow` now accepts multiple pages, applies project-level renderer fixes across them, exports proof packs per page, renders the current page with a manifest when requested, and reports warnings for deferred batch rerender cases instead of silently doing current-page-only work. | Koharu #535 custom export/include pages, #558/#587 PSD/export handoff, #610 batch projects; upstream #1134 export image workflow. | `ui/mainwindow.py`, `utils/lettering_workflow.py`, `docs/RENDERING_TEXT_FORMATTING.md` |
| Upstream dependency/shortcut compatibility port | Adapted upstream dev commit `a390d4c` by removing the hard `keyboard` import/dependency, adding a `pynput`-first SalaDict shortcut backend with graceful fallback, and porting `utils.structures.get_annotations` compatibility behavior. | dmMaze dev `a390d4c`, upstream packaging/shortcut compatibility. | `ui/textedit_area.py`, `requirements.txt`, `utils/structures.py`, `docs/UPSTREAM_BALLONSTRANSLATOR_DEV_SYNC.md` |

### Progress audit update for this pass

| Capability | Status | Newly implemented | Deferred with reason |
| --- | --- | --- | --- |
| Advanced text rendering / formatting | Implemented / advanced | Real Qt vertical layout now honors compact tate-chu-yoko groups; proof metrics include precise Qt bounds; selected textboxes have one-click diagnostic fixes. | HarfBuzz/OpenType shaping remains deferred because it requires optional dependency/design validation. |
| Layout review agent | Implemented / advanced | Batch workflow dialog uses renderer QA to decide when layout review is needed and exposes focus rows before applying fixes. | LLM-specific review quality remains provider/model dependent. |
| PSD/export workflow | Partially implemented / advanced | Multi-page workflow can export proof packs per page and current-page render manifests with explicit warnings for unsupported batch rerender. | Native editable PSD text writer remains deferred pending PSD validation. |
| Settings/config polish | Implemented / maintained | Live fixes and batch workflow consume existing persisted rendering defaults/fallback chains instead of creating duplicate settings. | Additional onboarding wizard remains a future pass. |
| Keyboard/tool UX | Implemented / advanced | Page-list context action, workflow dialog, selected textbox fix button, and upstream `keyboard` dependency removal improve discoverability and compatibility. | Full custom shortcut conflict UI remains deferred. |
| Automation/API/headless | Implemented / advanced | Multi-page `lettering_workflow` applies fixes and proof exports across pages and reports warnings for skipped batch rerender. | Streaming progress/events remain deferred. |
| Upstream BallonsTranslator sync | Active | Ported `a390d4c` manually without overwriting Pro shortcut systems; retained previous `6649de1` port and documented newer deferrals. | `c80eb81` transformers pin and provider/dependency commits remain deferred pending Pro compatibility testing. |

### Koharu issue-inspired items implemented/deferred

- Implemented/advanced: #597/#509/#583/#594 vertical/RTL/text formatting, #649/#640/#595 font/fixed rendering workflows, #601/#559/#519/#691 fewer-click editor/page workflow, #535/#558/#587 proof/export handoff status.
- Deferred: #558/#587 native editable PSD text layers, #591 XCF parity, #610 full parent/child CBZ queue, #698 double-outline effect controls beyond existing stroke/effect stack.

### Upstream BallonsTranslator issue/dev items implemented/deferred

- Implemented/advanced: upstream #1169/#1077 fit-to-box access, #1128/#1132 vertical/Latin text handling, #1122 font diagnostics, #1134 render/export workflow, dev `a390d4c` shortcut/dependency compatibility.
- Reviewed/deferred: dev `4c14019` direct replace-and-render UI freeze port pending conflict audit; `c80eb81`/`04c3414` transformer pins pending Pro module testing; `1958f66` Ollama provider pass pending settings integration.

### Next batch candidates

1. Native editable PSD text-layer writer validation for Koharu #558/#587.
2. True queued batch rerender after multi-page lettering workflow with progress/cancel.
3. Direct upstream `4c14019` replace-and-render freeze/save-state audit against Pro global search.
4. Visual regression captures for Qt tate-chu-yoko and vertical punctuation across Qt5/Qt6/Windows DPI.
5. Provider/model onboarding diagnostics for Ollama/Flux/gguf dependency issues from upstream #1167/#1171/#1175.
6. Double-outline/advanced effect controls for Koharu #698.


## Newly implemented in this pass (2026-05-13: lettering workflow, vertical diagnostics, issue navigation)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| Advanced vertical lettering diagnostics | Added vertical column orphan balancing and tate-chu-yoko orientation metadata to the renderer-neutral vertical cell plan so QA/proof/export handoffs can preserve compact `12`/`!?` runs and avoid single-glyph leftmost columns. | Koharu #597/#583/#509/#594, upstream #1128/#1132/#1169/#1077. | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| One-click lettering workflow | Added `utils.lettering_workflow` to convert rendering QA into ordered steps: typography polish, smart fit, layout review, proof pack, and render. Wired it to a Tools → Project action and `lettering_workflow` local API route with optional apply/export/render behavior. | Koharu #649/#640/#592/#519/#601, upstream #1169/#1077/#1134. | `utils/lettering_workflow.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py`, `tests/test_lettering_workflow.py` |
| Next rendering issue navigation | Added a user-visible Tools → Project command and `next_rendering_issue` API route to jump through current-page overflow/glyph/mask/writing-mode issues, reducing manual inspection clicks after QA. | Koharu #601/#559/#519 editor workflow requests; upstream #1122/#1128 font/vertical reports. | `utils/lettering_workflow.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py` |
| Export/render status manifest | Extended `render_current_page` automation export with an optional `.render-manifest.json` describing path, extension, quality, page, and warnings for headless workflows and export diagnostics. | Koharu #535/#558/#587/#591 export requests; upstream #1134 export workflow request. | `ui/mainwindow.py` |
| Backlog/sync pipeline refresh | Refreshed Koharu and upstream issue backlogs from paginated GitHub REST API results and upstream dev sync from fetched `dmMaze/BallonsTranslator dev` commits; documented implemented/deferred items and next candidates. | GitHub issue/commit research on 2026-05-13. | `docs/KOHARU_ISSUE_BACKLOG.md`, `docs/UPSTREAM_BALLONSTRANSLATOR_ISSUE_BACKLOG.md`, `docs/UPSTREAM_BALLONSTRANSLATOR_DEV_SYNC.md` |

### Progress audit update for this pass

| Capability | Status | Newly implemented | Deferred with reason |
| --- | --- | --- | --- |
| Advanced text rendering / formatting | Implemented / advanced | Vertical diagnostics now expose tate-chu-yoko orientation and avoid orphan columns; workflow applies polish + smart-fit across the page. | True HarfBuzz shaping and native editable PSD text layers remain deferred pending visual/PSD writer validation. |
| Layout review agent | Implemented / advanced | Lettering workflow now escalates high-risk QA rows into a layout-review step and can apply existing review-compatible fixes before proof/render. | Direct LLM provider quality still depends on configured API/model; no new provider was added this pass. |
| PSD/export workflow | Partially implemented / advanced | API render now can emit a manifest; lettering workflow includes proof-pack handoff step. | Native PSD text-layer writer remains deferred for #558/#587 because current safe handoff avoids corrupt PSD output. |
| Settings/config polish | Implemented / maintained | Existing rendering defaults/fallback settings are consumed by the new workflow and diagnostics. | No new settings added because relevant rendering settings already exist and adding duplicate UI would violate no-dead-UI constraints. |
| Keyboard/tool UX | Implemented / advanced | Added click-only menu actions owned by TitleBar/MainWindow without QAction shortcut duplication. | Full shortcut editor redesign deferred; existing conflict tests remain the guardrail. |
| Automation/API/headless | Implemented / advanced | Added `lettering_workflow` and `next_rendering_issue`; `render_current_page` can write a manifest. | Event-stream progress remains deferred. |
| Upstream BallonsTranslator sync | Active | Reviewed recent dev commits; adapted export/workflow status improvements safely while retaining prior `6649de1` port. | Dependency/provider commits (`c80eb81`, `04c3414`, `88d4969`, `1958f66`) deferred until Pro compatibility matrix is tested. |

### Koharu issue-inspired items implemented/deferred

- Implemented/advanced: #597 manual direction controls, #583 Arabic/RTL auto-size pain, #509 vertical wrapping, #594 advanced formatting, #649/#640 global/fixed rendering workflow, #601/#559/#519 fewer-click editor navigation, #535 export naming/status.
- Deferred: #558/#587 native editable PSD text fidelity, #591 XCF/SVG parity beyond current SVG handoff, #592 full cross-page workflow preset UI beyond current page lettering workflow.

### Upstream BallonsTranslator issue/dev items implemented/deferred

- Implemented/advanced: #1169/#1077 font-size based on block dimensions via workflow-accessible smart fit, #1128/#1132 vertical/Latin text diagnostics, #1122 font fallback diagnostics, #1134 current-page render manifest.
- Reviewed/deferred: dev `4c14019` replace-and-render freeze/save fix pending Pro conflict audit; `1958f66` Ollama OCR/translation provider changes pending settings integration; dependency commits `c80eb81`/`04c3414`/`88d4969` pending install matrix testing.

### Next batch candidates

1. Native editable PSD text-layer writer validation for Koharu #558/#587.
2. Cross-page/batch lettering workflow with progress/cancel and proof-pack queue.
3. Direct upstream `4c14019` file-level port if Pro still reproduces replace-and-render UI freezing/save-state issues.
4. Visual regression screenshots for vertical punctuation and tate-chu-yoko painting on Qt5/Qt6/Windows DPI.
5. Provider/model onboarding diagnostics for Ollama/Flux/gguf dependency issues from upstream #1167/#1171/#1175.
6. Shortcut conflict UI surfacing for custom single-key tool shortcuts.


## Newly implemented in this pass (2026-05-08: typography polish, workflow API, upstream save-state fix)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| Typography polish pipeline | Added renderer-neutral `plan_typography_cleanup` for script-aware Auto writing mode, vertical punctuation normalization, kinsoku/balanced line-break selection, safe padding, fallback-font repair, and structured diagnostics. | Koharu #624 advanced typesetting, #117 word splitting, #360 Arabic/manual editing, #589 renderer/font workflow requests; upstream #818 vertical text orientation and #1042 font problems. | `utils/text_rendering.py`, `tests/test_text_rendering.py` |
| User-visible typography workflow | Added **Format → Polish typography**, connected scene-manager undoable application, layout-review action support, rendering-QA suggestions/project fixes, text-panel diagnostics, and local API `polish_typography`. | Koharu #649 global text fixes, #486 render vertical text workflow, #589 fewer manual steps. | `ui/canvas.py`, `ui/context_menu_config_dialog.py`, `ui/scenetext_manager.py`, `ui/text_panel.py`, `utils/layout_review_agent.py`, `utils/rendering_qa.py`, `ui/mainwindow.py` |
| New OCR/detected textbox auto-polish setting | Added persistent **Auto-polish new OCR/detected textboxes** setting so run-created boxes can resolve CJK/RTL mode, line-break defaults, safe padding, and fallback fonts automatically without touching saved projects on reopen. | Koharu workflow/setup requests #360/#486/#589; upstream #818. | `utils/config.py`, `ui/configpanel.py`, `ui/scenetext_manager.py` |
| Automation workflow status | Added local API `project_status` for headless scripts to inspect current project, page/textbox counts, completion states, and unsaved state before running multi-step workflows. | Koharu #612 API workflow, #555 structured handoff, #610 batch project workflow. | `ui/mainwindow.py`, `docs/RENDERING_TEXT_FORMATTING.md` |
| Upstream dev fix port | Ported upstream dev commit `6649de1` / issue #1178 by updating save-state logic only after current-page render/save succeeds. | dmMaze/BallonsTranslator #1178, commit `6649de1`. | `ui/mainwindow.py`, `docs/UPSTREAM_BALLONSTRANSLATOR_DEV_SYNC.md` |

### Progress audit update for this pass

| Capability | Status | Newly implemented | Deferred with reason |
| --- | --- | --- | --- |
| Advanced text rendering / formatting | Implemented / advanced | Typography cleanup now precedes smart fit and project QA fixes; it handles CJK/RTL mode, vertical punctuation, balanced/kinsoku wrapping, padding, and fallback repair. | True HarfBuzz/OpenType shaping and real glyph ink bounds remain deferred because they need optional dependency and visual-regression validation. |
| Layout review agent | Implemented / advanced | Review snapshots include `typography_cleanup`; planner emits `polish_typography`; scene manager applies it before smart fit/resize. | External LLM review quality still depends on provider settings and model behavior. |
| Settings/config polish | Implemented / advanced | Persistent auto-polish setting wired in config panel and honored only for run-created/newly detected boxes. | Full onboarding wizard for renderer presets remains later. |
| Automation/API/headless | Implemented / advanced | Added `project_status` and `polish_typography` API routes. | Event streaming/progress websockets remain deferred. |
| Upstream BallonsTranslator sync | Active | Reviewed recent dev commits and ported safe save-state fix `6649de1`. | Dependency pins/provider/runtime commits deferred to avoid conflicts with Pro-specific module ranges and settings. |

## Deferred high-value research items

- Koharu #558/#454 native editable PSD text fidelity: high value but requires PSD writer validation.
- Koharu #555 global structured OCR with speaker assignment: useful follow-up to existing structured OCR/API.
- Koharu #610/#515 parent-child CBZ batch queues: requires queue/progress/cancel design.
- Upstream #1177/#1179 dependency pins: reviewed, deferred until Pro OCR/LLM module compatibility matrix is tested.
- Upstream #1167 Ollama OCR/translation and #1165 per-block translation: deferred for provider/settings integration pass.



## Latest implementation pass (2026-05-08 continued: lettering proof packs and API health)

| Area | Implemented | Research source | Files |
| --- | --- | --- | --- |
| Vertical CJK proof metrics | Added `vertical_layout_cells` and `lettering_proof_metrics` so QA/export manifests can show top-to-bottom/right-to-left glyph placement, punctuation rotation/hanging hints, overflow pixels, clearance, density, and recommended actions. | Koharu #624 advanced typesetting, #117 word splitting, #454/#558 export text fidelity. | `utils/text_rendering.py`, `utils/rendering_qa.py`, `tests/test_text_rendering.py` |
| Lettering proof export workflow | Added a current-page proof pack containing QA JSON/Markdown, editable SVG, PSD-helper manifest/layers when available, final composite reference, warnings, and next actions. Wired it to canvas **Review / QA → Export lettering proof pack** and local API. | Koharu #558 PSD text editability, #555 structured OCR/export handoff, #649 global font review. | `utils/lettering_proof_export.py`, `utils/svg_text_export.py`, `utils/layered_psd_export.py`, `ui/canvas.py`, `ui/mainwindow.py`, `tests/test_lettering_proof_export.py` |
| Automation discovery | Added local API `GET /health` and `GET /routes` so headless scripts can discover commands before running batch workflows. | Koharu #612 API workflow / MCP-style automation, #610 parent-child batch workflows. | `utils/local_automation_api.py`, `tests/test_local_automation_api.py` |
| Proof review UX follow-up | Added `lettering_proof_index.html` to proof packs and richer route metadata (`count` + method map) to API discovery so reviewers and scripts do not have to inspect raw JSON first. | Koharu #558/#555 export handoff, #612 API workflow. | `utils/lettering_proof_export.py`, `utils/local_automation_api.py`, `docs/RENDERING_TEXT_FORMATTING.md` |
| Upstream dev review | Re-reviewed `dmMaze/BallonsTranslator` dev commits through `6649de1..8577eaf`; previously ported stop/progress/undo-save-state fixes remain present, and dependency/provider commits stay deferred with reasons. | Upstream #1104, #1178, #1177/#1179, #1167. | `docs/UPSTREAM_BALLONSTRANSLATOR_DEV_SYNC.md` |

### New deferred items

- Native PSD text-layer writing is still deferred; proof packs preserve editable metadata and explicit warnings instead.
- Full OpenType shaping/HarfBuzz is still deferred; proof metrics are renderer-neutral diagnostics for current PyQt rendering/export paths.

<!-- Previous audit history below. -->

## Previous audit history

_Last audited: 2026-05-14 against BallonsTranslator-Pro current branch and public `mayocream/koharu` issues via GitHub REST API (660 all-state issues/PRs scanned, including #669 through #1)._

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

## Latest implementation pass (2026-05-08: mask-effective fitting and complete-page export)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Mask-effective fitting | Extended mask-safe diagnostics into effective fitting bounds so Typography QA can detect when text fits the full textbox but overflows the visible mask-safe area. QA rows now expose `mask_effective_box`, suggest `shrink_to_mask_safe_area`, and project fixes can shrink font size against that masked safe area. | #637 mask-aware collision/squeezing, #630 centering/fitting inconsistencies, #545 restricted render area / DPI clipping, #583 auto font-size overgrowth | `utils/text_masking.py`, `utils/rendering_qa.py`, `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `tests/test_text_rendering.py` |
| Live lettering diagnostics | The text formatting panel now includes mask coverage and effective safe-area size in its live lettering diagnostics, so users see mask-caused review status without opening the QA dialog. | #637 mask-aware fitting, #594 formatting diagnostics, #648 fewer manual style passes | `ui/text_panel.py`, `utils/text_masking.py` |
| Complete-page export fallback | Batch export can include pages without rendered results by falling back to inpainted/clean pages and then originals, preserving page count for ZIP/CBZ/PDF handoff and recording fallback sources in the manifest. The option is available in Settings, the export dialog, and local automation (`include_unrendered`). | #568 export skips pages without detections, #535 custom export/include unrendered pages, #626/#610 archive workflow, #541/#530 large batch reliability | `utils/config.py`, `ui/configpanel.py`, `ui/export_dialog.py`, `ui/mainwindow.py`, `utils/export_manifest.py`, `tests/test_export_manifest.py` |

### Progress audit update for this pass

| Capability | Status | Newly implemented | Deferred reason |
| --- | --- | --- | --- |
| Text fitting / overflow | Implemented / advanced | Mask-safe rectangles now become effective fit boxes for QA/review/fixes, not just warnings. | Full non-rectangular glyph flow and bubble contour collision still require renderer-level visual layout work. |
| Layout review agent | Implemented / advanced | Review snapshots carry `mask_effective_box`; heuristic review proposes shrink/padding actions for masked-safe overflow. | External provider behavior still depends on user settings/API quality. |
| Text formatting UI | Implemented / advanced | Live lettering diagnostics now report mask coverage and safe-area dimensions alongside fit, quality, mode, and missing glyphs. | Thumbnail/side-by-side render preview remains a later UI pass. |
| Export workflow | Implemented / advanced | Export no longer has to skip pages without rendered results when users enable fallback; manifest identifies rendered vs clean/original fallback sources. | True streaming ZIP/CBZ progress/cancel remains deferred. |


## Latest implementation pass (2026-05-07 continued: mask-safe lettering and overflow repair)

| Area | Implemented | Koharu issue inspiration | Files |
| --- | --- | --- | --- |
| Mask-safe lettering diagnostics | Added a reusable text-mask analyzer that reports visible coverage, mask-safe rectangle/insets, fully-masked boxes, narrow safe areas, and recommended padding. Typography QA and API issue listings now surface `mask_safe_area` warnings before final render/export. | #637 mask-aware collision/squeezing, #630 centering/fitting inconsistencies, #636 textbox movement/centering pain | `utils/text_masking.py`, `utils/rendering_qa.py`, `tests/test_text_rendering.py` |
| Renderer overlay and selected-box repair | Renderer diagnostics overlay now draws the measured text bounds plus the mask-safe rectangle, and the canvas context menu adds **Format → Apply mask-safe padding** for selected text boxes. The action increases persisted `FontFormat.text_padding`, updates the scene item, and posts status feedback. | #637 mask-aware fitting, #630 render centering, #648 fewer manual project/style adjustments | `ui/textitem.py`, `ui/canvas.py`, `ui/scenetext_manager.py` |
| Layout review mask awareness | Layout review snapshots include mask coverage, safe insets, and warnings; the heuristic planner proposes `increase_padding` and `recenter` actions for clipped/narrow mask areas, and project rendering fixes apply the recommended mask padding in batch/headless workflows. | #649 global text repair, #648 project-wide settings, #612 automation workflow | `utils/layout_review_agent.py`, `ui/scenetext_manager.py`, `utils/rendering_qa.py` |

### Progress audit update

| Capability | Status | Evidence / file references | Deferred reason |
| --- | --- | --- | --- |
| Text box overflow and effect-aware bounds | Implemented / improving | Effect-aware fit diagnostics, recommended box sizes, ink clip risk, and mask-safe warnings now feed QA/review. | True glyph ink bounds still require optional shaping/metrics work. |
| Mask-aware final lettering | Partially implemented → advanced | Text eraser masks now produce safe-area diagnostics, overlay visualization, selected repair, layout-review actions, and batch QA fixes. | Full non-rectangular text flow/squeezing around bubble masks remains large visual-layout work. |
| Selected textbox review | Implemented / verified | Review snapshots now carry writing mode, style, overflow, fallback, measured bounds, and mask-safe fields. | External model quality depends on user provider settings. |
| Whole-page/project workflow | Implemented / improving | Project Typography QA can find/apply mask-safe padding fixes, and local API issue listing includes the diagnostics. | Row thumbnails and cancelable streaming progress remain deferred. |
| PSD/export handoff | Partially implemented | Existing helper-layer/export manifest workflow is retained; mask QA is available before export to avoid silent clipped lettering. | Native editable PSD text layers and XCF writer remain deferred. |

### Deferred high-value issue-inspired requests

- #637 full mask-aware squeezing/collision remains deferred because safe production behavior needs a real irregular-shape line layout and visual regression images, not only rectangular insets.
- #587 native editable PSD text fidelity remains deferred because the current handoff path is safer than writing partially-compatible PSD text layers.
- #625/#652 provider/model/runtime diagnostics remain next-batch candidates; this pass prioritized text boxes, masking, overflow, and layout review repair.


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
