# Upstream BallonsTranslator Issue Backlog

_Refreshed: 2026-05-17. Source: GitHub REST API because `gh` is unavailable; scanned 300 all-state issues plus topic searches for bugs/compatibility (238), rendering/text/font/layout (300), shortcuts (25), workflow/export/setup (88), and install/dependency/GPU (259)._

## Implemented or advanced in this pass (2026-05-17)


### Current pass addendum (2026-05-17: export fidelity + dependency compatibility)

- [#1134](https://github.com/dmMaze/BallonsTranslator/issues/1134), [#1169](https://github.com/dmMaze/BallonsTranslator/issues/1169), [#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077), [#1128](https://github.com/dmMaze/BallonsTranslator/issues/1128), and [#995](https://github.com/dmMaze/BallonsTranslator/issues/995) advanced through export/proof parity: SVG and PSD-helper exports now preserve vertical layout plans, punctuation classes, tate-chu-yoko groups, and font-run metadata used by QA/fit logic.
- [#1179](https://github.com/dmMaze/BallonsTranslator/issues/1179) / dev commit `c80eb81` was adapted by raising Pro's `transformers` minimum to `>=4.57.6` instead of pinning exactly, preserving Pro's broader GLM/OCR/HF module compatibility while incorporating the upstream PaddleOCRVLManga compatibility floor.
- [#1020](https://github.com/dmMaze/BallonsTranslator/issues/1020) and [#1052](https://github.com/dmMaze/BallonsTranslator/issues/1052) remain deferred for a full batch queue, but proof packs now create portable ZIP archives that make per-page review/export artifacts easier to attach to batch logs and issue reports.

| Issue | Title | Category | Labels | Maps to Pro | Implemented in Pro | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#1179](https://github.com/dmMaze/BallonsTranslator/issues/1179) | PaddleOCRVLManga execution error | Compatibility/dependency/platform issues | bug, dependency | Yes | Yes/advanced this pass | High | Adapted upstream `c80eb81` by requiring `transformers>=4.57.6` while avoiding an exact pin that could conflict with Pro modules. |
| [#1134](https://github.com/dmMaze/BallonsTranslator/issues/1134) | Export/save image as JPG or PNG without translation | PSD/export/layers | feature request, export | Yes | Partial/advanced | Medium | Existing export paths and proof pack artifacts now include richer final-composite/proof metadata; direct no-translation export already exists through current-page download modes. |
| [#1128](https://github.com/dmMaze/BallonsTranslator/issues/1128) | Vertical Text not working | Vertical CJK / RTL / punctuation | bug, renderer | Yes | Partial/advanced this pass | High | Export/proof paths now serialize explicit vertical glyph placement, punctuation, and tate-chu-yoko metadata. Live renderer work remains ongoing. |
| [#995](https://github.com/dmMaze/BallonsTranslator/issues/995) | Full-width converted to half-width characters | Vertical CJK / RTL / punctuation | bug, text | Yes | Partial/advanced this pass | Medium | Vertical punctuation normalization is preserved in SVG/PSD handoff metadata; broader OCR normalization policy remains deferred. |

- [#1178](https://github.com/dmMaze/BallonsTranslator/issues/1178) / upstream dev `6649de1` review — fixed a Pro workflow regression in the selected-page **Detect only** helper: `_run_stages_restore` is now always the expected four-stage tuple, preventing pipeline-finished restore crashes after detector-only batch actions.
- [#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077), [#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138), [#35](https://github.com/dmMaze/BallonsTranslator/issues/35) — advanced text fitting/effects by adding persisted double-outline/back-stroke support, effect-aware fit margins, text-panel controls, and layout-review actions.
- [#1094](https://github.com/dmMaze/BallonsTranslator/issues/1094), [#1020](https://github.com/dmMaze/BallonsTranslator/issues/1020), [#841](https://github.com/dmMaze/BallonsTranslator/issues/841) — extended automation/page listing to include rendering QA so batch/headless tools can inspect textboxes and warnings before running exports or review.
- [#50](https://github.com/dmMaze/BallonsTranslator/issues/50), [#364](https://github.com/dmMaze/BallonsTranslator/issues/364), [#749](https://github.com/dmMaze/BallonsTranslator/issues/749) — SVG editable-text handoff now preserves back-outline metadata as layered SVG text; native PSD/XCF remains deferred.


- Follow-up 2026-05-17: [#50](https://github.com/dmMaze/BallonsTranslator/issues/50), [#364](https://github.com/dmMaze/BallonsTranslator/issues/364), and [#749](https://github.com/dmMaze/BallonsTranslator/issues/749) advanced by adding secondary-outline metadata to PSD handoff manifests/JSX and supporting multi-page API handoff export.
- Follow-up 2026-05-17: [#1180](https://github.com/dmMaze/BallonsTranslator/issues/1180), [#568](https://github.com/dmMaze/BallonsTranslator/issues/568), and upstream shortcut compatibility work advanced by canonicalizing shortcut conflicts and warning about single-key tool shortcuts while typing.
- Second follow-up 2026-05-17: [#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077), [#995](https://github.com/dmMaze/BallonsTranslator/issues/995), and font/layout bug reports such as [#862](https://github.com/dmMaze/BallonsTranslator/issues/862)/[#972](https://github.com/dmMaze/BallonsTranslator/issues/972) advanced with line-break quality diagnostics, safer RTL expand-to-fill, and a secondary-outline fit-margin regression fix.
- Second follow-up 2026-05-17: [#229](https://github.com/dmMaze/BallonsTranslator/issues/229) export suggestions advanced with Settings/dialog/API batch export filename templates and safe cross-platform filename sanitization.

- Third follow-up 2026-05-17: [#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077) and [#229](https://github.com/dmMaze/BallonsTranslator/issues/229) advanced with Atomic bubble fit plus an automation command for coherent bubble-internal formatting.

## Issue backlog

| Issue | Title | Category | Labels | Maps to Pro | Implemented in Pro | Priority | Notes / deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [#1178](https://github.com/dmMaze/BallonsTranslator/issues/1178) | Inpaint output contains non-grayscale pixels for grayscale input | Bugs/regressions/crashes | bug | Yes | Partial/advanced this pass | High | Upstream dev commit `6649de1` was reviewed; this pass adapted a discovered Pro stage-restore crash in the same workflow/stability area. The exact grayscale inpaint behavior remains deferred pending image fixture validation. |
| [#1179](https://github.com/dmMaze/BallonsTranslator/issues/1179) | PaddleOCRVLManga execution error | OCR/detection/inpainting | bug | Yes | Deferred | High | Upstream `c80eb81` changes requirements only; Pro pins differ and require module matrix validation. |
| [#1177](https://github.com/dmMaze/BallonsTranslator/issues/1177) | failed to set modules | Compatibility/dependency/platform issues | bug | Yes | Deferred | High | Upstream transformer pin conflicts with Pro's GLM/OCR dependency notes; deferred to compatibility pass. |
| [#1175](https://github.com/dmMaze/BallonsTranslator/issues/1175) | missing gguf dependency | Compatibility/dependency/platform issues | inferred | Yes | Deferred | Medium | `gguf` not added until Pro confirms an enabled module imports it by default. |
| [#1172](https://github.com/dmMaze/BallonsTranslator/issues/1172) | Bug Report | Bugs/regressions/crashes | bug | Yes | Deferred | High | Upstream Google translator fix reviewed; Pro translator customizations need conflict audit. |
| [#1181](https://github.com/dmMaze/BallonsTranslator/issues/1181) | Add 4 main module checkboxes in the left panel | UI/UX/editor workflow | none | Yes | Already/partial | Medium | Pro already has stage actions and presets; next pass can improve discoverability. |
| [#1180](https://github.com/dmMaze/BallonsTranslator/issues/1180) | Add spacebar panning in image-editing mode | Settings/config/keybinds | none | Yes | Deferred | Medium | Shortcut/keybinding conflict pass needed to avoid QAction/QShortcut ambiguity. |
| [#1077](https://github.com/dmMaze/BallonsTranslator/issues/1077) | Automatically adjust text size based on textbox dimensions | Text fitting / layout / overflow | none | Yes | Yes/advanced this pass | High | Fit diagnostics now account for second outlines; UI exposes back-outline controls for fitted SFX. |
| [#995](https://github.com/dmMaze/BallonsTranslator/issues/995) | full-width converted to half-width characters | Text rendering / typography / fonts | bug | Yes | Partial/advanced this pass | Medium | Line-break quality/proof metrics now flag poor punctuation wraps; full translator-side punctuation normalization remains deferred. |
| [#229](https://github.com/dmMaze/BallonsTranslator/issues/229) | A selection of bugs and suggestions | Feature requests/enhancements | none | Yes | Partial/advanced this pass | Medium | Export workflow improved with filename templates and safer batch naming; unrelated suggestions remain backlog items. |
| [#1138](https://github.com/dmMaze/BallonsTranslator/issues/1138) | Text on Path | Text rendering / typography / fonts | none | Yes | Yes/advanced this pass | Medium | Existing path text now paints second/back outline through the same effect cache. |
| [#1132](https://github.com/dmMaze/BallonsTranslator/issues/1132) | Latin characters added after original language | Text rendering / typography / fonts | bug | Yes | Partial | Medium | Existing vertical/RTL logic covers part of this; full upstream fix remains monitored. |
| [#1128](https://github.com/dmMaze/BallonsTranslator/issues/1128) | Vertical text rendering request/bug | Vertical CJK / RTL / punctuation | inferred | Yes | Yes/advanced previous passes | High | Continued by effect-aware vertical fit behavior. |
| [#1020](https://github.com/dmMaze/BallonsTranslator/issues/1020) | Batch Processing Queue with Pause/Cancel | Batch/project workflow | none | Yes | Partial/advanced this pass | Medium | QA-enriched API page listing helps batch tools; full pause/cancel queue deferred. |
| [#841](https://github.com/dmMaze/BallonsTranslator/issues/841) | Resume processing from a specific file | Batch/project workflow | none | Yes | Partial/advanced this pass | Medium | API/page-state tooling helps external resume logic; native queue resume deferred. |
| [#1094](https://github.com/dmMaze/BallonsTranslator/issues/1094) | CMD/batch translation automation | Automation/API/headless/MCP | none | Yes | Yes/advanced this pass | Medium | `list_pages(include_rendering_qa=true)` exposes textbox/rendering state for headless orchestration. |
| [#50](https://github.com/dmMaze/BallonsTranslator/issues/50) | Add export to PSD | PSD/export/layers | none | Yes | Partial | High | PSD handoff exists; true editable PSD text remains deferred. |
| [#364](https://github.com/dmMaze/BallonsTranslator/issues/364) | export to photoshop | PSD/export/layers | none | Yes | Partial/advanced this pass | High | SVG handoff now emits a separate back-outline text layer. |
| [#749](https://github.com/dmMaze/BallonsTranslator/issues/749) | Export in XCF format | PSD/export/layers | none | Yes | Deferred | Medium | XCF writer is out of scope until PSD/SVG handoff stabilizes. |
| [#1052](https://github.com/dmMaze/BallonsTranslator/issues/1052) | Print current filename when logging errors | UI/UX/editor workflow | none | Yes | Partial | Medium | Pipeline insights exist; comprehensive per-file error context remains next batch. |
| [#1041](https://github.com/dmMaze/BallonsTranslator/issues/1041) | Save entire project after importing translation text | Export/project workflow | none | Yes | Deferred | Medium | Needs audit of import/save side effects with Pro project compatibility. |

## Deferred high-value upstream items

- Dependency pins (`c80eb81`, `04c3414`, `88d4969`) are deferred until Pro's optional module matrix is checked; blind pins could break GLM/OCR/video modules.
- FLUX inpaint pipeline (`485bbe8`) is deferred because Pro already has custom inpainting settings and needs VRAM/platform validation.
- Shortcut requests (#1180/#568) are deferred to a consolidated keybinding owner/conflict pass.

## Next batch candidates

1. Validate and safely adapt upstream PaddleOCRVLManga/transformers pins.
2. Inspect upstream Google translator fix #1172 against Pro translator modules.
3. Add spacebar panning only after a shortcut ambiguity audit.
4. Add per-file error context to batch pipeline logs/status messages.
5. Import-translation full-save workflow with project compatibility checks.
6. Native PSD editable text writer or Photoshop JSX text reconstruction.
7. Batch pause/cancel/resume queue.
8. Confirm grayscale inpaint output behavior with fixtures.
