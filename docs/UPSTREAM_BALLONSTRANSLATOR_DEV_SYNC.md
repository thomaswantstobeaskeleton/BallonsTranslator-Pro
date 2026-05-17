# Upstream BallonsTranslator Dev Sync

_Refreshed: 2026-05-17. Source: `git fetch upstream-base dev --depth=120`, then `git log --oneline --stat --name-status` review of recent `dmMaze/BallonsTranslator` dev commits. Changes are manually adapted only when safe for BallonsTranslator-Pro._

## Ported/adapted in this pass

- Third follow-up 2026-05-17: no new direct cherry-pick; atomic bubble fit adapts upstream auto-sizing/workflow issue pressure while preserving Pro-specific rendering QA and context-menu architecture.
- Second follow-up 2026-05-17: refreshed `upstream-base/dev` and reviewed current head range `6649de1`..`91f5d13` again. No requirements/provider commit was safe to cherry-pick without Pro module-matrix validation, so this pass adapted upstream issue themes instead: safer fit/export behavior and batch filename workflow without overwriting Pro automation/rendering systems.
- Second follow-up 2026-05-17: rechecked `4c14019` (replace-all/render-all UI freeze/save bug) and kept the direct port deferred because Pro's global-search and batch-export code diverged; Pro-specific export manifest/naming improvements were implemented instead.
- Follow-up 2026-05-17: reviewed `d677d1b`, `1c6643b`, and `36fa134`; Pro already contains the progress-window and continue-mode fixes, while this pass safely advanced related shortcut/workflow polish without cherry-picking incompatible UI code.
- Follow-up 2026-05-17: advanced export workflow inspired by upstream PSD/export requests via multi-page API layered handoff and richer JSX/manifest style metadata.

- Reviewed upstream `6649de1` (`update project save state logic #1178`, changed `ui/mainwindow.py`) and audited adjacent Pro batch/save-stage helpers. Rather than blindly cherry-picking, this pass fixed a Pro-specific stage-restore tuple regression in `on_run_detect_images`: `_run_stages_restore` now stores exactly `(detect, ocr, translate, inpaint)`, preventing pipeline-finished unpack crashes after selected-page detect-only runs.
- Reviewed upstream `4c14019` (replace-all/render-all UI freeze and save regression) again. Direct port remains deferred due Pro's custom global-search/workflow code, but this pass adds QA-enriched automation page listing to reduce batch workflow blind spots.
- Reviewed dependency commits `c80eb81`, `04c3414`, and `88d4969`; deferred requirements changes because Pro's dependency set is intentionally broader and has module compatibility notes.

## Recent dev commit review

| Upstream commit | Summary | Changed files | Category | Relevant to Pro | Already implemented in Pro | Ported this pass | Porting notes / conflicts / deferred reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `6649de1` | update project save state logic #1178 | `ui/mainwindow.py` | Bugfix | Yes | Partial | Yes (adapted) | Manual audit found and fixed a Pro-specific selected-page detect-only `_run_stages_restore` tuple bug without overwriting Pro save/render logic. Exact grayscale inpaint fix remains fixture-deferred. |
| `c80eb81` | fix #1179 | `requirements.txt` | Dependency/platform compatibility | Yes | No | No | Requirements-only PaddleOCRVLManga fix; deferred until Pro optional module matrix confirms no GLM/OCR regression. |
| `04c3414` | pin transformers version, close #1177 | `requirements.txt`, `ui/textedit_area.py` | Dependency/platform compatibility | Yes | Partial | No | Pro intentionally requires newer transformers for several modules; pinning down blindly is unsafe. |
| `a05ef01` | remove deprecated deeplx | `modules/translators/trans_deeplx.py`, `requirements.txt` | OCR/detection/inpainting/translation module update | Maybe | No | No | Deferred until Pro translator list compatibility and migration UX are audited. |
| `88d4969` | add missing gguf dependency, close #1175 | `requirements.txt` | Dependency/platform compatibility | Yes | Unknown | No | Deferred until an enabled Pro module requires `gguf` by default. |
| `640c4bf` | add reference to Ballonstranslator-Pro, close #1151 | `README.md`, `README_EN.md` | Documentation only | Low | N/A | No | Upstream-only docs change. |
| `e1c93f2` | remove redundant code | `utils/torch_utils.py` | Not relevant / already superseded by Pro | Maybe | Unknown | No | Broad torch utility deletion is unsafe for Pro's custom GPU/runtime support. |
| `485bbe8` | add flux inpaint pipeline #1171 (#1173) | `launch.py`, `modules/base.py`, `modules/inpaint/base.py`, `modules/inpaint/flux_inpaint_pipeline.py`, `requirements.txt`, `utils/imgproc_utils.py`, `utils/torch_utils.py` | OCR/detection/inpainting/translation module update | Yes | Partial/custom | No | Large module addition deferred for VRAM/platform testing and Pro settings integration. |
| `a390d4c` | remove deprecated requirement keyboard | `requirements.txt`, `ui/textedit_area.py`, `utils/structures.py` | Dependency/platform compatibility | Yes | Yes | Previously ported | Pro already adapted by removing hard `keyboard` dependency and using a `pynput`-first shortcut backend. |
| `64a5713` | fix #1172 | `modules/translators/trans_google.py` | Bugfix | Yes | Unknown | No | Translator fix needs conflict audit with Pro custom translator/provider settings. |
| `4c14019` | optimize replace-all/render-all UI lag and page-save bug (#1170) | `ui/global_search_widget.py`, `ui/mainwindow.py` | UI/UX improvement | Yes | Partial/custom | Deferred | Direct patch conflicts with Pro workflow/search additions; batch QA/API improvements advanced separately. |
| `1958f66` | support Ollama as translation/OCR provider (#1167) | `modules/ocr/ocr_llm_api.py`, `modules/translators/trans_llm_api.py` | OCR/detection/inpainting/translation module update | Yes | Partial/custom | No | Pro provider settings are custom; defer to provider setup pass. |
| `10f5e8e` | translate each text block individually (#1165) | `modules/translators/base.py`, `modules/translators/trans_google.py`, `ui/module_manager.py`, `ui/module_parse_widgets.py`, `utils/config.py` | Settings/config | Yes | Partial | No | Pro has selected/batch translation pathways; needs UX merge plan. |
| `ec3ef9d` | low VRAM mode for llm_api (#1161) | `modules/translators/trans_llm_api.py` | Performance/runtime/GPU | Yes | Unknown | No | Defer to model/provider diagnostics pass. |
| `13fc383` | fix stop translation | `ui/module_manager.py` | Bugfix | Yes | Unknown | No | Needs thread lifecycle audit before manual adaptation. |
| `54ed6ae` / `91f5d13` / `00cd512` / `fc609c3` / `5b5c21f` | headless continuous mode series | `launch.py`, `ui/mainwindow.py`, `utils/message.py`, frameless utils, `utils/shared.py` | Export/project workflow | Yes | Partial/custom | No | Pro has local automation API; full continuous headless loop deferred until API/event progress design is consolidated. |
| `16e491c` | close #1144 | `requirements.txt` | Dependency/platform compatibility | Maybe | Unknown | No | Requirements-only; deferred to dependency matrix pass. |
| `09e2e5b` | fix #1133 | `utils/io_utils.py` | Bugfix | Yes | Unknown | No | Needs file-format regression fixtures before port. |

## Next batch candidates

1. Build a dependency/module matrix for `c80eb81`, `04c3414`, `88d4969`, and `16e491c`.
2. Manually inspect `64a5713` Google translator fix against Pro translators.
3. Reconcile `4c14019` replace-all/render-all responsiveness with Pro global search and page-save behavior.
4. Decide whether Pro should expose upstream Ollama OCR/provider settings from `1958f66`.
5. Evaluate low-VRAM LLM mode from `ec3ef9d` with Pro runtime settings.
6. Add fixtures for `09e2e5b` IO behavior.
