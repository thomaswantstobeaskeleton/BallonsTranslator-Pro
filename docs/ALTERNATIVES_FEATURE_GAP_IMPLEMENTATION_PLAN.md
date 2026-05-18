# Alternatives Feature Gap Implementation Plan

Last updated: 2026-05-18

This document is the safety baseline for closing gaps versus manga-image-translator, Koharu, ImageTrans, manga-translator-ui, and focused cleanup tools. The rule for every phase is: audit first, extend existing Pro systems, do not duplicate equivalent functionality, and do not remove existing Pro behavior.

## Safety and PR checklist baseline

Every implementation PR in this roadmap must verify:

- README.md and README_zh_CN.md parity for user-facing changes.
- Startup entrypoints are unchanged or explicitly verified (`launch.py`, `launch_win.bat`, `launcher.bat`, `launch_win_with_autoupdate.bat`).
- First-run model picker behavior remains compatible when startup/model code is touched.
- Local automation route discovery still reports existing and new routes through `GET /routes`.
- Export/proof-pack behavior and manifests remain backward compatible.
- Project save-state behavior remains compatible with existing project JSON.
- Manual UX validation path is documented in PR notes.
- Risks and rollback notes are included, especially for optional dependencies, PSD editability, renderer shaping, and config migration.

## Phase 0 — Audit and safety baseline

| Feature/check | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Audit plan | Partially implemented → updated here | `docs/FEATURE_PARITY_MATRIX.md` existed, but this detailed per-phase plan was missing | `docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md`, `docs/FEATURE_PARITY_MATRIX.md` | Low | Documentation review |
| Local automation API route discovery | Partially implemented | `utils/local_automation_api.py`, `tests/test_local_automation_api.py` | `utils/local_automation_api.py`, `utils/automation_api_contract.py`, `docs/LOCAL_AUTOMATION_API.md` | Low; JSON shape now includes `events`, `mcp_routes`, `job_routes`, `event_stream` | Route sorting, auth, health, `/routes`, SSE snapshot |
| Export/proof-pack behavior | Already implemented baseline | `utils/lettering_proof_export.py`, `utils/export_manifest.py`, `tests/test_lettering_proof_export.py`, `tests/test_export_manifest.py` | Same plus export callers | Medium when adding formats | Manifest correctness, proof pack contents, fallback source warnings |
| Project save-state behavior | Partially implemented | `ui/mainwindow.py` API edit path marks save state; project tests exist | `utils/proj_imgtrans.py`, `ui/mainwindow.py` | Medium; project JSON compatibility is critical | Save/load with new fields absent/present; undoable API edits |
| Startup entrypoints | Already implemented; untouched in this PR | `launch.py`, Windows batch files | Same | High if touched | Compile/argument smoke, Windows script snapshot |
| First-run model picker | Already implemented; untouched in this PR | model picker tests and startup translations | `ui/model_manager_dialog.py`, `launch.py`, `translate/startup_model_ui.*.json` | High if touched | Existing model package selector/defaulting tests |

## Phase 1 — Automation, headless, and MCP parity

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Stable JSON scene operation API | Partially implemented | `utils/api_edit_ops.py`, `ui/mainwindow.py` `_api_apply_edit`, `tests/test_api_edit_ops*.py` | Extend edit-op dataclasses, stable block IDs, undo command integration | Medium; current UI edit path mutates blocks directly for some ops | Payload validation, stable IDs, undo/redo, invalid page/index errors |
| Job manager | Partially implemented | `ui/mainwindow.py` job routes; new contract helper normalizes task aliases | Extract/extend job lifecycle helpers without duplicating UI behavior | Medium; cancellation is cooperative | job_start/status/cancel/logs/result/list; SSE snapshot; warning propagation |
| MCP-compatible command surface | Partially implemented | handlers include `open_project`, `project_status`, `list_pages`, `apply_edit`, `run_pipeline`, `render`, `export`, `undo`, `redo`; route discovery now marks MCP routes | `ui/mainwindow.py`, `utils/automation_api_contract.py`, docs | Low | Route discovery contract tests |
| CLI/headless mode | Partially implemented | `launch.py --headless`, `scripts/batch_translate.py` | Add stable CLI contract and exit codes | Medium; must not break current GUI launch | Headless open/run/export smoke with mocked modules |
| Route discovery docs/tests | Partially implemented → improved | `docs/LOCAL_AUTOMATION_API.md`, tests | Same | Low | Authenticated and unauthenticated discovery; `/events` content type |

## Phase 2 — Secure provider/model setup wizard

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| OS keyring credential storage | Partially implemented | `utils/credential_store.py`, `tests/test_credential_store.py` | Config migration path, provider settings UI/API | High; must not silently overwrite config or leak keys | keyring mocks, fallback opt-in, no-secret serialization |
| Provider setup wizard | Not implemented as full wizard | Provider modules/settings exist, but no unified onboarding wizard | `ui/configpanel.py`, provider utilities, local API | Medium | connection test mocks, endpoint preset serialization |
| Runtime profiles | Partially implemented | runtime/profile/data-path tests exist | config/runtime utilities, model manager UI | Medium | low VRAM/CPU fallback diagnostics |
| Translation options/retry policy | Partially implemented | translator batch mode tests exist | base translator/provider settings | Medium | per-block/page mode, retry behavior |

## Phase 3 — Professional export and interchange

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Truthful editable PSD strategy | Partially implemented | `utils/layered_psd_export.py`, `tests/test_layered_psd_export.py` | PSD handoff docs/manifests; optional PSD text only behind flag if feasible | Medium/high due PSD compatibility claims | no silent raster-only PSD claims |
| XLIFF/Excel/Word/LabelPlus/XML/HTML/searchable PDF/TXT/JSON | Partially implemented | structured OCR, SVG/PSD/proof exports exist | new interchange utils and export dialog/API | Medium | format snapshots and optional dependency skips |
| Roundtrip imports | Not implemented broadly | project ops protocol can update text | import utilities/API/UI | High; geometry/block matching can corrupt projects | stable ID/source/page/geometry matching tests |
| Export manifests | Already/partially implemented | `utils/export_manifest.py` | add renderer/pro format warnings | Low/medium | fallback source, missing font, clipped text warnings |

## Phase 4 — CAT-tool features and translation QA

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Translation memory | Not implemented as reusable TM | Some context/glossary docs exist | new TM storage utilities/UI/API | Medium | fuzzy match, import/export |
| Termbase/glossary | Partially implemented | `modules/llm_quality.py` glossary enforcement | glossary store/dialog/API | Medium | hard/soft violations, no destructive replacement |
| Corpus concordance | Not implemented | Search widgets exist but not CAT concordance | corpus index/search utilities | Low/medium | provenance search |
| SFX dictionary | Not implemented | style/prompt concepts exist | SFX dictionary data/UI | Low | lookup/import/export |
| Auto glossary extraction | Not implemented | LLM quality utilities can be extended | extraction utility + approval UI | Medium | candidate categories, approval persistence |
| Post-translation QA/retry | Partially implemented | `modules/llm_quality.py`, translation review tests | QA report/API/UI | Medium | thresholds, retry behavior |
| Prompt profiles | Partially implemented | prompt/profile docs and translator settings | prompt profile store | Low/medium | profile selection by block type |

## Phase 5 — OCR, reading order, and editor UX

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| OCR crop inspector | Partially implemented | `ui/ocr_crop_inspector_widget.py` and pipeline button exist | inspector rerun hooks | Medium | crop extraction/rerun metadata |
| Hybrid OCR workflow | Not implemented | multiple OCR engines exist | OCR compare utility/UI/API | Medium | primary/secondary compare |
| Reading-order editor | Partially implemented | `ui/reading_order_editor_dialog.py` exists | persistence/export integration | Medium | reorder persistence, structured export order |
| Batch find/replace | Partially implemented | regex profiles and global replace exist | preview/apply API and UI | Medium | regex preview/apply undo |
| Import translated image workflow | Not implemented | OCR/project utilities exist | alignment utility/UI/API | High | IoU/order matching tests |

## Phase 6 — Advanced renderer fidelity

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Renderer stack audit | Partially implemented | rendering docs/tests exist | text rendering docs and diagnostics | Low | doc + fixture review |
| Optional shaping backend | Not implemented fully | Qt rendering and text rendering utilities exist | optional uharfbuzz/freetype path | High; packaging/Windows risk | optional dependency skip tests |
| Keep Qt default | Already required | current renderer is Qt-based | config/render settings | Low | default config compatibility |
| Diagnostics | Partially implemented | rendering QA tests exist | renderer diagnostics/export manifests | Medium | missing glyph/clipping/shaping fallback |
| Visual regression tests | Partially implemented | text rendering tests exist | deterministic fixtures | Medium | image snapshots/tolerances |

## Phase 7 — Cleanup, segmentation, and mask quality

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Bubble-aware mask expansion | Partially implemented | mask diagnostics/textbox masking tests | mask generation utilities | Medium | inside/outside dilation fixtures |
| Edge-halo detector | Partially implemented | `tests/test_mask_diagnostics.py` | diagnostics utility/UI/API | Medium | halo fixtures |
| Mask-aware text flow | Partially implemented | auto layout/mask-safe diagnostics | text layout/rendering utilities | High | irregular flow fallback tests |
| Cleanup-only workflow | Partially implemented | inpaint/export modules; no stable CLI/API contract yet | API/headless/export | Medium | cleanup-only CLI/API path |

## Phase 8 — Batch, archive, and packaging parity

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Parent/child CBZ/folder processing | Partially implemented | `utils/zip_batch.py`, batch queue UI, headless dirs | batch queue/headless API | Medium | parent/child queue, resume state |
| Streaming ZIP/CBZ export | Partially implemented | API archive export exists; manifests exist | export utilities/job progress | Medium | progress/cancel/manifest |
| Docker/web/API mode docs | Not implemented fully | local API exists | docs, sample client | Low | docs smoke snippets |
| Data path manager | Partially implemented | `tests/test_data_path_manager.py` | config/UI migration helper | Medium | free-space and migration tests |

## Immediate implementation order

1. Finish Phase 1 API contract: stable IDs/errors for scene ops, deeper job cancellation checkpoints, and headless CLI exit-code contract.
2. Start Phase 2 secret migration using `utils/credential_store.py` and explicit insecure fallback warnings.
3. Expand Phase 3 interchange only after export/import schemas are documented and tests are in place.
