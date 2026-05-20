# Alternatives Feature Gap Implementation Plan

Last updated: 2026-05-19 (Phase 0 baseline audit pass 4)

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


## Phase 0 verification run (this pass)

Commands executed for baseline verification:

- `pytest -q tests/test_local_automation_api.py tests/test_local_automation_api_routes_contract.py tests/test_lettering_proof_export.py tests/test_export_manifest.py tests/test_project_ops_protocol.py tests/test_batch_parent_queue.py tests/test_model_package_selector_dialog.py`

Observed results:

- Route discovery, proof-pack/export manifest, project operation protocol, parent-batch save-state, and model-picker dialog tests all pass in this environment (`18 passed`).
- No startup scripts or first-run behavior code was changed in this pass; verification remains required for future startup/model-manager slices.

## Phase 1 — Automation, headless, and MCP parity

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Stable JSON scene operation API | Partially implemented | `utils/api_edit_ops.py`, `ui/mainwindow.py` `_api_apply_edit`, `tests/test_api_edit_ops*.py` | Extended dataclass validation with index-or-block_id targeting and detailed error payloads; next: full undo-command integration | Medium; current UI edit path mutates blocks directly for some ops | Payload validation, stable IDs, undo/redo, invalid page/index errors |
| Job manager | Partially implemented | `ui/mainwindow.py` job routes; new contract helper normalizes task aliases | Extract/extend job lifecycle helpers without duplicating UI behavior | Medium; cancellation is cooperative | job_start/status/cancel/logs/result/list; SSE snapshot; warning propagation |
| MCP-compatible command surface | Partially implemented | handlers include `open_project`, `project_status`, `list_pages`, `apply_edit`, `run_pipeline`, `render`, `export`, `undo`, `redo`; route discovery now marks MCP routes | `ui/mainwindow.py`, `utils/automation_api_contract.py`, docs | Low | Route discovery contract tests |
| CLI/headless mode | Partially implemented | `launch.py --headless`, `scripts/batch_translate.py` | `scripts/batch_translate.py` now returns stable headless exit codes and optional summary JSON; next: open/run/export unified CLI stages | Medium; must not break current GUI launch | Headless open/run/export smoke with mocked modules |
| Route discovery docs/tests | Partially implemented → improved | `docs/LOCAL_AUTOMATION_API.md`, tests | Same | Low | Authenticated and unauthenticated discovery; `/events` content type |

## Phase 2 — Secure provider/model setup wizard

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| OS keyring credential storage | Partially implemented → advanced | `utils/credential_store.py`, `utils/secret_migration.py`, `tests/test_credential_store.py`, `tests/test_secret_migration.py` | Config migration path + layout-review secret read/write UI path | High; must not silently overwrite config or leak keys | keyring mocks, migration clear/scrub tests, no-secret serialization |
| Provider setup wizard | Partially implemented → advanced | Layout-review provider onboarding now includes endpoint presets + test connection path; full multi-provider unified wizard still pending | `ui/mainwindow.py`, provider utilities, local API | Medium | connection-test mocks, endpoint preset serialization, keyring-backed credential checks |
| Runtime profiles | Partially implemented | runtime/profile/data-path tests exist | config/runtime utilities, model manager UI | Medium | low VRAM/CPU fallback diagnostics |
| Translation options/retry policy | Partially implemented | translator batch mode tests exist | base translator/provider settings | Medium | per-block/page mode, retry behavior |

## Phase 3 — Professional export and interchange

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Truthful editable PSD strategy | Partially implemented | `utils/layered_psd_export.py`, `tests/test_layered_psd_export.py` | PSD handoff docs/manifests; optional PSD text only behind flag if feasible | Medium/high due PSD compatibility claims | no silent raster-only PSD claims |
| XLIFF/Excel/Word/LabelPlus/XML/HTML/searchable PDF/TXT/JSON | Partially implemented → advanced | Added XLIFF export/import paths (UI + API + tests) on top of existing structured OCR/SVG/PSD/proof exports | `utils/xliff_interchange.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py` | Medium | roundtrip snapshots, mismatch reporting, optional dependency skips for other formats |
| Roundtrip imports | Partially implemented → advanced | project ops protocol can update text; XLIFF + translation JSON + translation CSV roundtrip now available | `utils/xliff_interchange.py`, `utils/translation_json_interchange.py`, import utilities/API/UI | High; geometry/block matching can corrupt projects | stable ID/source/page/geometry matching tests |
| Export manifests | Already/partially implemented | `utils/export_manifest.py` | add renderer/pro format warnings | Low/medium | fallback source, missing font, clipped text warnings |

## Phase 4 — CAT-tool features and translation QA

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Translation memory | Partially implemented → advanced | Added project TM store + fuzzy query + import/export API paths/tests (JSON) | `utils/translation_memory.py`, `utils/proj_imgtrans.py`, `ui/mainwindow.py` | Medium | fuzzy match tests, TM import/export tests, save/load compatibility tests |
| Termbase/glossary | Partially implemented → advanced | Added glossary CSV import/export helpers and API endpoints with merge/replace mode | `utils/glossary_io.py`, `ui/mainwindow.py` | Medium | CSV roundtrip tests, merge/replace behavior tests |
| Corpus concordance | Partially implemented → advanced | Added project concordance query utility + API/UI surface with page provenance output | `utils/translation_concordance.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py` | Low/medium | provenance query tests + API tests |
| SFX dictionary | Not implemented | style/prompt concepts exist | SFX dictionary data/UI | Low | lookup/import/export |
| Auto glossary extraction | Not implemented | LLM quality utilities can be extended | extraction utility + approval UI | Medium | candidate categories, approval persistence |
| Post-translation QA/retry | Partially implemented → advanced | Added profile-aware QA report + retry candidate API (`translation_qa_report`) | `utils/translation_qa_profiles.py`, `ui/mainwindow.py` | Medium | threshold/retry tests, profile behavior tests |
| Prompt profiles | Partially implemented → advanced | Added prompt profile registry + API exposure (`translation_prompt_profiles`) with default config selection | `utils/translation_qa_profiles.py`, `utils/config.py`, `ui/mainwindow.py` | Low/medium | profile selection tests and fallback tests |

## Phase 5 — OCR, reading order, and editor UX

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| OCR crop inspector | Partially implemented → advanced | Added per-block crop inspector rerun flow (engine selectable) in UI + API (`ocr_rerun_block`) | `ui/ocr_crop_inspector_widget.py`, `ui/mainwindow.py` | Medium | rerun metadata + index validation tests |
| Hybrid OCR workflow | Partially implemented → advanced | Added per-block OCR compare + apply flow (primary/secondary engine) in inspector UI and automation API | `ui/ocr_crop_inspector_widget.py`, `ui/mainwindow.py` | Medium | compare/apply API tests + persistence checks |
| Reading-order editor | Partially implemented | `ui/reading_order_editor_dialog.py` exists | persistence/export integration | Medium | reorder persistence, structured export order |
| Batch find/replace | Partially implemented → advanced | Added preview/apply batch find-replace API + project UI action with confirmation | `utils/batch_find_replace.py`, `ui/mainwindow.py`, `ui/mainwindowbars.py` | Medium | regex preview/apply tests, undo-path tests |
| Import translated image workflow | Partially implemented → advanced | Added translated-image IoU alignment utility and local API route (`import_translated_image_align`) to map OCR text back to raw blocks | `utils/translated_image_alignment.py`, `ui/mainwindow.py` | High | IoU alignment unit tests + API integration tests |

## Phase 6 — Advanced renderer fidelity

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Renderer stack audit | Partially implemented | rendering docs/tests exist | text rendering docs and diagnostics | Low | doc + fixture review |
| Optional shaping backend | Not implemented fully | Qt rendering and text rendering utilities exist | optional uharfbuzz/freetype path | High; packaging/Windows risk | optional dependency skip tests |
| Keep Qt default | Already required | current renderer is Qt-based | config/render settings | Low | default config compatibility |
| Diagnostics | Partially implemented → advanced | Added renderer capability diagnostics route + export manifest renderer metadata | `utils/renderer_diagnostics.py`, `ui/mainwindow.py`, `utils/export_manifest.py` | Medium | diagnostics shape tests + manifest renderer metadata tests |
| Visual regression tests | Partially implemented | text rendering tests exist | deterministic fixtures | Medium | image snapshots/tolerances |

## Phase 7 — Cleanup, segmentation, and mask quality

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Bubble-aware mask expansion | Partially implemented | mask diagnostics/textbox masking tests | mask generation utilities | Medium | inside/outside dilation fixtures |
| Edge-halo detector | Partially implemented → advanced | Cleanup-only workflow now computes mask edge-halo ratio, applies adaptive mask expansion/merge policy, and flags risky pages via API warnings/metadata | `modules/mask_diagnostics.py`, `utils/cleanup_only_workflow.py`, `ui/mainwindow.py` | Medium | halo-threshold cleanup workflow tests |
| Mask-aware text flow | Partially implemented | auto layout/mask-safe diagnostics | text layout/rendering utilities | High | irregular flow fallback tests |
| Cleanup-only workflow | Partially implemented → advanced | Added stable cleanup-only API route (`cleanup_only`) that runs detect+inpaint and optional clean export directory output | `utils/cleanup_only_workflow.py`, `ui/mainwindow.py` | Medium | cleanup-only workflow tests + route validation |

## Phase 8 — Batch, archive, and packaging parity

| Required feature | Status | Existing evidence | Target files | Migration risk | Tests to add/keep |
|---|---|---|---|---|---|
| Parent/child CBZ/folder processing | Partially implemented → advanced | Added parent-batch enumerate/save-state API for nested dirs + CBZ/ZIP child discovery and resumable status JSON | `utils/batch_parent_queue.py`, `ui/mainwindow.py` | Medium | parent/child enumerate + state persistence tests + status update/next-pending tests |
| Streaming ZIP/CBZ export | Partially implemented | API archive export exists; manifests exist | export utilities/job progress | Medium | progress/cancel/manifest |
| Docker/web/API mode docs | Partially implemented → advanced | Added `server_mode_info` API route and docs quickstart with curl/Python snippets and mount hints | `utils/server_mode_info.py`, `ui/mainwindow.py`, `docs/LOCAL_AUTOMATION_API.md`, `README.md`, `README_zh_CN.md` | Low | server-mode info shape tests + docs smoke snippets |
| Data path manager | Partially implemented → advanced | Added automation API routes for path status/free-space checks and dry-run/apply migration helper | `utils/data_path_manager.py`, `ui/mainwindow.py` | Medium | migration dry-run/apply tests + status threshold tests |

## Immediate implementation order

1. Finish Phase 1 API contract: stable IDs/errors for scene ops, deeper job cancellation checkpoints, and headless CLI exit-code contract.
2. Start Phase 2 secret migration using `utils/credential_store.py` and explicit insecure fallback warnings.
3. Expand Phase 3 interchange only after export/import schemas are documented and tests are in place.


## Phase PR slicing plan (review-sized)

To keep changes reviewable and reduce regression risk, implement each phase in multiple PR-sized slices with explicit rollback boundaries:

1. **Phase 1A**: route contract hardening (`/routes`, `/health`, `/events`) + tests + docs.
2. **Phase 1B**: stable scene edit IDs/errors + undo/redo invariants + API edit tests.
3. **Phase 1C**: job lifecycle and cancellation checkpoints + status/log/result tests.
4. **Phase 1D**: headless CLI contract and exit codes + batch smoke tests.
5. **Phase 2A**: keyring migration and insecure fallback warnings + migration tests.
6. **Phase 2B**: provider setup wizard and endpoint validation/test-connection flows.
7. **Phase 3+**: export/interchange and CAT/OCR/renderer/cleanup/batch features, each split into doc/schema/test-first slices before UI surfacing.

Each slice should include: (a) migration notes, (b) startup/model-picker impact statement, (c) README/README_zh_CN parity check when user-visible behavior changes.

### Pass 4 incremental completion notes

- Added dedicated `GET /mcp/commands` discovery endpoint in local automation server for MCP-only command enumeration.
- Extended route-discovery GET methods contract to include `mcp/commands`.
- Added API tests for the new endpoint and updated existing route-discovery shape assertions.

## 2026-05-20 Phase 0/1 checkpoint
- Realtime mode and Translation Assist were re-audited before implementation updates.
- Existing realtime/API scaffolding was extended (not duplicated) for first milestone route wiring and UI entry points.
- Next implementation slices remain phased to avoid startup/project compatibility regressions.
