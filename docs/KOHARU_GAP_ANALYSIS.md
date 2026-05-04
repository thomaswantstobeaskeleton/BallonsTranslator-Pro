# Koharu Deep-Dive Gap Analysis (for BallonsTranslator Pro)

## Scope
This analysis reviews `mayocream/koharu` architecture and documents concrete implementations we can port or adapt.

## High-value architecture differences

1. **Strict stage pipeline contracts**
   - Koharu isolates `Detect -> OCR -> Inpaint -> LLM Generate -> Render` with explicit stage outputs.
   - Recommendation: formalize per-stage typed payload schemas and add a stage-contract validator between modules.

2. **Multi-model detect stage split**
   - Koharu separates text+bubble detection from segmentation mask generation and font hints.
   - Recommendation: add an optional dual-detector orchestration mode that fuses current detector output with segmentation confidence maps before inpaint.

3. **Scene op/history protocol for automation**
   - Koharu exposes low-level mutation ops (`AddNode`, `UpdateNode`, `Batch`) and undo/redo through HTTP/MCP tools.
   - Recommendation: add a stable JSON op API over current project edits for scriptability and future MCP integration.

4. **MCP-first automation endpoint**
   - Koharu supports background pipeline jobs from MCP and external agents.
   - Recommendation: expose `open_project`, `run_pipeline`, `apply_edit`, `undo`, `redo`, and `export` actions in our local API.

5. **Credential storage separation**
   - Koharu stores API keys in platform credential storage, separate from config.
   - Recommendation: move translator/LLM API keys to OS keyring and keep only base URLs in config.

## Feature opportunities to implement in BallonsTranslator Pro

## 1) Pipeline UX & reliability
- **Per-stage retry policy** (different retries for OCR vs LLM vs downloads).
- **Stage-level warnings panel** with structured machine-readable warning codes.
- **Selective re-run**: rerun from OCR onward without repeating detection/inpaint.
- **Job IDs + event stream** for long runs (desktop and future headless modes).

## 2) Detection, segmentation, and cleanup quality
- **Mask diagnostics view**: overlay raw mask, thresholded mask, and dilated mask.
- **Auto-threshold tuner** using page histogram + confidence percentiles.
- **Bubble-aware cleanup expansion**: dilate differently inside/outside bubble regions.
- **Edge-halo detector** post-inpaint QA to auto-flag pages needing manual touch-up.

## 3) OCR and text editing workflow
- **OCR crop inspector** per text block (image crop + recognized text + confidence).
- **Reading-order editor** with quick reorder shortcuts and visual graph edges.
- **Text normalization module**: punctuation style, half/full-width normalization, ruby stripping presets.
- **Batch find/replace with regex profiles** scoped by page/chapter.

## 4) LLM procedure improvements
- **Prompt profiles per content type** (dialogue, narration, SFX, signboard).
- **Two-pass translation**: draft pass + style-consistency refinement pass.
- **Glossary enforcement mode** with hard/soft constraints and violation report.
- **Back-translation QA** (target->source sanity check for meaning drift).
- **Token-budget planner** that chunks oversized pages deterministically.

## 5) Rendering and typography
- **Font hint fusion**: combine OCR language/script detection + palette sampling + manual presets.
- **Vertical CJK layout presets** (line-gap defaults, punctuation hanging, rotate-latin toggles).
- **PSD export fidelity mode** preserving text layers and effect metadata.
- **Per-bubble style templates** auto-applied based on bubble geometry class.

## 6) Settings & runtime modules
- **Runtime HTTP controls** in UI (connect timeout, read timeout, retries) with restart hint.
- **Engine registry panel** showing active engine IDs and compatibility diagnostics.
- **Provider health status dots** (ready/missing/error/unknown) beside each API provider.
- **Data path manager** with migration helper + free-space check.

## Proposed implementation plan (phased)

### Phase A (2-3 weeks): automation & observability
- Add pipeline job IDs + progress events.
- Add stage warning codes and rerun-from-stage action.
- Add provider status diagnostics and keyring-backed key storage.

### Phase B (3-4 weeks): quality stack
- Add mask diagnostics panel and adaptive thresholding.
- Add OCR inspector + reading-order editor.
- Add glossary enforcement and translation drift checker.

### Phase C (3-4 weeks): advanced rendering
- Add vertical CJK layout preset system.
- Add per-bubble style template auto-apply.
- Improve PSD export metadata fidelity.

## Implementation progress snapshot (updated)

### Completed or substantially implemented
- ✅ **Layout review agent end-to-end wiring**: menu/shortcuts + Pipeline Insights trigger + local automation API route (`layout_review`) all invoke the same review flow.
- ✅ **Theme settings cleanup**: removed legacy Bubbly UI mode and unified light/dark switching so theme controls are no longer duplicated.
- ✅ **Edge-halo detector QA (initial implementation)** in mask diagnostics with edge-ring halo ratio scoring to flag likely cleanup artifacts.
- ✅ **Auto-threshold tuner foundation** in mask diagnostics (Otsu recommendation + live mask fill metrics).
- ✅ **Stage-level warnings panel with structured codes** (Pipeline Insights warnings: `MT_DRIFT`, `RERUN`, `HTTP_RETRY_FAIL`, etc.).
- ✅ **Pipeline job IDs + event stream basics** (job sequence IDs + timeline events emitted from ModuleManager to UI).
- ✅ **Selective stage intent controls** (rerun requests from pipeline panel; currently mapped to full run with stage intent retained in telemetry).
- ✅ **Mask diagnostics view foundation** (raw/thresholded/dilated mask preview + threshold/dilation controls).
- ✅ **OCR crop inspector (initial)** with per-block crop preview, OCR text, translation preview, and confidence listing from Pipeline Insights.
- ✅ **Glossary enforcement mode** with UI editor and postprocess integration.
- ✅ **Back-translation drift QA** with threshold setting and visible warnings.
- ✅ **Token-budget-aware helper utilities** for chunk planning.
- ✅ **Runtime HTTP controls in UI** (timeout/retry) applied to provider-backed review calls.
- ✅ **Data path manager (initial)** with custom data path picker and free-space health indicator in settings.
- ✅ **Engine/provider health surfaces** (provider readiness and engine registry snapshot in Pipeline Insights).
- ✅ **Text normalization + regex replace profiles** (configurable postprocess and reusable profile editor/apply flow).
- ✅ **Vertical CJK render controls (initial pass)**: rotate-latin toggle + punctuation hanging toggle exposed in settings and applied in scene text layout logic.

### In progress / partial
- 🟡 **Dual-detector orchestration**: partial groundwork via detector options/retries; confidence-map fusion policy still pending.
- 🟡 **Automation operation protocol**: introduced a stable ProjectOps schema (`UpdateText`, `Batch`) with undo/redo-ready operation chunks and a full JSON ProjectOps Console (apply/undo/redo/commit) in Pipeline Insights; local localhost automation API routes (`open_project`, `run_pipeline`, `apply_edit`, `undo`, `redo`, `export`) are now implemented as an initial command surface with optional API-key auth, queued UI-thread dispatch, and live queue status in Pipeline Insights.
- 🟡 **Credential storage separation**: keyring-backed storage added for video flow-fixer API keys with config fallback; full translator/LLM migration still pending.
- 🟡 **Reading-order/editor tooling**: helper workflows + OCR crop inspector are in place; a dedicated Reading Order Editor (manual reorder) is now implemented, while graph-edge visualization and auto-order suggestions remain pending.

### Not yet implemented
- ⏳ MCP-first command surface hardening (documented stable contract, richer validation, and external tool docs) for the new localhost routes (auth + queued UI-thread execution are now supported).
- ⏳ Keyring-backed credential storage migration for all translator/LLM keys (beyond current flow-fixer keyring support).
- ⏳ Bubble-aware adaptive cleanup expansion (inside/outside bubble differential dilation policy).
- ⏳ Vertical CJK policy engine and renderer-level punctuation hanging/rotate toggles.
- ⏳ PSD text/effect metadata fidelity mode.

## Notes on source references reviewed
- `docs/en-US/explanation/how-koharu-works.md`
- `docs/en-US/explanation/technical-deep-dive.md`
- `docs/en-US/reference/settings.md`
- `docs/en-US/reference/mcp-tools.md`
- Repository structure under `/tmp/koharu` for modules (`koharu-app`, `koharu-core`, `koharu-renderer`, `koharu-llm`, `koharu-psd`).
